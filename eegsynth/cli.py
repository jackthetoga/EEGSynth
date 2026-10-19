from __future__ import annotations

import argparse
import sys
import time
import webbrowser

from eegsynth.engine import DETECTORS, Engine, channel_quality
from eegsynth.mapping import load_config
from eegsynth.outputs import MidiOutput, UdpOutput, run_bridge
from eegsynth.sources import (
    NUM_CHANNELS,
    SAMPLE_RATE,
    SYNTHETIC_SCRIPT,
    CytonSource,
    SyntheticSource,
)


def make_source(args):
    if args.source == "synthetic":
        return SyntheticSource()
    return CytonSource(port=args.port)


def wait_for_samples(source, n: int) -> None:
    while source.buffer.count < n:
        if source.error:
            raise source.error
        time.sleep(0.05)


def report_contact(engine: Engine, cfg) -> None:
    quality = engine.quality()
    bad = [(cfg.general.labels[i], q) for i, q in enumerate(quality) if q["verdict"] != "ok"]
    if not engine.source.has_accel:
        print("Note: no accelerometer data from the board; motion cancellation is off.")
    if not bad:
        print("Electrode contact: all channels look OK.")
        return
    print("Electrode contact problems (detectors ignore these channels while they persist):")
    for label, q in bad:
        print(f"  {label:4} {q['verdict']:9}  p-p {q['ptp']:.0f} uV, mains hum {q['mains']:.0f} uV")
    if sum(q["verdict"] in ("railed", "no signal") for _, q in bad) >= len(quality) // 2:
        print("  Most channels have no signal: check the SRB2 (reference) and BIAS (ground) ear clips first.")


def cmd_run(args) -> None:
    cfg = load_config(args.config)
    only = None
    if args.only:
        only = {s.strip() for s in args.only.split(",")}
        unknown = only - set(DETECTORS) - {"bands"}
        if unknown:
            sys.exit(f"--only: unknown {', '.join(unknown)} (choose from {', '.join(DETECTORS)}, bands)")

    output = UdpOutput(args.udp) if args.udp else MidiOutput(args.midi)
    source = make_source(args)

    def log_event(t: float, ev) -> None:
        extra = f"  velocity {ev.velocity}" if ev.on and ev.detector in ("jaw", "eyebrow") else ""
        print(f"[{t:7.1f}s] {ev.label}{extra}" + ("" if ev.on else "  (released)"))

    engine = Engine(cfg, source, output.send, only=only, on_event=log_event)
    gui = None
    if not args.no_gui:
        from eegsynth.server import GuiServer

        gui = GuiServer(args.host, args.gui_port, engine.describe(), on_command=engine.command)

    source.start()
    print(f"Source: {args.source}" + (f" ({source.port})" if args.source == "cyton" else ""))
    if args.source == "synthetic":
        print(f"Synthetic script (repeats every 40 s):\n{SYNTHETIC_SCRIPT}")
    print(f"MIDI:   {output.description}" + (f"  (only: {', '.join(sorted(only))})" if only else ""))
    if gui:
        gui.start()
        print(f"GUI:    {gui.url}")
        if not args.no_browser:
            webbrowser.open(gui.url)

    period = 1.0 / cfg.general.update_hz
    try:
        wait_for_samples(source, 1)
        print("Streaming. Sit still and relaxed for the first few seconds while baselines settle. Ctrl+C to stop.")
        if cfg.enabled["alpha"] and gui:
            print("Alpha: press 'Calibrate (eyes open)' in the GUI; the alpha note starts when calibration succeeds.")
        elif cfg.enabled["alpha"]:
            print("Alpha: calibrating now. Keep your eyes open and head still for 7 seconds.")
            engine.command({"action": "calibrate"})
        next_tick = time.perf_counter()
        contact_report_at = time.perf_counter() + 1.5
        while True:
            if source.error:
                raise source.error
            if contact_report_at and time.perf_counter() > contact_report_at:
                contact_report_at = None
                report_contact(engine, cfg)
            frame = engine.tick()
            if frame and gui:
                gui.publish(frame)
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay < -1:
                next_tick = time.perf_counter()
            time.sleep(max(0.0, delay))
    except KeyboardInterrupt:
        pass
    finally:
        print()
        engine.release_all()
        source.stop()
        output.close()
        if gui:
            gui.stop()
        if isinstance(source, CytonSource) and source.dropped_packets:
            print(f"Dropped packets: {source.dropped_packets}")


def cmd_check(args) -> None:
    """Live per-channel signal quality, for getting electrodes seated."""
    source = make_source(args)
    source.start()
    n = int(SAMPLE_RATE)
    print("Per-channel signal over the last second. Ctrl+C to stop.")
    print("Good EEG: peak-to-peak roughly 10-100 uV, mains hum under ~20 uV, never RAILED or NO SIGNAL.\n")
    try:
        wait_for_samples(source, n)
        while True:
            if source.error:
                raise source.error
            raw = source.latest_eeg(n)
            explain = {
                "railed": "RAILED: input saturated (no contact / reference or ground off)",
                "no signal": "NO SIGNAL: flatlined (no contact / saturated)",
            }
            lines = [
                f"  ch{ch + 1}: p-p {q['ptp']:9.1f} uV   mains hum {q['mains']:7.1f} uV   "
                f"DC {q['dc']:9.0f} uV   {explain.get(q['verdict'], q['verdict'])}\033[K"
                for ch, q in enumerate(channel_quality(raw, SAMPLE_RATE, args.mains))
            ]
            sys.stdout.write("\n".join(lines) + f"\033[{NUM_CHANNELS - 1}A\r")
            sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(f"\033[{NUM_CHANNELS}B\n")
        source.stop()


def cmd_ports(_args) -> None:
    try:
        import mido
    except ImportError:
        sys.exit("Needs the optional deps: pip install 'eegsynth[midi]'")
    print("MIDI outputs:")
    for name in mido.get_output_names():
        print(f"  {name}")


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(prog="eegsynth", description="OpenBCI Cyton EEG -> MIDI")
    sub = p.add_subparsers(dest="command", required=True)

    def add_source_args(sp):
        sp.add_argument("--source", choices=["cyton", "synthetic"], default="cyton")
        sp.add_argument("--port", help="Cyton dongle serial port (auto-detected if omitted)")

    run = sub.add_parser("run", help="Detect gestures/EEG features and send MIDI, with a live GUI")
    add_source_args(run)
    run.add_argument("--config", default="config.toml")
    dest = run.add_mutually_exclusive_group()
    dest.add_argument(
        "--midi", default="EEGSynth",
        help="Existing MIDI output to use (substring match), else a virtual port "
             "with this name is created (default: EEGSynth)",
    )
    dest.add_argument("--udp", metavar="HOST:PORT", help="Send to an `eegsynth bridge` instead")
    run.add_argument(
        "--only", metavar="NAMES",
        help="Send MIDI only from these, comma-separated: "
             f"{', '.join(DETECTORS)}, bands. Use while MIDI-learning in Logic.",
    )
    run.add_argument("--no-gui", action="store_true")
    run.add_argument("--no-browser", action="store_true", help="Don't open the GUI automatically")
    run.add_argument("--gui-port", type=int, default=8765)
    run.add_argument("--host", default="127.0.0.1", help="GUI bind address (0.0.0.0 to view from other devices)")
    run.set_defaults(func=cmd_run)

    check = sub.add_parser("check", help="Show per-channel signal quality in the terminal")
    add_source_args(check)
    check.add_argument("--mains", type=float, default=60.0, help="Mains frequency (50 or 60)")
    check.set_defaults(func=cmd_check)

    bridge = sub.add_parser("bridge", help="Receive UDP from another machine, expose a virtual MIDI port")
    bridge.add_argument("--listen", default="0.0.0.0:9000")
    bridge.add_argument("--name", default="EEGSynth", help="Virtual MIDI port name")
    bridge.add_argument("--verbose", action="store_true", help="Print every message")
    bridge.set_defaults(func=lambda a: run_bridge(a.listen, a.name, a.verbose))

    ports = sub.add_parser("ports", help="List MIDI output ports")
    ports.set_defaults(func=cmd_ports)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, ValueError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
