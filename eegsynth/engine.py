"""Real-time pipeline:

samples -> 0.1 Hz high-pass -> accelerometer motion cancellation (NLMS)
        -> artifact monitor (pops, flatlines, saturation) + motion gate
        -> filters -> detectors -> MIDI + GUI frames."""

from __future__ import annotations

import queue
from typing import Callable

import numpy as np

from eegsynth.artifacts import ArtifactMonitor, MotionProcessor
from eegsynth.detectors import (
    AlphaDetector,
    BlinkDetector,
    Event,
    EyeMovementDetector,
    MuscleDetector,
)
from eegsynth.dsp import Envelope, StreamFilter, bandpass_sos, highpass_sos, notch_sos
from eegsynth.features import BandPowers
from eegsynth.mapping import Config
from eegsynth.sources import (
    ACCEL_ROWS,
    ADS1299_VREF,
    DEFAULT_GAIN,
    NUM_CHANNELS,
    SAMPLE_RATE,
    RingBuffer,
    Source,
)

RAIL_UV = ADS1299_VREF / DEFAULT_GAIN * 1e6
DETECTORS = ["jaw", "eyebrow", "blink", "eyes", "alpha"]
MUTABLE = DETECTORS + ["bands"]
MUSCLE_TRACE_DECIMATION = 5
ACCEL_TRACE_DECIMATION = 10


def note_on(ch: int, note: int, vel: int) -> bytes:
    return bytes([0x90 | (ch - 1), note, vel])


def note_off(ch: int, note: int) -> bytes:
    return bytes([0x80 | (ch - 1), note, 0])


def cc(ch: int, num: int, value: int) -> bytes:
    return bytes([0xB0 | (ch - 1), num, value])


PITCH_CENTER = 8192
PITCH_MAX = 16383


def pitch_bend(ch: int, value: int) -> bytes:
    value = min(max(value, 0), PITCH_MAX)
    return bytes([0xE0 | (ch - 1), value & 0x7F, value >> 7])


def _decimate(x, d: int, phase: int):
    return x[..., (-phase) % d :: d]


class Engine:
    def __init__(
        self,
        cfg: Config,
        source: Source,
        send: Callable[[list[bytes]], None],
        only: set[str] | None = None,
        on_event: Callable[[float, Event], None] | None = None,
    ):
        self.cfg, self.source, self.send = cfg, source, send
        self.on_event = on_event
        self.muted: set[str] = set(MUTABLE) - only if only else set()
        self.commands: queue.Queue[dict] = queue.Queue()
        g = cfg.general
        fs = SAMPLE_RATE
        self.fs = fs
        self.ch = g.midi_channel

        notch = notch_sos(g.mains_hz, fs)
        emg_hi = min(55.0, g.mains_hz - 5.0)
        self.highpass = StreamFilter(highpass_sos(0.1, fs))
        self.display = StreamFilter(highpass_sos(1.0, fs), notch)
        self.veog = StreamFilter(bandpass_sos(0.5, 15.0, fs))
        self.heog = StreamFilter(bandpass_sos(0.1, 8.0, fs))
        self.emg = Envelope(np.vstack([bandpass_sos(30.0, emg_hi, fs, order=4), notch]), 5.0, fs)
        self.lf = Envelope(bandpass_sos(1.0, 10.0, fs), 5.0, fs)

        self.motion = MotionProcessor(cfg.motion, NUM_CHANNELS, fs)
        self.artifacts = ArtifactMonitor(cfg.artifacts, NUM_CHANNELS, fs, RAIL_UV)
        self.muscle = MuscleDetector(cfg.jaw, cfg.eyebrow, NUM_CHANNELS, fs)
        self.blink = BlinkDetector(cfg.blink, fs)
        self.eyes = EyeMovementDetector(cfg.eyes, fs)
        self.alpha = AlphaDetector(cfg.alpha, fs)

        self.window_n = int(g.window_seconds * fs)
        self.cleaned = RingBuffer(NUM_CHANNELS, max(self.window_n, self.alpha.n) * 2)
        self.last_blocked = np.full(NUM_CHANNELS, -10**9)
        self.last_artifact = np.full(NUM_CHANNELS, -10**9)
        self.read_count = 0
        self.n = 0
        self.ticks = 0
        self.last_cc: dict[int, int] = {}
        self.pending_offs: list[tuple[float, int]] = []
        self.held: dict[int, str] = {}  # sounding note -> detector that owns it
        # The alpha drone: one long note whose pitch bend follows alpha strength.
        self.drone_playing = False
        self.bend = PITCH_CENTER
        self.last_bend_sent: int | None = None
        # Motion is often only detectable a moment after its artifact starts,
        # so short events wait out a veto window before being sent.
        self.vetoable: list[tuple[float, Event]] = []
        self.veto_channels = {
            "blink": [c - 1 for c in cfg.blink.channels],
            "eyes": [c - 1 for c in cfg.eyes.channels],
        }

    # ------------------------------------------------------------------ MIDI

    def _allowed(self, name: str) -> bool:
        return name not in self.muted

    def _cc(self, msgs: list[bytes], num: int, unit: float, owner: str) -> None:
        if num < 0 or not self._allowed(owner):
            return
        value = int(round(min(max(unit, 0.0), 1.0) * 127))
        if self.last_cc.get(num) != value:
            self.last_cc[num] = value
            msgs.append(cc(self.ch, num, value))

    def _event(self, msgs: list[bytes], ev: Event, t: float) -> None:
        if self.on_event:
            self.on_event(t, ev)
        if ev.note < 0 or not self._allowed(ev.detector):
            return
        if ev.on:
            msgs.append(note_on(self.ch, ev.note, ev.velocity))
            self.held[ev.note] = ev.detector
            if ev.duration:
                self.pending_offs.append((t + ev.duration, ev.note))
        elif ev.note in self.held:
            msgs.append(note_off(self.ch, ev.note))
            del self.held[ev.note]

    def _release(self, msgs: list[bytes], detector: str | None = None) -> None:
        for note, owner in list(self.held.items()):
            if detector is None or owner == detector:
                msgs.append(note_off(self.ch, note))
                del self.held[note]
        if detector in (None, "alpha"):
            self.drone_playing = False
            self._send_bend(msgs, PITCH_CENTER, force=True)

    def release_all(self) -> None:
        msgs: list[bytes] = []
        self._release(msgs)
        self.send(msgs)

    # ------------------------------------------------------------------ alpha drone

    def _send_bend(self, msgs: list[bytes], value: int, force: bool = False) -> None:
        if force or value != self.last_bend_sent:
            msgs.append(pitch_bend(self.ch, value))
            self.last_bend_sent = value

    def _start_drone(self, msgs: list[bytes]) -> None:
        a = self.cfg.alpha
        if a.note < 0 or not self._allowed("alpha") or not self.cfg.enabled["alpha"]:
            return
        if a.note in self.held:
            msgs.append(note_off(self.ch, a.note))
        self._send_bend(msgs, round(self.bend), force=True)
        msgs.append(note_on(self.ch, a.note, a.velocity))
        self.held[a.note] = "alpha"
        self.drone_playing = True

    def _update_drone(self, msgs: list[bytes], period: float) -> None:
        """Glide the bend toward the alpha target every tick, so the 4 Hz alpha
        updates don't produce audible steps."""
        a = self.cfg.alpha
        target = PITCH_CENTER + self.alpha.strength * a.bend_up * (PITCH_MAX - PITCH_CENTER)
        self.bend += (target - self.bend) * min(1.0, period / a.update_seconds)
        if self.drone_playing and self._allowed("alpha"):
            self._send_bend(msgs, int(round(self.bend)))

    # ------------------------------------------------------------------ GUI commands

    def command(self, cmd: dict) -> None:
        """Thread-safe: queued from the web server, applied on the next tick."""
        self.commands.put(cmd)

    def _apply_commands(self, msgs: list[bytes], now: float) -> None:
        while not self.commands.empty():
            cmd = self.commands.get_nowait()
            action = cmd.get("action")
            if action == "calibrate":
                self.alpha.calibrate(now)
            elif action == "restart_note":
                self._start_drone(msgs)
            elif action == "stop_note":
                self._release(msgs, "alpha")
            elif action == "mute" and cmd.get("target") in MUTABLE:
                target = cmd["target"]
                if cmd.get("muted"):
                    self.muted.add(target)
                    self._release(msgs, target)
                else:
                    self.muted.discard(target)
                    if target == "alpha" and self.alpha.baseline is not None:
                        self._start_drone(msgs)

    # ------------------------------------------------------------------ tick

    def _blocked_reason(self, channels: list[int], moving: bool) -> str:
        if moving:
            return "head movement"
        hits = [
            f"{self.cfg.general.labels[c - 1]} {self.artifacts.reason[c - 1]}"
            for c in channels
            if self.artifacts.reason[c - 1]
        ]
        return ", ".join(hits)

    def tick(self) -> dict | None:
        rows, self.read_count = self.source.buffer.read_since(self.read_count)
        k = rows.shape[1]
        if k == 0:
            return None
        raw, accel = rows[:NUM_CHANNELS], rows[ACCEL_ROWS]
        t0 = self.n / self.fs
        n_before = self.n
        self.n += k
        t_now = self.n / self.fs
        self.ticks += 1
        enabled = self.cfg.enabled
        events: list[Event] = []
        traces: dict[str, list] = {}
        msgs: list[bytes] = []
        self._apply_commands(msgs, t_now)

        hp = self.highpass(raw)
        blocked_ch = self.artifacts.process(raw, hp)
        if self.source.has_accel:
            clean, moving, accel_ref = self.motion.process(hp, accel, blocked_ch)
            traces["accel"] = _decimate(accel_ref, ACCEL_TRACE_DECIMATION, n_before % ACCEL_TRACE_DECIMATION).round(4).T.tolist()
        else:
            clean, moving = hp, np.zeros(k, dtype=bool)
        blocked = blocked_ch | moving[None, :]
        for ch in range(NUM_CHANNELS):
            hits = np.flatnonzero(blocked[ch])
            if len(hits):
                self.last_blocked[ch] = n_before + hits[-1]
            hits = np.flatnonzero(blocked_ch[ch])
            if len(hits):
                self.last_artifact[ch] = n_before + hits[-1]
        self.cleaned.push_block(clean)

        display = self.display(clean)
        env = self.emg(clean)
        lf = self.lf(clean)
        veog = self.veog(clean)
        heog = self.heog(clean)

        if enabled["jaw"] or enabled["eyebrow"]:
            mt = self.muscle.process(t0, env, lf, blocked, events)
            phase = n_before % MUSCLE_TRACE_DECIMATION
            traces["jaw"] = _decimate(np.array(mt["jaw"]), MUSCLE_TRACE_DECIMATION, phase).tolist()
            traces["eyebrow"] = _decimate(np.array(mt["eyebrow"]), MUSCLE_TRACE_DECIMATION, phase).tolist()
            events[:] = [e for e in events if enabled[e.detector]]
        brow_active = self.muscle.brow.active
        if enabled["blink"]:
            idx = [c - 1 for c in self.cfg.blink.channels]
            suppressed = blocked[idx].any(axis=0) | brow_active
            traces["veog"] = self.blink.process(t0, veog, suppressed, events)
        if enabled["eyes"]:
            idx = [c - 1 for c in self.cfg.eyes.channels]
            suppressed = blocked[idx].any(axis=0) | self.blink.busy | brow_active
            traces["heog"] = self.eyes.process(t0, heog, suppressed, events)

        window = self.cleaned.latest(self.window_n)
        bp = None
        if window is not None:
            bp = BandPowers(window, self.fs, self.cfg.general.bands)

        if enabled["alpha"]:
            a = self.alpha
            accel_window = None
            if self.source.has_accel:
                full = self.source.buffer.latest(a.n)
                accel_window = None if full is None else full[ACCEL_ROWS]
            artifact = bool((self.n - self.last_artifact[a.idx] <= a.n).any())
            if a.update(t_now, self.cleaned.latest(a.n), accel_window, artifact):
                traces["alpha"] = [a.gain]
            if a.just_calibrated:
                self._start_drone(msgs)
            self._update_drone(msgs, 1.0 / self.cfg.general.update_hz)

        veto = self.cfg.motion.veto_seconds
        if veto > 0:
            self.vetoable = [
                (t, ev) for t, ev in self.vetoable
                if not blocked[self.veto_channels[ev.detector]].any()
            ]
            ready = [ev for t, ev in self.vetoable if t <= t_now]
            self.vetoable = [(t, ev) for t, ev in self.vetoable if t > t_now]
            self.vetoable += [(t_now + veto, ev) for ev in events if ev.detector in self.veto_channels]
            events = [ev for ev in events if ev.detector not in self.veto_channels] + ready

        for ev in events:
            self._event(msgs, ev, t_now)
        due = [n for t, n in self.pending_offs if t <= t_now]
        self.pending_offs = [(t, n) for t, n in self.pending_offs if t > t_now]
        for n in due:
            if n in self.held:
                msgs.append(note_off(self.ch, n))
                del self.held[n]

        if enabled["jaw"]:
            self._cc(msgs, self.cfg.jaw.cc, self.muscle.jaw.intensity, "jaw")
        if enabled["eyebrow"]:
            self._cc(msgs, self.cfg.eyebrow.cc, self.muscle.brow.intensity, "eyebrow")
        if enabled["alpha"]:
            self._cc(msgs, self.cfg.alpha.cc, self.alpha.strength, "alpha")

        band_levels = {}
        if bp is not None:
            for m in self.cfg.mappings:
                idx = m.channel_indices
                if (self.n - self.last_blocked[idx] > self.window_n).all():
                    m.update(bp.feature(m.feature, idx, m.relative), self.cfg.general)
                self._cc(msgs, m.cc, m.level, "bands")
                band_levels[m.name] = m.level

        if msgs:
            self.send(msgs)

        is_moving = bool(moving[-1])
        c = self.cfg
        muscle_state = self.muscle.state()
        for name in ("jaw", "eyebrow"):
            why = self._blocked_reason(getattr(c, name).channels, is_moving)
            muscle_state[name]["blocked"] = why or muscle_state[name].pop("rejected")
        frame = {
            "type": "frame",
            "t": t_now,
            "wave": np.round(display, 1).tolist(),
            "traces": traces,
            "artifact": [bool(b) for b in blocked_ch[:, -1]],
            "artifact_reason": list(self.artifacts.reason),
            "state": {
                **muscle_state,
                "blink": {**self.blink.state(), "blocked": self._blocked_reason(c.blink.channels, is_moving)},
                "eyes": {**self.eyes.state(), "blocked": self._blocked_reason(c.eyes.channels, is_moving)},
                "alpha": {
                    **self.alpha.state(),
                    "blocked": "",
                    "drone_playing": self.drone_playing,
                    "bend": (self.bend - PITCH_CENTER) / (PITCH_MAX - PITCH_CENTER),
                },
                "bands": band_levels,
                "muted": sorted(self.muted),
                "motion": {
                    "available": self.source.has_accel,
                    "magnitude": self.motion.magnitude,
                    "moving": is_moving,
                    "threshold": c.motion.threshold_g,
                    "removed_uv": self.motion.removed_uv,
                    "cancel": c.motion.cancel,
                    "gate": c.motion.gate,
                },
            },
            "events": [
                {"detector": e.detector, "label": e.label, "velocity": e.velocity, "on": e.on}
                for e in events
            ],
            "midi": [_describe(m) for m in msgs],
        }
        if bp is not None and self.ticks % 5 == 0:
            mask = bp.freqs <= 60
            psd = bp.psd[self.alpha.idx][:, mask].mean(axis=0)
            frame["spectrum"] = {
                "freqs": bp.freqs[mask].round(2).tolist(),
                "power": np.log10(np.maximum(psd, 1e-6)).round(3).tolist(),
            }
        if self.ticks % 25 == 0:
            frame["quality"] = self.quality()
        return frame

    def quality(self) -> list[dict]:
        raw = self.source.latest_eeg(int(self.fs))
        if raw is None:
            return []
        return channel_quality(raw, self.fs, self.cfg.general.mains_hz)

    def describe(self) -> dict:
        """Static info for the GUI: labels, assignments, trace sample rates."""
        c = self.cfg
        names = c.channel_names
        return {
            "type": "config",
            "fs": self.fs,
            "labels": c.general.labels,
            "update_hz": c.general.update_hz,
            "muscle_trace_hz": self.fs / MUSCLE_TRACE_DECIMATION,
            "accel_trace_hz": self.fs / ACCEL_TRACE_DECIMATION,
            "alpha_update_hz": 1.0 / c.alpha.update_seconds,
            "enabled": c.enabled,
            "detectors": {
                "jaw": {"channels": names(c.jaw.channels), "midi": f"CC{c.jaw.cc} intensity, note {c.jaw.note} (velocity = strength)"},
                "eyebrow": {"channels": names(c.eyebrow.channels), "midi": f"CC{c.eyebrow.cc} intensity, note {c.eyebrow.note}"},
                "blink": {"channels": names(c.blink.channels), "midi": f"single: note {c.blink.note_single}, double: note {c.blink.note_double}"},
                "eyes": {"channels": f"{names(c.eyes.channels[:1])} − {names(c.eyes.channels[1:])}", "midi": f"left: note {c.eyes.note_left}, right: note {c.eyes.note_right}"},
                "alpha": {"channels": names(c.alpha.channels), "midi": f"note {c.alpha.note} held, pitch bend + CC{c.alpha.cc} = alpha strength"},
            },
            "bands": [{"name": m.name, "midi": f"CC{m.cc}", "channels": names(m.channels)} for m in c.mappings],
        }


def channel_quality(raw: np.ndarray, fs: float, mains_hz: float) -> list[dict]:
    """Per-channel contact check over a short raw window."""
    bp = BandPowers(raw, fs, {"mains": (mains_hz - 2, mains_hz + 2)})
    noise_floor = np.median(np.abs(np.diff(raw, axis=1)), axis=1)
    out = []
    for ch in range(raw.shape[0]):
        ptp = float(bp.peak_to_peak[ch])
        mains = float(np.sqrt(10 ** bp.band("mains", [ch], False)))
        if np.abs(raw[ch]).max() > 0.95 * RAIL_UV:
            verdict = "railed"
        elif noise_floor[ch] < 0.05:
            verdict = "no signal"
        elif ptp > 200:
            verdict = "noisy"
        else:
            verdict = "ok"
        out.append({
            "ptp": round(ptp, 1),
            "mains": round(mains, 1),
            "dc": round(float(raw[ch].mean()), 0),
            "verdict": verdict,
        })
    return out


def _describe(m: bytes) -> str:
    kind, ch = m[0] & 0xF0, (m[0] & 0x0F) + 1
    if kind == 0xB0:
        return f"ch{ch} CC{m[1]} = {m[2]}"
    if kind == 0x90:
        return f"ch{ch} note {m[1]} on  vel {m[2]}"
    if kind == 0xE0:
        return f"ch{ch} pitch bend {((m[2] << 7) | m[1]) - PITCH_CENTER:+d}"
    return f"ch{ch} note {m[1]} off"
