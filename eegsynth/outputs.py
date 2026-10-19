"""Where CC messages go: a local MIDI port, or UDP to the bridge on another machine.

UDP datagrams are raw MIDI bytes, one or more 3-byte messages per packet
(note on/off, control change and pitch bend only)."""

from __future__ import annotations

import socket


def _open_mido_output(name: str):
    try:
        import mido
    except ImportError as e:
        raise SystemExit(
            "MIDI output needs the optional deps: pip install 'eegsynth[midi]'"
        ) from e
    existing = mido.get_output_names()
    match = next((p for p in existing if name.lower() in p.lower()), None)
    if match:
        return mido.open_output(match)
    return mido.open_output(name, virtual=True)


class MidiOutput:
    """Connects to an existing port whose name contains `name`, otherwise
    creates a virtual port called `name` (works on macOS and Linux/ALSA)."""

    def __init__(self, name: str):
        self.port = _open_mido_output(name)
        import mido

        self._mido = mido
        self.description = f"MIDI port '{self.port.name}'"

    def send(self, messages: list[bytes]) -> None:
        for m in messages:
            self.port.send(self._mido.Message.from_bytes(m))

    def close(self) -> None:
        self.port.close()


class UdpOutput:
    def __init__(self, target: str):
        host, _, port = target.rpartition(":")
        if not host or not port:
            raise ValueError(f"--udp expects HOST:PORT, got {target!r}")
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.description = f"UDP {host}:{port}"

    def send(self, messages: list[bytes]) -> None:
        if not messages:
            return
        try:
            self.sock.sendto(b"".join(messages), self.addr)
        except OSError:
            # Bridge host unreachable or not resolved yet; keep streaming.
            pass

    def close(self) -> None:
        self.sock.close()


def run_bridge(listen: str, port_name: str, verbose: bool = False) -> None:
    """Receive raw MIDI over UDP and emit it from a virtual MIDI port."""
    host, _, port = listen.rpartition(":")
    out = MidiOutput(port_name)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host or "0.0.0.0", int(port)))
    print(f"Bridge: UDP {host or '0.0.0.0'}:{port} -> {out.description}. Ctrl+C to stop.")
    sender = None
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            if addr != sender:
                sender = addr
                print(f"Receiving from {addr[0]}")
            msgs = [data[i : i + 3] for i in range(0, len(data) - len(data) % 3, 3)]
            msgs = [m for m in msgs if m[0] & 0xF0 in (0x80, 0x90, 0xB0, 0xE0) and m[1] < 128 and m[2] < 128]
            out.send(msgs)
            if verbose:
                for m in msgs:
                    print(f"  {m.hex(' ')}")
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        sock.close()
