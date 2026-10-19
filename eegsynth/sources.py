"""EEG sample sources: the OpenBCI Cyton over its USB dongle, and a synthetic generator."""

from __future__ import annotations

import glob
import sys
import threading
import time

import numpy as np

SAMPLE_RATE = 250.0
NUM_CHANNELS = 8
# Buffer rows: 8 EEG channels (µV) followed by accelerometer X, Y, Z (g).
ACCEL_ROWS = slice(NUM_CHANNELS, NUM_CHANNELS + 3)
NUM_ROWS = NUM_CHANNELS + 3

# Cyton packet: 0xA0, sample#, 8 x 24-bit channels, 6 aux bytes, 0xCX footer.
# With footer 0xC0 the aux bytes are accelerometer X/Y/Z (16-bit big-endian),
# updated at 25 Hz; packets without a new reading carry zeros.
PACKET_LEN = 33
HEADER = 0xA0
ADS1299_VREF = 4.5
DEFAULT_GAIN = 24
UV_PER_COUNT = ADS1299_VREF / DEFAULT_GAIN / (2**23 - 1) * 1e6
G_PER_COUNT = 0.002 / 16


class RingBuffer:
    """Thread-safe fixed-size buffer of the most recent samples, shape (channels, n)."""

    def __init__(self, channels: int, size: int):
        self._data = np.zeros((channels, size))
        self._size = size
        self._count = 0
        self._lock = threading.Lock()

    def push(self, sample: np.ndarray) -> None:
        with self._lock:
            self._data[:, self._count % self._size] = sample
            self._count += 1

    def push_block(self, block: np.ndarray) -> None:
        with self._lock:
            block = block[:, -self._size:]
            k = block.shape[1]
            idx = np.arange(self._count, self._count + k) % self._size
            self._data[:, idx] = block
            self._count += k

    @property
    def count(self) -> int:
        return self._count

    def latest(self, n: int) -> np.ndarray | None:
        with self._lock:
            if self._count < n:
                return None
            end = self._count % self._size
            idx = (np.arange(end - n, end)) % self._size
            return self._data[:, idx].copy()

    def read_since(self, count: int) -> tuple[np.ndarray, int]:
        """Samples pushed after `count` (at most a full buffer), and the new count."""
        with self._lock:
            now = self._count
            start = max(count, now - self._size)
            idx = np.arange(start, now) % self._size
            return self._data[:, idx].copy(), now


class Source:
    def __init__(self, buffer_seconds: float = 10.0):
        self.buffer = RingBuffer(NUM_ROWS, int(SAMPLE_RATE * buffer_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.has_accel = False

    def latest_eeg(self, n: int) -> np.ndarray | None:
        data = self.buffer.latest(n)
        return None if data is None else data[:NUM_CHANNELS]

    def start(self) -> None:
        self._thread = threading.Thread(target=self._guarded_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _guarded_run(self) -> None:
        try:
            self._run()
        except BaseException as e:
            self.error = e

    def _run(self) -> None:
        raise NotImplementedError


def find_cyton_port() -> str | None:
    if sys.platform == "darwin":
        candidates = glob.glob("/dev/cu.usbserial-*")
    else:
        candidates = glob.glob("/dev/ttyUSB*")
    return sorted(candidates)[0] if candidates else None


def _int24(b0: int, b1: int, b2: int) -> int:
    v = (b0 << 16) | (b1 << 8) | b2
    return v - (1 << 24) if v & 0x800000 else v


def parse_packet(pkt: bytes) -> np.ndarray:
    return np.array(
        [_int24(*pkt[2 + 3 * i : 5 + 3 * i]) for i in range(NUM_CHANNELS)],
        dtype=float,
    ) * UV_PER_COUNT


def parse_accel(pkt: bytes) -> np.ndarray | None:
    """Accelerometer reading in g, or None if this packet carries no new one."""
    if pkt[PACKET_LEN - 1] != 0xC0:
        return None
    raw = [int.from_bytes(pkt[26 + 2 * i : 28 + 2 * i], "big", signed=True) for i in range(3)]
    if raw == [0, 0, 0]:
        return None
    return np.array(raw, dtype=float) * G_PER_COUNT


class CytonSource(Source):
    def __init__(self, port: str | None = None, **kw):
        super().__init__(**kw)
        self.port = port or find_cyton_port()
        if not self.port:
            raise RuntimeError(
                "No Cyton dongle found. Plug it in, set its switch to GPIO_6, "
                "or pass --port explicitly."
            )
        self.dropped_packets = 0

    def _run(self) -> None:
        import serial

        with serial.Serial(self.port, 115200, timeout=1) as ser:
            ser.write(b"s")
            time.sleep(0.2)
            ser.reset_input_buffer()
            ser.write(b"v")
            banner = self._read_until(ser, b"$$$", timeout=5)
            if b"$$$" not in banner:
                raise RuntimeError(
                    f"Cyton did not answer on {self.port}. Is the board switched "
                    "to PC and the dongle switch on GPIO_6?"
                )
            # Board mode persists across soft resets; force default (accelerometer in aux bytes).
            ser.write(b"/0")
            self._read_until(ser, b"$$$", timeout=2)
            ser.write(b"b")
            try:
                self._stream(ser)
            finally:
                ser.write(b"s")

    @staticmethod
    def _read_until(ser, token: bytes, timeout: float) -> bytes:
        buf = b""
        deadline = time.time() + timeout
        while time.time() < deadline and token not in buf:
            buf += ser.read(ser.in_waiting or 1)
        return buf

    def _stream(self, ser) -> None:
        buf = bytearray()
        last_sample_num = None
        accel = np.zeros(3)
        while not self._stop.is_set():
            buf += ser.read(ser.in_waiting or 1)
            while len(buf) >= PACKET_LEN:
                if buf[0] != HEADER or (buf[PACKET_LEN - 1] & 0xF0) != 0xC0:
                    del buf[0]
                    continue
                pkt = bytes(buf[:PACKET_LEN])
                del buf[:PACKET_LEN]
                sample_num = pkt[1]
                if last_sample_num is not None:
                    self.dropped_packets += (sample_num - last_sample_num - 1) % 256
                last_sample_num = sample_num
                new_accel = parse_accel(pkt)
                if new_accel is not None:
                    accel = new_accel
                    self.has_accel = True
                self.buffer.push(np.concatenate([parse_packet(pkt), accel]))


def _bump(t, t0, dur):
    x = (t - t0) / dur
    return np.where((x > 0) & (x < 1), np.sin(np.pi * x) ** 2, 0.0)


def _gate(t, a, b, ramp=0.08):
    return np.clip((t - a) / ramp, 0, 1) * np.clip((b - t) / ramp, 0, 1)


def _step(t, t0, ramp=0.05):
    return np.clip((t - t0) / ramp, 0, 1)


def _pop(t, t0, tau=0.4):
    """Electrode pop: instant step that relaxes back exponentially."""
    dt = t - t0
    return np.where(dt >= 0, np.exp(-np.maximum(dt, 0) / tau), 0.0)


SYNTHETIC_CYCLE = 40.0
SYNTHETIC_SCRIPT = """\
  2.0s single blink        5.0s double blink
  8.0s light jaw clench   11.0s hard jaw clench
 14.0s eyebrow raise      17.0s look left (back 18.5s)
 20.0s look right (back 21.5s)
 22.6s electrode pop on T8 (should NOT trigger anything)
 24-32s eyes closed (alpha)   35.0s single blink
 36.5-38.5s head shake: motion artifacts + pops (detectors should pause)"""


class SyntheticSource(Source):
    """Fake EEG for the default montage (Fp1 Fp2 F7 F8 T7 T8 O1 O2) that acts
    out every gesture the detectors look for on a 40 s loop (SYNTHETIC_SCRIPT)."""

    def __init__(self, seed: int | None = None, **kw):
        super().__init__(**kw)
        self.rng = np.random.default_rng(seed)
        self.offsets = self.rng.uniform(-5000, 5000, (NUM_CHANNELS, 1))
        self.alpha_phase = self.rng.uniform(0, 2 * np.pi)
        self.brown_zi = np.zeros((NUM_CHANNELS, 1))
        # µV of artifact per g of head acceleration on each axis, per channel.
        self.motion_gain = self.rng.uniform(-1500, 1500, (NUM_CHANNELS, 3))
        self.generated = 0
        self.has_accel = True

    def generate(self, target: int) -> None:
        """Push samples up to absolute sample index `target`."""
        from scipy.signal import lfilter

        if target <= self.generated:
            return
        rng = self.rng
        t = np.arange(self.generated, target) / SAMPLE_RATE
        c = t % SYNTHETIC_CYCLE
        k = len(t)
        brown, self.brown_zi = lfilter(
            [1.0], [1.0, -0.98], rng.normal(0, 1.5, (NUM_CHANNELS, k)), zi=self.brown_zi
        )
        x = self.offsets + brown + rng.normal(0, 2, (NUM_CHANNELS, k))
        x += 4 * np.sin(2 * np.pi * 60 * t)

        eyes_closed = _gate(c, 24, 32, ramp=0.8)
        alpha_spread = np.array([0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 1.0, 1.0])[:, None]
        x += alpha_spread * (1.5 + 18.5 * eyes_closed) * np.sin(2 * np.pi * 10 * t + self.alpha_phase)

        blink = _bump(c, 2.0, 0.3) + _bump(c, 5.0, 0.28) + _bump(c, 5.38, 0.28) + _bump(c, 35.0, 0.3)
        x[0:2] += 220 * blink
        x[2:4] += 60 * blink

        gaze = (_step(c, 17) - _step(c, 18.5)) - (_step(c, 20) - _step(c, 21.5))
        x[2] += 90 * gaze
        x[3] -= 90 * gaze
        x[0] += 25 * gaze
        x[1] -= 25 * gaze

        jaw = 0.3 * _gate(c, 8, 9.5) + 1.0 * _gate(c, 11, 12.5)
        jaw_gain = np.array([15, 15, 20, 20, 45, 45, 12, 12])[:, None]
        x += jaw_gain * jaw * rng.normal(0, 1, (NUM_CHANNELS, k))

        brow = _gate(c, 14, 15.5)
        brow_gain = np.array([40, 40, 12, 12, 2, 2, 0, 0])[:, None]
        x += brow_gain * brow * rng.normal(0, 1, (NUM_CHANNELS, k))

        x[5] += 3000 * _pop(c, 22.6)

        shake = _gate(c, 36.5, 38.5, ramp=0.2)
        motion = np.vstack([
            0.35 * shake * np.sin(2 * np.pi * 2.5 * t),
            0.15 * shake * np.sin(2 * np.pi * 2.5 * t + 1.0),
            0.10 * shake * np.sin(2 * np.pi * 5.0 * t),
        ])
        x += self.motion_gain @ motion
        for ch, t_pop in [(0, 36.9), (4, 37.3), (6, 37.8), (5, 38.1)]:
            x[ch] += 2500 * _pop(c, t_pop, tau=0.3)

        # Accelerometer: gravity on Z plus motion, sampled and held at 25 Hz.
        accel = motion + np.array([[0.02], [-0.05], [1.0]]) + rng.normal(0, 0.003, (3, k))
        held = (np.arange(self.generated, target) // 10) * 10
        accel = accel[:, np.clip(held - self.generated, 0, k - 1)]

        self.buffer.push_block(np.vstack([x, accel]))
        self.generated = target

    def _run(self) -> None:
        t0 = time.perf_counter()
        while not self._stop.is_set():
            self.generate(int((time.perf_counter() - t0) * SAMPLE_RATE))
            time.sleep(0.004)
