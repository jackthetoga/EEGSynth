"""Causal streaming filters that keep state between chunks."""

from __future__ import annotations

import numpy as np
from scipy import signal


def notch_sos(mains_hz: float, fs: float) -> np.ndarray:
    sections = []
    f = mains_hz
    while f < fs / 2 - 1:
        b, a = signal.iirnotch(f, Q=30, fs=fs)
        sections.append(signal.tf2sos(b, a))
        f += mains_hz
    return np.vstack(sections)


def bandpass_sos(lo: float, hi: float, fs: float, order: int = 2) -> np.ndarray:
    return signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def highpass_sos(f: float, fs: float, order: int = 2) -> np.ndarray:
    return signal.butter(order, f, btype="highpass", fs=fs, output="sos")


def lowpass_sos(f: float, fs: float, order: int = 2) -> np.ndarray:
    return signal.butter(order, f, btype="lowpass", fs=fs, output="sos")


class StreamFilter:
    """Applies an SOS filter along the last axis of successive chunks.
    Initial state assumes the signal has been at its first value forever, so the
    Cyton's large DC offsets don't ring through on startup."""

    def __init__(self, *sos: np.ndarray):
        self.sos = np.vstack(sos)
        self.zi: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] == 0:
            return x
        if self.zi is None:
            base = signal.sosfilt_zi(self.sos)
            self.zi = base.reshape(base.shape[0], *([1] * (x.ndim - 1)), 2) * x[..., :1][None]
        y, self.zi = signal.sosfilt(self.sos, x, axis=-1, zi=self.zi)
        return y


class Envelope:
    """RMS envelope: band-limit, square, low-pass, square root."""

    def __init__(self, band_sos: np.ndarray, smooth_hz: float, fs: float):
        self.band = StreamFilter(band_sos)
        self.smooth = StreamFilter(lowpass_sos(smooth_hz, fs))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = self.band(x)
        return np.sqrt(np.maximum(self.smooth(y * y), 0.0))
