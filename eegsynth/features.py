"""Band power estimation from a window of multichannel EEG."""

from __future__ import annotations

import numpy as np

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
TOTAL_RANGE = (1.0, 45.0)


def _detrend(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    t = np.arange(n) - (n - 1) / 2
    slope = (x * t).sum(axis=-1, keepdims=True) / (t * t).sum()
    return x - x.mean(axis=-1, keepdims=True) - slope * t


class BandPowers:
    """Power spectrum of one analysis window, queried by band and channel."""

    def __init__(self, window: np.ndarray, fs: float, bands: dict[str, tuple[float, float]]):
        self.bands = bands
        x = _detrend(window)
        self.peak_to_peak = x.max(axis=-1) - x.min(axis=-1)
        taper = np.hanning(x.shape[-1])
        spectrum = np.abs(np.fft.rfft(x * taper, axis=-1)) ** 2
        self.freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / fs)
        self.psd = spectrum / (fs * (taper**2).sum())

    def _integrate(self, lo: float, hi: float, channels: list[int]) -> np.ndarray:
        mask = (self.freqs >= lo) & (self.freqs < hi)
        return self.psd[channels][:, mask].sum(axis=-1)

    def band(self, name: str, channels: list[int], relative: bool) -> float:
        lo, hi = self.bands[name]
        p = self._integrate(lo, hi, channels)
        if relative:
            p = p / np.maximum(self._integrate(*TOTAL_RANGE, channels), 1e-12)
        return float(np.log10(np.maximum(p, 1e-12)).mean())

    def feature(self, expr: str, channels: list[int], relative: bool) -> float:
        """`expr` is a band name ("alpha") or a ratio of two bands ("alpha/beta").
        Values are log10, so a ratio is a difference of log powers."""
        if "/" in expr:
            num, den = (s.strip() for s in expr.split("/", 1))
            return self.band(num, channels, False) - self.band(den, channels, False)
        return self.band(expr, channels, relative)
