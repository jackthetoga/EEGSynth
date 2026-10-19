"""Artifact handling: accelerometer-based motion cancellation and gating, and
per-channel detection of electrode pops, flatlines and saturation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eegsynth.dsp import StreamFilter, highpass_sos, lowpass_sos


@dataclass
class MotionConfig:
    cancel: bool = True          # adaptively subtract accelerometer-correlated signal
    gate: bool = True            # pause detectors while the head is moving
    threshold_g: float = 0.04    # head acceleration (gravity removed) counted as movement
    hold_seconds: float = 0.8    # keep detectors paused this long after movement stops
    veto_seconds: float = 0.15   # blink/eye events wait this long and are dropped if motion or an artifact follows
    step_size: float = 0.01      # NLMS adaptation rate; higher = faster but noisier


@dataclass
class ArtifactConfig:
    step_uv: float = 500.0       # sample-to-sample jump that means an electrode pop
    amplitude_uv: float = 1000.0 # slow-wave excursion no blink or eye movement reaches
    flat_seconds: float = 0.25   # identical samples for this long = no contact / saturated
    hold_seconds: float = 1.0    # keep a channel blocked this long after an artifact


class MotionProcessor:
    """Turns the 25 Hz sample-and-hold accelerometer into motion references,
    gates detectors while moving, and removes motion-correlated signal from
    each EEG channel with a normalized LMS adaptive filter.

    NLMS learns, per channel, the linear relation between recent head
    acceleration and the EEG. Brain activity is uncorrelated with head motion,
    so only the artifact part is predicted and subtracted. It cannot remove
    electrode pops or cable tugs that aren't a linear function of the
    acceleration, which is why the gate and the artifact monitor exist too."""

    TAP_DELAYS = (0, 6, 12, 18, 24)
    MOTION_FLOOR_G = 0.01
    # Adaptation sees EEG above 0.5 Hz (where head-motion artifact lives, and
    # pop recovery tails don't), with errors clipped so outliers can't throw
    # the weights off.
    ADAPT_CLIP_UV = 200.0

    def __init__(self, cfg: MotionConfig, n_channels: int, fs: float):
        self.cfg, self.fs = cfg, fs
        self.accel_filter = StreamFilter(highpass_sos(0.3, fs), lowpass_sos(10.0, fs))
        self.adapt_band = StreamFilter(highpass_sos(0.5, fs))
        self.magnitude_filter = StreamFilter(lowpass_sos(5.0, fs))
        self.n_refs = 6
        n_taps = len(self.TAP_DELAYS)
        self.hist_len = max(self.TAP_DELAYS) + 1
        self.history = np.zeros((self.n_refs, self.hist_len))
        self.hist_pos = 0
        self.weights = np.zeros((n_channels, self.n_refs * n_taps))
        self.eps = self.weights.shape[1] * self.MOTION_FLOOR_G**2
        self.prev_ref = np.zeros(3)
        self.deriv_scale = fs / (2 * np.pi * 3.0)
        self.hold_n = int(cfg.hold_seconds * fs)
        self.quiet_for = self.hold_n
        self.magnitude = 0.0
        self.removed_uv = 0.0
        self.active = False

    def process(self, eeg_hp: np.ndarray, accel: np.ndarray, frozen: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`frozen` (channels x samples) marks samples the filter must not learn
        from (artifact-flagged). Returns (cleaned EEG, moving mask per sample,
        accel with gravity removed)."""
        k = eeg_hp.shape[1]
        target = self.adapt_band(eeg_hp)
        ref = self.accel_filter(accel)
        mag = self.magnitude_filter(np.sqrt((ref**2).sum(axis=0)))
        moving = np.zeros(k, dtype=bool)
        cleaned = eeg_hp.copy()
        removed = np.zeros(k)
        delays = np.array(self.TAP_DELAYS)
        for i in range(k):
            if mag[i] > self.cfg.threshold_g:
                self.quiet_for = 0
            else:
                self.quiet_for += 1
            moving[i] = self.cfg.gate and self.quiet_for < self.hold_n

            if not self.cfg.cancel:
                continue
            deriv = (ref[:, i] - self.prev_ref) * self.deriv_scale
            self.prev_ref = ref[:, i]
            self.hist_pos = (self.hist_pos + 1) % self.hist_len
            self.history[:3, self.hist_pos] = ref[:, i]
            self.history[3:, self.hist_pos] = deriv
            x = self.history[:, (self.hist_pos - delays) % self.hist_len].ravel()
            y = self.weights @ x
            cleaned[:, i] = eeg_hp[:, i] - y
            removed[i] = np.sqrt((y**2).mean())
            e = np.clip(target[:, i] - y, -self.ADAPT_CLIP_UV, self.ADAPT_CLIP_UV)
            ok = ~frozen[:, i]
            self.weights[ok] += self.cfg.step_size * np.outer(e[ok], x) / (self.eps + x @ x)
        self.magnitude = float(mag[-1]) if k else self.magnitude
        self.removed_uv = float(removed.mean()) if k else 0.0
        self.active = bool(moving[-1]) if k else self.active
        return cleaned, moving, ref


class ArtifactMonitor:
    """Flags each channel, per sample, as unusable while it shows a pop
    (sudden jump), an excursion too large to be physiological, saturation at
    the ADC rail, or a flatline (no contact). Flags are held for
    `hold_seconds` so filter ringing after the artifact is covered too.

    Pop and amplitude checks look at a 20 Hz low-passed copy, so strong muscle
    activity (30+ Hz, e.g. a hard jaw clench under T7/T8) isn't mistaken for an
    artifact. Steps are measured across STEP_SECONDS, which a pop's step
    still spans after low-passing."""

    LOWPASS_HZ = 20.0
    STEP_SECONDS = 0.04

    def __init__(self, cfg: ArtifactConfig, n_channels: int, fs: float, rail_uv: float):
        self.cfg = cfg
        self.lowpass = StreamFilter(lowpass_sos(self.LOWPASS_HZ, fs))
        self.step_n = max(1, int(self.STEP_SECONDS * fs))
        self.prev_lp: np.ndarray | None = None
        self.rail = 0.95 * rail_uv
        self.hold_n = int(cfg.hold_seconds * fs)
        self.flat_n = int(cfg.flat_seconds * fs)
        self.n = 0
        self.last_trigger = np.full(n_channels, -10**9)
        self.last_moving = np.full(n_channels, -1)  # index of the last non-still sample
        self.prev_raw: np.ndarray | None = None
        self.reason = ["" for _ in range(n_channels)]

    def process(self, raw: np.ndarray, hp: np.ndarray) -> np.ndarray:
        """`raw` is unfiltered EEG, `hp` the high-passed (and motion-cleaned) EEG."""
        c, k = raw.shape
        if self.prev_raw is None:
            self.prev_raw = raw[:, 0].copy()
        raw_ext = np.concatenate([self.prev_raw[:, None], raw], axis=1)
        self.prev_raw = raw[:, -1].copy()
        idx = self.n + np.arange(k)
        self.n += k

        lp = self.lowpass(hp)
        if self.prev_lp is None:
            self.prev_lp = np.repeat(lp[:, :1], self.step_n, axis=1)
        lp_ext = np.concatenate([self.prev_lp, lp], axis=1)
        self.prev_lp = lp_ext[:, -self.step_n:]
        pop = np.abs(lp_ext[:, self.step_n:] - lp_ext[:, :-self.step_n]) > self.cfg.step_uv
        big = np.abs(lp) > self.cfg.amplitude_uv
        railed = np.abs(raw) > self.rail
        # Flat: length of the current run of (near-)identical samples.
        changing = np.abs(np.diff(raw_ext, axis=1)) >= 0.05
        last_change = np.maximum.accumulate(
            np.where(changing, idx, -1), axis=1
        )
        last_change = np.maximum(last_change, self.last_moving[:, None])
        self.last_moving = last_change[:, -1]
        flat = (idx - last_change) >= self.flat_n

        trigger = pop | big | railed | flat
        last_trigger = np.maximum.accumulate(np.where(trigger, idx, -10**9), axis=1)
        last_trigger = np.maximum(last_trigger, self.last_trigger[:, None])
        self.last_trigger = last_trigger[:, -1]
        blocked = (idx - last_trigger) < self.hold_n

        for ch in range(c):
            if trigger[ch].any():
                i = np.flatnonzero(trigger[ch])[-1]
                self.reason[ch] = (
                    "flat" if flat[ch, i] else "railed" if railed[ch, i] else "pop" if pop[ch, i] else "too large"
                )
            elif not blocked[ch, -1]:
                self.reason[ch] = ""
        return blocked
