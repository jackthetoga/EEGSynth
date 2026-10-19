"""Gesture detectors: jaw clench and eyebrow raise (EMG), blinks and horizontal
eye movements (EOG), and occipital alpha (eyes closed / relaxation).

Every detector works on sample time (sample index / fs) so behaviour is
independent of how the incoming data happens to be chunked."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Filters settle and baselines are learned during this time; nothing triggers.
WARMUP_SECONDS = 3.0


@dataclass
class Event:
    detector: str
    label: str
    note: int  # < 0 means no MIDI note assigned
    velocity: int = 100
    on: bool = True
    duration: float | None = None  # seconds until automatic note-off; None = held until an off event


def _ratio_to_unit(ratio: float, lo: float, hi: float) -> float:
    """Position of `ratio` between lo and hi on a log scale, clipped to 0..1."""
    if ratio <= lo:
        return 0.0
    return min(math.log(ratio / lo) / math.log(hi / lo), 1.0)


# --------------------------------------------------------------------------- EMG


@dataclass
class JawConfig:
    channels: list[int] = field(default_factory=lambda: [5, 6])
    threshold_ratio: float = 4.0
    full_scale_ratio: float = 40.0
    require_all: bool = True       # every channel must show it (pops are one electrode)
    min_hf_lf: float = 0.3         # 30-55 Hz must be at least this fraction of 1-10 Hz
    max_uv: float = 600.0          # EMG envelope above this is an artifact, not muscle
    min_baseline_uv: float = 0.5   # floor for the resting level, so a flat channel can't give huge ratios
    min_duration: float = 0.2      # must stay above threshold this long to count
    refractory: float = 0.3        # no new onset this soon after a release
    cc: int = 23
    note: int = 36


@dataclass
class EyebrowConfig:
    channels: list[int] = field(default_factory=lambda: [1, 2])
    threshold_ratio: float = 4.0
    full_scale_ratio: float = 40.0
    dominance: float = 2.5
    require_all: bool = True
    min_hf_lf: float = 0.2
    max_uv: float = 300.0
    min_baseline_uv: float = 0.5
    min_duration: float = 0.2
    refractory: float = 0.3
    cc: int = 24
    note: int = 38


class _MuscleState:
    """idle -> pending (must hold for `min_duration`; strength measured) ->
    active -> release, then a refractory period before the next onset."""

    RELEASE_DELAY = 0.25

    def __init__(self, name: str, label: str, note: int, cfg):
        self.name, self.label, self.note = name, label, note
        self.threshold, self.full_scale = cfg.threshold_ratio, cfg.full_scale_ratio
        self.min_duration, self.refractory = cfg.min_duration, cfg.refractory
        self.phase = "idle"
        self.since = 0.0
        self.released_at = -1e9
        self.peak = 0.0
        self.intensity = 0.0
        self.last_velocity = 0

    @property
    def active(self) -> bool:
        return self.phase in ("pending", "active")

    def step(self, t: float, ratio: float, allowed: bool, events: list[Event]) -> None:
        unit = _ratio_to_unit(ratio, self.threshold, self.full_scale)
        above = allowed and ratio > self.threshold
        if self.phase == "idle":
            self.intensity = 0.0
            if above and t - self.released_at >= self.refractory:
                self.phase, self.since, self.peak = "pending", t, unit
        elif self.phase == "pending":
            self.peak = max(self.peak, unit)
            self.intensity = 0.0
            if not (allowed and ratio > self.threshold * 0.8):
                self.phase = "idle"
            elif t - self.since >= self.min_duration:
                self.last_velocity = max(1, round(1 + 126 * self.peak))
                events.append(Event(self.name, self.label, self.note, self.last_velocity))
                self.phase = "active"
                self.since = t
        else:
            self.intensity = unit if allowed else self.intensity
            if allowed and ratio > self.threshold * 0.6:
                self.since = t
            elif t - self.since >= self.RELEASE_DELAY:
                self.phase = "idle"
                self.intensity = 0.0
                self.released_at = t
                events.append(Event(self.name, self.label, self.note, 0, on=False))


class MuscleDetector:
    """Jaw clench vs. eyebrow raise from 30-55 Hz EMG envelopes.

    Jaw EMG (temporalis/masseter) is picked up everywhere, partly through the
    ear reference, so it is identified by the temporal channels. An eyebrow
    raise (frontalis) is strong on the forehead but weak at the temples, so it
    requires frontal activity `dominance` times larger than temporal.

    To reject electrode pops and motion, which also put energy into 30-55 Hz,
    a sample only counts as muscle when: none of the channels are flagged by
    the artifact monitor or motion gate, every channel is elevated (with
    `require_all`), the envelope is within physiological range (`max_uv`), and
    the spectrum is EMG-shaped: pops and movement are dominated by slow waves,
    so 30-55 Hz must be at least `min_hf_lf` of the 1-10 Hz envelope."""

    def __init__(self, jaw: JawConfig, brow: EyebrowConfig, n_channels: int, fs: float):
        self.fs = fs
        self.cfgs = {"jaw": jaw, "eyebrow": brow}
        self.idx = {"jaw": [c - 1 for c in jaw.channels], "eyebrow": [c - 1 for c in brow.channels]}
        self.jaw = _MuscleState("jaw", "Jaw clench", jaw.note, jaw)
        self.brow = _MuscleState("eyebrow", "Eyebrow raise", brow.note, brow)
        self.states = {"jaw": self.jaw, "eyebrow": self.brow}
        self.base_log = np.zeros(n_channels)
        self.floors = np.full(n_channels, jaw.min_baseline_uv)
        for name, cfg in self.cfgs.items():
            self.floors[self.idx[name]] = cfg.min_baseline_uv
        self.n = 0
        self.warmup_n = int(WARMUP_SECONDS * fs)
        self.rest_alpha = 1.0 / (20.0 * fs)
        # Still adapts (slowly) while "active", so a baseline learned from a bad
        # start can't lock a detector on forever.
        self.active_alpha = 1.0 / (120.0 * fs)
        self.ratio = {"jaw": 1.0, "eyebrow": 1.0}
        self.rejected = {"jaw": "", "eyebrow": ""}

    def _group(self, name: str, ratios: np.ndarray, env: np.ndarray, lf: np.ndarray,
               blocked: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Per-sample group ratio and rejection reason ("" = usable)."""
        cfg, idx = self.cfgs[name], self.idx[name]
        r = ratios[idx]
        ratio = r.min(axis=0) if cfg.require_all else r.mean(axis=0)
        elevated = ratio > cfg.threshold_ratio * 0.6
        artifact = blocked[idx].any(axis=0)
        too_large = elevated & (env[idx] > cfg.max_uv).any(axis=0)
        not_emg = elevated & (env[idx] < cfg.min_hf_lf * lf[idx]).any(axis=0)
        why = np.where(artifact, "artifact", np.where(too_large, "too large", np.where(not_emg, "not EMG-like", "")))
        return ratio, why.tolist()

    def process(self, t0: float, env: np.ndarray, lf: np.ndarray, blocked: np.ndarray,
                events: list[Event]) -> dict[str, list[float]]:
        k = env.shape[1]
        ratios = env / np.maximum(np.exp(self.base_log), self.floors)[:, None]
        rj, why_j = self._group("jaw", ratios, env, lf, blocked)
        rb, why_b = self._group("eyebrow", ratios, env, lf, blocked)
        dominance = self.cfgs["eyebrow"].dominance

        for i in range(k):
            self.n += 1
            t = t0 + i / self.fs
            warm = self.n > self.warmup_n
            brow_dominant = rb[i] > rj[i] * dominance
            self.jaw.step(t, float(rj[i]), warm and not why_j[i] and not brow_dominant, events)
            self.brow.step(t, float(rb[i]), warm and not why_b[i] and brow_dominant, events)
        self.ratio = {"jaw": float(rj[-1]), "eyebrow": float(rb[-1])}
        self.rejected = {"jaw": why_j[-1], "eyebrow": why_b[-1]}

        # Resting-level update, once per chunk (chunks are far shorter than
        # the time constant). Artifact-flagged samples are never learned from.
        lv = np.log(np.maximum(env, 1e-3))
        if self.n <= self.warmup_n:
            mask = ~blocked
            weight = mask.sum(axis=1) / self.n
        else:
            resting = not (self.jaw.active or self.brow.active)
            mask = ~blocked & (ratios.max(axis=0) < 2.0)[None, :]
            weight = mask.sum(axis=1) * (self.rest_alpha if resting else self.active_alpha)
        mean_lv = np.where(mask, lv, 0).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        self.base_log += np.minimum(weight, 1.0) * (mean_lv - self.base_log)
        return {"jaw": rj.tolist(), "eyebrow": rb.tolist()}

    def state(self) -> dict:
        return {
            name: {
                "active": s.active,
                "intensity": s.intensity,
                "ratio": self.ratio[name],
                "velocity": s.last_velocity,
                "threshold": s.threshold,
                "full_scale": s.full_scale,
                "rejected": self.rejected[name],
            }
            for name, s in self.states.items()
        }


# --------------------------------------------------------------------------- blinks


@dataclass
class BlinkConfig:
    channels: list[int] = field(default_factory=lambda: [1, 2])
    min_uv: float = 60.0
    noise_factor: float = 6.0
    max_duration: float = 0.5
    double_window: float = 0.6
    note_single: int = 40
    note_double: int = 41


class BlinkDetector:
    """Blinks are large same-polarity deflections on Fp1/Fp2 (averaging the two
    also cancels horizontal eye movements, which have opposite polarity there).
    A blink followed by another within `double_window` is a double blink; a
    single blink is only reported once that window has passed."""

    # The high-pass filter leaves an opposite-polarity dip after each blink;
    # crossings of the other sign this soon after a blink are that dip.
    REBOUND_SECONDS = 0.5

    def __init__(self, cfg: BlinkConfig, fs: float):
        self.cfg, self.fs = cfg, fs
        self.idx = [c - 1 for c in cfg.channels]
        self.noise = 0.0
        self.noise_alpha = 1.0 / (5.0 * fs)
        self.n = 0
        self.warmup_n = int(WARMUP_SECONDS * fs)
        self.in_peak = False
        self.peak_sign = 0.0
        self.peak_start = 0.0
        self.ignore_peak = False
        self.last_sign = 0.0
        self.last_end = -1e9
        self.pending: float | None = None
        self.singles = 0
        self.doubles = 0
        self.threshold = cfg.min_uv

    @property
    def busy(self) -> bool:
        return self.in_peak

    def process(self, t0: float, veog: np.ndarray, suppressed: np.ndarray, events: list[Event]) -> list[float]:
        """`suppressed` is a per-sample mask (motion, artifacts, eyebrow raise)."""
        v = veog[self.idx].mean(axis=0)
        for i, x in enumerate(v):
            t = t0 + i / self.fs
            self.n += 1
            ax = abs(x)
            if self.n <= self.warmup_n:
                self.noise += (ax - self.noise) / self.n
                continue
            self.threshold = max(self.cfg.min_uv, self.cfg.noise_factor * self.noise)
            if self.in_peak and suppressed[i]:
                self.ignore_peak = True
            if not self.in_peak:
                if not suppressed[i]:
                    self.noise += self.noise_alpha * (min(ax, self.threshold) - self.noise)
                if ax > self.threshold:
                    sign = math.copysign(1.0, x)
                    rebound = sign != self.last_sign and t - self.last_end < self.REBOUND_SECONDS
                    self.in_peak, self.peak_start, self.peak_sign = True, t, sign
                    self.ignore_peak = bool(suppressed[i]) or rebound
            elif ax < self.threshold * 0.5 or math.copysign(1.0, x) != self.peak_sign:
                self.in_peak = False
                duration = t - self.peak_start
                if not self.ignore_peak and 0.04 <= duration <= self.cfg.max_duration:
                    self.last_sign, self.last_end = self.peak_sign, t
                    self._blink(t, events)
            if self.pending is not None and t - self.pending > self.cfg.double_window:
                self.pending = None
                self.singles += 1
                events.append(Event("blink", "Single blink", self.cfg.note_single, 100, duration=0.15))
        return v.tolist()

    def _blink(self, t: float, events: list[Event]) -> None:
        if self.pending is not None and t - self.pending <= self.cfg.double_window:
            self.pending = None
            self.doubles += 1
            events.append(Event("blink", "Double blink", self.cfg.note_double, 100, duration=0.15))
        else:
            self.pending = t

    def state(self) -> dict:
        return {"threshold": self.threshold, "singles": self.singles, "doubles": self.doubles}


# --------------------------------------------------------------------------- eye movements


@dataclass
class EyesConfig:
    channels: list[int] = field(default_factory=lambda: [3, 4])  # [left temple, right temple]
    threshold_uv: float = 60.0
    invert: bool = False
    return_seconds: float = 3.0
    note_left: int = 43
    note_right: int = 45


class EyeMovementDetector:
    """Horizontal EOG: the cornea is positive, so looking left makes the left
    temple electrode more positive relative to the right one. A saccade is a
    fast step in (left - right); its sign gives the direction. Looking back to
    centre is the opposite step and is not reported."""

    STEP_SECONDS = 0.1
    REFRACTORY = 0.25

    def __init__(self, cfg: EyesConfig, fs: float):
        if len(cfg.channels) != 2:
            raise ValueError("[eyes] channels must be [left, right]")
        self.cfg, self.fs = cfg, fs
        self.left, self.right = cfg.channels[0] - 1, cfg.channels[1] - 1
        self.history: deque[float] = deque(maxlen=int(self.STEP_SECONDS * fs) + 1)
        self.n = 0
        self.warmup_n = int(WARMUP_SECONDS * fs)
        self.armed = True
        self.last_step = -1e9
        self.position = "center"
        self.position_since = 0.0

    def process(self, t0: float, heog_all: np.ndarray, suppressed: np.ndarray, events: list[Event]) -> list[float]:
        h = heog_all[self.left] - heog_all[self.right]
        if self.cfg.invert:
            h = -h
        thr = self.cfg.threshold_uv
        for i, x in enumerate(h):
            t = t0 + i / self.fs
            self.history.append(x)
            self.n += 1
            if self.n <= self.warmup_n:
                continue
            d = x - self.history[0]
            if not self.armed:
                if abs(d) < thr * 0.5 and t - self.last_step > self.REFRACTORY:
                    self.armed = True
            elif suppressed[i]:
                # Disarm so the tail of an artifact can't register as a saccade.
                self.armed, self.last_step = False, t
            elif abs(d) > thr:
                self.armed, self.last_step = False, t
                direction = "left" if d > 0 else "right"
                if self.position == "center":
                    self.position, self.position_since = direction, t
                    note = self.cfg.note_left if direction == "left" else self.cfg.note_right
                    events.append(Event("eyes", f"Look {direction}", note, 100, duration=0.15))
                elif self.position != direction:
                    self.position = "center"
                else:
                    self.position_since = t
            if self.position != "center" and t - self.position_since > self.cfg.return_seconds:
                self.position = "center"
        return h.tolist()

    def state(self) -> dict:
        return {"position": self.position, "threshold": self.cfg.threshold_uv}


# --------------------------------------------------------------------------- alpha


@dataclass
class AlphaConfig:
    channels: list[int] = field(default_factory=lambda: [7])
    band: list[float] = field(default_factory=lambda: [8.0, 13.0])
    broadband: list[float] = field(default_factory=lambda: [4.0, 30.0])
    window_seconds: float = 2.0
    update_seconds: float = 0.25
    smooth_seconds: float = 3.0         # larger = steadier, but slower response
    motion_threshold_g: float = 0.12    # accelerometer deviation that rejects a reading
    min_accel_samples: int = 8
    require_accel: bool = True          # pause alpha when motion can't be verified
    calibration_delay: float = 2.0
    calibration_seconds: float = 5.0
    min_calibration_values: int = 8
    full_scale_gain: float = 2.5        # gain over baseline that gives full strength
    cc: int = 1                         # alpha strength 0-127
    note: int = 60                      # the long note whose pitch alpha bends
    velocity: int = 100
    bend_up: float = 1.0                # fraction of the synth's upward bend range at full strength


class AlphaDetector:
    """Alpha fraction (8-13 Hz power / 4-30 Hz power, Hann-windowed FFT of a
    2 s window) with accelerometer motion rejection, an exponential moving
    average that only updates on motion-free readings, and an eyes-open
    calibration whose median becomes the baseline. Strength = how far the
    smoothed fraction is above baseline, reaching 1 at `full_scale_gain`."""

    def __init__(self, cfg: AlphaConfig, fs: float):
        self.cfg, self.fs = cfg, fs
        self.idx = [c - 1 for c in cfg.channels]
        self.n = int(cfg.window_seconds * fs)
        self.freqs = np.fft.rfftfreq(self.n, d=1 / fs)
        self.window = np.hanning(self.n)
        self.band_mask = (self.freqs >= cfg.band[0]) & (self.freqs <= cfg.band[1])
        self.broad_mask = (self.freqs >= cfg.broadband[0]) & (self.freqs <= cfg.broadband[1])
        self.smoothing_weight = 1 - np.exp(-cfg.update_seconds / cfg.smooth_seconds)
        self.smoothed_fraction: float | None = None
        self.fraction: float | None = None
        self.baseline: float | None = None
        self.cal_start: float | None = None
        self.cal_end: float | None = None
        self.cal_values: list[float] = []
        self.last_update = -1e9
        self.motion: float | None = None
        self.gain: float | None = None
        self.strength = 0.0
        self.status = "uncalibrated"
        self.message = "Calibrate with your eyes open."
        self.just_calibrated = False

    @property
    def calibrating(self) -> bool:
        return self.cal_end is not None

    def calibrate(self, now: float) -> None:
        self.baseline = None
        self.smoothed_fraction = None
        self.gain = None
        self.strength = 0.0
        self.cal_values = []
        self.cal_start = now + self.cfg.calibration_delay
        self.cal_end = self.cal_start + self.cfg.calibration_seconds

    def motion_score(self, accel: np.ndarray | None) -> float | None:
        """95th percentile of the accelerometer's deviation from its median."""
        if accel is None:
            return None
        a = accel.T
        a = a[np.all(np.isfinite(a), axis=1)]
        a = a[np.any(a != 0, axis=1)]
        if len(a) < self.cfg.min_accel_samples:
            return None
        deviation = np.linalg.norm(a - np.median(a, axis=0), axis=1)
        return float(np.percentile(deviation, 95))

    def update(self, now: float, eeg: np.ndarray | None, accel: np.ndarray | None,
               artifact: bool) -> bool:
        """Called every engine tick; recomputes every `update_seconds`.
        `eeg` is the last window (all channels), `accel` the matching
        accelerometer window or None. Returns True when it recomputed."""
        self.just_calibrated = False
        if eeg is None or now - self.last_update < self.cfg.update_seconds:
            return False
        self.last_update = now
        self.motion = self.motion_score(accel)

        if self.motion is None and self.cfg.require_accel:
            self._set("no_accel", "No accelerometer data: can't verify motion, alpha paused.")
        elif self.motion is not None and self.motion > self.cfg.motion_threshold_g:
            self._set("movement", f"Movement ({self.motion:.3f} g): alpha reading rejected.")
        elif artifact:
            self._set("artifact", "Electrode artifact on the alpha channel: reading rejected.")
        else:
            fractions = []
            for ch in self.idx:
                x = eeg[ch]
                spectrum = np.abs(np.fft.rfft((x - x.mean()) * self.window)) ** 2
                broadband = spectrum[self.broad_mask].sum()
                if broadband > 0:
                    fractions.append(spectrum[self.band_mask].sum() / broadband)
            if fractions:
                self._display(now, float(np.mean(fractions)))
        self._check_calibration_end(now)
        return True

    def _set(self, status: str, message: str) -> None:
        if not self.calibrating:
            self.status, self.message = status, message

    def _check_calibration_end(self, now: float) -> None:
        if self.cal_end is None:
            return
        if now >= self.cal_end:
            if len(self.cal_values) >= self.cfg.min_calibration_values:
                self.baseline = float(np.median(self.cal_values))
                self.status, self.message = "calibrated", "Calibrated. Close your eyes: alpha should rise."
                self.just_calibrated = True
            else:
                self.baseline = None
                self.status, self.message = "cal_failed", "Calibration failed: not enough clean data. Keep still and retry."
            self.cal_end = None
            self.smoothed_fraction = None
        elif now < self.cal_start:
            self.status, self.message = "preparing", "Preparing... keep your eyes open and head still."
        else:
            seconds = max(0, int(self.cal_end - now) + 1)
            self.status, self.message = "calibrating", f"Calibrating: eyes OPEN, keep still. {seconds}s"

    def _display(self, now: float, fraction: float) -> None:
        self.fraction = fraction
        if self.cal_end is not None:
            if self.cal_start <= now < self.cal_end:
                self.cal_values.append(fraction)
            return

        if self.smoothed_fraction is None:
            self.smoothed_fraction = fraction
        else:
            self.smoothed_fraction += self.smoothing_weight * (fraction - self.smoothed_fraction)

        if self.baseline is None or self.baseline <= 0:
            self.gain, self.strength = None, 0.0
            self.status, self.message = "uncalibrated", "Calibrate with your eyes open."
            return
        self.gain = self.smoothed_fraction / self.baseline
        self.strength = float(np.clip((self.gain - 1) / (self.cfg.full_scale_gain - 1), 0, 1))
        self.status, self.message = "ok", "Motion rejection + alpha smoothing active."

    def state(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "fraction": self.fraction,
            "smoothed": self.smoothed_fraction,
            "baseline": self.baseline,
            "gain": self.gain,
            "strength": self.strength,
            "motion": self.motion,
            "motion_threshold": self.cfg.motion_threshold_g,
            "calibrating": self.calibrating,
            "full_scale": self.cfg.full_scale_gain,
        }
