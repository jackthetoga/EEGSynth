"""Config loading and the band power -> MIDI CC pipeline."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from eegsynth.artifacts import ArtifactConfig, MotionConfig
from eegsynth.detectors import AlphaConfig, BlinkConfig, EyebrowConfig, EyesConfig, JawConfig
from eegsynth.features import DEFAULT_BANDS

DEFAULT_LABELS = ["Fp1", "Fp2", "F7", "F8", "T7", "T8", "O1", "O2"]


@dataclass
class General:
    labels: list[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    mains_hz: float = 60.0
    midi_channel: int = 1
    window_seconds: float = 2.0
    update_hz: float = 25.0
    adapt_seconds: float = 30.0
    artifact_uv: float = 150.0
    bands: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BANDS))


class Normalizer:
    """Adaptive z-score: tracks the running mean/variance of a feature so its
    output stays centred no matter the person, electrode contact or gain.
    Starts as a cumulative average, then becomes an EMA with `adapt_seconds`."""

    def __init__(self, adapt_seconds: float, update_hz: float):
        self.ema_alpha = 1.0 / max(adapt_seconds * update_hz, 1.0)
        self.n = 0
        self.mean = 0.0
        self.var = 0.0

    def __call__(self, x: float) -> float:
        self.n += 1
        a = max(1.0 / self.n, self.ema_alpha)
        delta = x - self.mean
        self.mean += a * delta
        self.var = (1 - a) * (self.var + a * delta * delta)
        return delta / math.sqrt(self.var) if self.var > 1e-12 else 0.0


@dataclass(eq=False)
class Mapping:
    feature: str
    cc: int
    channels: list[int]  # 1-based, as labelled on the Cyton (N1P..N8P)
    name: str = ""
    relative: bool = True
    invert: bool = False
    smoothing: float = 0.8
    z_range: float = 2.5

    def __post_init__(self):
        self.name = self.name or f"{self.feature} -> CC{self.cc}"
        self._norm: Normalizer | None = None
        self._smoothed: float | None = None
        self.level = 0.0

    @property
    def channel_indices(self) -> list[int]:
        return [c - 1 for c in self.channels]

    def update(self, raw: float, general: General) -> float:
        """Returns the smoothed 0..1 level."""
        if self._norm is None:
            self._norm = Normalizer(general.adapt_seconds, general.update_hz)
        z = self._norm(raw)
        if self.invert:
            z = -z
        unit = min(max(0.5 + z / (2 * self.z_range), 0.0), 1.0)
        if self._smoothed is None:
            self._smoothed = unit
        else:
            self._smoothed = self.smoothing * self._smoothed + (1 - self.smoothing) * unit
        self.level = self._smoothed
        return self.level


@dataclass
class Config:
    general: General
    jaw: JawConfig
    eyebrow: EyebrowConfig
    blink: BlinkConfig
    eyes: EyesConfig
    alpha: AlphaConfig
    motion: MotionConfig
    artifacts: ArtifactConfig
    mappings: list[Mapping]
    enabled: dict[str, bool]

    def channel_names(self, channels: list[int]) -> str:
        return "/".join(self.general.labels[c - 1] for c in channels)


def _build(cls, section: dict, name: str):
    known = {f.name for f in fields(cls)}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in [{name}]: {', '.join(sorted(unknown))}")
    obj = cls(**section)
    chans = getattr(obj, "channels", [])
    if not all(1 <= c <= 8 for c in chans):
        raise ValueError(f"[{name}] channels must be between 1 and 8")
    return obj


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    g = dict(data.get("general", {}))
    bands = dict(DEFAULT_BANDS)
    bands.update({k: tuple(v) for k, v in data.get("bands", {}).items()})
    g["bands"] = bands
    general = _build(General, g, "general")
    if len(general.labels) != 8:
        raise ValueError("[general] labels must list 8 electrode names")
    if not 1 <= general.midi_channel <= 16:
        raise ValueError("[general] midi_channel must be 1-16")

    sections = {}
    enabled = {}
    for name, cls in [("jaw", JawConfig), ("eyebrow", EyebrowConfig), ("blink", BlinkConfig),
                      ("eyes", EyesConfig), ("alpha", AlphaConfig)]:
        section = dict(data.get(name, {}))
        enabled[name] = section.pop("enabled", True)
        sections[name] = _build(cls, section, name)
    sections["motion"] = _build(MotionConfig, data.get("motion", {}), "motion")
    sections["artifacts"] = _build(ArtifactConfig, data.get("artifacts", {}), "artifacts")

    mappings = [_build(Mapping, m, "mapping") for m in data.get("mapping", [])]
    for m in mappings:
        for n in m.feature.split("/"):
            if n.strip() not in bands:
                raise ValueError(f"Unknown band {n.strip()!r} in mapping {m.name!r}")
        if not 0 <= m.cc <= 127:
            raise ValueError(f"Bad cc in mapping {m.name!r}")

    return Config(general=general, mappings=mappings, enabled=enabled, **sections)
