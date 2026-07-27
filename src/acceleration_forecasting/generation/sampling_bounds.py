from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from acceleration_forecasting.datasets.normalization import AccelerationNormalization


DEFAULT_PHYSICAL_MIN = 0.3
DEFAULT_PHYSICAL_MAX = 5.0


@dataclass(frozen=True)
class SamplingBounds:
    physical_min: float
    physical_max: float
    normalized_min: float
    normalized_max: float
    valid_training_value_count: int
    source_split: str = "model_train"
    bounds_policy: str = "fixed_physical"

    @property
    def normalized(self):
        return self.normalized_min, self.normalized_max

    @property
    def physical_width(self):
        return self.physical_max - self.physical_min

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


def fit_sampling_bounds(
    dataset_dir, physical_min=DEFAULT_PHYSICAL_MIN, physical_max=DEFAULT_PHYSICAL_MAX
):
    dataset_dir = Path(dataset_dir)
    values = np.load(dataset_dir / "model_train" / "target_values.npy", mmap_mode="r")
    masks = np.load(dataset_dir / "model_train" / "target_masks.npy", mmap_mode="r")
    valid = (np.asarray(masks) > 0) & np.isfinite(values)
    physical = np.asarray(values)[valid].astype(np.float64, copy=False)
    if physical.size == 0:
        raise ValueError("model_train has no finite masked target values for sampling bounds")
    physical_min = float(physical_min)
    physical_max = float(physical_max)
    if not np.isfinite(physical_min) or not np.isfinite(physical_max) or physical_min >= physical_max:
        raise ValueError("model_train sampling bounds must be finite and non-degenerate")
    normalization = AccelerationNormalization.load(dataset_dir / "normalization.json")
    normalized = normalization.normalize([physical_min, physical_max]).astype(float)
    return SamplingBounds(
        physical_min=physical_min,
        physical_max=physical_max,
        normalized_min=float(normalized[0]),
        normalized_max=float(normalized[1]),
        valid_training_value_count=int(physical.size),
    )
