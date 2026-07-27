from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AccelerationNormalization:
    mean: float
    std: float
    fitted_observation_count: int
    split: str = "model_train"

    def normalize(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def denormalize(self, values, clip_nonnegative=False):
        result = np.asarray(values, dtype=np.float32) * self.std + self.mean
        return np.maximum(result, 0.0) if clip_nonnegative else result

    def normalize_delta(self, values):
        return np.asarray(values, dtype=np.float32) / self.std

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_normalization(frame):
    unique = frame.drop_duplicates(["dataset_id", "measurement_date"])
    values = unique["current_acc_z_max"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite model_train observations are required")
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        raise ValueError("Training acceleration standard deviation must be positive")
    return AccelerationNormalization(float(values.mean()), std, int(values.size))

