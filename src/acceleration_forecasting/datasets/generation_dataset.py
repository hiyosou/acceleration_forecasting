from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .normalization import AccelerationNormalization


INPUT_ARRAYS = (
    "current_values",
    "history_values",
    "history_masks",
    "guide_values",
    "guide_deltas",
    "guide_masks",
    "guide_similarities",
    "retrieval_masks",
)


class GenerationDataset(Dataset):
    def __init__(self, split_dir, normalization_path, include_targets=True):
        self.split_dir = Path(split_dir)
        self.metadata = pd.read_csv(self.split_dir / "metadata.csv", encoding="utf-8-sig")
        self.arrays = {
            name: np.load(self.split_dir / f"{name}.npy", mmap_mode="r")
            for name in INPUT_ARRAYS
        }
        self.include_targets = include_targets
        if include_targets:
            self.targets = np.load(self.split_dir / "target_values.npy", mmap_mode="r")
            self.target_masks = np.load(self.split_dir / "target_masks.npy", mmap_mode="r")
        self.normalization = AccelerationNormalization.load(normalization_path)
        condition_path = Path(normalization_path).parent / "condition_normalization.json"
        self.condition_normalization = (
            AccelerationNormalization.load(condition_path)
            if condition_path.is_file() else self.normalization
        )
        baseline_path = self.split_dir / "guide_baselines.npy"
        self.guide_baselines = (
            np.load(baseline_path, mmap_mode="r") if baseline_path.is_file() else None
        )
        self.is_residual = self.guide_baselines is not None

    def __len__(self):
        return len(self.metadata)

    def _normalized(self, values, normalization=None):
        normalized = (normalization or self.normalization).normalize(values)
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(self, index):
        current = self._normalized(self.arrays["current_values"][index], self.condition_normalization)
        history = self._normalized(self.arrays["history_values"][index], self.condition_normalization)
        guide = self._normalized(self.arrays["guide_values"][index], self.condition_normalization)
        delta = np.nan_to_num(
            self.condition_normalization.normalize_delta(self.arrays["guide_deltas"][index]), nan=0.0
        )
        item = {
            "current": torch.as_tensor(current, dtype=torch.float32),
            "history": torch.as_tensor(history, dtype=torch.float32),
            "history_mask": torch.as_tensor(self.arrays["history_masks"][index].copy(), dtype=torch.float32),
            "guide_values": torch.as_tensor(guide, dtype=torch.float32),
            "guide_deltas": torch.as_tensor(delta, dtype=torch.float32),
            "guide_mask": torch.as_tensor(self.arrays["guide_masks"][index].copy(), dtype=torch.float32),
            "guide_similarities": torch.as_tensor(self.arrays["guide_similarities"][index].copy(), dtype=torch.float32),
            "retrieval_mask": torch.as_tensor(self.arrays["retrieval_masks"][index].copy(), dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.include_targets:
            item["target"] = torch.as_tensor(self._normalized(self.targets[index]), dtype=torch.float32)
            item["target_mask"] = torch.as_tensor(self.target_masks[index].copy(), dtype=torch.float32)
        return item

    def denormalize_prediction(self, normalized, index, clip_bounds=(0.3, 5.0)):
        values = self.normalization.denormalize(normalized, clip_nonnegative=False)
        if self.is_residual:
            values = values + np.asarray(self.guide_baselines[index], dtype=np.float32)
        if clip_bounds is not None:
            values = np.clip(values, float(clip_bounds[0]), float(clip_bounds[1]))
        return values

    def physical_target(self, index):
        if not self.include_targets:
            raise ValueError("targets are not loaded")
        values = self.normalization.denormalize(self.targets[index], clip_nonnegative=False)
        if self.is_residual:
            values = values + np.asarray(self.guide_baselines[index], dtype=np.float32)
        return values
