from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .normalization import AccelerationNormalization


@dataclass(frozen=True)
class ResidualConfig:
    mode: str
    softmax_temperature: float
    clip_quantile_low: float
    clip_quantile_high: float
    residual_clip_physical: tuple[float, float]
    residual_clip_normalized: tuple[float, float]
    final_physical_bounds: tuple[float, float]
    source_dataset_dir: str
    source_dataset_build_id: str
    dataset_build_id: str

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def masked_softmax_baseline(values, masks, similarities, retrieval_masks, current, temperature=0.1):
    values = np.asarray(values, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.float64)
    similarities = np.asarray(similarities, dtype=np.float64)
    retrieval_masks = np.asarray(retrieval_masks, dtype=np.float64)
    if values.shape != masks.shape or values.ndim != 2:
        raise ValueError("guide values and masks must have shape [guide, month]")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("softmax temperature must be finite and positive")
    valid = (masks > 0) & np.isfinite(values) & (retrieval_masks[:, None] > 0)
    scores = similarities / float(temperature)
    baseline = np.full(values.shape[1], np.nan, dtype=np.float64)
    weights = np.zeros_like(values, dtype=np.float64)
    source = np.full(values.shape[1], "weighted_guides", dtype="U24")
    for month in range(values.shape[1]):
        selected = valid[:, month]
        if not selected.any():
            continue
        logits = scores[selected]
        logits = logits - np.max(logits)
        month_weights = np.exp(logits)
        month_weights /= month_weights.sum()
        weights[selected, month] = month_weights
        baseline[month] = float(np.sum(month_weights * values[selected, month]))
    finite = np.isfinite(baseline)
    if finite.any():
        indices = np.arange(len(baseline))
        first, last = int(indices[finite][0]), int(indices[finite][-1])
        internal = (~finite) & (indices > first) & (indices < last)
        if internal.any():
            baseline[internal] = np.interp(indices[internal], indices[finite], baseline[finite])
            source[internal] = "interpolated"
        if first > 0:
            baseline[:first] = baseline[first]
            source[:first] = "edge_filled"
        if last + 1 < len(baseline):
            baseline[last + 1:] = baseline[last]
            source[last + 1:] = "edge_filled"
    else:
        baseline.fill(float(current))
        source[:] = "current_value_fallback"
    if not np.isfinite(baseline).all():
        raise ValueError("failed to construct a finite residual baseline")
    return baseline.astype(np.float32), weights.astype(np.float32), source


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_inputs(source, destination, include_targets):
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and (include_targets or path.name not in {"target_values.npy", "target_masks.npy"}):
            shutil.copy2(path, destination / path.name)


def _derive_split(source, destination, temperature, include_targets):
    guide_values = np.load(source / "guide_values.npy", mmap_mode="r")
    guide_masks = np.load(source / "guide_masks.npy", mmap_mode="r")
    similarities = np.load(source / "guide_similarities.npy", mmap_mode="r")
    retrieval_masks = np.load(source / "retrieval_masks.npy", mmap_mode="r")
    current = np.load(source / "current_values.npy", mmap_mode="r")
    baselines, weights, sources = [], [], []
    for index in range(len(current)):
        baseline, weight, source_kind = masked_softmax_baseline(
            guide_values[index], guide_masks[index], similarities[index],
            retrieval_masks[index], float(np.ravel(current[index])[0]), temperature,
        )
        baselines.append(baseline)
        weights.append(weight)
        sources.append(source_kind)
    _copy_inputs(source, destination, include_targets=False)
    np.save(destination / "guide_baselines.npy", np.asarray(baselines, dtype=np.float32))
    np.save(destination / "guide_softmax_weights.npy", np.asarray(weights, dtype=np.float32))
    np.save(destination / "guide_baseline_sources.npy", np.asarray(sources))
    if include_targets:
        targets = np.load(source / "target_values.npy", mmap_mode="r")
        masks = np.load(source / "target_masks.npy", mmap_mode="r")
        residuals = np.asarray(targets, dtype=np.float32) - np.asarray(baselines, dtype=np.float32)
        residuals = np.where(np.asarray(masks) > 0, residuals, np.nan).astype(np.float32)
        np.save(destination / "target_values.npy", residuals)
        np.save(destination / "target_masks.npy", np.asarray(masks, dtype=np.float32))
        return residuals, np.asarray(masks, dtype=np.float32)
    return None, None


def prepare_residual_dataset(
    source_dataset_dir, output_dir, *, temperature=0.1,
    clip_quantile_low=0.5, clip_quantile_high=99.5,
):
    source, output = Path(source_dataset_dir).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_config = json.loads((source / "guide_search_config.json").read_text(encoding="utf-8"))
    train_values, train_masks = _derive_split(
        source / "model_train", output / "model_train", temperature, True
    )
    _derive_split(source / "model_validation", output / "model_validation", temperature, True)
    _derive_split(source / "inference" / "inputs", output / "inference" / "inputs", temperature, False)
    targets_dir = output / "inference" / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    for path in (source / "inference" / "targets").iterdir():
        if path.is_file():
            shutil.copy2(path, targets_dir / path.name)
    condition_norm = AccelerationNormalization.load(source / "normalization.json")
    condition_norm.save(output / "condition_normalization.json")
    valid = (train_masks > 0) & np.isfinite(train_values)
    physical = train_values[valid].astype(np.float64)
    if physical.size < 2:
        raise ValueError("model_train has insufficient finite residual targets")
    residual_norm = AccelerationNormalization(
        float(physical.mean()), float(physical.std(ddof=0)), int(physical.size), "model_train_residual"
    )
    residual_norm.save(output / "normalization.json")
    q_low, q_high = np.percentile(physical, [clip_quantile_low, clip_quantile_high])
    radius = float(max(abs(q_low), abs(q_high)))
    normalized_clip = tuple(float(x) for x in residual_norm.normalize([-radius, radius]))
    identity = {
        "source_dataset_build_id": source_config["dataset_build_id"],
        "temperature": float(temperature), "clip_quantile_low": float(clip_quantile_low),
        "clip_quantile_high": float(clip_quantile_high), "mode": "guide_baseline_residual",
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    config = ResidualConfig(
        "guide_baseline_residual", float(temperature), float(clip_quantile_low),
        float(clip_quantile_high), (-radius, radius), normalized_clip, (0.3, 5.0),
        str(source), source_config["dataset_build_id"], build_id,
    )
    config.save(output / "residual_config.json")
    residual_search = {**source_config, "dataset_build_id": build_id, "residual": asdict(config)}
    (output / "guide_search_config.json").write_text(
        json.dumps(residual_search, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name in ("guide_assignments.csv", "source_retrieval_artifacts.json", "dataset_summary.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    audit = {
        "source_dataset_dir": str(source),
        "source_files": {
            "guide_search_config.json": _sha256(source / "guide_search_config.json"),
            "normalization.json": _sha256(source / "normalization.json"),
        },
        "residual_mean": residual_norm.mean, "residual_std": residual_norm.std,
        "valid_residual_count": residual_norm.fitted_observation_count,
        "residual_clip_physical": [-radius, radius],
    }
    (output / "residual_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**audit, "dataset_build_id": build_id, "softmax_temperature": float(temperature)}
