from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting.common.progress import progress_bar, progress_message
from acceleration_forecasting.datasets.generation_dataset import GenerationDataset

from .sampling import load_ema_checkpoint, sample_one


def _condition_batch(item):
    return {
        key: value.unsqueeze(0)
        for key, value in item.items()
        if key not in ("index", "target", "target_mask")
    }


def predict(
    dataset_dir,
    selection_file,
    output_dir,
    *,
    device=None,
    num_samples=100,
    sampling_steps=50,
    eta=0.0,
    seed=42,
    save_samples=True,
    chunk_size=100,
    max_records=None,
    progress=True,
):
    started = time.time()
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(selection_file).read_text(encoding="utf-8"))
    model, model_name, actual_device = load_ema_checkpoint(selection["selected_checkpoint"], device)
    dataset = GenerationDataset(
        dataset_dir / "inference" / "inputs", dataset_dir / "normalization.json",
        include_targets=False,
    )
    prediction_path = output_dir / "predictions.csv"
    existing = pd.read_csv(prediction_path, encoding="utf-8-sig") if prediction_path.is_file() else pd.DataFrame()
    completed = set(existing["target_id"].astype(str)) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []
    sample_dir = output_dir / "samples"
    if save_samples:
        sample_dir.mkdir(parents=True, exist_ok=True)
    pending_samples, pending_ids, chunk_index = [], [], len(list(sample_dir.glob("samples_*.npz")))
    count = min(len(dataset), max_records or len(dataset))
    scoped_ids = set(dataset.metadata.iloc[:count]["target_id"].astype(str))
    completed_in_scope = len(completed & scoped_ids)
    progress_message(
        f"推論対象={count}, 完了済み={completed_in_scope}, 未処理={count - completed_in_scope}",
        enabled=progress,
    )
    target_progress = progress_bar(
        enabled=progress, total=count, initial=completed_in_scope,
        desc=f"{model_name} 推論", unit="target",
    )
    for index in range(count):
        meta = dataset.metadata.iloc[index]
        target_id = str(meta["target_id"])
        if target_id in completed:
            continue
        item = dataset[index]
        normalized = sample_one(
            model, _condition_batch(item), target_id, num_samples=num_samples,
            sampling_steps=sampling_steps, eta=eta, seed=seed,
        )
        samples = dataset.normalization.denormalize(normalized, clip_nonnegative=True)
        mean = samples.mean(axis=0)
        median = np.median(samples, axis=0)
        p10, p90 = np.percentile(samples, [10, 90], axis=0)
        std = samples.std(axis=0)
        anchor = pd.Timestamp(meta["anchor_date"])
        for month in range(18):
            rows.append({
                "model_name": model_name, "target_id": target_id, "dataset_id": meta["dataset_id"],
                "anchor_date": meta["anchor_date"], "direction": meta["direction"],
                "bin_start_m": meta["bin_start_m"], "bin_end_m": meta["bin_end_m"],
                "month_index": month + 1,
                "target_month": (anchor.replace(day=1) + pd.DateOffset(months=month + 1)).strftime("%Y-%m-%d"),
                "prediction_mean": float(mean[month]), "prediction_median": float(median[month]),
                "prediction_p10": float(p10[month]), "prediction_p90": float(p90[month]),
                "prediction_std": float(std[month]), "sample_count": int(num_samples),
                "guide_count": int(meta["guide_count"]),
            })
        completed.add(target_id)
        if save_samples:
            pending_ids.append(target_id)
            pending_samples.append(samples.astype(np.float32))
            if len(pending_ids) >= chunk_size:
                target_progress.set_postfix(state="保存中", refresh=progress)
                np.savez_compressed(sample_dir / f"samples_{chunk_index:05d}.npz", target_ids=np.asarray(pending_ids), samples=np.stack(pending_samples))
                pending_ids, pending_samples, chunk_index = [], [], chunk_index + 1
        if len(completed) % chunk_size == 0:
            pd.DataFrame(rows).drop_duplicates(["target_id", "month_index"], keep="last").to_csv(prediction_path, index=False, encoding="utf-8-sig")
        target_progress.set_postfix(state="推論中", refresh=False)
        target_progress.update(1)
    if pending_ids:
        target_progress.set_postfix(state="保存中", refresh=progress)
        np.savez_compressed(sample_dir / f"samples_{chunk_index:05d}.npz", target_ids=np.asarray(pending_ids), samples=np.stack(pending_samples))
    target_progress.close()
    final = pd.DataFrame(rows).drop_duplicates(["target_id", "month_index"], keep="last")
    final.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    assignments = dataset_dir / "guide_assignments.csv"
    if assignments.is_file():
        guides = pd.read_csv(assignments, encoding="utf-8-sig")
        guides.loc[guides["split"] == "inference"].to_csv(output_dir / "prediction_guides.csv", index=False, encoding="utf-8-sig")
    run = {
        "model_name": model_name, "checkpoint": selection["selected_checkpoint"],
        "device": str(actual_device), "num_samples": num_samples,
        "sampling_steps": sampling_steps, "eta": eta, "seed": seed,
        "target_count": int(final["target_id"].nunique()), "elapsed_seconds": time.time() - started,
        "targets_read": False,
    }
    (output_dir / "prediction_run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    return run
