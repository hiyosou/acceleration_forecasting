from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GuideSearchConfig:
    top_k: int = 3
    max_current_difference: float = 0.5
    min_valid_months: int = 12
    strict_time: bool = True
    guide_search_mode: str = "strict_time"
    near_distance_m: float = 100.0
    spatial_tolerance_m: float = 1e-6
    near_candidates_require_complete_past: bool = False
    far_candidates_strict_time: bool = True
    exclude_same_dataset: bool = True


def _load_rows(connection: sqlite3.Connection, allowed_dataset_ids=None):
    sql = """
        SELECT w.record_id, w.measurement_id, w.measurement_date,
               w.embedding, w.embedding_dim, w.dataset_id,
               t.trend_id, t.current_acc_z_max, t.future_values,
               t.future_mask, t.selected_dates, t.guide_available_date,
               t.cutoff_maintenance_date, t.maintenance_type,
               t.maintenance_description, t.direction,
               t.bin_start_m, t.bin_end_m
        FROM waveform_records AS w
        JOIN trends AS t ON t.trend_id = w.trend_id
    """
    rows = connection.execute(sql).fetchall()
    if allowed_dataset_ids is None:
        return rows
    allowed = {str(value) for value in allowed_dataset_ids}
    return [row for row in rows if str(row[5]) in allowed]


def search_guides_for_embeddings(
    connection: sqlite3.Connection,
    query_embeddings,
    *,
    query_date,
    query_dataset_id,
    query_current_acc_z_max,
    query_bin_start_m=None,
    allowed_dataset_ids=None,
    config: GuideSearchConfig | None = None,
):
    """Search all query waveforms and return at most three distinct guide dates."""
    config = config or GuideSearchConfig()
    embeddings = np.asarray(query_embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("Query embeddings must be finite and non-zero")
    embeddings = embeddings / norms
    query_current = float(query_current_acc_z_max)
    if not np.isfinite(query_current):
        raise ValueError("Query current acceleration must be finite")

    best_by_date = {}
    for row in _load_rows(connection, allowed_dataset_ids):
        candidate_date = str(row[2])
        candidate_dataset = str(row[5])
        if candidate_date == str(query_date):
            continue
        if config.exclude_same_dataset and candidate_dataset == str(query_dataset_id):
            continue
        available_date = str(row[11])
        distance_difference = (
            abs(float(row[16]) - float(query_bin_start_m))
            if query_bin_start_m is not None else float("inf")
        )
        spatially_near = distance_difference <= (
            config.near_distance_m + config.spatial_tolerance_m
        )
        require_past = config.strict_time or (
            spatially_near and config.near_candidates_require_complete_past
        ) or (not spatially_near and config.far_candidates_strict_time)
        if require_past and available_date >= str(query_date):
            continue
        current = float(row[7])
        difference = current - query_current
        if not np.isfinite(current) or abs(difference) > config.max_current_difference + 1e-12:
            continue
        mask = np.asarray(json.loads(row[9]), dtype=np.int8)
        valid_months = int(mask.sum())
        if valid_months < config.min_valid_months:
            continue
        candidate_embedding = np.frombuffer(row[3], dtype=np.float32, count=int(row[4]))
        if candidate_embedding.size != embeddings.shape[1]:
            continue
        similarities = embeddings @ candidate_embedding
        similarity = float(np.max(similarities))
        if not np.isfinite(similarity):
            continue
        item = {
            "record_id": row[0],
            "measurement_id": row[1],
            "measurement_date": candidate_date,
            "dataset_id": candidate_dataset,
            "trend_id": row[6],
            "current_acc_z_max": current,
            "current_max_difference": difference,
            "future_values": json.loads(row[8]),
            "future_mask": mask.astype(int).tolist(),
            "selected_dates": json.loads(row[10]),
            "guide_available_date": available_date,
            "cutoff_maintenance_date": row[12] or "",
            "maintenance_type": row[13] or "",
            "maintenance_description": row[14] or "",
            "direction": row[15],
            "bin_start_m": float(row[16]),
            "bin_end_m": float(row[17]),
            "valid_months": valid_months,
            "similarity": similarity,
            "distance_difference_m": distance_difference,
            "spatially_near": spatially_near,
            "temporal_condition_applied": require_past,
        }
        existing = best_by_date.get(candidate_date)
        if existing is None or (similarity, str(row[0])) > (
            existing["similarity"], str(existing["record_id"])
        ):
            best_by_date[candidate_date] = item

    ranked = sorted(
        best_by_date.values(),
        key=lambda item: (-item["similarity"], item["measurement_date"], item["record_id"]),
    )
    return ranked[: config.top_k]


def connect_read_only(path: str | Path):
    path = Path(path).resolve()
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
