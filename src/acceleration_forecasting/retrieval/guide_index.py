from __future__ import annotations

import json
import sqlite3

import numpy as np

from .guide_search import GuideSearchConfig


class GuideIndex:
    """In-memory, read-only view of the development vector database."""

    def __init__(self, connection: sqlite3.Connection):
        rows = connection.execute(
            """
            SELECT w.record_id, w.measurement_id, w.measurement_date, w.dataset_id,
                   w.embedding, w.embedding_dim, t.trend_id, t.current_acc_z_max,
                   t.future_values, t.future_mask, t.selected_dates,
                   t.guide_available_date, t.cutoff_maintenance_date,
                   t.maintenance_type, t.maintenance_description,
                   t.direction, t.bin_start_m, t.bin_end_m
            FROM waveform_records w JOIN trends t ON t.trend_id=w.trend_id
            """
        ).fetchall()
        if not rows:
            raise ValueError("Retrieval database contains no waveform records")
        dim = int(rows[0][5])
        self.embeddings = np.stack(
            [np.frombuffer(row[4], dtype=np.float32, count=dim) for row in rows]
        )
        self.record_ids = np.asarray([row[0] for row in rows], dtype=object)
        self.measurement_ids = np.asarray([row[1] for row in rows], dtype=object)
        self.dates = np.asarray([str(row[2]) for row in rows], dtype="U10")
        self.datasets = np.asarray([str(row[3]) for row in rows], dtype=object)
        self.trend_ids = np.asarray([row[6] for row in rows], dtype=object)
        self.current = np.asarray([row[7] for row in rows], dtype=np.float32)
        self.future_values = [json.loads(row[8]) for row in rows]
        self.future_masks = [json.loads(row[9]) for row in rows]
        self.selected_dates = [json.loads(row[10]) for row in rows]
        self.available_dates = np.asarray([str(row[11]) for row in rows], dtype="U10")
        self.cutoffs = [row[12] or "" for row in rows]
        self.maintenance_types = [row[13] or "" for row in rows]
        self.maintenance_descriptions = [row[14] or "" for row in rows]
        self.directions = [row[15] for row in rows]
        self.bin_starts = np.asarray([row[16] for row in rows], dtype=np.float32)
        self.bin_ends = np.asarray([row[17] for row in rows], dtype=np.float32)
        self.valid_months = np.asarray(
            [int(np.asarray(mask, dtype=np.int8).sum()) for mask in self.future_masks],
            dtype=np.int16,
        )

    def search(
        self,
        query_embeddings,
        *,
        query_date,
        query_dataset_id,
        query_current,
        allowed_dataset_ids,
        config: GuideSearchConfig,
    ):
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.maximum(norms, 1e-12)
        allowed = set(str(value) for value in allowed_dataset_ids)
        eligible = np.fromiter((dataset in allowed for dataset in self.datasets), dtype=bool)
        eligible &= self.datasets != str(query_dataset_id)
        eligible &= self.dates != str(query_date)
        eligible &= np.isfinite(self.current)
        eligible &= np.abs(self.current - float(query_current)) <= config.max_current_difference + 1e-12
        eligible &= self.valid_months >= config.min_valid_months
        if config.strict_time:
            eligible &= self.available_dates < str(query_date)
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return []
        similarities = queries @ self.embeddings[indices].T
        best_similarity = similarities.max(axis=0)
        order = np.argsort(-best_similarity, kind="stable")
        selected = []
        selected_dates = set()
        for offset in order:
            index = int(indices[int(offset)])
            date = str(self.dates[index])
            if date in selected_dates:
                continue
            selected_dates.add(date)
            selected.append(
                {
                    "record_id": self.record_ids[index],
                    "measurement_id": self.measurement_ids[index],
                    "measurement_date": date,
                    "dataset_id": self.datasets[index],
                    "trend_id": self.trend_ids[index],
                    "current_acc_z_max": float(self.current[index]),
                    "current_max_difference": float(self.current[index] - query_current),
                    "future_values": self.future_values[index],
                    "future_mask": self.future_masks[index],
                    "selected_dates": self.selected_dates[index],
                    "guide_available_date": str(self.available_dates[index]),
                    "cutoff_maintenance_date": self.cutoffs[index],
                    "maintenance_type": self.maintenance_types[index],
                    "maintenance_description": self.maintenance_descriptions[index],
                    "direction": self.directions[index],
                    "bin_start_m": float(self.bin_starts[index]),
                    "bin_end_m": float(self.bin_ends[index]),
                    "valid_months": int(self.valid_months[index]),
                    "similarity": float(best_similarity[int(offset)]),
                }
            )
            if len(selected) >= config.top_k:
                break
        return selected

