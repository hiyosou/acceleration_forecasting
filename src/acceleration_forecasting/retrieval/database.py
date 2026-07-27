from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trends (
    trend_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    direction TEXT NOT NULL,
    bin_start_m REAL NOT NULL,
    bin_end_m REAL NOT NULL,
    current_acc_z_max REAL NOT NULL,
    future_values TEXT NOT NULL,
    future_mask TEXT NOT NULL,
    selected_dates TEXT NOT NULL,
    guide_available_date TEXT NOT NULL,
    cutoff_maintenance_date TEXT,
    maintenance_type TEXT,
    maintenance_description TEXT
);
CREATE TABLE IF NOT EXISTS waveform_records (
    record_id TEXT PRIMARY KEY,
    measurement_id TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    direction TEXT NOT NULL,
    bin_start_m REAL NOT NULL,
    bin_end_m REAL NOT NULL,
    mean_velocity_kmh REAL NOT NULL,
    source_csv_path TEXT NOT NULL,
    waveform_sha256 TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    trend_id TEXT NOT NULL REFERENCES trends(trend_id),
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_waveform_date
ON waveform_records(measurement_date);
CREATE INDEX IF NOT EXISTS idx_waveform_trend
ON waveform_records(trend_id);
"""


def connect_database(path):
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(path, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    connection = connect_database(path)
    connection.executescript(SCHEMA)
    return connection


def insert_trends(connection, trend_frame):
    trend_frame = trend_frame.copy()
    if "guide_available_date" not in trend_frame.columns:
        anchor_month = pd.to_datetime(
            trend_frame["measurement_date"], errors="raise"
        ).dt.to_period("M").dt.to_timestamp()
        trend_frame["guide_available_date"] = (
            anchor_month + pd.DateOffset(months=18) + pd.Timedelta(days=15)
        ).dt.strftime("%Y-%m-%d")
    columns = [
        "trend_id",
        "dataset_id",
        "measurement_date",
        "direction",
        "bin_start_m",
        "bin_end_m",
        "current_acc_z_max",
        "future_values",
        "future_mask",
        "selected_dates",
        "guide_available_date",
        "cutoff_maintenance_date",
        "maintenance_type",
        "maintenance_description",
    ]
    connection.executemany(
        f"INSERT OR REPLACE INTO trends ({','.join(columns)}) "
        f"VALUES ({','.join('?' for _ in columns)})",
        [
            tuple(None if pd.isna(row[column]) else row[column] for column in columns)
            for _, row in trend_frame.iterrows()
        ],
    )


def insert_waveform_record(connection, row, embedding):
    embedding = np.asarray(embedding, dtype=np.float32)
    connection.execute(
        """
        INSERT INTO waveform_records (
            record_id, measurement_id, measurement_date, direction,
            bin_start_m, bin_end_m, mean_velocity_kmh, source_csv_path,
            waveform_sha256, dataset_id, trend_id, embedding, embedding_dim
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["record_id"],
            row["measurement_id"],
            row["measurement_date"],
            row["direction"],
            float(row["bin_start_m"]),
            float(row["bin_end_m"]),
            float(row["mean_velocity_kmh"]),
            row["source_csv_path"],
            row["waveform_sha256"],
            row.get("dataset_id", ""),
            row["trend_id"],
            embedding.tobytes(order="C"),
            int(embedding.size),
        ),
    )


def store_metadata(connection, values):
    connection.executemany(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        [
            (
                str(key),
                json.dumps(value, ensure_ascii=False)
                if not isinstance(value, str)
                else value,
            )
            for key, value in values.items()
        ],
    )


def search_embedding(
    connection,
    query_embedding,
    query_date,
    top_k=3,
    query_dataset_id=None,
    strict_time=False,
):
    query = np.asarray(query_embedding, dtype=np.float32)
    norm = float(np.linalg.norm(query))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("検索ベクトルのノルムが0または非有限です。")
    query = query / norm

    rows = connection.execute(
        """
        SELECT w.record_id, w.measurement_id, w.measurement_date, w.direction,
               w.bin_start_m, w.bin_end_m, w.mean_velocity_kmh, w.embedding,
               w.embedding_dim, t.trend_id, t.current_acc_z_max,
               t.future_values, t.future_mask, t.selected_dates,
               t.cutoff_maintenance_date, t.maintenance_type,
               w.dataset_id, t.dataset_id, t.guide_available_date
        FROM waveform_records AS w
        JOIN trends AS t ON t.trend_id = w.trend_id
        """
    ).fetchall()
    candidates = []
    for row in rows:
        if str(row[2]) == str(query_date):
            continue
        if query_dataset_id and str(row[17]) == str(query_dataset_id):
            continue
        if strict_time and str(row[18]) >= str(query_date):
            continue
        embedding = np.frombuffer(row[7], dtype=np.float32, count=int(row[8]))
        if embedding.size != query.size:
            continue
        similarity = float(np.dot(query, embedding))
        candidates.append((similarity, row))
    candidates.sort(key=lambda item: (-item[0], item[1][2], item[1][0]))

    selected = []
    selected_dates = set()
    for similarity, row in candidates:
        date = str(row[2])
        if date in selected_dates:
            continue
        selected_dates.add(date)
        selected.append(
            {
                "similarity": similarity,
                "record_id": row[0],
                "measurement_id": row[1],
                "measurement_date": date,
                "direction": row[3],
                "bin_start_m": float(row[4]),
                "bin_end_m": float(row[5]),
                "mean_velocity_kmh": float(row[6]),
                "trend_id": row[9],
                "current_acc_z_max": float(row[10]),
                "future_values": json.loads(row[11]),
                "future_mask": json.loads(row[12]),
                "selected_dates": json.loads(row[13]),
                "cutoff_maintenance_date": row[14] or "",
                "maintenance_type": row[15] or "",
                "dataset_id": row[17],
                "guide_available_date": row[18],
            }
        )
        if len(selected) >= int(top_k):
            break
    return selected
