from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .artifacts import resolve_artifact_layout
from .constants import EMBEDDING_DIM, SAMPLES_PER_BIN
from .data import open_waveforms
from .database import (
    initialize_database,
    insert_trends,
    insert_waveform_record,
    search_embedding,
    store_metadata,
)
from .model import l2_normalize
from .training import MemmapWaveformDataset, load_trained_model


def build_database(artifact_dir, device=None, batch_size=512, overwrite=True):
    artifact_dir = Path(artifact_dir)
    layout = resolve_artifact_layout(artifact_dir)
    manifest = pd.read_csv(layout["manifest_path"], encoding="utf-8-sig")
    trends = pd.read_csv(layout["trend_path"], encoding="utf-8-sig")
    if layout["format"] == "dataset_split":
        database_rows = manifest.loc[
            manifest["outer_split"] == "development"
        ].copy()
        split_label = "development"
    else:
        database_rows = manifest.loc[manifest["split"] == "database"].copy()
        split_label = "database"
    if database_rows.empty:
        raise ValueError("DBへ登録するdatabaseレコードがありません。")
    referenced_trends = set(database_rows["trend_id"].astype(str))
    trends = trends.loc[trends["trend_id"].astype(str).isin(referenced_trends)].copy()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model, mean, std = load_trained_model(
        artifact_dir / "autoencoder.pt", device
    )

    dataset = MemmapWaveformDataset(
        layout["waveform_path"],
        len(manifest),
        database_rows["waveform_index"].astype(int).to_numpy(),
        mean,
        std,
    )
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=0
    )
    embeddings = []
    with torch.no_grad():
        for waveform in loader:
            embedding = l2_normalize(model.encode(waveform.to(device)))
            embeddings.append(embedding.cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(embeddings, axis=0)

    db_path = artifact_dir / "vector_database.sqlite"
    connection = initialize_database(db_path, overwrite=overwrite)
    try:
        insert_trends(connection, trends)
        for (_, row), embedding in zip(database_rows.iterrows(), embeddings):
            insert_waveform_record(connection, row, embedding)
        store_metadata(
            connection,
            {
                "embedding_dim": EMBEDDING_DIM,
                "samples_per_waveform": SAMPLES_PER_BIN,
                "record_count": int(len(database_rows)),
                "similarity": "cosine",
                "split": split_label,
                "artifact_format": layout["format"],
            },
        )
        connection.commit()
    finally:
        connection.close()
    return {
        "database_path": str(db_path),
        "records": int(len(database_rows)),
        "trends": int(len(trends)),
    }


def search_manifest_record(
    artifact_dir,
    record_id,
    device=None,
    top_k=3,
    strict_time=True,
):
    from .database import connect_database

    artifact_dir = Path(artifact_dir)
    layout = resolve_artifact_layout(artifact_dir)
    if layout["format"] == "dataset_split":
        manifest = pd.read_csv(
            artifact_dir / "inference_inputs.csv", encoding="utf-8-sig"
        )
    else:
        manifest = pd.read_csv(layout["manifest_path"], encoding="utf-8-sig")
    matching = manifest.loc[manifest["record_id"].astype(str) == str(record_id)]
    if matching.empty:
        raise KeyError(f"record_id がmanifestにありません: {record_id}")
    row = matching.iloc[0]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model, mean, std = load_trained_model(
        artifact_dir / "autoencoder.pt", device
    )
    waveforms = open_waveforms(
        layout["waveform_path"], int(layout["record_count"] or len(manifest))
    )
    waveform = np.asarray(
        waveforms[int(row["waveform_index"])], dtype=np.float32
    ).copy()
    tensor = torch.from_numpy((waveform - mean) / std).reshape(1, 1, -1).to(device)
    with torch.no_grad():
        query_embedding = l2_normalize(model.encode(tensor))[0].cpu().numpy()

    connection = connect_database(artifact_dir / "vector_database.sqlite")
    try:
        results = search_embedding(
            connection,
            query_embedding,
            query_date=row["measurement_date"],
            top_k=top_k,
            query_dataset_id=row.get("dataset_id", None),
            strict_time=bool(strict_time and layout["format"] == "dataset_split"),
        )
    finally:
        connection.close()
    output = {
        "query": {
            "record_id": row["record_id"],
            "measurement_id": row["measurement_id"],
            "measurement_date": row["measurement_date"],
            "direction": row["direction"],
            "bin_start_m": float(row["bin_start_m"]),
            "bin_end_m": float(row["bin_end_m"]),
            "split": (
                "inference"
                if layout["format"] == "dataset_split"
                else row["split"]
            ),
            "dataset_id": row.get("dataset_id", ""),
        },
        "results": results,
        "requested_top_k": int(top_k),
        "returned_count": int(len(results)),
        "shortfall_reason": (
            ""
            if len(results) >= int(top_k)
            else "causal_and_uniqueness_filters_left_fewer_candidates"
        ),
    }
    return output


def save_search_result(result, output_path):
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
