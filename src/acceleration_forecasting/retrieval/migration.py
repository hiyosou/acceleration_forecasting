from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


TRACKED_ARTIFACTS = (
    "autoencoder.pt",
    "dataset_split_manifest.csv",
    "development_trends.csv",
    "inference_inputs.csv",
    "inference_targets.csv",
    "normalization.json",
    "source_artifacts.json",
    "split_summary.json",
    "training_history.csv",
    "vector_database.sqlite",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_migration(artifact_dir: str | Path, output: str | Path | None = None):
    artifact_dir = Path(artifact_dir).resolve()
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"Retrieval artifact directory not found: {artifact_dir}")
    files = {}
    for name in TRACKED_ARTIFACTS:
        path = artifact_dir / name
        if path.is_file():
            files[name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}

    database_path = artifact_dir / "vector_database.sqlite"
    counts = {}
    if database_path.is_file():
        uri = f"file:{database_path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            for table in ("waveform_records", "trends", "metadata"):
                counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            dimensions = connection.execute(
                "SELECT MIN(embedding_dim), MAX(embedding_dim) FROM waveform_records"
            ).fetchone()
            counts["embedding_dim_min"] = int(dimensions[0] or 0)
            counts["embedding_dim_max"] = int(dimensions[1] or 0)
        finally:
            connection.close()

    result = {
        "artifact_dir": str(artifact_dir),
        "files": files,
        "database_counts": counts,
        "read_only": True,
    }
    if output:
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

