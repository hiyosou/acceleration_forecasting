from __future__ import annotations

import json
from pathlib import Path


def resolve_artifact_layout(artifact_dir):
    artifact_dir = Path(artifact_dir)
    source_config = artifact_dir / "source_artifacts.json"
    if source_config.is_file():
        config = json.loads(source_config.read_text(encoding="utf-8"))
        return {
            "format": "dataset_split",
            "artifact_dir": artifact_dir,
            "manifest_path": artifact_dir / "dataset_split_manifest.csv",
            "waveform_path": Path(config["waveforms_path"]),
            "trend_path": artifact_dir / "development_trends.csv",
            "record_count": int(config["record_count"]),
        }
    return {
        "format": "date_split",
        "artifact_dir": artifact_dir,
        "manifest_path": artifact_dir / "split_manifest.csv",
        "waveform_path": artifact_dir / "waveforms.bin",
        "trend_path": artifact_dir / "trend_catalog.csv",
        "record_count": None,
    }
