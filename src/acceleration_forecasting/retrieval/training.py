from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .artifacts import resolve_artifact_layout
from .constants import EMBEDDING_DIM, RANDOM_SEED, SAMPLES_PER_BIN
from .data import open_waveforms
from .model import WaveformAutoencoder


class MemmapWaveformDataset(Dataset):
    def __init__(self, waveform_path, total_count, indices, mean, std):
        self.waveforms = open_waveforms(waveform_path, total_count)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        waveform = np.asarray(
            self.waveforms[int(self.indices[index])], dtype=np.float32
        ).copy()
        waveform = (waveform - self.mean) / self.std
        return torch.from_numpy(waveform).unsqueeze(0)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_normalization(waveform_path, total_count, indices, chunk_size=4096):
    waveforms = open_waveforms(waveform_path, total_count)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("正規化に使用できるdatabase波形がありません。")
    total = 0.0
    square_total = 0.0
    value_count = 0
    for start in range(0, len(indices), chunk_size):
        chunk = np.asarray(
            waveforms[indices[start : start + chunk_size]], dtype=np.float64
        )
        total += float(chunk.sum())
        square_total += float(np.square(chunk).sum())
        value_count += int(chunk.size)
    mean = total / value_count
    variance = max(square_total / value_count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if not np.isfinite(std) or std <= 1e-12:
        raise ValueError("波形の標準偏差が0または非有限です。")
    return float(mean), std


def split_training_dates(manifest, seed=RANDOM_SEED, train_ratio=0.9):
    database = manifest.loc[manifest["split"] == "database"].copy()
    dates = sorted(database["measurement_date"].astype(str).unique())
    random.Random(int(seed) + 1).shuffle(dates)
    if len(dates) < 2:
        return set(dates), set()
    train_count = min(max(int(np.floor(len(dates) * train_ratio)), 1), len(dates) - 1)
    return set(dates[:train_count]), set(dates[train_count:])


def _average_loss(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batch_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for waveform in loader:
            waveform = waveform.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(waveform)
            loss = criterion(reconstruction, waveform)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            batch_count += 1
    return total_loss / max(batch_count, 1)


def train_autoencoder(
    artifact_dir,
    device=None,
    epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=1e-5,
    patience=10,
    seed=RANDOM_SEED,
):
    artifact_dir = Path(artifact_dir)
    layout = resolve_artifact_layout(artifact_dir)
    manifest = pd.read_csv(layout["manifest_path"], encoding="utf-8-sig")
    waveform_path = layout["waveform_path"]
    total_count = len(manifest)
    if layout["format"] == "dataset_split":
        train_rows = manifest.loc[
            manifest["model_split"] == "model_train"
        ].copy()
        valid_rows = manifest.loc[
            manifest["model_split"] == "model_validation"
        ].copy()
        if train_rows.empty or valid_rows.empty:
            raise ValueError("model_trainまたはmodel_validationが空です。")
        train_indices = train_rows["waveform_index"].astype(int)
        valid_indices = valid_rows["waveform_index"].astype(int)
        normalization_indices = train_indices.to_numpy()
        normalization_split = "model_train"
    else:
        database = manifest.loc[manifest["split"] == "database"].copy()
        if database.empty:
            raise ValueError("Autoencoder学習用のdatabaseレコードがありません。")
        train_dates, valid_dates = split_training_dates(manifest, seed=seed)
        train_indices = database.loc[
            database["measurement_date"].astype(str).isin(train_dates),
            "waveform_index",
        ].astype(int)
        valid_indices = database.loc[
            database["measurement_date"].astype(str).isin(valid_dates),
            "waveform_index",
        ].astype(int)
        if valid_indices.empty:
            valid_indices = train_indices
        normalization_indices = database["waveform_index"].astype(int).to_numpy()
        normalization_split = "database"
    mean, std = calculate_normalization(
        waveform_path, total_count, normalization_indices
    )

    train_dataset = MemmapWaveformDataset(
        waveform_path, total_count, train_indices, mean, std
    )
    valid_dataset = MemmapWaveformDataset(
        waveform_path, total_count, valid_indices, mean, std
    )
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )

    set_random_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = WaveformAutoencoder(EMBEDDING_DIM).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    criterion = nn.SmoothL1Loss()
    checkpoint_path = artifact_dir / "autoencoder.pt"
    history = []
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        train_loss = _average_loss(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        valid_loss = _average_loss(model, valid_loader, criterion, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.8f} "
            f"valid_loss={valid_loss:.8f}"
        )
        if valid_loss < best_loss - 1e-10:
            best_loss = valid_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": EMBEDDING_DIM,
                    "mean": mean,
                    "std": std,
                    "seed": int(seed),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    normalization = {
        "mean": mean,
        "std": std,
        "signal_column": "acc_z[m/s2]",
        "fitted_record_count": int(len(normalization_indices)),
        "samples_per_waveform": SAMPLES_PER_BIN,
        "split": normalization_split,
        "artifact_format": layout["format"],
    }
    (artifact_dir / "normalization.json").write_text(
        json.dumps(normalization, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(
        artifact_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "best_valid_loss": best_loss,
        "epochs_completed": len(history),
        "device": str(device),
    }


def load_trained_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = WaveformAutoencoder(
        int(checkpoint.get("embedding_dim", EMBEDDING_DIM))
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, float(checkpoint["mean"]), float(checkpoint["std"])
