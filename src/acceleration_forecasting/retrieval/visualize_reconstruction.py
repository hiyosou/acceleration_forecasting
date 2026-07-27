from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .artifacts import resolve_artifact_layout
from .constants import (
    DEFAULT_ARTIFACT_DIR,
    EMBEDDING_DIM,
    RANDOM_SEED,
    SAMPLES_PER_BIN,
    STEP_M,
)
from .data import open_waveforms
from .model import WaveformAutoencoder
from .training import load_trained_model


METRIC_COLUMNS = [
    "record_id",
    "waveform_index",
    "measurement_id",
    "measurement_date",
    "direction",
    "bin_start_m",
    "bin_end_m",
    "mean_velocity_kmh",
    "source_csv_path",
    "split",
    "model_source",
    "random_seed",
    "mae",
    "rmse",
    "correlation",
    "peak_amplitude_original",
    "peak_amplitude_reconstructed",
    "peak_amplitude_error",
    "peak_distance_original",
    "peak_distance_reconstructed",
    "peak_distance_error_m",
]
MODEL_SOURCES = ("trained", "random", "mean")
METADATA_COLUMNS = METRIC_COLUMNS[:10]


def inverse_normalize(values, mean, std):
    return np.asarray(values) * float(std) + float(mean)


def calculate_reconstruction_metrics(original, reconstructed, bin_start_m, step_m=STEP_M):
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if original.shape != reconstructed.shape or original.ndim != 1:
        raise ValueError("元波形と復元波形は同じ長さの1次元配列である必要があります。")
    residual = reconstructed - original
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    original_std = float(np.std(original))
    reconstructed_std = float(np.std(reconstructed))
    if original_std <= 1e-12 or reconstructed_std <= 1e-12:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(original, reconstructed)[0, 1])
    original_peak_index = int(np.argmax(np.abs(original)))
    reconstructed_peak_index = int(np.argmax(np.abs(reconstructed)))
    original_peak = float(abs(original[original_peak_index]))
    reconstructed_peak = float(abs(reconstructed[reconstructed_peak_index]))
    original_distance = float(bin_start_m + original_peak_index * step_m)
    reconstructed_distance = float(bin_start_m + reconstructed_peak_index * step_m)
    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "peak_amplitude_original": original_peak,
        "peak_amplitude_reconstructed": reconstructed_peak,
        "peak_amplitude_error": float(abs(reconstructed_peak - original_peak)),
        "peak_distance_original": original_distance,
        "peak_distance_reconstructed": reconstructed_distance,
        "peak_distance_error_m": float(abs(reconstructed_distance - original_distance)),
    }


def select_manifest_rows(
    manifest,
    split="query",
    record_id=None,
    source_csv=None,
    direction=None,
    bin_start_m=None,
    max_records=None,
    seed=RANDOM_SEED,
):
    selected = manifest.copy()
    direct_selection = bool(record_id or source_csv)
    if record_id:
        selected = selected.loc[selected["record_id"].astype(str) == str(record_id)]
    if source_csv:
        requested = os.path.normcase(os.path.abspath(os.path.normpath(str(source_csv))))
        normalized = selected["source_csv_path"].astype(str).map(
            lambda value: os.path.normcase(
                os.path.abspath(os.path.normpath(value))
            )
        )
        selected = selected.loc[normalized == requested]
    if not direct_selection and split != "all":
        selected = selected.loc[selected["split"].astype(str) == split]
    if direction:
        selected = selected.loc[
            selected["direction"].astype(str).str.upper() == direction.upper()
        ]
    if bin_start_m is not None:
        values = pd.to_numeric(selected["bin_start_m"], errors="coerce")
        selected = selected.loc[np.isclose(values, float(bin_start_m), atol=1e-6)]
    if selected.empty:
        raise ValueError("指定条件に該当する波形レコードがありません。")
    if max_records is not None and len(selected) > int(max_records):
        selected = selected.sample(n=int(max_records), random_state=int(seed))
    return selected.sort_values("waveform_index", kind="mergesort").reset_index(drop=True)


def _validate_artifacts(artifact_dir):
    artifact_dir = Path(artifact_dir)
    layout = resolve_artifact_layout(artifact_dir)
    required = [
        layout["waveform_path"],
        layout["manifest_path"],
        artifact_dir / "autoencoder.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("必要な成果物がありません: " + ", ".join(missing))
    manifest = pd.read_csv(
        layout["manifest_path"], encoding="utf-8-sig"
    )
    if "split" not in manifest.columns and "outer_split" in manifest.columns:
        manifest["split"] = manifest["outer_split"].map(
            {"development": "database", "inference": "query"}
        )
    required_columns = {
        "record_id",
        "waveform_index",
        "measurement_id",
        "measurement_date",
        "direction",
        "bin_start_m",
        "bin_end_m",
        "mean_velocity_kmh",
        "source_csv_path",
        "split",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            "manifestに必要な列がありません: " + ", ".join(sorted(missing_columns))
        )
    expected_size = len(manifest) * SAMPLES_PER_BIN * np.dtype(np.float32).itemsize
    actual_size = layout["waveform_path"].stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"waveforms.binとmanifestの件数が一致しません: "
            f"expected={expected_size}, actual={actual_size}"
        )
    indices = pd.to_numeric(manifest["waveform_index"], errors="raise").astype(int)
    if (
        indices.nunique() != len(manifest)
        or indices.min() != 0
        or indices.max() != len(manifest) - 1
    ):
        raise ValueError("manifestのwaveform_indexが一意な連番ではありません。")
    return manifest


def _load_checkpoint_metadata(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    mean = float(checkpoint["mean"])
    std = float(checkpoint["std"])
    embedding_dim = int(checkpoint.get("embedding_dim", EMBEDDING_DIM))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-12:
        raise ValueError("チェックポイントの正規化情報が不正です。")
    return mean, std, embedding_dim


def create_random_model(embedding_dim, seed, device):
    # CPU上で初期化することで、同じseedならCUDAの有無によらず同じ重みにする。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = WaveformAutoencoder(int(embedding_dim))
    return model.to(device).eval()


def reconstruct_batch(
    normalized,
    original,
    model_source,
    mean,
    std,
    device,
    trained_model=None,
    random_model=None,
):
    if model_source == "mean":
        return np.full_like(original, float(mean), dtype=np.float32)
    model = trained_model if model_source == "trained" else random_model
    if model is None:
        raise ValueError(f"{model_source}モデルが初期化されていません。")
    tensor = torch.from_numpy(normalized).unsqueeze(1).to(device)
    with torch.inference_mode():
        reconstructed_normalized, _ = model(tensor)
    return inverse_normalize(
        reconstructed_normalized.squeeze(1).cpu().numpy(), mean, std
    ).astype(np.float32)


def evaluate_reconstructions(
    artifact_dir,
    selected_manifest,
    device=None,
    batch_size=512,
    model_source="trained",
    seed=RANDOM_SEED,
):
    artifact_dir = Path(artifact_dir)
    layout = resolve_artifact_layout(artifact_dir)
    full_manifest = pd.read_csv(
        layout["manifest_path"], encoding="utf-8-sig"
    )
    waveforms = open_waveforms(
        layout["waveform_path"], len(full_manifest)
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if model_source not in MODEL_SOURCES and model_source != "compare":
        raise ValueError(f"未対応のmodel_sourceです: {model_source}")
    mean, std, embedding_dim = _load_checkpoint_metadata(
        artifact_dir / "autoencoder.pt"
    )
    requested_sources = (
        MODEL_SOURCES if model_source == "compare" else (model_source,)
    )
    trained_model = (
        load_trained_model(artifact_dir / "autoencoder.pt", device)[0]
        if "trained" in requested_sources
        else None
    )
    random_model = (
        create_random_model(embedding_dim, seed, device)
        if "random" in requested_sources
        else None
    )

    records = []
    batch_size = max(int(batch_size), 1)
    progress = tqdm(
        range(0, len(selected_manifest), batch_size),
        desc="復元性能を評価",
        unit="batch",
        dynamic_ncols=True,
    )
    with torch.inference_mode():
        for start in progress:
            rows = selected_manifest.iloc[start : start + batch_size]
            indices = rows["waveform_index"].astype(int).to_numpy()
            original = np.asarray(waveforms[indices], dtype=np.float32).copy()
            normalized = (original - mean) / std
            for source in requested_sources:
                reconstructed = reconstruct_batch(
                    normalized,
                    original,
                    source,
                    mean,
                    std,
                    device,
                    trained_model=trained_model,
                    random_model=random_model,
                )
                for row_position, (_, row) in enumerate(rows.iterrows()):
                    metrics = calculate_reconstruction_metrics(
                        original[row_position],
                        reconstructed[row_position],
                        float(row["bin_start_m"]),
                    )
                    records.append(
                        {
                            **{
                                column: row[column]
                                for column in METADATA_COLUMNS
                            },
                            "model_source": source,
                            "random_seed": int(seed) if source == "random" else np.nan,
                            **metrics,
                        }
                    )
    result = pd.DataFrame.from_records(records, columns=METRIC_COLUMNS)
    del waveforms
    return result, str(device)


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = [
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _save_figure_atomic(figure, output_path, dpi):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    figure.savefig(temporary, format="png", dpi=int(dpi), bbox_inches="tight")
    os.replace(temporary, output_path)


def _safe_filename(value):
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    )


def plot_individual_reconstruction(
    row,
    original,
    reconstructed,
    output_path,
    dpi=150,
):
    plt = _import_matplotlib()
    bin_start = float(row["bin_start_m"])
    distance = bin_start + np.arange(len(original)) * STEP_M
    residual = reconstructed - original
    original_peak_index = int(np.argmax(np.abs(original)))
    reconstructed_peak_index = int(np.argmax(np.abs(reconstructed)))
    frequency = np.fft.rfftfreq(len(original), d=STEP_M)
    original_spectrum = np.abs(np.fft.rfft(original)) / len(original)
    reconstructed_spectrum = np.abs(np.fft.rfft(reconstructed)) / len(reconstructed)

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    axes[0].plot(distance, original, color="black", linewidth=1.1, label="元波形")
    axes[0].plot(
        distance, reconstructed, color="tab:red", linewidth=1.0, label="復元波形"
    )
    axes[0].scatter(
        distance[original_peak_index],
        original[original_peak_index],
        color="black",
        s=28,
        zorder=3,
        label="元ピーク",
    )
    axes[0].scatter(
        distance[reconstructed_peak_index],
        reconstructed[reconstructed_peak_index],
        color="tab:red",
        marker="x",
        s=42,
        zorder=3,
        label="復元ピーク",
    )
    axes[0].set_ylabel("acc_z [m/s²]")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(distance, residual, color="tab:blue", linewidth=1.0)
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("補正後距離 [m]")
    axes[1].set_ylabel("復元値 − 元値 [m/s²]")
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        frequency, original_spectrum, color="black", linewidth=1.1, label="元波形"
    )
    axes[2].plot(
        frequency,
        reconstructed_spectrum,
        color="tab:red",
        linewidth=1.0,
        label="復元波形",
    )
    axes[2].set_xlabel("空間周波数 [cycles/m]")
    axes[2].set_ylabel("振幅")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)

    correlation_text = (
        f"{float(row['correlation']):.4f}"
        if pd.notna(row["correlation"])
        else "NaN"
    )
    figure.suptitle(
        f"{row['measurement_id']} / {row['direction']}方向 / "
        f"{int(float(row['bin_start_m']))}-{int(float(row['bin_end_m']))}m\n"
        f"平均速度={float(row['mean_velocity_kmh']):.2f} km/h  "
        f"RMSE={float(row['rmse']):.4f} m/s²  相関={correlation_text}"
    )
    figure.tight_layout()
    _save_figure_atomic(figure, output_path, dpi)
    plt.close(figure)


def plot_summary(metrics, output_path, dpi=150):
    plt = _import_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(metrics["rmse"].dropna(), bins=50, color="tab:blue", alpha=0.8)
    axes[0, 0].set_xlabel("RMSE [m/s²]")
    axes[0, 0].set_ylabel("波形数")
    axes[0, 0].set_title("RMSE分布")

    axes[0, 1].hist(
        metrics["correlation"].dropna(), bins=50, color="tab:green", alpha=0.8
    )
    axes[0, 1].set_xlabel("相関係数")
    axes[0, 1].set_ylabel("波形数")
    axes[0, 1].set_title("相関係数分布")

    axes[1, 0].scatter(
        metrics["peak_amplitude_original"],
        metrics["peak_amplitude_reconstructed"],
        s=6,
        alpha=0.25,
        color="tab:red",
    )
    maximum = float(
        np.nanmax(
            [
                metrics["peak_amplitude_original"].max(),
                metrics["peak_amplitude_reconstructed"].max(),
            ]
        )
    )
    axes[1, 0].plot([0, maximum], [0, maximum], "--", color="gray", linewidth=1)
    axes[1, 0].set_xlabel("元ピーク振幅 [m/s²]")
    axes[1, 0].set_ylabel("復元ピーク振幅 [m/s²]")
    axes[1, 0].set_title("ピーク振幅の再現性")

    axes[1, 1].scatter(
        metrics["mean_velocity_kmh"],
        metrics["rmse"],
        s=6,
        alpha=0.25,
        color="tab:purple",
    )
    axes[1, 1].set_xlabel("平均速度 [km/h]")
    axes[1, 1].set_ylabel("RMSE [m/s²]")
    axes[1, 1].set_title("平均速度と復元誤差")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(f"Autoencoder波形復元性能（{len(metrics):,}波形）")
    figure.tight_layout()
    _save_figure_atomic(figure, output_path, dpi)
    plt.close(figure)


def plot_comparison_summary(metrics, output_path, dpi=150):
    plt = _import_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    sources = list(MODEL_SOURCES)
    colors = ["tab:red", "tab:blue", "gray"]
    for axis, metric, title, ylabel in (
        (axes[0, 0], "rmse", "方式別RMSE", "RMSE [m/s²]"),
        (axes[0, 1], "correlation", "方式別相関係数", "相関係数"),
        (
            axes[1, 0],
            "peak_amplitude_error",
            "方式別ピーク振幅誤差",
            "ピーク振幅誤差 [m/s²]",
        ),
    ):
        values = [
            metrics.loc[metrics["model_source"] == source, metric]
            .dropna()
            .to_numpy()
            for source in sources
        ]
        boxes = axis.boxplot(values, tick_labels=sources, showfliers=False, patch_artist=True)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)

    pivot = metrics.pivot(index="record_id", columns="model_source", values="rmse")
    trained = pivot["trained"]
    random_values = pivot["random"]
    mean_values = pivot["mean"]
    maximum = float(np.nanmax(pivot.to_numpy()))
    axes[1, 1].scatter(
        random_values, trained, s=7, alpha=0.25, color="tab:blue", label="random"
    )
    axes[1, 1].scatter(
        mean_values, trained, s=7, alpha=0.25, color="gray", label="mean"
    )
    axes[1, 1].plot([0, maximum], [0, maximum], "--", color="black", linewidth=1)
    axes[1, 1].set_xlabel("ベースラインRMSE [m/s²]")
    axes[1, 1].set_ylabel("trained RMSE [m/s²]")
    axes[1, 1].set_title(
        "trainedとベースラインの比較\n"
        f"randomより良好={(trained < random_values).mean():.1%} / "
        f"meanより良好={(trained < mean_values).mean():.1%}"
    )
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)
    figure.suptitle(f"学習効果の比較（{len(pivot):,}波形）")
    figure.tight_layout()
    _save_figure_atomic(figure, output_path, dpi)
    plt.close(figure)


def plot_comparison_reconstruction(
    row,
    original,
    reconstructions,
    source_metrics,
    output_path,
    seed=RANDOM_SEED,
    dpi=150,
):
    plt = _import_matplotlib()
    bin_start = float(row["bin_start_m"])
    distance = bin_start + np.arange(len(original)) * STEP_M
    frequency = np.fft.rfftfreq(len(original), d=STEP_M)
    colors = {"trained": "tab:red", "random": "tab:blue", "mean": "gray"}
    labels = {"trained": "学習済み", "random": "未学習", "mean": "平均値"}

    figure, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=False)
    axes[0].plot(distance, original, color="black", linewidth=1.2, label="元波形")
    for source in MODEL_SOURCES:
        axes[0].plot(
            distance,
            reconstructions[source],
            color=colors[source],
            linewidth=1.0,
            label=labels[source],
        )
    axes[0].set_ylabel("acc_z [m/s²]")
    axes[0].legend(ncol=4)
    axes[0].grid(alpha=0.2)

    for source in MODEL_SOURCES:
        axes[1].plot(
            distance,
            reconstructions[source] - original,
            color=colors[source],
            linewidth=1.0,
            label=labels[source],
        )
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_xlabel("補正後距離 [m]")
    axes[1].set_ylabel("復元値 − 元値 [m/s²]")
    axes[1].legend(ncol=3)
    axes[1].grid(alpha=0.2)

    original_spectrum = np.abs(np.fft.rfft(original)) / len(original)
    axes[2].plot(
        frequency, original_spectrum, color="black", linewidth=1.2, label="元波形"
    )
    for source in MODEL_SOURCES:
        spectrum = np.abs(np.fft.rfft(reconstructions[source])) / len(original)
        axes[2].plot(
            frequency,
            spectrum,
            color=colors[source],
            linewidth=1.0,
            label=labels[source],
        )
    axes[2].set_xlabel("空間周波数 [cycles/m]")
    axes[2].set_ylabel("振幅")
    axes[2].legend(ncol=4)
    axes[2].grid(alpha=0.2)

    metric_names = ["mae", "rmse", "correlation", "peak_amplitude_error"]
    metric_labels = ["MAE", "RMSE", "相関係数", "ピーク誤差"]
    x = np.arange(len(metric_names))
    width = 0.24
    for offset, source in enumerate(MODEL_SOURCES):
        metric_row = source_metrics.loc[
            source_metrics["model_source"] == source
        ].iloc[0]
        values = [float(metric_row[name]) for name in metric_names]
        axes[3].bar(
            x + (offset - 1) * width,
            values,
            width,
            color=colors[source],
            alpha=0.75,
            label=labels[source],
        )
    axes[3].set_xticks(x, metric_labels)
    axes[3].set_ylabel("指標値")
    axes[3].legend(ncol=3)
    axes[3].grid(axis="y", alpha=0.2)

    figure.suptitle(
        f"{row['measurement_id']} / {row['direction']}方向 / "
        f"{int(float(row['bin_start_m']))}-{int(float(row['bin_end_m']))}m / "
        f"平均速度={float(row['mean_velocity_kmh']):.2f} km/h / random seed={seed}"
    )
    figure.tight_layout()
    _save_figure_atomic(figure, output_path, dpi)
    plt.close(figure)


def select_representatives(metrics, count=5):
    count = max(int(count), 1)
    median_rmse = float(metrics["rmse"].median())
    candidates = {
        "best": metrics.nsmallest(count, "rmse"),
        "median": metrics.assign(
            _median_distance=(metrics["rmse"] - median_rmse).abs()
        ).nsmallest(count, "_median_distance"),
        "worst": metrics.nlargest(count, "rmse"),
        "peak_error": metrics.nlargest(count, "peak_amplitude_error"),
    }
    selected = {}
    used = set()
    for category, rows in candidates.items():
        unique_rows = rows.loc[~rows["record_id"].astype(str).isin(used)].copy()
        selected[category] = unique_rows
        used.update(unique_rows["record_id"].astype(str))
    return selected


def _reconstruct_rows(artifact_dir, rows, device, model_source="trained", seed=RANDOM_SEED):
    artifact_dir = Path(artifact_dir)
    layout = resolve_artifact_layout(artifact_dir)
    manifest = pd.read_csv(
        layout["manifest_path"], encoding="utf-8-sig"
    )
    waveforms = open_waveforms(layout["waveform_path"], len(manifest))
    device = torch.device(device)
    mean, std, embedding_dim = _load_checkpoint_metadata(
        artifact_dir / "autoencoder.pt"
    )
    trained_model = (
        load_trained_model(artifact_dir / "autoencoder.pt", device)[0]
        if model_source == "trained"
        else None
    )
    random_model = (
        create_random_model(embedding_dim, seed, device)
        if model_source == "random"
        else None
    )
    indices = rows["waveform_index"].astype(int).to_numpy()
    original = np.asarray(waveforms[indices], dtype=np.float32).copy()
    result = reconstruct_batch(
        (original - mean) / std,
        original,
        model_source,
        mean,
        std,
        device,
        trained_model=trained_model,
        random_model=random_model,
    )
    del waveforms
    return original, result


def _write_metrics_atomic(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    metrics.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, output_path)


def _source_directory_name(model_source, seed):
    return f"random_seed{int(seed)}" if model_source == "random" else model_source


def _image_name(row):
    return (
        f"{_safe_filename(row['measurement_id'])}_"
        f"{int(float(row['bin_start_m']))}-"
        f"{int(float(row['bin_end_m']))}m_"
        f"{_safe_filename(row['record_id'])}.png"
    )


def _write_source_outputs(
    artifact_dir,
    output_root,
    metrics,
    model_source,
    device,
    explicit,
    seed,
    dpi,
):
    source_dir = Path(output_root) / _source_directory_name(model_source, seed)
    metrics_path = source_dir / "reconstruction_metrics.csv"
    summary_path = source_dir / "reconstruction_summary.png"
    _write_metrics_atomic(metrics, metrics_path)
    plot_summary(metrics, summary_path, dpi=dpi)
    groups = (
        {"selected": metrics}
        if explicit
        else select_representatives(metrics, count=5)
    )
    image_count = 0
    for category, rows in groups.items():
        if rows.empty:
            continue
        original, reconstructed = _reconstruct_rows(
            artifact_dir,
            rows,
            device,
            model_source=model_source,
            seed=seed,
        )
        for position, (_, row) in enumerate(rows.iterrows()):
            plot_individual_reconstruction(
                row,
                original[position],
                reconstructed[position],
                source_dir / "individual" / category / _image_name(row),
                dpi=dpi,
            )
            image_count += 1
    return {
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "individual_images": image_count,
        "output_dir": str(source_dir),
    }


def run_visualization(
    artifact_dir=DEFAULT_ARTIFACT_DIR,
    output_dir=None,
    split="query",
    record_id=None,
    source_csv=None,
    direction=None,
    bin_start_m=None,
    max_records=None,
    batch_size=512,
    device=None,
    dpi=150,
    seed=RANDOM_SEED,
    model_source="trained",
):
    artifact_dir = Path(artifact_dir)
    manifest = _validate_artifacts(artifact_dir)
    selected = select_manifest_rows(
        manifest,
        split=split,
        record_id=record_id,
        source_csv=source_csv,
        direction=direction,
        bin_start_m=bin_start_m,
        max_records=max_records,
        seed=seed,
    )
    metrics, actual_device = evaluate_reconstructions(
        artifact_dir,
        selected,
        device=device,
        batch_size=batch_size,
        model_source=model_source,
        seed=seed,
    )
    output_root = (
        Path(output_dir)
        if output_dir
        else artifact_dir / "reconstruction"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    explicit = bool(record_id or source_csv)
    requested_sources = (
        MODEL_SOURCES if model_source == "compare" else (model_source,)
    )
    source_results = {}
    for source in requested_sources:
        source_metrics = metrics.loc[
            metrics["model_source"] == source
        ].reset_index(drop=True)
        source_results[source] = _write_source_outputs(
            artifact_dir,
            output_root,
            source_metrics,
            source,
            actual_device,
            explicit,
            seed,
            dpi,
        )

    comparison_result = None
    if model_source == "compare":
        comparison_dir = output_root / "comparison"
        comparison_metrics_path = (
            comparison_dir / "reconstruction_metrics_comparison.csv"
        )
        comparison_summary_path = comparison_dir / "reconstruction_summary.png"
        _write_metrics_atomic(metrics, comparison_metrics_path)
        plot_comparison_summary(metrics, comparison_summary_path, dpi=dpi)
        trained_metrics = metrics.loc[
            metrics["model_source"] == "trained"
        ].reset_index(drop=True)
        comparison_groups = (
            {"selected": trained_metrics}
            if explicit
            else select_representatives(trained_metrics, count=5)
        )
        comparison_images = 0
        for category, rows in comparison_groups.items():
            if rows.empty:
                continue
            original = None
            reconstructions = {}
            for source in MODEL_SOURCES:
                source_original, source_reconstruction = _reconstruct_rows(
                    artifact_dir,
                    rows,
                    actual_device,
                    model_source=source,
                    seed=seed,
                )
                if original is None:
                    original = source_original
                reconstructions[source] = source_reconstruction
            for position, (_, row) in enumerate(rows.iterrows()):
                record_metrics = metrics.loc[
                    metrics["record_id"].astype(str) == str(row["record_id"])
                ]
                plot_comparison_reconstruction(
                    row,
                    original[position],
                    {
                        source: reconstructions[source][position]
                        for source in MODEL_SOURCES
                    },
                    record_metrics,
                    comparison_dir / "individual" / category / _image_name(row),
                    seed=seed,
                    dpi=dpi,
                )
                comparison_images += 1
        comparison_result = {
            "metrics_path": str(comparison_metrics_path),
            "summary_path": str(comparison_summary_path),
            "individual_images": comparison_images,
            "output_dir": str(comparison_dir),
        }

    result = {
        "evaluated_records": int(len(selected)),
        "model_source": model_source,
        "device": actual_device,
        "sources": source_results,
        "output_dir": str(output_root),
    }
    if comparison_result is not None:
        result["comparison"] = comparison_result
    elif len(requested_sources) == 1:
        # 既存利用者が参照していた主要キーも維持する。
        only = source_results[requested_sources[0]]
        result.update(
            {
                "metrics_path": only["metrics_path"],
                "summary_path": only["summary_path"],
                "individual_images": only["individual_images"],
            }
        )
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Autoencoderの100m波形復元性能を可視化します。"
    )
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--split", choices=["query", "database", "all"], default="query"
    )
    parser.add_argument("--record-id", default="")
    parser.add_argument("--source-csv", default="")
    parser.add_argument("--direction", choices=["U", "D"], default=None)
    parser.add_argument("--bin-start-m", type=float, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--model-source",
        choices=["trained", "random", "mean", "compare"],
        default="trained",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_visualization(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir or None,
        split=args.split,
        record_id=args.record_id or None,
        source_csv=args.source_csv or None,
        direction=args.direction,
        bin_start_m=args.bin_start_m,
        max_records=args.max_records,
        batch_size=args.batch_size,
        device=args.device,
        dpi=args.dpi,
        seed=args.seed,
        model_source=args.model_source,
    )
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
