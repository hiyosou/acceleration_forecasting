from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from acceleration_forecasting.common.reproducibility import set_seed
from acceleration_forecasting.common.progress import progress_bar, progress_message
from acceleration_forecasting.datasets.generation_dataset import GenerationDataset

from .diffusion import create_model
from .ema import EMA
from .utils import choose_device, move_batch


def _autocast(device):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _fixed_validation(dataset, seed, steps):
    generator = torch.Generator().manual_seed(seed)
    timesteps = torch.randint(0, steps, (len(dataset),), generator=generator)
    noise = torch.randn((len(dataset), 18), generator=generator)
    return timesteps, noise


@torch.no_grad()
def _validate(model, loader, device, timesteps, noise, progress=True):
    model.eval()
    losses = []
    for batch in progress_bar(
        loader, enabled=progress, total=len(loader), desc="validation",
        unit="batch", leave=False,
    ):
        indices = batch["index"].long()
        batch = move_batch(batch, device)
        losses.append(
            float(model.masked_loss(batch, timesteps[indices].to(device), noise[indices].to(device)))
        )
    return float(sum(losses) / max(len(losses), 1))


def train_model(
    dataset_dir,
    artifact_dir,
    model_name,
    *,
    device=None,
    epochs=200,
    learning_rate=1e-4,
    min_learning_rate=1e-6,
    weight_decay=1e-4,
    warmup_epochs=5,
    patience=20,
    min_delta=1e-4,
    gradient_clip=1.0,
    batch_size=None,
    effective_batch_size=256,
    ema_decay=0.999,
    seed=42,
    resume=True,
    progress=True,
):
    set_seed(seed)
    device = choose_device(device)
    dataset_dir = Path(dataset_dir)
    output = Path(artifact_dir) / model_name
    output.mkdir(parents=True, exist_ok=True)
    normalization_path = dataset_dir / "normalization.json"
    train_data = GenerationDataset(dataset_dir / "model_train", normalization_path)
    valid_data = GenerationDataset(dataset_dir / "model_validation", normalization_path)
    if batch_size is None:
        batch_size = 256 if model_name == "mlp" else 128
    accumulation = max(1, math.ceil(effective_batch_size / batch_size))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
    model = create_model(model_name).to(device)
    ema = EMA(model, ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    use_fp16_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max((epoch + 1) / max(warmup_epochs, 1), min_learning_rate / learning_rate)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate / learning_rate + (1 - min_learning_rate / learning_rate) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_epoch, best_loss, stale = 0, float("inf"), 0
    last_path = output / "last_model.pt"
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_loss"])
        stale = int(checkpoint.get("stale_epochs", 0))
        progress_message(
            f"{model_name}: epoch {start_epoch}/{epochs} から再開 "
            f"(best validation loss={best_loss:.6f})",
            enabled=progress,
        )
    valid_t, valid_noise = _fixed_validation(valid_data, seed, model.steps)
    history = []
    history_path = output / "training_history.csv"
    if history_path.is_file() and start_epoch:
        history = pd.read_csv(history_path).to_dict("records")
    epoch_progress = progress_bar(
        range(start_epoch, int(epochs)), enabled=progress, total=int(epochs),
        initial=start_epoch, desc=f"{model_name} 学習", unit="epoch",
    )
    for epoch in epoch_progress:
        started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses, gradient_norms = [], []
        batch_progress = progress_bar(
            train_loader, enabled=progress, total=len(train_loader),
            desc=f"epoch {epoch + 1}/{epochs}", unit="batch", leave=False,
        )
        last_progress_update = time.monotonic()
        for step, batch in enumerate(batch_progress):
            batch = move_batch(batch, device)
            with _autocast(device):
                loss = model.masked_loss(batch) / accumulation
            scaler.scale(loss).backward()
            train_losses.append(float(loss.detach()) * accumulation)
            now = time.monotonic()
            if progress and now - last_progress_update >= 1.0:
                batch_progress.set_postfix(loss=f"{train_losses[-1]:.6f}", refresh=False)
                last_progress_update = now
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                norm = clip_grad_norm_(model.parameters(), gradient_clip)
                gradient_norms.append(float(norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
        validation_loss = _validate(
            ema.model, valid_loader, device, valid_t, valid_noise, progress=progress
        )
        scheduler.step()
        improved = validation_loss < best_loss - min_delta
        if improved:
            best_loss, stale = validation_loss, 0
        else:
            stale += 1
        state = {
            "model_name": model_name, "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch,
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_loss": best_loss, "stale_epochs": stale, "seed": seed,
        }
        torch.save(state, last_path)
        if improved:
            torch.save(state, output / "best_model.pt")
            torch.save({"model_name": model_name, "model_state_dict": ema.state_dict(), "epoch": epoch, "seed": seed}, output / "best_ema_model.pt")
        history.append({
            "epoch": epoch, "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "validation_loss": validation_loss, "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm": sum(gradient_norms) / max(len(gradient_norms), 1),
            "elapsed_seconds": time.time() - started, "is_best": improved,
        })
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
        epoch_progress.set_postfix(
            train=f"{history[-1]['train_loss']:.6f}",
            validation=f"{validation_loss:.6f}", best=f"{best_loss:.6f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            patience_left=max(0, patience - stale),
            elapsed=f"{history[-1]['elapsed_seconds']:.1f}s", refresh=False,
        )
        if stale >= patience:
            break
    resolved = {
        "model": model_name, "epochs": epochs, "batch_size": batch_size,
        "gradient_accumulation": accumulation, "effective_batch_size": batch_size * accumulation,
        "learning_rate": learning_rate, "weight_decay": weight_decay, "ema_decay": ema_decay,
        "seed": seed, "device": str(device), "best_validation_loss": best_loss,
    }
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    return resolved
