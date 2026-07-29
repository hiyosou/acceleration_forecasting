from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from acceleration_forecasting.common.reproducibility import stable_seed

from .diffusion import create_model
from .utils import choose_device, move_batch, repeat_conditions


def load_ema_checkpoint(path, device=None):
    device = choose_device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    model = create_model(model_name, **checkpoint.get("model_kwargs", {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_name, device


@torch.no_grad()
def sample_one(
    model, batch, target_id, num_samples=100, sampling_steps=50, eta=0.0,
    seed=42, sample_batch=100, clean_clip=None,
):
    device = next(model.parameters()).device
    batch = move_batch(batch, device)
    samples = []
    for start in range(0, num_samples, sample_batch):
        count = min(sample_batch, num_samples - start)
        repeated = repeat_conditions(batch, count)
        initial = []
        for sample_index in range(start, start + count):
            generator = torch.Generator(device=device).manual_seed(
                stable_seed(seed, target_id, sample_index)
            )
            initial.append(torch.randn((18,), device=device, generator=generator))
        initial = torch.stack(initial)
        generated = model.ddim_sample(
            repeated, (count, 18), sampling_steps=sampling_steps, eta=eta,
            initial_noise=initial, clean_clip=clean_clip,
        )
        samples.append(generated.cpu().numpy())
    return np.concatenate(samples, axis=0)
