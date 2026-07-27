from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def cosine_beta_schedule(steps=1000, s=0.008):
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alphas = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    return betas.clamp(1e-5, 0.999).float()


class GaussianDiffusion(nn.Module):
    def __init__(self, denoiser, steps=1000):
        super().__init__()
        self.denoiser = denoiser
        betas = cosine_beta_schedule(steps)
        alphas = 1 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.steps = int(steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumulative", cumulative)

    def add_noise(self, clean, timesteps, noise=None):
        noise = torch.randn_like(clean) if noise is None else noise
        alpha = self.alpha_cumulative[timesteps].unsqueeze(-1)
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise, noise

    def masked_loss(self, batch, timesteps=None, noise=None):
        clean = batch["target"]
        mask = batch["target_mask"]
        if timesteps is None:
            timesteps = torch.randint(0, self.steps, (clean.shape[0],), device=clean.device)
        noisy, noise = self.add_noise(clean, timesteps, noise)
        missing = mask <= 0
        if missing.any():
            noisy = torch.where(missing, noise, noisy)
        prediction = self.denoiser(noisy, timesteps, batch)
        per_record = (((prediction - noise) ** 2) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return per_record.mean()

    @torch.no_grad()
    def ddim_sample(
        self, batch, shape, sampling_steps=50, eta=0.0, generator=None,
        initial_noise=None, clean_clip=None,
    ):
        device = self.alpha_cumulative.device
        sample = (
            torch.randn(shape, device=device, generator=generator)
            if initial_noise is None else initial_noise.to(device)
        )
        times = torch.linspace(self.steps - 1, 0, sampling_steps, device=device).long()
        for index, timestep in enumerate(times):
            t = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
            predicted_noise = self.denoiser(sample, t, batch)
            alpha = self.alpha_cumulative[timestep]
            if index + 1 < len(times):
                previous = times[index + 1]
                alpha_previous = self.alpha_cumulative[previous]
            else:
                alpha_previous = torch.tensor(1.0, device=device)
            predicted_clean = (sample - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            if clean_clip is not None:
                lower, upper = map(float, clean_clip)
                if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                    raise ValueError("clean_clip must contain finite increasing bounds")
                predicted_clean = predicted_clean.clamp(lower, upper)
            sigma = eta * torch.sqrt(
                ((1 - alpha_previous) / (1 - alpha)) * (1 - alpha / alpha_previous)
            ).clamp_min(0)
            direction = torch.sqrt((1 - alpha_previous - sigma**2).clamp_min(0)) * predicted_noise
            random = torch.randn(sample.shape, device=device, generator=generator) if eta > 0 else 0
            sample = alpha_previous.sqrt() * predicted_clean + direction + sigma * random
        return sample


def create_model(model_name, **kwargs):
    if model_name == "mlp":
        from .mlp_denoiser import MLPDenoiser
        denoiser = MLPDenoiser(dropout=kwargs.get("dropout", 0.1))
    elif model_name == "unet":
        from .unet1d_denoiser import UNet1DDenoiser
        denoiser = UNet1DDenoiser(dropout=kwargs.get("dropout", 0.1))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return GaussianDiffusion(denoiser, steps=kwargs.get("steps", 1000))
