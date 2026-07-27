import numpy as np
import pytest
import torch

from acceleration_forecasting.generation.diffusion import create_model
from acceleration_forecasting.generation.guide_encoder import GuideEncoder
from acceleration_forecasting.generation.masked_cross_attention import MaskedCrossAttention


def batch(batch_size=2):
    return {
        "current": torch.randn(batch_size, 1),
        "history": torch.randn(batch_size, 3),
        "history_mask": torch.ones(batch_size, 3),
        "guide_values": torch.randn(batch_size, 3, 18),
        "guide_deltas": torch.randn(batch_size, 3, 18),
        "guide_mask": torch.ones(batch_size, 3, 18),
        "guide_similarities": torch.tensor([[0.9, 0.8, 0.7]]).repeat(batch_size, 1),
        "retrieval_mask": torch.ones(batch_size, 3),
        "target": torch.randn(batch_size, 18),
        "target_mask": torch.ones(batch_size, 18),
    }


def test_guide_encoder_shape():
    data = batch()
    output = GuideEncoder()(data["guide_values"], data["guide_deltas"], data["guide_similarities"])
    assert output.shape == (2, 3, 18, 64)


def test_attention_masks_invalid_tokens_and_all_missing_is_finite():
    attention = MaskedCrossAttention(64)
    query = torch.randn(2, 18, 64)
    guides = torch.randn(2, 3, 18, 64)
    guide_mask = torch.ones(2, 3, 18)
    retrieval = torch.ones(2, 3)
    guide_mask[0, 1, 5] = 0
    guide_mask[1] = 0
    output, weights = attention(query, guides, guide_mask, retrieval, torch.ones(2, 3), True)
    assert torch.all(weights[0, :, :, 18 + 5] == 0)
    assert torch.all(weights[1] == 0)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("name", ["mlp", "unet"])
def test_denoiser_and_masked_loss(name):
    model = create_model(name, steps=20)
    data = batch()
    timestep = torch.tensor([1, 10])
    output = model.denoiser(torch.randn(2, 18), timestep, data)
    assert output.shape == (2, 18)
    loss = model.masked_loss(data, timestep, torch.randn(2, 18))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.denoiser.guide_encoder.parameters())


def test_masked_loss_ignores_missing_error():
    model = create_model("mlp", steps=20)
    data = batch(1)
    data["target_mask"][0, 10:] = 0
    timestep = torch.tensor([4])
    noise = torch.randn(1, 18)
    torch.manual_seed(9)
    first = model.masked_loss(data, timestep, noise)
    changed = {key: value.clone() for key, value in data.items()}
    changed["target"][0, 10:] = 100000
    torch.manual_seed(9)
    second = model.masked_loss(changed, timestep, noise)
    assert torch.allclose(first, second)


def test_ddim_reproducible_with_initial_noise():
    model = create_model("mlp", steps=20)
    model.eval()
    data = batch(1)
    initial = torch.randn(1, 18)
    one = model.ddim_sample(data, (1, 18), sampling_steps=5, initial_noise=initial)
    two = model.ddim_sample(data, (1, 18), sampling_steps=5, initial_noise=initial)
    assert torch.allclose(one, two)

