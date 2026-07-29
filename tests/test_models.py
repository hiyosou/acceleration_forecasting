import numpy as np
import pytest
import torch
import json

from acceleration_forecasting.generation.diffusion import create_model
from acceleration_forecasting.generation.guide_encoder import GuideEncoder
from acceleration_forecasting.generation.masked_cross_attention import MaskedCrossAttention
from acceleration_forecasting.generation.predict import predict
from acceleration_forecasting.generation.sampling_bounds import fit_sampling_bounds


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


@pytest.mark.parametrize("name", ["mlp", "unet"])
@pytest.mark.parametrize("sampling_steps", [5, 10, 20])
def test_ddim_clean_prediction_clipping_is_finite_and_bounded(name, sampling_steps):
    model = create_model(name, steps=20)
    model.eval()
    data = batch(1)
    initial = torch.randn(1, 18)
    bounds = (-1.25, 2.5)
    one = model.ddim_sample(
        data, (1, 18), sampling_steps=sampling_steps,
        initial_noise=initial, clean_clip=bounds,
    )
    two = model.ddim_sample(
        data, (1, 18), sampling_steps=sampling_steps,
        initial_noise=initial, clean_clip=bounds,
    )
    assert torch.isfinite(one).all()
    assert float(one.min()) >= bounds[0]
    assert float(one.max()) <= bounds[1]
    assert torch.allclose(one, two)


def test_sampling_bounds_use_fixed_physical_range_and_train_normalization(tmp_path):
    train = tmp_path / "model_train"
    train.mkdir()
    np.save(train / "target_values.npy", np.array([[1.0, 2.0, 999.0], [3.0, np.nan, -999.0]], dtype=np.float32))
    np.save(train / "target_masks.npy", np.array([[1, 1, 0], [1, 1, 0]], dtype=np.float32))
    (tmp_path / "normalization.json").write_text(json.dumps({
        "mean": 2.0, "std": 0.5, "fitted_observation_count": 3,
        "split": "model_train",
    }), encoding="utf-8")
    bounds = fit_sampling_bounds(tmp_path)
    assert bounds.physical_min == 0.3
    assert bounds.physical_max == 5.0
    assert bounds.normalized == pytest.approx((-3.4, 6.0))
    assert bounds.valid_training_value_count == 3
    assert bounds.bounds_policy == "fixed_physical"


def test_predict_rejects_legacy_selection_without_sampling_bounds(tmp_path):
    selection = tmp_path / "selected_model.json"
    selection.write_text(json.dumps({"selected_checkpoint": "unused.pt"}), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling_bounds"):
        predict(tmp_path, selection, tmp_path / "output", progress=False)


def test_unet_without_cross_attention_has_no_guide_modules_and_is_guide_independent():
    model = create_model("unet", steps=20, use_cross_attention=False)
    model.eval()
    assert model.denoiser.guide_encoder is None
    assert model.denoiser.attn0 is None
    assert model.denoiser.attn1 is None
    assert model.denoiser.attn_mid is None
    assert not any("guide_encoder" in name or "attn" in name for name, _ in model.named_parameters())
    data = batch(1)
    noisy = torch.randn(1, 18)
    timestep = torch.tensor([5])
    first = model.denoiser(noisy, timestep, data)
    changed = {key: value.clone() for key, value in data.items()}
    changed["guide_values"] += 1000
    changed["guide_deltas"] -= 1000
    changed["guide_similarities"][:] = -100
    changed["guide_mask"][:] = 0
    changed["retrieval_mask"][:] = 0
    second = model.denoiser(noisy, timestep, changed)
    assert first.shape == (1, 18)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)


def test_unet_attention_flag_changes_parameter_count():
    with_attention = create_model("unet", steps=20, use_cross_attention=True)
    without_attention = create_model("unet", steps=20, use_cross_attention=False)
    count_with = sum(parameter.numel() for parameter in with_attention.parameters())
    count_without = sum(parameter.numel() for parameter in without_attention.parameters())
    assert count_without < count_with
