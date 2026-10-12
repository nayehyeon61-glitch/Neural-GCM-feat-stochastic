"""Behavioral checks for stage guidance; these do not establish forecast skill."""

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from ngcm_memory.kai import (  # noqa: E402
    StageAdapterConfig,
    StageManifoldKAI,
    load_kai_checkpoint,
    stage_alignment_loss,
)
from ngcm_memory.kai_backbone import KAI  # noqa: E402


@pytest.fixture(autouse=True)
def deterministic_small_cpu_checks():
    """Avoid large OpenMP overhead for these small convolutions."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(319)
    yield
    torch.set_num_threads(previous)


def make_backbone():
    return KAI(
        img_size=(6, 12), in_chans=3, out_chans=3,
        depths=(1, 1, 1, 1), dims=(8, 8, 8, 8),
    )


def make_model(mode="residual", stages=(0, 1, 2, 3)):
    return StageManifoldKAI(
        make_backbone(),
        StageAdapterConfig(
            physical_channels=3, latent_channels=2,
            stage_indices=stages, mode=mode, hidden_dim=8,
            max_correction=.1, gate_max=.7,
        ),
    )


def fields(batch=2):
    return (
        torch.randn(batch, 3, 6, 12),
        torch.randn(batch, 3, 6, 12),
        torch.randn(batch, 2, 6, 12),
    )


def activate_adapters(model):
    """Stand in for learned nonzero adapters, without changing the backbone."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith("backbone."):
                parameter.normal_(mean=0, std=.15)


@pytest.mark.parametrize("mode", ["residual", "filter"])
def test_disabled_path_is_exact_original_without_guides_or_rng(mode):
    model = make_model(mode).eval()
    activate_adapters(model)
    x, _, _ = fields()
    expected = model.backbone(x)
    random_state = torch.random.get_rng_state().clone()
    actual = model(x, enabled=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(random_state, torch.random.get_rng_state())
    actual, diagnostics = model.forward_with_diagnostics(x, enabled=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert diagnostics == {}


@pytest.mark.parametrize("mode", ["residual", "filter"])
def test_stage_selection_preserves_resolution_and_reaches_later_stages(mode):
    model = make_model(mode, stages=(0, 2)).eval()
    activate_adapters(model)
    x, physical, latent = fields()
    captured = []
    handle = model.backbone.core_model.stages[3].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        baseline = model(x, enabled=False)
        prediction, diagnostics = model.forward_with_diagnostics(x, physical, latent)
    finally:
        handle.remove()
    assert prediction.shape == x.shape
    assert set(diagnostics) == {0, 2}
    assert all(readout.shape == physical.shape for readout in diagnostics.values())
    assert len(captured) == 2
    assert captured[0].shape[-2:] == (6, 12)
    assert not torch.equal(captured[0], captured[1])
    assert not torch.equal(prediction, baseline)


@pytest.mark.parametrize("mode", ["residual", "filter"])
def test_physical_and_latent_guides_each_affect_forecast(mode):
    model = make_model(mode).eval()
    activate_adapters(model)
    x, physical, latent = fields()
    prediction = model(x, physical, latent)
    different_physics = model(x, physical + .5, latent)
    different_memory = model(x, physical, latent + .5)
    assert not torch.equal(prediction, different_physics)
    assert not torch.equal(prediction, different_memory)


@pytest.mark.parametrize("mode", ["residual", "filter"])
def test_each_stage_correction_obeys_feature_space_bound(mode):
    model = make_model(mode).eval()
    activate_adapters(model)
    x, physical, latent = fields()
    corrections = []
    handles = [
        adapter.register_forward_hook(
            lambda _module, inputs, output: corrections.append(
                (output[0] - inputs[0]).detach()
            )
        )
        for adapter in model.adapters.values()
    ]
    try:
        before = torch.random.get_rng_state().clone()
        model(x, physical * 1e3, latent * 1e3)
        assert torch.equal(before, torch.random.get_rng_state())
    finally:
        for handle in handles:
            handle.remove()
    bound = model.config.max_correction * model.config.gate_max
    assert len(corrections) == 4
    for correction in corrections:
        assert torch.isfinite(correction).all()
        assert correction.abs().max().item() <= bound + 1e-6
        assert correction.abs().sum().item() > 0


def test_empty_stage_selection_is_exact_bypass():
    model = make_model(stages=()).eval()
    x, _, _ = fields()
    actual, diagnostics = model.forward_with_diagnostics(x)
    torch.testing.assert_close(actual, model.backbone(x), rtol=0, atol=0)
    assert diagnostics == {}


def test_channel_expanding_backbone_and_frozen_training_mode():
    backbone = KAI(
        img_size=(6, 12), in_chans=3, out_chans=3,
        depths=(1, 1, 1, 1), dims=(8, 12, 16, 24), drop_path_rate=.2,
    )
    model = StageManifoldKAI(
        backbone, StageAdapterConfig(physical_channels=3, latent_channels=2, hidden_dim=8),
    )
    model.freeze_backbone().train()
    assert model.training and model.adapters.training
    assert all(not module.training for module in backbone.modules())
    x, physical, latent = fields()
    prediction, diagnostics = model.forward_with_diagnostics(x, physical, latent)
    assert prediction.shape == x.shape
    assert set(diagnostics) == {0, 1, 2, 3}
    assert all(readout.shape == physical.shape for readout in diagnostics.values())
    model.freeze_backbone(False)
    assert all(module.training for module in backbone.modules())


@pytest.mark.parametrize("zero_setting", ["max_correction", "gate_max"])
def test_zero_bound_preserves_original_forecast(zero_setting):
    config = StageAdapterConfig(
        physical_channels=3, latent_channels=2, hidden_dim=8,
        **{zero_setting: 0.},
    )
    model = StageManifoldKAI(make_backbone(), config).eval()
    activate_adapters(model)
    x, physical, latent = fields()
    torch.testing.assert_close(model(x, physical, latent), model.backbone(x), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["residual", "filter"])
def test_frozen_backbone_keeps_gradient_path_through_multistep_rollout(mode):
    model = make_model(mode)
    activate_adapters(model)
    model.freeze_backbone()
    original_weights = deepcopy(model.backbone.state_dict())
    x, physical, latent = fields(batch=1)
    x.requires_grad_()
    physical.requires_grad_()
    latent.requires_grad_()
    state = x
    first_prediction = None
    for step in range(3):
        state = model(state, physical, latent)
        if step == 0:
            state.retain_grad()
            first_prediction = state
    # Only a terminal forecast loss: earlier predictions must remain in the graph.
    (state - .7).square().mean().backward()
    assert first_prediction.grad is not None
    assert first_prediction.grad.abs().sum().item() > 0
    for guide in (x, physical, latent):
        assert guide.grad is not None and torch.isfinite(guide.grad).all()
        assert guide.grad.abs().sum().item() > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    adapter_grads = [
        parameter.grad for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.grad is not None
    ]
    assert adapter_grads and all(torch.isfinite(grad).all() for grad in adapter_grads)
    assert any(grad.abs().sum().item() > 0 for grad in adapter_grads)
    for key, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, original_weights[key], rtol=0, atol=0)
    model.freeze_backbone(False)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


@pytest.mark.parametrize("ddp", [False, True])
def test_original_kai_checkpoint_loads_exactly(tmp_path, ddp):
    expected = make_backbone().eval()
    state = expected.state_dict()
    if ddp:
        state = {"module." + key: value for key, value in state.items()}
    path = tmp_path / "original_kai.pt"
    torch.save({"model_state": state, "epoch": 19}, path)
    restored = make_backbone().eval()
    load_kai_checkpoint(restored, path)
    x, _, _ = fields()
    torch.testing.assert_close(restored(x), expected(x), rtol=0, atol=0)


def test_incompatible_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"model_state": {"unknown.weight": torch.zeros(1)}}, path)
    with pytest.raises((RuntimeError, ValueError)):
        load_kai_checkpoint(make_backbone(), path)


@pytest.mark.parametrize(
    "change",
    [
        {"physical_channels": 0}, {"latent_channels": -1},
        {"hidden_dim": 0}, {"stage_indices": (0, 0)},
        {"stage_indices": (-1,)}, {"stage_indices": (4,)},
        {"mode": "unknown"}, {"max_correction": -1.},
        {"gate_max": -0.1}, {"gate_max": 1.1},
        {"max_correction": float("nan")}, {"gate_max": float("inf")},
    ],
)
def test_invalid_config_rejected(change):
    kwargs = {"physical_channels": 3, "latent_channels": 2, **change}
    with pytest.raises((ValueError, TypeError)):
        StageManifoldKAI(make_backbone(), StageAdapterConfig(**kwargs))


@pytest.mark.parametrize("bad_field", ["batch", "physical_channels", "latent_channels", "grid", "rank"])
def test_guide_shape_mismatch_rejected(bad_field):
    model = make_model()
    x, physical, latent = fields()
    if bad_field == "batch":
        physical = physical[:1]
    elif bad_field == "physical_channels":
        physical = physical[:, :2]
    elif bad_field == "latent_channels":
        latent = latent[:, :1]
    elif bad_field == "grid":
        latent = latent[..., :-1]
    else:
        physical = physical[0]
    with pytest.raises(ValueError):
        model(x, physical, latent)


def test_enabled_requires_the_configured_guides():
    model = make_model()
    x, physical, latent = fields()
    with pytest.raises(ValueError):
        model(x)
    with pytest.raises(ValueError):
        model(x, physical)
    with pytest.raises(ValueError):
        model(x, latent_guide=latent)


def test_stage_alignment_is_area_weighted_and_differentiable():
    target = torch.zeros(1, 1, 2, 2)
    first = torch.tensor([[[[1., 1.], [3., 3.]]]], requires_grad=True)
    second = torch.tensor([[[[2., 2.], [2., 2.]]]], requires_grad=True)
    # Latitude weights 1 and 3: first stage MSE=7, second=4, stage mean=5.5.
    weights = torch.tensor([[1., 1.], [3., 3.]])
    loss = stage_alignment_loss({0: first, 3: second}, target, weights)
    assert loss.item() == pytest.approx(5.5)
    scaled = stage_alignment_loss({0: first, 3: second}, target, weights * 17)
    assert scaled.item() == pytest.approx(loss.item())
    loss.backward()
    assert first.grad is not None and second.grad is not None
    assert first.grad[0, 0, 1, 0] == pytest.approx(9 * first.grad[0, 0, 0, 0].item())


@pytest.mark.parametrize("weight", [-1., 0., float("nan"), float("inf")])
def test_stage_alignment_rejects_invalid_area_weights(weight):
    guide = torch.zeros(1, 3, 6, 12)
    with pytest.raises(ValueError):
        stage_alignment_loss({0: guide + 1}, guide, torch.full((6, 12), weight))
