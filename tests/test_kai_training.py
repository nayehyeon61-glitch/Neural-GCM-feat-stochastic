"""End-to-end causal KAI stage training on small synthetic guide archives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ngcm_memory.kai import KAI, StageAdapterConfig, StageManifoldKAI
from ngcm_memory.kai_training import (
    GUIDE_FORMAT, GuideArchive, forecast_loss, load_training_checkpoint,
    predict_archive, rollout_kai, split_archives, train_kai,
    validate_archive_collection,
)


@pytest.fixture(autouse=True)
def small_torch_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def make_archive(path: Path, origin: str = "2000-01-01", *, targets: bool = True, step_hours: int = 24, **metadata_changes) -> Path:
    rng = np.random.default_rng(12)
    leads = np.array([step_hours, 2 * step_hours])
    metadata = {
        "format": GUIDE_FORMAT, "origin_time": origin,
        "valid_times": [str(np.datetime64(origin, "h") + np.timedelta64(int(h), "h")) for h in leads],
        "channels": ["temperature_500", "u_500"], "pressure_levels": [500], "temperature_channels": [0],
        "means": [280.0, 0.0], "stds": [10.0, 5.0], "normalized": True,
        "base_checkpoint_sha256": "base-hash", "closure_checkpoint_sha256": "closure-hash",
        "source_encoder_sha256": "encoder-hash", "guide_fit_end": "1999-12-31", "forcing_mode": "persistence", "seed": 12,
    }
    metadata.update(metadata_changes)
    arrays = {
        "initial": rng.normal(size=(2, 6, 8)).astype(np.float32),
        "physical_guides": rng.normal(size=(2, 2, 1, 6, 8)).astype(np.float32),
        "latent_guides": rng.normal(size=(2, 2, 3, 6, 8)).astype(np.float32),
        "latitude": np.linspace(-75, 75, 6), "longitude": np.arange(8) * 45,
        "lead_hours": leads, "metadata_json": np.asarray(json.dumps(metadata)),
    }
    if targets:
        arrays["targets"] = rng.normal(size=(2, 2, 6, 8)).astype(np.float32)
    np.savez(path, **arrays)
    return path


def make_model():
    args = dict(img_size=(6, 8), in_chans=2, out_chans=2, depths=(1, 1, 1, 1), dims=(8, 8, 8, 8), drop_path_rate=0.0)
    config = StageAdapterConfig(physical_channels=1, latent_channels=3, hidden_dim=4)
    return StageManifoldKAI(KAI(**args), config).freeze_backbone(), args, config


def test_archive_times_and_temperature_mapping_rejected(tmp_path):
    path = make_archive(tmp_path / "one.npz", valid_times=["2000-01-03", "2000-01-04"])
    with pytest.raises(ValueError, match="valid_times"):
        GuideArchive.from_file(path)
    path = make_archive(path, temperature_channels=[2])
    with pytest.raises(ValueError, match="temperature_channels"):
        GuideArchive.from_file(path)


def test_hourly_spacing_is_supported_and_schema_tracks_interval(tmp_path):
    one = GuideArchive.from_file(make_archive(tmp_path / "one.npz", step_hours=6))
    assert one.schema()["step_hours"] == 6.0
    two = make_archive(tmp_path / "two.npz", origin="2000-01-04", step_hours=24)
    with pytest.raises(ValueError, match="Inconsistent"):
        validate_archive_collection([one.path, two])


def test_temporal_split_discards_crossings_and_rejects_fitted_future(tmp_path):
    train = make_archive(tmp_path / "train.npz")
    crossing = make_archive(tmp_path / "crossing.npz", origin="2000-01-03")
    validation = make_archive(tmp_path / "val.npz", origin="2000-01-05")
    test = make_archive(tmp_path / "test.npz", origin="2000-01-09")
    assert split_archives([train, crossing, validation, test], "2000-01-05", "2000-01-09") == ([train], [validation], [crossing, test])
    leaked = make_archive(tmp_path / "leaked.npz", guide_fit_end="2000-01-05")
    with pytest.raises(ValueError, match="leakage"):
        split_archives([leaked, validation], "2000-01-05", "2000-01-09")


def test_grid_normalization_and_generator_identity_checked(tmp_path):
    first = make_archive(tmp_path / "one.npz")
    second = make_archive(tmp_path / "two.npz", origin="2000-01-05", means=[281., 0.])
    with pytest.raises(ValueError, match="Inconsistent"):
        validate_archive_collection([first, second])
    second = make_archive(second, origin="2000-01-05", source_encoder_sha256="other")
    with pytest.raises(ValueError, match="Inconsistent"):
        validate_archive_collection([first, second])


def test_forecast_never_reads_future_targets_and_has_multi_step_gradients(tmp_path):
    archive = GuideArchive.from_file(make_archive(tmp_path / "one.npz"))
    model, _, _ = make_model()
    model.eval()
    seen_outputs = []
    original_forward = model.forward_with_diagnostics

    def recording_forward(*args, **kwargs):
        prediction, diagnostics = original_forward(*args, **kwargs)
        prediction.retain_grad()
        seen_outputs.append(prediction)
        return prediction, diagnostics

    model.forward_with_diagnostics = recording_forward
    predictions, alignment = rollout_kai(model, archive)
    predictions[:, -1].square().mean().backward()
    assert seen_outputs[0].grad is not None and seen_outputs[0].grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapters.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())
    assert alignment.isfinite()
    model.forward_with_diagnostics = original_forward
    archive.targets[:] = 9999
    changed, _ = rollout_kai(model, archive)
    torch.testing.assert_close(predictions, changed, rtol=0, atol=0)
    archive.targets = None
    missing, _ = rollout_kai(model, archive)
    torch.testing.assert_close(predictions, missing, rtol=0, atol=0)


def test_empirical_crps_preserves_ensemble_pair_term():
    pred = torch.tensor([0., 2.]).reshape(2, 1, 1, 1, 1)
    target = torch.ones((1, 1, 1, 1))
    # Mean absolute error = 1; half pair expectation = .5.
    assert forecast_loss(pred, target, np.array([0.]), mean_mse_weight=0).item() == pytest.approx(0.5)


def test_train_checkpoint_predict_and_exact_baseline(tmp_path):
    train_path = make_archive(tmp_path / "train.npz")
    val_path = make_archive(tmp_path / "val.npz", origin="2000-01-05")
    schema = validate_archive_collection([train_path, val_path], require_targets=True)
    model, args, config = make_model()
    output = tmp_path / "best.pt"
    history = train_kai(model, [train_path], [val_path], output=output, schema=schema, backbone_args=args, stage_config=config, epochs=1)
    assert len(history) == 1 and np.isfinite(history[0]["validation"])
    restored, checkpoint = load_training_checkpoint(output)
    assert checkpoint["schema"] == schema
    archive = GuideArchive.from_file(val_path)
    predictions, _ = rollout_kai(restored, archive, enabled=False)
    state = torch.tensor(archive.initial).unsqueeze(0)
    with torch.no_grad():
        first = restored.backbone(state)
        second = restored.backbone(first)
    torch.testing.assert_close(predictions[0, 0], first[0])
    torch.testing.assert_close(predictions[0, 1], second[0])
    torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
    predict_archive(restored, archive, schema=schema, output=tmp_path / "prediction.npz", validation_before="2000-01-05")
    with np.load(tmp_path / "prediction.npz") as data:
        assert data["forecast"].shape == (2, 2, 2, 6, 8)
        np.testing.assert_allclose(data["forecast"][:, :, 0], data["forecast_normalized"][:, :, 0] * 10 + 280)
        assert json.loads(str(data["metadata_json"].item()))["held_out_after_validation"] is True


def test_cli_training_and_target_free_prediction(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "guides"
    data.mkdir()
    make_archive(data / "train.npz")
    make_archive(data / "val.npz", origin="2000-01-05")
    checkpoint = tmp_path / "best.pt"
    subprocess.run([sys.executable, str(root / "scripts/train_kai.py"), "--data", str(data),
                    "--train-before", "2000-01-05", "--validation-before", "2000-01-09", "--train-backbone",
                    "--dims", "8,8,8,8", "--depths", "1,1,1,1", "--hidden-dim", "4", "--epochs", "1", "--output", str(checkpoint)], check=True, capture_output=True, text=True)
    inference = make_archive(tmp_path / "inference.npz", origin="2000-01-09", targets=False)
    subprocess.run([sys.executable, str(root / "scripts/predict_kai.py"), "--checkpoint", str(checkpoint),
                    "--data", str(inference), "--output", str(tmp_path / "predictions"), "--baseline"], check=True, capture_output=True, text=True)
    with np.load(tmp_path / "predictions/inference.predictions.npz") as data:
        np.testing.assert_array_equal(data["forecast"][0], data["forecast"][1])
        assert json.loads(str(data["metadata_json"].item()))["held_out_after_validation"] is True


def test_pretrained_checkpoint_lead_mismatch_rejected_and_explicit_lead_accepted(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "six_hour_guides"
    data.mkdir()
    make_archive(data / "train.npz", step_hours=6)
    make_archive(data / "val.npz", origin="2000-01-05", step_hours=6)
    model, _, _ = make_model()
    pretrained = tmp_path / "pretrained.pt"
    torch.save(model.backbone.state_dict(), pretrained)
    output = tmp_path / "trained.pt"
    command = [sys.executable, str(root / "scripts/train_kai.py"), "--data", str(data),
               "--train-before", "2000-01-05", "--validation-before", "2000-01-09",
               "--kai-checkpoint", str(pretrained), "--dims", "8,8,8,8", "--depths", "1,1,1,1",
               "--hidden-dim", "4", "--epochs", "1", "--output", str(output)]
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "Guide interval differs" in rejected.stderr
    assert not output.exists()
    subprocess.run([*command, "--kai-step-hours", "6"], check=True, capture_output=True, text=True)
    checkpoint = torch.load(output, weights_only=True)
    assert checkpoint["schema"]["step_hours"] == 6
    assert checkpoint["split"]["pretrained_kai_step_hours"] == 6
    assert checkpoint["split"]["pretrained_kai_sha256"] == hashlib.sha256(pretrained.read_bytes()).hexdigest()
    assert checkpoint["split"]["train_backbone"] is False
