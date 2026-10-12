"""Causal KAI training with frozen, precomputed NeuralGCM ensemble guides.

The archive stores guide forecasts, never future observations as model inputs.
One origin is loaded at a time; each ensemble trajectory feeds its own KAI
prediction into the next day. All members and lead times retain gradients.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

GUIDE_FORMAT = "ngcm_memory.kai_guides.v1"
CHECKPOINT_FORMAT = "ngcm_memory.kai_stage_checkpoint.v1"
IDENTITY_KEYS = ("base_checkpoint_sha256", "closure_checkpoint_sha256", "source_encoder_sha256", "guide_fit_end", "forcing_mode")


def _time(value: Any) -> np.datetime64:
    try:
        time = np.datetime64(str(value).removesuffix("Z"), "ns")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid UTC timestamp: {value!r}") from exc
    if np.isnat(time):
        raise ValueError("NaT timestamps are forbidden")
    return time


@dataclass
class GuideArchive:
    initial: np.ndarray
    physical_guides: np.ndarray
    latent_guides: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    lead_hours: np.ndarray
    metadata: dict[str, Any]
    targets: np.ndarray | None = None
    path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path, *, require_targets: bool = False) -> "GuideArchive":
        with np.load(path, allow_pickle=False) as data:
            required = ("initial", "physical_guides", "latent_guides", "latitude", "longitude", "lead_hours", "metadata_json")
            missing = set(required) - set(data.files)
            if missing:
                raise ValueError(f"{path}: missing archive arrays {sorted(missing)}")
            archive = cls(
                initial=data["initial"], physical_guides=data["physical_guides"],
                latent_guides=data["latent_guides"], latitude=data["latitude"],
                longitude=data["longitude"], lead_hours=data["lead_hours"],
                metadata=json.loads(str(data["metadata_json"].item())),
                targets=data["targets"] if "targets" in data else None, path=Path(path),
            )
        archive.validate(require_targets=require_targets)
        return archive

    def validate(self, *, require_targets: bool = False) -> None:
        meta = self.metadata
        if meta.get("format") != GUIDE_FORMAT or meta.get("normalized") is not True:
            raise ValueError(f"Expected normalized {GUIDE_FORMAT} archive")
        for key in (*IDENTITY_KEYS, "origin_time", "valid_times", "channels", "pressure_levels", "temperature_channels", "means", "stds"):
            if key not in meta:
                raise ValueError(f"Missing archive metadata: {key}")
        if meta["forcing_mode"] != "persistence":
            raise ValueError("Only causal persistence forcings are supported")
        for key in IDENTITY_KEYS[:3]:
            if not isinstance(meta[key], str) or not meta[key]:
                raise ValueError(f"{key} must identify the guide generator")
        if self.initial.ndim != 3:
            raise ValueError("initial must have shape [C,H,W]")
        c, h, w = self.initial.shape
        if min(c, h, w) < 1 or self.physical_guides.ndim != 5 or self.latent_guides.ndim != 5:
            raise ValueError("Guides must have shape [M,L,channels,H,W]")
        m, length, p, gh, gw = self.physical_guides.shape
        if min(m, length, p) < 1 or (gh, gw) != (h, w):
            raise ValueError("Physical guide ensemble, horizon, or spatial shape mismatch")
        if self.latent_guides.shape[:2] != (m, length) or self.latent_guides.shape[2] < 1 or self.latent_guides.shape[-2:] != (h, w):
            raise ValueError("Latent guide ensemble, horizon, or spatial shape mismatch")
        if self.latitude.shape != (h,) or self.longitude.shape != (w,):
            raise ValueError("Coordinate shapes do not match the KAI grid")
        if np.any(np.abs(self.latitude) > 90) or len(np.unique(self.latitude)) != h or len(np.unique(self.longitude)) != w:
            raise ValueError("Grid coordinates must be unique with latitude in [-90,90]")
        if (self.lead_hours.shape != (length,) or not np.isfinite(self.lead_hours).all()
                or self.lead_hours[0] <= 0
                or not np.allclose(self.lead_hours, self.lead_hours[0] * np.arange(1, length + 1), rtol=0, atol=1e-9)):
            raise ValueError("KAI requires positive, regularly spaced forecast lead hours")
        if len(meta["channels"]) != c or len(set(meta["channels"])) != c:
            raise ValueError("Ordered unique channel names must match initial")
        indexes = meta["temperature_channels"]
        if len(indexes) != p or len(set(indexes)) != p or any(type(i) is not int or not 0 <= i < c for i in indexes):
            raise ValueError("temperature_channels must map every physical guide to a unique KAI channel")
        levels = np.asarray(meta["pressure_levels"], dtype=float)
        if levels.shape != (p,) or not np.isfinite(levels).all() or np.any(levels <= 0) or len(np.unique(levels)) != p:
            raise ValueError("pressure_levels must give one unique positive hPa level per physical guide")
        means, stds = np.asarray(meta["means"]), np.asarray(meta["stds"])
        if means.shape != (c,) or stds.shape != (c,) or not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
            raise ValueError("One finite mean and positive standard deviation required per KAI channel")
        origin, valid = self.origin, [_time(t) for t in meta["valid_times"]]
        expected = origin + np.rint(self.lead_hours * 3_600_000_000_000).astype("timedelta64[ns]")
        if len(valid) != length or not np.array_equal(valid, expected):
            raise ValueError("valid_times must exactly equal origin_time + lead_hours")
        _time(meta["guide_fit_end"])
        if require_targets and self.targets is None:
            raise ValueError("Training and validation archives require targets")
        if self.targets is not None and self.targets.shape != (length, c, h, w):
            raise ValueError("targets must have shape [L,C,H,W]")
        arrays = (self.initial, self.physical_guides, self.latent_guides, self.latitude, self.longitude, self.lead_hours)
        if any(not np.isfinite(array).all() for array in arrays) or (self.targets is not None and not np.isfinite(self.targets).all()):
            raise ValueError("Archives may not contain non-finite values")

    @property
    def origin(self) -> np.datetime64:
        return _time(self.metadata["origin_time"])

    @property
    def last_valid(self) -> np.datetime64:
        return _time(self.metadata["valid_times"][-1])

    def schema(self) -> dict[str, Any]:
        """Strict grid/normalization/guide identity, independent of origin and horizon."""
        return {
            **{key: self.metadata[key] for key in (*IDENTITY_KEYS, "channels", "pressure_levels", "temperature_channels", "means", "stds")},
            "latitude": self.latitude.tolist(), "longitude": self.longitude.tolist(),
            "physical_channels": self.physical_guides.shape[2],
            "latent_channels": self.latent_guides.shape[2], "step_hours": float(self.lead_hours[0]),
        }


def discover_archives(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    paths = [directory] if directory.is_file() else sorted(directory.glob("*.npz"))
    if not paths:
        raise ValueError(f"No guide NPZ archives found: {directory}")
    return paths


def validate_archive_collection(paths: list[Path], *, require_targets: bool = False) -> dict[str, Any]:
    schema, origins = None, set()
    for path in paths:
        archive = GuideArchive.from_file(path, require_targets=require_targets)
        if schema is not None and archive.schema() != schema:
            raise ValueError(f"Inconsistent grid, normalization, channels, or guide identity: {path}")
        if archive.origin in origins:
            raise ValueError(f"Duplicate forecast origin: {archive.origin}")
        schema = archive.schema()
        origins.add(archive.origin)
    if schema is None:
        raise ValueError("At least one archive is required")
    return schema


def split_archives(paths: list[Path], train_before: str, validation_before: str) -> tuple[list[Path], list[Path], list[Path]]:
    """Discard windows crossing either boundary; never split a target sequence."""
    train_cut, validation_cut = _time(train_before), _time(validation_before)
    if validation_cut <= train_cut:
        raise ValueError("validation-before must be later than train-before")
    training, validation, discarded = [], [], []
    for path in paths:
        archive = GuideArchive.from_file(path, require_targets=True)
        if _time(archive.metadata["guide_fit_end"]) >= train_cut:
            raise ValueError("Guide generator fit must end before train-before; validation/test leakage detected")
        if archive.last_valid < train_cut:
            training.append(path)
        elif archive.origin >= train_cut and archive.last_valid < validation_cut:
            validation.append(path)
        else:
            discarded.append(path)
    if not training or not validation:
        raise ValueError("Both train and validation require complete non-crossing forecast windows")
    return training, validation, discarded


def _tensor(value: np.ndarray, model: torch.nn.Module) -> torch.Tensor:
    parameter = next(model.parameters())
    return torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)


def rollout_kai(model: torch.nn.Module, archive: GuideArchive, *, enabled: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [M,L,C,H,W] predictions and mean alignment loss; no targets read."""
    from ngcm_memory.kai import stage_alignment_loss
    members, length = archive.physical_guides.shape[:2]
    state = _tensor(archive.initial, model).unsqueeze(0).expand(members, -1, -1, -1)
    physical, latent = _tensor(archive.physical_guides, model), _tensor(archive.latent_guides, model)
    weights = _tensor(np.broadcast_to(np.cos(np.deg2rad(archive.latitude))[:, None], archive.initial.shape[-2:]).copy(), model).clamp_min(0)
    predictions, alignments = [], []
    for lead in range(length):
        state, diagnostics = model.forward_with_diagnostics(state, physical[:, lead], latent[:, lead], enabled=enabled)
        predictions.append(state)
        if diagnostics:
            alignments.append(stage_alignment_loss(diagnostics, physical[:, lead], weights))
    alignment = torch.stack(alignments).mean() if alignments else state.sum() * 0
    return torch.stack(predictions, dim=1), alignment


def _area_mean(field: torch.Tensor, latitude: np.ndarray) -> torch.Tensor:
    weight = torch.as_tensor(np.cos(np.deg2rad(latitude)), device=field.device, dtype=field.dtype).clamp_min(0)
    if weight.sum() <= 0:
        raise ValueError("Latitude weights have zero total area")
    weight = weight / weight.mean()
    return (field * weight[:, None]).mean()


def forecast_loss(predictions: torch.Tensor, targets: torch.Tensor, latitude: np.ndarray, *, loss: str = "crps", mean_mse_weight: float = 0.1) -> torch.Tensor:
    """Empirical-ensemble CRPS (including diagonal pairs), plus mean-field MSE.

    This is the proper CRPS of the finite empirical predictive distribution;
    it is not the unbiased/fair estimator for an underlying continuous law.
    """
    if predictions.ndim != 5 or targets.shape != predictions.shape[1:]:
        raise ValueError("Expected predictions [M,L,C,H,W] and targets [L,C,H,W]")
    if loss == "mse":
        return _area_mean((predictions - targets).square(), latitude)
    if loss == "relative-l2":
        difference = torch.linalg.vector_norm((predictions - targets).flatten(2), dim=-1)
        denominator = torch.linalg.vector_norm(targets.flatten(1), dim=-1).clamp_min(1e-12)
        return (difference / denominator).mean()
    if loss != "crps":
        raise ValueError(f"Unknown loss: {loss}")
    members = predictions.shape[0]
    score = (predictions - targets).abs().mean(dim=0)
    for first in range(members):
        for second in range(first + 1, members):
            score = score - (predictions[first] - predictions[second]).abs() / members**2
    return _area_mean(score, latitude) + mean_mse_weight * _area_mean((predictions.mean(dim=0) - targets).square(), latitude)


def load_training_checkpoint(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    from ngcm_memory.kai import KAI, StageAdapterConfig, StageManifoldKAI
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Not a KAI stage-training checkpoint")
    config_args = dict(checkpoint["stage_config"])
    config_args["stage_indices"] = tuple(config_args["stage_indices"])
    model = StageManifoldKAI(KAI(**checkpoint["backbone_args"]), StageAdapterConfig(**config_args))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def train_kai(model: torch.nn.Module, training: list[Path], validation: list[Path], *, output: str | Path, schema: dict[str, Any], backbone_args: dict[str, Any], stage_config: Any, epochs: int = 1, lr: float = 1e-3, stage_loss_weight: float = 0.1, mean_mse_weight: float = 0.1, loss: str = "crps", seed: int = 0, split_metadata: dict[str, Any] | None = None) -> list[dict[str, float]]:
    if epochs < 1 or lr <= 0 or stage_loss_weight < 0 or mean_mse_weight < 0:
        raise ValueError("Invalid epochs, learning rate, or loss weights")
    if not training or not validation:
        raise ValueError("Train and validation sets must be nonempty")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    output = Path(output)
    if output.resolve() in {p.resolve() for p in training + validation}:
        raise ValueError("Checkpoint may not overwrite a guide archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    rng, history, best = np.random.default_rng(seed), [], float("inf")
    for epoch in range(epochs):
        losses: dict[str, float] = {}
        for phase, paths in (("train", training), ("validation", validation)):
            model.train(phase == "train")
            # Frozen pretrained normalization/dropout must remain in evaluation mode.
            if not any(p.requires_grad for p in model.backbone.parameters()):
                model.backbone.eval()
            total = 0.0
            order = rng.permutation(len(paths)) if phase == "train" else range(len(paths))
            for index in order:
                archive = GuideArchive.from_file(paths[index], require_targets=True)
                if archive.schema() != schema:
                    raise ValueError(f"Archive schema changed: {paths[index]}")
                with torch.set_grad_enabled(phase == "train"):
                    predictions, alignment = rollout_kai(model, archive)
                    value = forecast_loss(predictions, _tensor(archive.targets, model), archive.latitude, loss=loss, mean_mse_weight=mean_mse_weight) + stage_loss_weight * alignment
                    if not torch.isfinite(value):
                        raise FloatingPointError(f"Non-finite {phase} loss: {paths[index]}")
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        value.backward()
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
                        optimizer.step()
                total += float(value.detach())
            losses[phase] = total / len(paths)
        history.append({"epoch": epoch + 1, **losses})
        print(json.dumps(history[-1]), flush=True)
        if losses["validation"] < best:
            best = losses["validation"]
            torch.save({
                "format": CHECKPOINT_FORMAT,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "backbone_args": backbone_args, "stage_config": asdict(stage_config),
                "schema": schema, "epoch": epoch + 1, "validation_loss": best,
                "objective": {"name": loss, "crps_estimator": "empirical", "mean_mse_weight": mean_mse_weight, "stage_loss_weight": stage_loss_weight},
                "seed": seed, "split": split_metadata or {}, "history": history,
            }, output)
    return history


def predict_archive(model: torch.nn.Module, archive: GuideArchive, *, schema: dict[str, Any], output: str | Path, baseline: bool = False, validation_before: str | None = None) -> dict[str, Any]:
    if archive.schema() != schema:
        raise ValueError("Inference grid, normalization, channels, or guide-generator identity differs from checkpoint")
    if archive.path is not None and Path(output).resolve() == archive.path.resolve():
        raise ValueError("Prediction output may not overwrite its input archive")
    model.eval()
    with torch.no_grad():
        predictions, _ = rollout_kai(model, archive, enabled=not baseline)
    if not torch.isfinite(predictions).all():
        raise FloatingPointError("Non-finite KAI forecast; output has not been saved")
    normalized = predictions.cpu().numpy()
    means = np.asarray(schema["means"], dtype=np.float32)[None, None, :, None, None]
    stds = np.asarray(schema["stds"], dtype=np.float32)[None, None, :, None, None]
    metadata = {**archive.metadata, "format": "ngcm_memory.kai_prediction.v1", "baseline": baseline, "forecast_normalized": True, "forecast_units": "original channel units"}
    if validation_before is not None:
        metadata.update(validation_before=validation_before, held_out_after_validation=bool(archive.origin >= _time(validation_before)))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, forecast_normalized=normalized, forecast=normalized * stds + means,
                        latitude=archive.latitude, longitude=archive.longitude, lead_hours=archive.lead_hours,
                        metadata_json=np.asarray(json.dumps(metadata)))
    return metadata
