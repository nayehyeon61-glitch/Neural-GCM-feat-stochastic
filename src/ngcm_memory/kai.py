"""Optional PyTorch stage adapters for the unmodified KAI-Atmos backbone.

The manifold generator owns physical time, recurrent memory, and stochastic
noise. Advance that generator *once per forecast lead* and pass the same pair
of target-lead guides to every enabled KAI stage. These adapters have no
temporal state and do not sample additional noise.

``physical_guide`` is a decoded physical forecast, normalized with the training
data's channel ordering, means and scales. ``latent_guide`` is a spatial latent
map from the same ensemble member and lead, already aligned to the KAI grid.
Both use BCHW layout. A learned readout gives each stage a physical coordinate
system, anchored by :func:`stage_alignment_loss`; arbitrary feature channels
are not treated as conserved physical quantities.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import math
from os import PathLike

import torch
from torch import Tensor, nn

from .kai_backbone import KAI


@dataclass(frozen=True)
class StageAdapterConfig:
    """Stage correction settings; bounds are in backbone feature units.

    ``residual`` learns a correction from stage features, physical innovation,
    and the latent guide. ``filter`` nudges features toward a learned lift of
    the guides. Neither mode is an exact physical projection or a Bayesian
    data-assimilation filter. ``gate_max`` is at most one, so each element's
    correction is bounded by ``max_correction`` (more tightly by their product).
    Stage indices are zero-based; an empty tuple selects no adapters.
    """

    physical_channels: int
    latent_channels: int
    stage_indices: tuple[int, ...] = (0, 1, 2, 3)
    mode: str = "residual"
    hidden_dim: int = 32
    max_correction: float = 0.1
    gate_max: float = 1.0

    def __post_init__(self) -> None:
        for name in ("physical_channels", "latent_channels", "hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.stage_indices, tuple):
            raise ValueError("stage_indices must be an immutable tuple")
        if any(type(i) is not int or i not in range(4) for i in self.stage_indices):
            raise ValueError("stage_indices must contain integers from 0 through 3")
        if len(set(self.stage_indices)) != len(self.stage_indices):
            raise ValueError("stage_indices must not contain duplicates")
        if self.mode not in ("residual", "filter"):
            raise ValueError("mode must be 'residual' or 'filter'")
        for name in ("max_correction", "gate_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite nonnegative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.gate_max > 1:
            raise ValueError("gate_max must be in [0, 1]")


def _validate_bchw(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or any(size < 1 for size in value.shape):
        raise ValueError(f"{name} must have nonempty BCHW shape; got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a real floating-point dtype")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _validate_guide(name: str, guide: Tensor, x: Tensor, channels: int) -> None:
    _validate_bchw(name, guide)
    expected = (x.shape[0], channels, x.shape[2], x.shape[3])
    if tuple(guide.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}; got {tuple(guide.shape)}")
    if guide.device != x.device:
        raise ValueError(f"{name} and input must be on the same device")
    if guide.dtype != x.dtype:
        raise ValueError(f"{name} and input must have the same dtype")


class _StageAdapter(nn.Module):
    def __init__(self, channels: int, config: StageAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.physical_readout = nn.Conv2d(channels, config.physical_channels, 1)
        condition_channels = channels + config.physical_channels + config.latent_channels
        self.condition = nn.Sequential(
            nn.Conv2d(condition_channels, config.hidden_dim, 1),
            nn.GELU(),
        )
        self.gate = nn.Conv2d(config.hidden_dim, channels, 1)
        # Begin with a conservative, still trainable correction gate.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        if config.mode == "residual":
            self.delta = nn.Conv2d(config.hidden_dim, channels, 1)
        else:
            self.lift = nn.Sequential(
                nn.Conv2d(config.physical_channels + config.latent_channels,
                          config.hidden_dim, 1),
                nn.GELU(),
                nn.Conv2d(config.hidden_dim, channels, 1),
            )

    def forward(self, h: Tensor, physical: Tensor, latent: Tensor) -> tuple[Tensor, Tensor]:
        # Input guides have been checked against x. Autocast may give later
        # features a lower precision, so align internally without moving devices.
        physical = physical.to(dtype=h.dtype)
        latent = latent.to(dtype=h.dtype)
        innovation = physical - self.physical_readout(h)
        condition = self.condition(torch.cat((h, innovation, latent), dim=1))
        gate = self.config.gate_max * torch.sigmoid(self.gate(condition))
        if self.config.mode == "residual":
            direction = self.delta(condition)
        else:
            candidate = self.lift(torch.cat((physical, latent), dim=1))
            direction = candidate - h
        corrected = h + self.config.max_correction * gate * torch.tanh(direction)
        # Anchor corrected stage features, not only their uncorrected inputs.
        return corrected, self.physical_readout(corrected)


class StageManifoldKAI(nn.Module):
    """Wrap KAI's four full-resolution stages with independent learned adapters.

    The backbone and final head are reused verbatim. Call with ``enabled=False``
    for an exact direct call to the original backbone: no guides, validation of
    unused guides, adapter execution, or additional random sampling occurs.
    Equality in training with stochastic dropout additionally requires the same
    RNG state, as for any two calls to the original backbone itself.
    """

    def __init__(self, backbone: KAI, config: StageAdapterConfig) -> None:
        super().__init__()
        if not isinstance(backbone, KAI):
            raise TypeError("backbone must be a ngcm_memory.kai_backbone.KAI")
        if not isinstance(config, StageAdapterConfig):
            raise TypeError("config must be a StageAdapterConfig")
        self.backbone = backbone
        self.config = config
        self._backbone_frozen = False
        core = backbone.core_model
        if len(core.stages) != 4 or len(core.downsample_layers) != 4:
            raise ValueError("KAI must have exactly four full-resolution stages")
        widths = [core.downsample_layers[0][0].pointwise.out_channels]
        widths.extend(core.downsample_layers[i][1].pointwise.out_channels for i in range(1, 4))
        self.adapters = nn.ModuleDict({
            str(i): _StageAdapter(widths[i], config) for i in config.stage_indices
        })
        # A wrapper around a pretrained GPU/double backbone starts on its device
        # and dtype; normal Module.to() remains available afterward.
        reference = next(backbone.parameters())
        self.adapters.to(device=reference.device, dtype=reference.dtype)

    def freeze_backbone(self, freeze: bool = True) -> StageManifoldKAI:
        """Freeze backbone parameters and dropout; adapters remain trainable."""
        if not isinstance(freeze, bool):
            raise TypeError("freeze must be bool")
        self._backbone_frozen = freeze
        self.backbone.requires_grad_(not freeze)
        self.backbone.train(False if freeze else self.training)
        return self

    def train(self, mode: bool = True) -> StageManifoldKAI:
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward(
        self,
        x: Tensor,
        physical_guide: Tensor | None = None,
        latent_guide: Tensor | None = None,
        *,
        enabled: bool = True,
    ) -> Tensor:
        prediction, _ = self.forward_with_diagnostics(
            x, physical_guide, latent_guide, enabled=enabled,
        )
        return prediction

    def forward_with_diagnostics(
        self,
        x: Tensor,
        physical_guide: Tensor | None = None,
        latent_guide: Tensor | None = None,
        *,
        enabled: bool = True,
    ) -> tuple[Tensor, dict[int, Tensor]]:
        """Return forecast and each adapted stage's BCHW physical readout."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        if not enabled or not self.adapters:
            return self.backbone(x), {}
        _validate_bchw("x", x)
        if x.shape[1] != self.backbone.in_chans:
            raise ValueError(f"x must have {self.backbone.in_chans} channels")
        # The original band kernels need at least five latitude/longitude cells.
        if min(x.shape[-2:]) < 5:
            raise ValueError("KAI requires at least five cells in each spatial dimension")
        reference = next(self.backbone.parameters())
        if x.device != reference.device or x.dtype != reference.dtype:
            raise ValueError("x must match the backbone parameter device and dtype")
        if physical_guide is None or latent_guide is None:
            raise ValueError("enabled adapters require both physical_guide and latent_guide")
        _validate_guide("physical_guide", physical_guide, x, self.config.physical_channels)
        _validate_guide("latent_guide", latent_guide, x, self.config.latent_channels)
        h = x
        diagnostics: dict[int, Tensor] = {}
        core = self.backbone.core_model
        for i, (transition, stage) in enumerate(zip(core.downsample_layers, core.stages)):
            h = stage(transition(h))
            if h.shape[0] != x.shape[0] or h.shape[-2:] != x.shape[-2:]:
                raise ValueError(f"KAI stage {i} changed batch or spatial resolution")
            if str(i) in self.adapters:
                h, diagnostics[i] = self.adapters[str(i)](h, physical_guide, latent_guide)
                _validate_bchw(f"stage {i} physical readout", diagnostics[i])
        prediction = core.head(h)
        _validate_bchw("prediction", prediction)
        return prediction, diagnostics


def stage_alignment_loss(
    diagnostics: Mapping[int, Tensor],
    physical_guide: Tensor,
    area_weights: Tensor,
) -> Tensor:
    """Area-weighted MSE averaged over stages, batches, and physical channels.

    ``area_weights`` has exactly shape ``(H, W)``, is finite and nonnegative,
    and has a positive total. Scaling all weights by a constant does not change
    the loss. All tensors must share a device. Mixed floating precision (e.g.
    autocast readouts and float32 guides) is accumulated in at least float32.
    An empty diagnostics mapping returns a differentiable scalar zero linked
    to the guide. Detach the guide explicitly if it should act as a fixed target.
    """
    if not isinstance(diagnostics, Mapping):
        raise TypeError("diagnostics must be a mapping from stage index to tensor")
    _validate_bchw("physical_guide", physical_guide)
    if not isinstance(area_weights, Tensor) or not area_weights.is_floating_point():
        raise TypeError("area_weights must be a real floating-point torch.Tensor")
    if tuple(area_weights.shape) != tuple(physical_guide.shape[-2:]):
        raise ValueError("area_weights must have exactly shape (H, W)")
    if area_weights.device != physical_guide.device:
        raise ValueError("area_weights and physical_guide must share a device")
    if not torch.isfinite(area_weights).all().item() or (area_weights < 0).any().item():
        raise ValueError("area_weights must be finite and nonnegative")
    # Normalize by the maximum first to avoid overflow in a large or highly
    # scaled finite weight map; the normalization cancels in the weighted mean.
    maximum = area_weights.max()
    if maximum.item() <= 0:
        raise ValueError("area_weights must have a positive sum")
    dtype = torch.promote_types(physical_guide.dtype, torch.float32)
    dtype = torch.promote_types(dtype, area_weights.dtype)
    for stage, readout in diagnostics.items():
        if type(stage) is not int or stage not in range(4):
            raise ValueError("diagnostics keys must be stage indices 0 through 3")
        _validate_bchw(f"stage {stage} physical readout", readout)
        if readout.shape != physical_guide.shape:
            raise ValueError("every physical readout must match physical_guide shape")
        if readout.device != physical_guide.device:
            raise ValueError("physical readouts and physical_guide must share a device")
        dtype = torch.promote_types(dtype, readout.dtype)
    guide = physical_guide.to(dtype=dtype)
    if not diagnostics:
        # Avoid overflow from guide.sum() on finite, large inputs.
        return (guide.reshape(-1)[0] * 0.0).sum()
    weights = area_weights.to(dtype=dtype) / maximum.to(dtype=dtype)
    weights = weights / weights.sum()
    losses = [
        ((readout.to(dtype=dtype) - guide).square() * weights).sum(dim=(-2, -1)).mean()
        for readout in diagnostics.values()
    ]
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss).item():
        raise ValueError("stage alignment loss overflowed; check physical normalization")
    return loss


def load_kai_checkpoint(backbone: KAI, path: str | PathLike[str]) -> object:
    """Strictly load original KAI tensor weights, without unsafe pickle fallback.

    Accept an original ``model_state`` or ``state_dict`` checkpoint wrapper, or
    a plain state dictionary. Strip only a leading DataParallel ``module.``;
    other prefixes are not guessed. Load the backbone before wrapping it, or
    pass ``wrapped_model.backbone``. Original KAI optimizer metadata can contain
    a ruamel YAML ``ScalarFloat`` learning rate. When ruamel.yaml is installed,
    only that class is temporarily allowlisted; all other unsupported pickle
    objects are still rejected by ``torch.load(weights_only=True)``.
    """
    if not isinstance(backbone, KAI):
        raise TypeError("backbone must be a KAI instance")
    # Lazy import keeps tensor-only checkpoints usable without the YAML extra.
    # This is a narrow compatibility allowance for the original KAI trainer's
    # AdamW lr=params.lr, not permission to deserialize arbitrary Python objects.
    try:
        from ruamel.yaml.scalarfloat import ScalarFloat
    except ModuleNotFoundError:
        yaml_context = nullcontext()
    else:
        yaml_context = torch.serialization.safe_globals([ScalarFloat])
    with yaml_context:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a state dictionary")
    if "model_state" in payload:
        state = payload["model_state"]
    elif "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint state dictionary must be a nonempty mapping")
    normalized: dict[str, Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError("checkpoint state dictionary must map names to tensors")
        name = key.removeprefix("module.")
        if name in normalized:
            raise ValueError(f"duplicate checkpoint parameter after prefix removal: {name}")
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ValueError(f"checkpoint parameter {name} contains nonfinite values")
        normalized[name] = value
    return backbone.load_state_dict(normalized, strict=True)


__all__ = ["KAI", "StageAdapterConfig", "StageManifoldKAI", "stage_alignment_loss",
           "load_kai_checkpoint"]
