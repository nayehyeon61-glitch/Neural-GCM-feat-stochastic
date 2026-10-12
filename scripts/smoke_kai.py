#!/usr/bin/env python3
"""Numerical stage-integration check; synthetic inputs, not forecast skill.

--real-neuralgcm runs the official bundled toy NeuralGCM with untrained closure
weights and a synthetic initial latent. It checks the actual JAX->NumPy->Torch
boundary. Without that flag, only the inexpensive PyTorch stage path runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from ngcm_memory.kai import KAI, StageAdapterConfig, StageManifoldKAI, stage_alignment_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-neuralgcm", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    h, w, c, latent_channels = 8, 16, 3, 2
    if a.real_neuralgcm:
        import jax
        import jax.numpy as jnp
        import neuralgcm
        from ngcm_memory.closure import ClosureConfig, init_params
        from ngcm_memory.neuralgcm_adapter import NeuralGCMMemoryAdapter
        from ngcm_memory.stage_guides import NeuralGCMStageGuide

        base = neuralgcm.PressureLevelModel.from_checkpoint(neuralgcm.demo.load_checkpoint_tl63_stochastic())
        data = neuralgcm.demo.load_data(base.data_coords).isel(time=0)
        inputs, forcing = base.inputs_from_xarray(data), base.forcings_from_xarray(data)
        levels = base.model_coords.vertical.layers
        config = ClosureConfig(feature_dim=levels + latent_channels, levels=levels,
                               latent_dim=latent_channels, hidden_dim=4, latent_hidden_dim=4)
        adapter = NeuralGCMMemoryAdapter(base, config)
        params = init_params(jax.random.PRNGKey(9), config)
        state = adapter.encode(inputs, forcing, jax.random.PRNGKey(17))
        state = state._replace(latent_state=state.latent_state._replace(
            latent=jnp.full_like(state.latent_state.latent, .2)))
        bridge = NeuralGCMStageGuide(adapter, params,
            target_longitude=np.arange(w) * 360. / w,
            target_latitude=np.linspace(78.75, -78.75, h),
            pressure_levels=[500., 850.], temperature_means=[260., 280.], temperature_stds=[10., 10.])
        guides = []
        for lead in (1., 2.):
            print(f"Actual NeuralGCM guide: lead {lead:g} h", flush=True)
            state, frame = bridge.advance(state, forcing, interval_hours=1., lead_hours=lead)
            guides.append((torch.from_numpy(frame.physical.copy())[None], torch.from_numpy(frame.latent.copy())[None]))
        kind = "official_toy_neuralgcm_untrained_closure_synthetic_origin_latent"
    else:
        guides = [(torch.randn(1, 2, h, w), torch.randn(1, latent_channels, h, w)) for _ in range(2)]
        kind = "synthetic_guide_interface_only"
    model = StageManifoldKAI(KAI(in_chans=c, out_chans=c, dims=[8]*4, depths=[1]*4),
                            StageAdapterConfig(physical_channels=2, latent_channels=latent_channels)).freeze_backbone()
    x = torch.randn(1, c, h, w)
    with torch.no_grad():
        baseline = model.backbone(x)
        if not torch.equal(baseline, model(x, enabled=False)):
            raise AssertionError("Disabled adapters changed the KAI prediction")
    prediction = x
    loss = x.new_zeros(())
    for physical, latent in guides:
        prediction, diagnostics = model.forward_with_diagnostics(prediction, physical, latent)
        loss = loss + stage_alignment_loss(diagnostics, physical, torch.ones(h, w))
    loss = loss + prediction.square().mean()
    loss.backward()
    gradient = sum(float(parameter.grad.square().sum()) for parameter in model.adapters.parameters() if parameter.grad is not None)**.5
    if not torch.isfinite(prediction).all() or not np.isfinite(gradient) or gradient <= 0:
        raise AssertionError("Nonfinite prediction or no stage-adapter gradient")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise AssertionError("Frozen KAI backbone received parameter gradients")
    print(json.dumps(dict(validation_kind=kind, leads=len(guides), stages=list(diagnostics),
                          output_shape=list(prediction.shape), disabled_exact=True,
                          finite=True, adapter_gradient_norm=gradient, forecast_skill_validated=False), indent=2))


if __name__ == "__main__":
    main()
