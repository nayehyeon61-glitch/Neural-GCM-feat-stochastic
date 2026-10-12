#!/usr/bin/env python3
"""Train only memory-closure parameters through a frozen NeuralGCM forecast.

Requires a trusted local upstream checkpoint and ERA5 already prepared on that
checkpoint's pressure-level grid, with its exact variable names and units.
No checkpoint download, source training, or ERA5 regridding is implicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import numpy as np
import optax

from ngcm_memory.closure import ClosureConfig, init_params
from ngcm_memory.features import SpatialHistory, thermal_with_empty_spatial
from ngcm_memory.neuralgcm_adapter import NeuralGCMMemoryAdapter
from ngcm_memory.training import (
    filter_spatial_origins, joint_forecast_loss, load_local_problem, make_train_step, rollout_ensemble,
    save_closure_checkpoint, sha256_file, slice_window, temporal_window_split,
    validate_time_axis,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", required=True, type=Path)
    p.add_argument("--trust-base-checkpoint", action="store_true", help="Acknowledge that the local upstream checkpoint is trusted pickle")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="Safe NPZ closure checkpoint; saves the best validation epoch")
    p.add_argument("--warmup-steps", type=int, default=4, help="Observed historical intervals before each origin")
    p.add_argument("--horizon-steps", type=int, default=4, help="Future observed intervals scored jointly")
    p.add_argument("--ensemble-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-train-windows", type=int)
    p.add_argument("--max-validation-windows", type=int)
    p.add_argument("--variables", default="temperature", help="Comma-separated decoded input variable names")
    p.add_argument("--scales", default="10", help="Positive physical-unit scales, one per variable; defaults to temperature/10 K")
    p.add_argument("--mean-mse-weight", type=float, default=0.1)
    p.add_argument("--forcing-mode", choices=("persistence", "observed"), default="persistence")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--tau-hours", default="6,24,120")
    p.add_argument("--stochastic-tau-hours", type=float, default=24.0)
    p.add_argument("--max-tendency-k-per-hour", type=float, default=0.2)
    p.add_argument("--max-diffusion-k-sqrt-hour", type=float, default=0.1)
    p.add_argument("--feature-scale-kelvin", type=float, default=10.)
    p.add_argument("--remove-global-mean", action="store_true", help="Remove each sigma layer's global mean correction")
    p.add_argument("--no-memory", action="store_true", help="Train a separate ablation without deterministic memory")
    p.add_argument("--no-stochastic", action="store_true", help="Train a separate ablation without OU forcing")
    p.add_argument("--no-spatial", action="store_true", help="Train without the instantaneous term; source history remains optional")
    p.add_argument("--spatial-history", type=Path, help="Causal observed history from the exported Climate Manifold encoder")
    p.add_argument("--spatial-history-steps", type=int, default=8)
    p.add_argument("--spatial-mode", choices=("recurrent", "initialization"), default="recurrent",
                   help="With source history: recurrent latent dynamics (default), or legacy initialization only")
    p.add_argument("--latent-loss-weight", type=float, default=None,
                   help="Default .1 in recurrent mode, 0 otherwise; explicit 0 disables latent supervision as an ablation")
    p.add_argument("--latent-hidden-dim", type=int, default=16)
    p.add_argument("--latent-tau-hours", type=float, default=24.)
    p.add_argument("--latent-memory-tau-hours", default="24,120,720")
    p.add_argument("--latent-max-diffusion-per-sqrt-hour", type=float, default=.1)
    p.add_argument("--latent-target-bound", type=float, default=5.)
    return p


def main(argv: list[str] | None = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    protected = {args.base_checkpoint.resolve(), args.data.resolve()}
    if args.spatial_history:
        protected.add(args.spatial_history.resolve())
    if {args.output.resolve(), args.output.with_suffix('.training.json').resolve()} & protected:
        p.error("Output must not overwrite an input dataset or checkpoint")
    if args.ensemble_size < 2 or args.epochs < 1:
        p.error("Training requires ensemble-size >= 2 and epochs >= 1")
    if args.learning_rate <= 0 or args.gradient_clip <= 0 or args.mean_mse_weight < 0:
        p.error("Learning rate/gradient clip must be positive; MSE weight nonnegative")
    for name in ("max_train_windows", "max_validation_windows"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    variables = tuple(args.variables.split(","))
    scale_values = [float(value) for value in args.scales.split(",")]
    if len(scale_values) != len(variables) or any(not np.isfinite(value) or value <= 0 for value in scale_values):
        p.error("--scales requires one finite positive scale per --variables entry")
    scales = dict(zip(variables, scale_values))
    recurrent = bool(args.spatial_history) and args.spatial_mode == "recurrent"
    latent_loss_weight = args.latent_loss_weight if args.latent_loss_weight is not None else (.1 if recurrent else 0.)
    if not np.isfinite(latent_loss_weight) or latent_loss_weight < 0:
        p.error("latent-loss-weight must be finite and nonnegative")
    if latent_loss_weight > 0 and not recurrent:
        p.error("Positive latent-loss-weight requires --spatial-history and --spatial-mode recurrent")
    if recurrent and latent_loss_weight == 0:
        print("Latent supervision is explicitly disabled: recurrent latents are trained only through physical forecast loss.",
              file=sys.stderr, flush=True)
    base, dataset, inputs, forcings = load_local_problem(
        args.base_checkpoint, args.data, trust_checkpoint=args.trust_base_checkpoint,
    )
    missing = set(variables) - set(inputs)
    if missing:
        p.error(f"Selected variables absent from checkpoint inputs: {sorted(missing)}")
    hours, internal_steps = validate_time_axis(dataset.time.values, base.timestep)
    train_origins, val_origins, cut = temporal_window_split(
        len(dataset.time), args.warmup_steps, args.horizon_steps,
        args.train_fraction, args.stride,
    )
    history = SpatialHistory.from_npz(args.spatial_history, base) if args.spatial_history else None
    spatial_dim = history.feature_dim if history else 0
    levels = base.model_coords.vertical.layers
    config = ClosureConfig(
        feature_dim=levels + spatial_dim, levels=levels, hidden_dim=args.hidden_dim,
        tau_hours=tuple(map(float, args.tau_hours.split(","))),
        stochastic_tau_hours=args.stochastic_tau_hours,
        max_tendency_k_per_hour=args.max_tendency_k_per_hour,
        max_diffusion_k_sqrt_hour=args.max_diffusion_k_sqrt_hour,
        enable_memory=not args.no_memory, enable_stochastic=not args.no_stochastic,
        enable_spatial=not args.no_spatial,
        latent_dim=spatial_dim if recurrent else 0,
        latent_hidden_dim=args.latent_hidden_dim,
        latent_tau_hours=args.latent_tau_hours,
        latent_memory_tau_hours=tuple(map(float, args.latent_memory_tau_hours.split(","))),
        latent_max_diffusion_per_sqrt_hour=args.latent_max_diffusion_per_sqrt_hour,
        latent_target_bound=args.latent_target_bound,
    )
    feature_fn = thermal_with_empty_spatial(base, spatial_dim, args.feature_scale_kelvin) if history and not recurrent else None
    adapter = NeuralGCMMemoryAdapter(
        base, config, feature_fn, feature_scale_kelvin=args.feature_scale_kelvin,
        remove_global_mean=args.remove_global_mean,
    )
    initializer = (history.transition_initializer(config) if recurrent else history.initializer(config)) if history else None
    if history:
        if args.spatial_history_steps < 1:
            p.error("spatial-history-steps must be positive")
        filter_kwargs = dict(recurrent=recurrent, horizon_steps=args.horizon_steps,
                             require_future_targets=latent_loss_weight > 0)
        train_origins, train_history_filter = filter_spatial_origins(
            history, dataset.time.values, train_origins, args.spatial_history_steps, **filter_kwargs)
        val_origins, validation_history_filter = filter_spatial_origins(
            history, dataset.time.values, val_origins, args.spatial_history_steps, **filter_kwargs)
        for partition, details in (("training", train_history_filter), ("validation", validation_history_filter)):
            if details["rejected_windows"]:
                print(json.dumps({"spatial_history_filter": partition, **details}), file=sys.stderr, flush=True)
        if not len(train_origins) or not len(val_origins):
            failed = train_history_filter if not len(train_origins) else validation_history_filter
            p.error(f"No complete train/validation windows with valid spatial history: {failed['first_error']}")
    train_origins = train_origins[:args.max_train_windows]
    val_origins = val_origins[:args.max_validation_windows]
    if not len(train_origins) or not len(val_origins):
        p.error("No complete train/validation windows with sufficient causal history")
    # Only one regridded context is live at a time. Caching all overlapping
    # windows would duplicate large source grids onto the accelerator.
    def context_for(origin):
        if history is None:
            return None
        context = (history.transition_context if recurrent else history.context)(
            dataset.time.values[origin], args.spatial_history_steps)
        if latent_loss_weight > 0:
            context["latent_targets"] = history.targets_at(dataset.time.values[origin + 1:origin + args.horizon_steps + 1])
        return context
    area_weights = base.data_coords.horizontal.quadrature_weights
    latent_area_weights = base.model_coords.horizontal.quadrature_weights if recurrent else None
    params = init_params(jax.random.PRNGKey(args.seed), config)
    optimizer = optax.chain(optax.clip_by_global_norm(args.gradient_clip), optax.adam(args.learning_rate))
    opt_state = optimizer.init(params)
    train_step = make_train_step(
        adapter, optimizer, warmup_steps=args.warmup_steps, horizon_steps=args.horizon_steps,
        internal_steps=internal_steps, variables=variables, scales=scales,
        area_weights=area_weights, forcing_mode=args.forcing_mode,
        mean_mse_weight=args.mean_mse_weight, memory_initializer=initializer,
        latent_loss_weight=latent_loss_weight, latent_area_weights=latent_area_weights,
    )

    @jax.jit
    def validation_loss(parameters, x, forcing, keys, context):
        output = rollout_ensemble(
            parameters, adapter, x, forcing, keys, warmup_steps=args.warmup_steps,
            horizon_steps=args.horizon_steps, internal_steps=internal_steps,
            variables=variables, forcing_mode=args.forcing_mode,
            memory_initializer=initializer, memory_context=context,
            return_latents=latent_loss_weight > 0,
        )
        prediction, latent_prediction = output if latent_loss_weight > 0 else (output, None)
        targets = {name: x[name][args.warmup_steps + 1:] for name in variables}
        return joint_forecast_loss(
            prediction, targets, area_weights, scales, args.mean_mse_weight,
            latent_predictions=latent_prediction,
            latent_targets=context["latent_targets"] if latent_loss_weight > 0 else None,
            latent_loss_weight=latent_loss_weight, latent_area_weights=latent_area_weights,
        )

    metadata = {
        "implementation": "ngcm_memory.recurrent_latent_rollout.v2" if recurrent else "ngcm_memory.frozen_base_rollout.v1",
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "training_data_sha256": sha256_file(args.data),
        "training_last_observation": str(dataset.time.values[cut - 1]),
        "validation_first_observation": str(dataset.time.values[cut]),
        "validation_last_observation": str(dataset.time.values[int(max(val_origins)) + args.horizon_steps]),
        "observation_interval_hours": hours, "internal_steps": internal_steps,
        "warmup_steps": args.warmup_steps, "horizon_steps": args.horizon_steps,
        "variables": list(variables), "scales": scales,
        "forcing_mode": args.forcing_mode, "ensemble_size": args.ensemble_size,
        "mean_mse_weight": args.mean_mse_weight, "seed": args.seed,
        "probabilistic_training_score": "fair_crps",
        "mean_mse_estimator": "sample_mean_mse_minus_unbiased_sample_variance_over_members",
        "feature_scale_kelvin": args.feature_scale_kelvin,
        "remove_global_mean": args.remove_global_mean,
        "spatial_history_steps": args.spatial_history_steps if history else None,
        "spatial_history": history.metadata if history else None,
        "spatial_history_artifact_sha256": history.artifact_sha256 if history else None,
        "spatial_mode": args.spatial_mode if history else "none",
        "latent_initialization": "observed_exact_origin_and_past_memory" if recurrent else None,
        "latent_loss_weight": latent_loss_weight,
        "latent_supervision": "normalized_frozen_source_at_observation_leads" if latent_loss_weight > 0 else "none",
        "latent_memory_tau_hours": list(config.latent_memory_tau_hours) if recurrent else None,
        "latent_tau_hours": config.latent_tau_hours if recurrent else None,
        "train_window_count": len(train_origins), "validation_window_count": len(val_origins),
        "base_parameters_frozen": True,
    }
    best = float("inf")
    records = []
    order_rng = np.random.default_rng(args.seed)
    step_number = 0
    print(json.dumps({"train_windows": len(train_origins), "validation_windows": len(val_origins),
                      "forecast_hours": hours * args.horizon_steps,
                      "forcing_mode": args.forcing_mode, "spatial_mode": metadata["spatial_mode"],
                      "latent_loss_weight": latent_loss_weight}), flush=True)
    for epoch in range(args.epochs):
        epoch_losses, gradient_norms = [], []
        for origin in order_rng.permutation(train_origins):
            x = slice_window(inputs, int(origin), args.warmup_steps, args.horizon_steps)
            forcing = slice_window(forcings, int(origin), args.warmup_steps, args.horizon_steps)
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 1), step_number)
            keys = jax.random.split(key, args.ensemble_size)
            params, opt_state, loss, norm = train_step(params, opt_state, x, forcing, keys, context_for(int(origin)))
            loss, norm = float(loss), float(norm)
            if not np.isfinite(loss) or not np.isfinite(norm):
                raise FloatingPointError(f"Nonfinite rollout loss/gradient at epoch={epoch}, origin={origin}")
            epoch_losses.append(loss)
            gradient_norms.append(norm)
            step_number += 1
        validation, validation_physical, validation_latent = [], [], []
        for origin in val_origins:
            # Identical validation random seeds across epochs for comparability.
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 2), int(origin))
            loss, physical_loss, latent_loss = validation_loss(
                params, slice_window(inputs, int(origin), args.warmup_steps, args.horizon_steps),
                slice_window(forcings, int(origin), args.warmup_steps, args.horizon_steps),
                jax.random.split(key, args.ensemble_size), context_for(int(origin)),
            )
            validation.append(float(loss))
            validation_physical.append(float(physical_loss))
            validation_latent.append(float(latent_loss))
        record = {"epoch": epoch + 1, "updates": step_number,
                  "train_loss": float(np.mean(epoch_losses)),
                  "validation_loss": float(np.mean(validation)),
                  "validation_physical_loss": float(np.mean(validation_physical)),
                  "validation_latent_loss": float(np.mean(validation_latent)),
                  "mean_gradient_norm": float(np.mean(gradient_norms))}
        if not np.isfinite(record["validation_loss"]):
            raise FloatingPointError("Nonfinite validation forecast")
        records.append(record)
        if record["validation_loss"] < best:
            best = record["validation_loss"]
            save_closure_checkpoint(args.output, params, config, {**metadata, **record})
        print(json.dumps(record), flush=True)
        args.output.with_suffix(".training.json").write_text(json.dumps({"metadata": metadata, "epochs": records}, indent=2))
    print(f"Saved best validation closure to {args.output}", flush=True)


if __name__ == "__main__":
    main()
