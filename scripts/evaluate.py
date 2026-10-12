#!/usr/bin/env python3
"""Long-horizon, probabilistic hindcast evaluation of a trained closure.

Verification dates and warmup begin strictly after training and model-selection
validation observations, requiring an independent later test period.
Observed boundary forcing is an explicitly labeled conditional hindcast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import numpy as np

from ngcm_memory.features import SpatialHistory, thermal_with_empty_spatial
from ngcm_memory.metrics import aggregate_window_metrics, forecast_metrics
from ngcm_memory.neuralgcm_adapter import NeuralGCMMemoryAdapter
from ngcm_memory.training import (
    evaluation_origins, filter_spatial_origins, independent_test_start_index, load_closure_checkpoint, load_local_problem,
    rollout_ensemble, sha256_file, slice_window, validate_time_axis,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--trust-base-checkpoint", action="store_true")
    p.add_argument("--closure", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="JSON scores by variable and lead")
    p.add_argument("--horizon-steps", type=int, default=30)
    p.add_argument("--warmup-steps", type=int, help="Default is the training checkpoint recipe")
    p.add_argument("--ensemble-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-windows", type=int)
    p.add_argument("--start-time", help="Earliest verification partition timestamp; history starts here")
    p.add_argument("--forcing-mode", choices=("persistence", "observed"), help="Default matches training")
    p.add_argument("--baseline", action="store_true", help="Exact unmodified base_model.advance comparison; uses same origins and random keys")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--spatial-history", type=Path)
    p.add_argument("--spatial-history-steps", type=int, help="Default matches training")
    return p


def main(argv: list[str] | None = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    protected = {args.base_checkpoint.resolve(), args.closure.resolve(), args.data.resolve()}
    if args.spatial_history:
        protected.add(args.spatial_history.resolve())
    if args.output.resolve() in protected:
        p.error("Output must not overwrite an input dataset or checkpoint")
    if args.ensemble_size < 2 or (args.max_windows is not None and args.max_windows < 1):
        p.error("ensemble-size >= 2 and max-windows >= 1 required")
    params, config, trained = load_closure_checkpoint(args.closure)
    # Schema, rather than an evaluation switch, selects the training recipe.
    # Older checkpoints omit latent_dim and restore its legacy default of 0.
    recurrent = config.latent_dim > 0
    if sha256_file(args.base_checkpoint) != trained.get("base_checkpoint_sha256"):
        p.error("Base checkpoint differs from the frozen checkpoint used in training")
    base, dataset, inputs, forcings = load_local_problem(
        args.base_checkpoint, args.data, trust_checkpoint=args.trust_base_checkpoint,
    )
    hours, internal_steps = validate_time_axis(dataset.time.values, base.timestep)
    if not np.isclose(hours, trained["observation_interval_hours"]):
        p.error("Evaluation observation interval must match training's memory warmup recipe")
    warmup = args.warmup_steps if args.warmup_steps is not None else trained["warmup_steps"]
    forcing_mode = args.forcing_mode or trained["forcing_mode"]
    train_end = np.datetime64(trained["training_last_observation"], "ns")
    start_index = independent_test_start_index(dataset.time.values, trained, args.start_time)
    origins = evaluation_origins(len(dataset.time), warmup, args.horizon_steps, start_index, args.stride)
    trained_history = trained.get("spatial_history")
    if recurrent and not trained_history:
        p.error("Recurrent checkpoint lacks the source-history initialization metadata")
    if bool(trained_history) != bool(args.spatial_history):
        p.error("Supply --spatial-history exactly when the trained checkpoint used source history")
    history = SpatialHistory.from_npz(args.spatial_history, base) if args.spatial_history else None
    if history:
        for field in ("encoder_sha256",):
            if not history.metadata.get(field) or history.metadata[field] != trained_history.get(field):
                p.error(f"Spatial-history {field} must match the trained source encoder")
        trained_gap = trained_history.get("max_history_gap_hours")
        if trained_gap is not None and not np.isclose(
                history.max_history_gap_hours, trained_gap, rtol=0, atol=1. / 3_600_000_000_000):
            p.error("Spatial-history max_history_gap_hours must match training; prepare evaluation history "
                    "with the same explicit --max-history-gap-hours")
    history_steps = args.spatial_history_steps if args.spatial_history_steps is not None else trained.get("spatial_history_steps")
    if history:
        if history_steps is None or history_steps < 1:
            p.error("spatial-history-steps must be positive")
        origins, history_filter = filter_spatial_origins(
            history, dataset.time.values, origins, history_steps, recurrent=recurrent)
        if history_filter["rejected_windows"]:
            print(json.dumps({"spatial_history_filter": "evaluation", **history_filter}), file=sys.stderr, flush=True)
        if not len(origins):
            p.error(f"No eligible held-out windows with valid spatial history: {history_filter['first_error']}")
    origins = origins[:args.max_windows]
    if not len(origins):
        p.error("No eligible held-out forecast windows")
    features = thermal_with_empty_spatial(base, history.feature_dim, trained["feature_scale_kelvin"]) if history and not recurrent else None
    adapter = NeuralGCMMemoryAdapter(
        base, config, features, feature_scale_kelvin=trained["feature_scale_kelvin"],
        enabled=not args.baseline, remove_global_mean=trained["remove_global_mean"],
    )
    initializer = (history.transition_initializer(config) if recurrent else history.initializer(config)) if history else None
    variables = tuple(trained["variables"])

    @jax.jit
    def predict(x, forcing, keys, context):
        return rollout_ensemble(
            params, adapter, x, forcing, keys, warmup_steps=warmup,
            horizon_steps=args.horizon_steps, internal_steps=internal_steps,
            variables=variables, forcing_mode=forcing_mode,
            memory_initializer=initializer, memory_context=context,
        )

    collected = {name: [] for name in variables}
    for number, origin in enumerate(origins):
        x = slice_window(inputs, int(origin), warmup, args.horizon_steps)
        forcing = slice_window(forcings, int(origin), warmup, args.horizon_steps)
        # Fold timestamp, rather than local index, so separate files can retain
        # matched initialization seeds when comparing the same origin dates.
        stamp = int(dataset.time.values[origin].astype("datetime64[s]").astype(np.int64))
        key = jax.random.fold_in(jax.random.PRNGKey(args.seed), stamp & 0xFFFFFFFF)
        context = ((history.transition_context if recurrent else history.context)(dataset.time.values[origin], history_steps)
                   if history else None)
        predictions = predict(x, forcing, jax.random.split(key, args.ensemble_size), context)
        for name in variables:
            prediction = np.asarray(predictions[name])
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"Nonfinite {name} forecast at {dataset.time.values[origin]}")
            collected[name].append(forecast_metrics(
                prediction, x[name][warmup + 1:], x[name][warmup],
                base.data_coords.horizontal.quadrature_weights,
            ))
        print(json.dumps({"window": number + 1, "of": len(origins), "origin": str(dataset.time.values[origin])}), flush=True)
    report = {
        "evaluation_kind": "held_out_conditional_hindcast" if forcing_mode == "observed" else "held_out_persistent_boundary_forecast",
        "model": "unmodified_neuralgcm" if args.baseline else "neuralgcm_memory_closure",
        "base_checkpoint_sha256": trained["base_checkpoint_sha256"],
        "closure_checkpoint_sha256": sha256_file(args.closure),
        "evaluation_data_sha256": sha256_file(args.data),
        "training_last_observation": str(train_end),
        "validation_last_observation": trained["validation_last_observation"],
        "origins": [str(dataset.time.values[origin]) for origin in origins],
        "ensemble_size": args.ensemble_size, "seed": args.seed,
        "warmup_steps": warmup, "forcing_mode": forcing_mode,
        "lead_hours": (np.arange(1, args.horizon_steps + 1) * hours).tolist(),
        "spatial_history": history.metadata if history else None,
        "spatial_history_steps": history_steps,
        "spatial_mode": ("recurrent" if recurrent else "initialization") if history else "none",
        "future_source_latents_used": False,
        "latent_loss_weight_at_training": trained.get("latent_loss_weight", 0.),
        "score_notes": "crps scores the empirical ensemble; fair_crps is its iid finite-member unbiased counterpart. Equal pressure-level weighting; exact Gaussian-grid spatial quadrature. RMSE/spread pooled across windows. Native variable units.",
        "metrics": {name: aggregate_window_metrics(windows) for name, windows in collected.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved evaluation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
