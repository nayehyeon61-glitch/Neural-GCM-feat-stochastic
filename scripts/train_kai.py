#!/usr/bin/env python3
"""Train KAI stage adapters using causal precomputed NeuralGCM ensemble guides."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from ngcm_memory.kai import KAI, StageAdapterConfig, StageManifoldKAI, load_kai_checkpoint
from ngcm_memory.kai_training import (
    discover_archives, split_archives, train_kai, validate_archive_collection,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path, help="Directory of per-origin guide NPZ archives")
    p.add_argument("--train-before", required=True, help="Exclusive UTC timestamp ending KAI training")
    p.add_argument("--validation-before", required=True, help="Exclusive UTC timestamp ending validation")
    p.add_argument("--kai-checkpoint", type=Path, help="Original pretrained KAI state dict (weights-only load)")
    p.add_argument("--kai-step-hours", type=float, default=24., help="Declared native lead of the pretrained KAI checkpoint; must match guide interval")
    p.add_argument("--train-backbone", action="store_true", help="Explicitly allow backbone training; otherwise pretrained weights are required and frozen")
    p.add_argument("--mode", choices=("residual", "filter"), default="residual")
    p.add_argument("--stages", default="0,1,2,3", help="Comma-separated zero-based stage indices")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--max-correction", type=float, default=0.1)
    p.add_argument("--gate-max", type=float, default=1.0)
    p.add_argument("--stage-loss-weight", type=float, default=0.1)
    p.add_argument("--mean-mse-weight", type=float, default=0.1)
    p.add_argument("--loss", choices=("crps", "mse", "relative-l2"), default="crps", help="Default empirical ensemble CRPS + ensemble-mean MSE")
    p.add_argument("--depths", default="3,3,15,3", help="Must match the supplied KAI checkpoint")
    p.add_argument("--dims", default="48,96,192,288", help="Must match the supplied KAI checkpoint")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1, help="CPU torch threads, useful for small-grid smoke tests")
    p.add_argument("--output", required=True, type=Path, help="Best-validation .pt checkpoint")
    return p


def main(argv: list[str] | None = None) -> None:
    p = parser()
    args = p.parse_args(argv)
    if not args.train_backbone and args.kai_checkpoint is None:
        p.error("--kai-checkpoint is required unless --train-backbone is explicitly selected")
    if args.kai_checkpoint and args.output.resolve() == args.kai_checkpoint.resolve():
        p.error("Output must not overwrite the pretrained KAI checkpoint")
    if args.threads < 1:
        p.error("threads must be positive")
    if not np.isfinite(args.kai_step_hours) or args.kai_step_hours <= 0:
        p.error("kai-step-hours must be finite and positive")
    if any(not np.isfinite(value) for value in (args.lr, args.stage_loss_weight, args.mean_mse_weight)):
        p.error("Learning rate and loss weights must be finite")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        depths, dims = tuple(map(int, args.depths.split(","))), tuple(map(int, args.dims.split(",")))
        if len(depths) != 4 or len(dims) != 4 or any(d < 1 for d in depths) or any(d < 8 or d % 8 for d in dims):
            raise ValueError("Four positive depths and four channel widths divisible by 8 are required")
        stages = tuple(int(value) for value in args.stages.split(",") if value.strip())
        paths = discover_archives(args.data)
        schema = validate_archive_collection(paths, require_targets=True)
        if args.kai_checkpoint and not np.isclose(schema["step_hours"], args.kai_step_hours, rtol=0, atol=1e-9):
            raise ValueError("Guide interval differs from the declared pretrained --kai-step-hours; use a checkpoint trained for this lead")
        training, validation, discarded = split_archives(paths, args.train_before, args.validation_before)
        config = StageAdapterConfig(physical_channels=schema["physical_channels"], latent_channels=schema["latent_channels"], stage_indices=stages,
                                    mode=args.mode, hidden_dim=args.hidden_dim, max_correction=args.max_correction, gate_max=args.gate_max)
        if not stages and not args.train_backbone:
            raise ValueError("No adapters selected and backbone frozen: no trainable model")
        backbone_args = {"img_size": (len(schema["latitude"]), len(schema["longitude"])), "in_chans": len(schema["channels"]),
                         "out_chans": len(schema["channels"]), "depths": depths, "dims": dims, "drop_path_rate": 0.0}
        backbone = KAI(**backbone_args)
        if args.kai_checkpoint:
            load_kai_checkpoint(backbone, args.kai_checkpoint)
        model = StageManifoldKAI(backbone, config).freeze_backbone(not args.train_backbone).to(args.device)
        split = {"train_before": args.train_before, "validation_before": args.validation_before,
                 "train_origins": len(training), "validation_origins": len(validation), "discarded_origins": len(discarded),
                 "train_backbone": args.train_backbone}
        if args.kai_checkpoint:
            with args.kai_checkpoint.open("rb") as stream:
                split["pretrained_kai_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            split["pretrained_kai_step_hours"] = args.kai_step_hours
        print(json.dumps(split), flush=True)
        train_kai(model, training, validation, output=args.output, schema=schema, backbone_args=backbone_args,
                  stage_config=config, epochs=args.epochs, lr=args.lr, stage_loss_weight=args.stage_loss_weight,
                  mean_mse_weight=args.mean_mse_weight, loss=args.loss, seed=args.seed, split_metadata=split)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    print(f"Best-validation checkpoint: {args.output}", flush=True)


if __name__ == "__main__":
    main()
