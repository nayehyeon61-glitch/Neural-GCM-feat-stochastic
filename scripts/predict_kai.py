#!/usr/bin/env python3
"""Free-run a trained KAI ensemble from guide archives, without target labels."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ngcm_memory.kai_training import (
    GuideArchive, discover_archives, load_training_checkpoint, predict_archive,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path, help="One NPZ archive or directory of archives")
    p.add_argument("--output", required=True, type=Path, help="New prediction directory")
    p.add_argument("--device", default="cpu")
    p.add_argument("--baseline", action="store_true", help="Bypass all stage adapters; every ensemble member follows the same backbone")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)
    if args.threads < 1:
        p.error("threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        model, checkpoint = load_training_checkpoint(args.checkpoint, args.device)
        paths = discover_archives(args.data)
        outputs = [args.output / f"{path.stem}.predictions.npz" for path in paths]
        protected = {path.resolve() for path in paths} | {args.checkpoint.resolve()}
        if any(path.resolve() in protected for path in outputs):
            raise ValueError("Prediction outputs must not overwrite input archives or the checkpoint")
        # Validate every origin before writing any predictions.
        for path in paths:
            if GuideArchive.from_file(path).schema() != checkpoint["schema"]:
                raise ValueError(f"Guide schema or identity differs from checkpoint: {path}")
        for path, output in zip(paths, outputs):
            archive = GuideArchive.from_file(path)
            predict_archive(model, archive, schema=checkpoint["schema"], output=output, baseline=args.baseline,
                            validation_before=checkpoint.get("split", {}).get("validation_before"))
            print(output, flush=True)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
