#!/usr/bin/env python3
"""Export a verified Climate Manifold spatial checkpoint to Torch-free JAX NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ngcm_memory.spatial import export_payload, file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-pickle", action="store_true",
                        help="Allow legacy pickle loading ONLY for your own trusted checkpoint; may execute code")
    args = parser.parse_args(argv)
    import torch  # Optional offline dependency; never needed in JAX rollout.

    manifest = args.checkpoint.with_suffix(".manifest.json")
    actual_hash = file_sha256(args.checkpoint)
    if not manifest.exists() or json.loads(manifest.read_text())["checkpoint_sha256"] != actual_hash:
        raise ValueError("Source checkpoint manifest/hash mismatch; retain the source .manifest.json")
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=not args.trusted_pickle)
    except Exception as exc:
        if not args.trusted_pickle:
            raise ValueError("Safe checkpoint loading failed. Legacy NumPy metadata may require --trusted-pickle "
                             "for a checkpoint you created and trust.") from exc
        raise
    contract = export_payload(payload, args.output, checkpoint_sha256=actual_hash)
    print(json.dumps({"output": str(args.output), "sha256": file_sha256(args.output),
                      "surface_fields": contract["surface_fields"],
                      "information_fields": contract["information_fields"],
                      "latent_grid": contract["latent_grid"],
                      "checkpoint_source_commit": contract["checkpoint_source_commit"]}, indent=2))


if __name__ == "__main__":
    main()
