"""Regression for YAML scalar metadata in original KAI-Atmos checkpoints."""

import pickle

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ruamel.yaml")
from ruamel.yaml import YAML  # noqa: E402
from ruamel.yaml.scalarfloat import ScalarFloat  # noqa: E402

from ngcm_memory.kai import load_kai_checkpoint  # noqa: E402
from ngcm_memory.kai_backbone import KAI  # noqa: E402


class UnsupportedMetadata:
    """An inert class that must not be added to the safe unpickler allowlist."""


def _backbone():
    return KAI(in_chans=3, out_chans=3, depths=(0, 0, 0, 0), dims=(8, 8, 8, 8))


def test_original_yaml_adamw_checkpoint_loads_without_unsafe_pickle(tmp_path):
    # Match KAI's round-trip YAML parser and AdamW configuration, rather than
    # constructing a metadata dictionary that bypasses the actual scalar type.
    config = YAML().load("lr: 1.0e-3\n")
    assert isinstance(config["lr"], ScalarFloat)
    source = _backbone()
    optimizer = torch.optim.AdamW(source.parameters(), lr=config["lr"])
    checkpoint = tmp_path / "original_kai.pt"
    torch.save({
        "iters": 100,
        "epoch": 5,
        "model_state": {"module." + k: v for k, v in source.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint)
    # Establish that this fixture reproduces the original compatibility issue.
    with pytest.raises(pickle.UnpicklingError, match="ScalarFloat"):
        torch.load(checkpoint, weights_only=True)

    destination = _backbone()
    prior_allowlist = torch.serialization.get_safe_globals().copy()
    result = load_kai_checkpoint(destination, checkpoint)
    assert not result.missing_keys and not result.unexpected_keys
    for name, actual in destination.state_dict().items():
        torch.testing.assert_close(actual, source.state_dict()[name], rtol=0, atol=0)
    assert torch.serialization.get_safe_globals() == prior_allowlist


def test_yaml_compatibility_does_not_allow_unrelated_pickle_classes(tmp_path):
    backbone = _backbone()
    checkpoint = tmp_path / "unsupported_metadata.pt"
    torch.save({
        "model_state": backbone.state_dict(),
        "metadata": UnsupportedMetadata(),
    }, checkpoint)
    prior_allowlist = torch.serialization.get_safe_globals().copy()
    with pytest.raises(pickle.UnpicklingError, match="UnsupportedMetadata"):
        load_kai_checkpoint(backbone, checkpoint)
    # Both successful and failed loads must leave global deserialization policy
    # unchanged after the temporary compatibility context exits.
    assert torch.serialization.get_safe_globals() == prior_allowlist
