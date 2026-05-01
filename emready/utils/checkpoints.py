"""Checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path

import torch


STATE_KEYS = ("ema", "model", "state_dict")


def extract_state_dict(checkpoint):
    """Return a model state dict from clean weights or a training checkpoint."""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dict")
    for key in STATE_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def clean_state_dict(state_dict: dict) -> dict:
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_state_dict_file(path: str | Path, map_location="cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    return clean_state_dict(extract_state_dict(checkpoint))


def save_clean_checkpoint(src: str | Path, dst: str | Path) -> None:
    state_dict = load_state_dict_file(src, map_location="cpu")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, dst)

