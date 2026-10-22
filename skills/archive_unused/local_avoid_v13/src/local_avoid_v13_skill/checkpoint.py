"""Restricted reader for the one verified historical V13 checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from numpy._core.multiarray import _reconstruct, scalar


EXPECTED_CHECKPOINT_SHA256 = "632662d47fb35edfdb10d680a359cced03fab75af09cff666631a97f0ccf87f3"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_checkpoint(path: str | Path) -> dict:
    checkpoint = Path(path).resolve(strict=True)
    actual = file_sha256(checkpoint)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected V13 checkpoint SHA-256: {actual}")
    safe = [_reconstruct, scalar, np.ndarray, np.dtype,
            type(np.dtype(np.float64)), type(np.dtype(np.float32))]
    with torch.serialization.safe_globals(safe):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("algorithm") != "local_avoid_asymmetric_ppo_v1":
        raise ValueError("Unexpected V13 checkpoint algorithm")
    return payload
