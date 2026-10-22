"""Trusted built-in backends; importing does not register or load models."""

from .numpy_affine import NumpyAffineBackend

__all__ = ["NumpyAffineBackend"]
