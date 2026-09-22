"""Pure search-space math for the frozen R12-SC4 outer policy.

R12 deliberately keeps the already-audited R11 runtime controller in eight
dimensions.  This module owns only the four-dimensional search coordinates,
their immutable embedding into the R11 ``theta`` vector, deterministic
experiment scheduling, and relation-checked checkpoints.  It has no simulator,
reward, target-outcome, or commander dependency.

The JSON checkpoint written here is intentionally an R11 theta transport
checkpoint with a strict R12 metadata envelope.  Consequently the unchanged
R11 commander can load it, while the outer runner must use
``load_sc4_checkpoint`` before trusting its ``phi`` identity.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import paos_policy as paos


RUNTIME_DIMENSION = 8
SEARCH_DIMENSION = 4
SC4_RADIUS = 1.0
SC4_METHOD = "R12-SC4"
SC4_METADATA_SCHEMA = "r12_sc4_metadata_v1"
SC4_SEED_NAMESPACE = "R12-SC4-v1"
SC4_EXPECTED_BASIS_SHA256 = (
    "c983a7471691bc3a2c4c7a42631a22946013bd8ae3ee2e6ce30f004fa6d98aec"
)

# Keep this exact evaluation path.  In particular, do not replace it with a
# second decimal/literal source: the L contrast is consumed by an integer
# largest-remainder projector, where a tiny tie-breaking residual can matter.
_SQRT_6 = math.sqrt(6.0)
_SQRT_2 = math.sqrt(2.0)
SC4_C_STAR = 0.5 * (3.0 / _SQRT_6) / (1.0 / _SQRT_2)


def _finite_vector(name: str, values: Sequence[float], dimension: int) -> tuple[float, ...]:
    if len(values) != dimension:
        raise ValueError(f"{name} must have exactly {dimension} entries")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be finite")
    return result


def _little_f8_sha256(values: Sequence[float] | np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8", order="C")
    if not np.isfinite(array).all():
        raise ValueError("digest input must be finite")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def build_sc4_basis() -> np.ndarray:
    """Return the immutable 8x4 semantic embedding frozen by R12.

    Columns are H 9400/9600, M 9400/9600, the pure L
    ``(-1, 2, -1)/sqrt(6)`` target-type contrast, and time-dependent reserve
    pressure.  ``SC4_C_STAR`` combines the existing R11 e2/e3 features without
    changing the runtime feature map.
    """

    basis = np.zeros((RUNTIME_DIMENSION, SEARCH_DIMENSION), dtype=np.float64)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    basis[2, 2] = -0.5
    basis[3, 2] = SC4_C_STAR
    basis[7, 3] = 1.0
    if not np.allclose(basis.T @ basis, np.eye(SEARCH_DIMENSION), atol=1e-12, rtol=0.0):
        raise AssertionError("frozen SC4 basis is not orthonormal")
    actual = basis_sha256(basis)
    if actual != SC4_EXPECTED_BASIS_SHA256:
        raise AssertionError(
            f"frozen SC4 basis digest changed: {actual} != {SC4_EXPECTED_BASIS_SHA256}"
        )
    basis.setflags(write=False)
    return basis


def basis_sha256(basis: Sequence[Sequence[float]] | np.ndarray) -> str:
    """Hash C-contiguous little-endian float64 basis bytes."""

    array = np.asarray(basis, dtype=np.float64)
    if array.shape != (RUNTIME_DIMENSION, SEARCH_DIMENSION):
        raise ValueError("SC4 basis must have shape (8, 4)")
    return _little_f8_sha256(array)


SC4_BASIS = build_sc4_basis()
SC4_BASIS_SHA256 = basis_sha256(SC4_BASIS)


def phi_sha256(phi: Sequence[float]) -> str:
    """Hash a four-dimensional search vector as little-endian float64 bytes."""

    return _little_f8_sha256(_finite_vector("phi", phi, SEARCH_DIMENSION))


def phi_to_theta(phi: Sequence[float]) -> tuple[float, ...]:
    """Embed a four-dimensional SC4 vector into the frozen R11 theta space."""

    values = np.asarray(_finite_vector("phi", phi, SEARCH_DIMENSION), dtype=np.float64)
    theta = SC4_BASIS @ values
    return tuple(float(value) for value in theta)


def theta_to_phi(theta: Sequence[float], *, atol: float = 1e-12) -> tuple[float, ...]:
    """Recover phi and reject theta values outside the frozen SC4 plane."""

    values = np.asarray(_finite_vector("theta", theta, RUNTIME_DIMENSION), dtype=np.float64)
    phi = SC4_BASIS.T @ values
    reconstructed = SC4_BASIS @ phi
    if not np.allclose(values, reconstructed, atol=atol, rtol=0.0):
        raise ValueError("theta does not lie in the frozen SC4 subspace")
    return tuple(float(value) for value in phi)


@dataclass(frozen=True, slots=True)
class SC4SeedTuple:
    blue: int
    red: int
    simulation: int

    def to_dict(self) -> dict[str, int]:
        return {"blue": self.blue, "red": self.red, "simulation": self.simulation}


def cycle_seed_tuple(cycle: int) -> SC4SeedTuple:
    """Derive the one CRN seed tuple shared by all four axes in a cycle.

    The first four SHA-256 bytes are interpreted as an *unsigned big-endian*
    integer.  Only the two preregistered training cycles are valid.
    """

    if type(cycle) is not int or cycle not in (0, 1):
        raise ValueError("SC4 cycle must be exactly 0 or 1")
    payload = f"{SC4_SEED_NAMESPACE}|cycle={cycle}".encode("utf-8")
    prefix = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)
    simulation = 44_000_000 + prefix % 100_000
    return SC4SeedTuple(
        blue=simulation + 2_000_000,
        red=simulation + 1_000_000,
        simulation=simulation,
    )


def sign_order(axis: int) -> tuple[str, str]:
    """Return the frozen alternating antithetic branch order for an axis."""

    if type(axis) is not int or not 0 <= axis < SEARCH_DIMENSION:
        raise ValueError("SC4 axis must be an integer in [0, 3]")
    return ("plus", "minus") if axis % 2 == 0 else ("minus", "plus")


@dataclass(frozen=True, slots=True)
class SC4AxisCandidate:
    axis: int
    sign: str
    radius: float
    phi: tuple[float, ...]
    theta: tuple[float, ...]
    phi_sha256: str
    theta_sha256: str


def axis_candidate(phi: Sequence[float], axis: int, sign: str) -> SC4AxisCandidate:
    """Create one fixed-radius candidate around an immutable cycle center."""

    center = np.asarray(_finite_vector("phi", phi, SEARCH_DIMENSION), dtype=np.float64)
    if type(axis) is not int or not 0 <= axis < SEARCH_DIMENSION:
        raise ValueError("SC4 axis must be an integer in [0, 3]")
    if sign not in {"plus", "minus"}:
        raise ValueError("SC4 sign must be 'plus' or 'minus'")
    multiplier = 1.0 if sign == "plus" else -1.0
    candidate_phi_array = center.copy()
    candidate_phi_array[axis] += multiplier * SC4_RADIUS
    candidate_phi = tuple(float(value) for value in candidate_phi_array)
    theta = phi_to_theta(candidate_phi)
    return SC4AxisCandidate(
        axis=axis,
        sign=sign,
        radius=SC4_RADIUS,
        phi=candidate_phi,
        theta=theta,
        phi_sha256=phi_sha256(candidate_phi),
        theta_sha256=paos.theta_sha256(theta),
    )


@dataclass(frozen=True, slots=True)
class SC4Center:
    """An immutable cycle-local search center."""

    cycle: int
    phi: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.cycle) is not int or not 0 <= self.cycle <= 2:
            raise ValueError("SC4 center cycle must lie in [0, 2]")
        object.__setattr__(self, "phi", _finite_vector("phi", self.phi, SEARCH_DIMENSION))

    @property
    def theta(self) -> tuple[float, ...]:
        return phi_to_theta(self.phi)

    def candidate(self, axis: int, sign: str) -> SC4AxisCandidate:
        if self.cycle >= 2:
            raise ValueError("the frozen post-q16 center has no training candidates")
        return axis_candidate(self.phi, axis, sign)


@dataclass(frozen=True, slots=True)
class SC4ReturnUpdate:
    return_differences: tuple[float, ...]
    rms_return_difference: float
    secants: tuple[float, ...]
    gradient_phi: tuple[float, ...]
    gradient_norm: float
    unit_phi: tuple[float, ...] | None
    unit_theta: tuple[float, ...] | None


def paired_return_update(
    returns_plus: Sequence[float], returns_minus: Sequence[float]
) -> SC4ReturnUpdate:
    """Compute the complete four-pair fixed-r coordinate-search update."""

    plus = np.asarray(
        _finite_vector("returns_plus", returns_plus, SEARCH_DIMENSION), dtype=np.float64
    )
    minus = np.asarray(
        _finite_vector("returns_minus", returns_minus, SEARCH_DIMENSION), dtype=np.float64
    )
    differences = plus - minus
    secants = differences / (2.0 * SC4_RADIUS)
    # Search directions are the four canonical phi axes, so the secant vector
    # itself is g_phi.
    gradient = secants.copy()
    norm = float(np.linalg.norm(gradient))
    rms = float(np.sqrt(np.mean(np.square(differences))))
    unit_phi: tuple[float, ...] | None = None
    unit_theta: tuple[float, ...] | None = None
    if norm > 0.0 and math.isfinite(norm):
        normalized = gradient / norm
        unit_phi = tuple(float(value) for value in normalized)
        unit_theta = phi_to_theta(unit_phi)
    return SC4ReturnUpdate(
        return_differences=tuple(float(value) for value in differences),
        rms_return_difference=rms,
        secants=tuple(float(value) for value in secants),
        gradient_phi=tuple(float(value) for value in gradient),
        gradient_norm=norm,
        unit_phi=unit_phi,
        unit_theta=unit_theta,
    )


def advance_center(center: SC4Center, update: SC4ReturnUpdate, beta: float) -> SC4Center:
    """Apply a trusted beta in phi and preserve the exact theta=U@phi relation."""

    if center.cycle not in (0, 1):
        raise ValueError("only training-cycle centers can be advanced")
    if update.unit_phi is None or update.unit_theta is None:
        raise ValueError("cannot advance with a zero/non-finite search direction")
    step = float(beta)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("beta must be finite and positive")
    phi = tuple(
        float(value + step * direction)
        for value, direction in zip(center.phi, update.unit_phi)
    )
    theta = np.asarray(phi_to_theta(phi), dtype=np.float64)
    expected = np.asarray(center.theta, dtype=np.float64) + step * np.asarray(
        update.unit_theta, dtype=np.float64
    )
    if not np.allclose(theta, expected, atol=1e-12, rtol=0.0):
        raise AssertionError("SC4 phi update and theta update diverged")
    return SC4Center(cycle=center.cycle + 1, phi=phi)


@dataclass(frozen=True, slots=True)
class SC4Checkpoint:
    phi: tuple[float, ...]
    theta: tuple[float, ...]
    phi_sha256: str
    theta_sha256: str
    basis_sha256: str
    context: Mapping[str, Any]


def _sc4_metadata(phi: tuple[float, ...], context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sc4": {
            "basis_sha256": SC4_BASIS_SHA256,
            "phi": list(phi),
            "phi_sha256": phi_sha256(phi),
            "radius": SC4_RADIUS,
            "runtime_dimension": RUNTIME_DIMENSION,
            "schema_version": SC4_METADATA_SCHEMA,
            "search_dimension": SEARCH_DIMENSION,
        },
        "context": dict(context),
    }


def save_sc4_checkpoint(
    path: str | Path,
    phi: Sequence[float],
    *,
    context: Mapping[str, Any] | None = None,
    config: paos.PAOSConfig = paos.DEFAULT_CONFIG,
    bounds: str = "center",
) -> SC4Checkpoint:
    """Write an R11-loadable theta file carrying strict R12 identity metadata."""

    if config.dimension != RUNTIME_DIMENSION:
        raise ValueError("SC4 requires the unchanged eight-dimensional PAOS runtime")
    phi_values = _finite_vector("phi", phi, SEARCH_DIMENSION)
    theta = phi_to_theta(phi_values)
    metadata = _sc4_metadata(phi_values, context or {})
    transport = paos.save_theta_checkpoint(
        path, theta, metadata=metadata, config=config, bounds=bounds
    )
    return SC4Checkpoint(
        phi=phi_values,
        theta=transport.theta,
        phi_sha256=phi_sha256(phi_values),
        theta_sha256=transport.theta_sha256,
        basis_sha256=SC4_BASIS_SHA256,
        context=dict(context or {}),
    )


def load_sc4_checkpoint(
    path: str | Path,
    *,
    config: paos.PAOSConfig = paos.DEFAULT_CONFIG,
    bounds: str = "center",
    relation_atol: float = 1e-12,
) -> SC4Checkpoint:
    """Load a transport checkpoint and verify every SC4 relation and digest."""

    if config.dimension != RUNTIME_DIMENSION:
        raise ValueError("SC4 requires the unchanged eight-dimensional PAOS runtime")
    transport = paos.load_theta_checkpoint(path, config=config, bounds=bounds)
    metadata = transport.metadata
    if set(metadata) != {"sc4", "context"}:
        raise ValueError("SC4 checkpoint metadata has an unexpected schema")
    sc4 = metadata["sc4"]
    expected_keys = {
        "basis_sha256",
        "phi",
        "phi_sha256",
        "radius",
        "runtime_dimension",
        "schema_version",
        "search_dimension",
    }
    if not isinstance(sc4, Mapping) or set(sc4) != expected_keys:
        raise ValueError("SC4 checkpoint identity metadata has an unexpected schema")
    if (
        sc4["schema_version"] != SC4_METADATA_SCHEMA
        or int(sc4["runtime_dimension"]) != RUNTIME_DIMENSION
        or int(sc4["search_dimension"]) != SEARCH_DIMENSION
        or float(sc4["radius"]) != SC4_RADIUS
        or sc4["basis_sha256"] != SC4_BASIS_SHA256
    ):
        raise ValueError("SC4 checkpoint frozen identity mismatch")
    phi = _finite_vector("phi", sc4["phi"], SEARCH_DIMENSION)
    digest = phi_sha256(phi)
    if sc4["phi_sha256"] != digest:
        raise ValueError("SC4 checkpoint phi digest mismatch")
    expected_theta = np.asarray(phi_to_theta(phi), dtype=np.float64)
    actual_theta = np.asarray(transport.theta, dtype=np.float64)
    if not np.allclose(actual_theta, expected_theta, atol=relation_atol, rtol=0.0):
        raise ValueError("SC4 checkpoint violates theta = U @ phi")
    context = metadata["context"]
    if not isinstance(context, Mapping):
        raise ValueError("SC4 checkpoint context must be an object")
    return SC4Checkpoint(
        phi=phi,
        theta=transport.theta,
        phi_sha256=digest,
        theta_sha256=transport.theta_sha256,
        basis_sha256=SC4_BASIS_SHA256,
        context=dict(context),
    )


def sealed_quota_projection(
    snapshot: paos.PolicySnapshot,
    theta: Sequence[float],
    *,
    config: paos.PAOSConfig = paos.DEFAULT_CONFIG,
) -> paos.ProjectedQuota:
    """Return the quota vector that the audited R11 plan builder seals.

    R11's plan builder assigns exactly the projected target counts and reserve
    counts; it adds platform identities, due slots, and leaders but does not
    alter these counts.  Contract tests compare this pure representation with
    the real commander ``preview_seal`` and mutating ``seal`` outputs.
    """

    return paos.project_quotas(snapshot, theta, config=config)


def sealed_quota_vector(
    snapshot: paos.PolicySnapshot,
    theta: Sequence[float],
    *,
    config: paos.PAOSConfig = paos.DEFAULT_CONFIG,
) -> tuple[int, ...]:
    return tuple(int(value) for value in sealed_quota_projection(snapshot, theta, config=config).vector())


def axis_quota_pair(
    snapshot: paos.PolicySnapshot,
    center_phi: Sequence[float],
    axis: int,
    *,
    config: paos.PAOSConfig = paos.DEFAULT_CONFIG,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    plus = axis_candidate(center_phi, axis, "plus")
    minus = axis_candidate(center_phi, axis, "minus")
    return (
        sealed_quota_vector(snapshot, plus.theta, config=config),
        sealed_quota_vector(snapshot, minus.theta, config=config),
    )


__all__ = [
    "RUNTIME_DIMENSION",
    "SC4AxisCandidate",
    "SC4_BASIS",
    "SC4_BASIS_SHA256",
    "SC4_C_STAR",
    "SC4Center",
    "SC4Checkpoint",
    "SC4_EXPECTED_BASIS_SHA256",
    "SC4_METHOD",
    "SC4_METADATA_SCHEMA",
    "SC4_RADIUS",
    "SC4ReturnUpdate",
    "SC4SeedTuple",
    "SEARCH_DIMENSION",
    "advance_center",
    "axis_candidate",
    "axis_quota_pair",
    "basis_sha256",
    "build_sc4_basis",
    "cycle_seed_tuple",
    "load_sc4_checkpoint",
    "paired_return_update",
    "phi_sha256",
    "phi_to_theta",
    "save_sc4_checkpoint",
    "sealed_quota_projection",
    "sealed_quota_vector",
    "sign_order",
    "theta_to_phi",
]
