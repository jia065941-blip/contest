"""Pure policy and projected-action math for R11-PAOS.

This module intentionally has no simulator, commander, reward, or target-outcome
dependency.  The caller supplies only legal catalogue ordinals, target types,
distances, public commitment counts, and decision-pool sizes.  Atomic execution
and lifecycle bookkeeping live in :mod:`policies.red.commander`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


KIND_ORDER = ("H", "M", "L")
TARGET_TYPES = (9400, 9500, 9600)
OFFICIAL_WEIGHTS = {9400: 3.0, 9500: 1.0, 9600: 1.0}
THETA_DIMENSION = 8
RESERVE = "reserve"


@dataclass(frozen=True, slots=True)
class PAOSConfig:
    """Immutable preregistered constants for the eight-dimensional allocator."""

    dimension: int = THETA_DIMENSION
    center_bounds: tuple[float, float] = (-2.5, 2.5)
    candidate_bounds: tuple[float, float] = (-4.0, 4.0)
    radius_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    radius_tv_bounds: tuple[float, float] = (0.05, 0.20)
    beta_grid: tuple[float, ...] = (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625)
    update_mean_tv_bounds: tuple[float, float] = (0.01, 0.05)
    update_max_tv: float = 0.10
    rank_tolerance: float = 1e-6
    minimum_rank_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.dimension != THETA_DIMENSION:
            raise ValueError("R11-PAOS policy dimension is immutable at 8")
        _validate_bounds("center_bounds", self.center_bounds)
        _validate_bounds("candidate_bounds", self.candidate_bounds)
        if not (
            self.candidate_bounds[0] <= self.center_bounds[0]
            and self.center_bounds[1] <= self.candidate_bounds[1]
        ):
            raise ValueError("candidate_bounds must contain center_bounds")
        _validate_positive_descending("radius_grid", self.radius_grid, descending=False)
        _validate_positive_descending("beta_grid", self.beta_grid, descending=True)
        _validate_unit_interval("radius_tv_bounds", self.radius_tv_bounds)
        _validate_unit_interval("update_mean_tv_bounds", self.update_mean_tv_bounds)
        if not (0.0 < self.update_max_tv <= 1.0):
            raise ValueError("update_max_tv must lie in (0, 1]")
        if self.rank_tolerance <= 0.0:
            raise ValueError("rank_tolerance must be positive")
        if not (0.0 < self.minimum_rank_ratio <= 1.0):
            raise ValueError("minimum_rank_ratio must lie in (0, 1]")



def _validate_bounds(name: str, bounds: tuple[float, float]) -> None:
    if len(bounds) != 2 or not all(math.isfinite(float(x)) for x in bounds):
        raise ValueError(f"{name} must contain two finite values")
    if float(bounds[0]) >= float(bounds[1]):
        raise ValueError(f"{name} lower bound must be smaller than upper bound")


def _validate_positive_descending(
    name: str, values: tuple[float, ...], *, descending: bool
) -> None:
    if not values or not all(math.isfinite(float(x)) and float(x) > 0.0 for x in values):
        raise ValueError(f"{name} must contain finite positive values")
    ordered = tuple(sorted((float(x) for x in values), reverse=descending))
    if tuple(float(x) for x in values) != ordered or len(set(ordered)) != len(ordered):
        direction = "descending" if descending else "ascending"
        raise ValueError(f"{name} must be unique and {direction}")


def _validate_unit_interval(name: str, bounds: tuple[float, float]) -> None:
    _validate_bounds(name, bounds)
    if bounds[0] < 0.0 or bounds[1] > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _triple(name: str, values: Sequence[int]) -> tuple[int, int, int]:
    if len(values) != len(KIND_ORDER):
        raise ValueError(f"{name} must be ordered exactly as H, M, L")
    result = tuple(int(x) for x in values)
    if any(x < 0 for x in result):
        raise ValueError(f"{name} cannot contain negative counts")
    return result  # type: ignore[return-value]


DEFAULT_CONFIG = PAOSConfig()

@dataclass(frozen=True, slots=True)
class TargetCandidate:
    """A legal target catalogue entry, deliberately without entity ID/outcome."""

    admission_ordinal: int
    entity_type: int
    distance: float
    committed_counts: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        if int(self.admission_ordinal) != self.admission_ordinal or self.admission_ordinal < 0:
            raise ValueError("admission_ordinal must be a non-negative integer")
        if self.entity_type not in TARGET_TYPES:
            raise ValueError(f"unsupported legal target type: {self.entity_type}")
        if not math.isfinite(float(self.distance)) or self.distance < 0.0:
            raise ValueError("distance must be finite and non-negative")
        counts = _triple("committed_counts", self.committed_counts)
        object.__setattr__(self, "committed_counts", counts)


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """Minimal legal state consumed by the pure PAOS allocator."""

    time_fraction: float
    initial_inventory: tuple[int, int, int]
    decision_pool_counts: tuple[int, int, int]
    targets: tuple[TargetCandidate, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time_fraction)) or not 0.0 <= self.time_fraction <= 1.0:
            raise ValueError("time_fraction must lie in [0, 1]")
        initial = _triple("initial_inventory", self.initial_inventory)
        pools = _triple("decision_pool_counts", self.decision_pool_counts)
        if any(pool > total for pool, total in zip(pools, initial)):
            raise ValueError("a decision pool cannot exceed initial inventory")
        ordinals = [item.admission_ordinal for item in self.targets]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("target admission ordinals must be unique")
        for target in self.targets:
            if any(count > initial[i] for i, count in enumerate(target.committed_counts)):
                raise ValueError("committed target count cannot exceed initial inventory")
        object.__setattr__(self, "initial_inventory", initial)
        object.__setattr__(self, "decision_pool_counts", pools)
        object.__setattr__(self, "targets", tuple(self.targets))


@dataclass(frozen=True, slots=True)
class CandidateDistribution:
    """Centered real-target features and the reserve-inclusive distribution."""

    kind: str
    admission_ordinals: tuple[int, ...]
    features: tuple[tuple[float, ...], ...]
    base_logits: tuple[float, ...]
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]

    @property
    def reserve_index(self) -> int:
        return len(self.admission_ordinals)


@dataclass(frozen=True, slots=True)
class KindQuota:
    kind: str
    decision_pool_count: int
    target_counts: tuple[int, ...]
    reserve_count: int

    def __post_init__(self) -> None:
        if self.kind not in KIND_ORDER:
            raise ValueError(f"unsupported platform kind: {self.kind}")
        if self.decision_pool_count < 0 or self.reserve_count < 0:
            raise ValueError("quota counts cannot be negative")
        if any(int(x) != x or x < 0 for x in self.target_counts):
            raise ValueError("target quotas must be non-negative integers")
        if sum(self.target_counts) + self.reserve_count != self.decision_pool_count:
            raise ValueError("target plus reserve quotas must equal decision-pool size")


@dataclass(frozen=True, slots=True)
class ProjectedQuota:
    """Integer quotas aligned to a caller-owned stable catalogue ordering."""

    catalogue_ordinals: tuple[int, ...]
    kinds: tuple[KindQuota, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.catalogue_ordinals)) != self.catalogue_ordinals:
            raise ValueError("catalogue_ordinals must be sorted")
        if len(set(self.catalogue_ordinals)) != len(self.catalogue_ordinals):
            raise ValueError("catalogue_ordinals must be unique")
        if tuple(item.kind for item in self.kinds) != KIND_ORDER:
            raise ValueError("quota kinds must be ordered exactly as H, M, L")
        if any(len(item.target_counts) != len(self.catalogue_ordinals) for item in self.kinds):
            raise ValueError("every kind must include every catalogue target cell")

    def vector(self) -> tuple[int, ...]:
        cells: list[int] = []
        for item in self.kinds:
            cells.extend(int(x) for x in item.target_counts)
            cells.append(int(item.reserve_count))
        return tuple(cells)

    def vector_hash(self) -> str:
        return canonical_vector_hash(self.vector())

    def kind(self, kind: str) -> KindQuota:
        try:
            return self.kinds[KIND_ORDER.index(kind)]
        except ValueError as exc:
            raise ValueError(f"unsupported platform kind: {kind}") from exc

    def target_count(self, kind: str, admission_ordinal: int) -> int:
        try:
            index = self.catalogue_ordinals.index(int(admission_ordinal))
        except ValueError as exc:
            raise KeyError(admission_ordinal) from exc
        return self.kind(kind).target_counts[index]


def _kind_index(kind: str) -> int:
    try:
        return KIND_ORDER.index(kind)
    except ValueError as exc:
        raise ValueError(f"unsupported platform kind: {kind}") from exc


def compatible(kind: str, entity_type: int) -> bool:
    """Repository physical mask: H/M have zero effectiveness against 9500."""

    _kind_index(kind)
    if entity_type not in TARGET_TYPES:
        return False
    return not (kind in {"H", "M"} and entity_type == 9500)


def _theta_tuple(
    theta: Sequence[float],
    *,
    config: PAOSConfig = DEFAULT_CONFIG,
    bounds: str | None = None,
) -> tuple[float, ...]:
    if len(theta) != config.dimension:
        raise ValueError(f"theta must have exactly {config.dimension} entries")
    values = tuple(float(x) for x in theta)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("theta must be finite")
    if bounds is not None:
        if bounds == "center":
            lower, upper = config.center_bounds
        elif bounds == "candidate":
            lower, upper = config.candidate_bounds
        else:
            raise ValueError("bounds must be 'center', 'candidate', or None")
        if any(x < lower or x > upper for x in values):
            raise ValueError(f"theta lies outside {bounds} bounds [{lower}, {upper}]")
    return values


def theta_hash(theta: Sequence[float], *, config: PAOSConfig = DEFAULT_CONFIG) -> str:
    values = _theta_tuple(theta, config=config)
    payload = [format(x, ".17g") for x in values]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def features_for_kind(snapshot: PolicySnapshot, kind: str) -> CandidateDistribution:
    """Build six centered real features and two reserve-only features."""

    kind_index = _kind_index(kind)
    candidates = tuple(
        sorted(
            (item for item in snapshot.targets if compatible(kind, item.entity_type)),
            key=lambda item: item.admission_ordinal,
        )
    )
    max_distance = max((item.distance for item in snapshot.targets), default=0.0)
    distance_scale = max(1.0, float(max_distance))
    initial_count = snapshot.initial_inventory[kind_index]
    raw: list[list[float]] = []
    for item in candidates:
        is_9400 = float(item.entity_type == 9400)
        is_9500 = float(item.entity_type == 9500)
        is_9600 = float(item.entity_type == 9600)
        row = [
            (is_9400 - is_9600) / math.sqrt(2.0) if kind == "H" else 0.0,
            (is_9400 - is_9600) / math.sqrt(2.0) if kind == "M" else 0.0,
            (2.0 * is_9400 - is_9500 - is_9600) / math.sqrt(6.0)
            if kind == "L"
            else 0.0,
            (is_9500 - is_9600) / math.sqrt(2.0) if kind == "L" else 0.0,
            1.0 - float(item.distance) / distance_scale,
            1.0 - float(item.committed_counts[kind_index]) / max(1, initial_count),
        ]
        raw.append(row)
    if raw:
        matrix = np.asarray(raw, dtype=np.float64)
        matrix -= matrix.mean(axis=0, keepdims=True)
        real_features = tuple(tuple(float(x) for x in row) + (0.0, 0.0) for row in matrix)
    else:
        real_features = ()
    reserve_features = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 - snapshot.time_fraction)
    features = real_features + (reserve_features,)
    base_logits = tuple(math.log(OFFICIAL_WEIGHTS[item.entity_type]) for item in candidates) + (0.0,)
    # Logits and probabilities are populated by distribution_for_kind.
    return CandidateDistribution(
        kind=kind,
        admission_ordinals=tuple(item.admission_ordinal for item in candidates),
        features=features,
        base_logits=base_logits,
        logits=(),
        probabilities=(),
    )


def distribution_for_kind(
    snapshot: PolicySnapshot,
    kind: str,
    theta: Sequence[float],
    *,
    config: PAOSConfig = DEFAULT_CONFIG,
) -> CandidateDistribution:
    values = np.asarray(_theta_tuple(theta, config=config, bounds="candidate"), dtype=np.float64)
    partial = features_for_kind(snapshot, kind)
    feature_matrix = np.asarray(partial.features, dtype=np.float64)
    base = np.asarray(partial.base_logits, dtype=np.float64)
    logits = base + feature_matrix @ values
    shifted = logits - float(np.max(logits))
    exponentials = np.exp(shifted)
    probabilities = exponentials / float(exponentials.sum())
    return CandidateDistribution(
        kind=kind,
        admission_ordinals=partial.admission_ordinals,
        features=partial.features,
        base_logits=partial.base_logits,
        logits=tuple(float(x) for x in logits),
        probabilities=tuple(float(x) for x in probabilities),
    )


def _largest_remainder(
    count: int, probabilities: Sequence[float], real_ordinals: Sequence[int]
) -> tuple[int, ...]:
    if count < 0:
        raise ValueError("count cannot be negative")
    if len(probabilities) != len(real_ordinals) + 1:
        raise ValueError("probabilities must include exactly one reserve cell")
    if count == 0:
        return (0,) * len(probabilities)
    scaled = np.asarray(probabilities, dtype=np.float64) * count
    floors = np.floor(scaled).astype(np.int64)
    remaining = int(count - int(floors.sum()))
    if remaining < 0 or remaining >= len(probabilities):
        raise AssertionError("invalid largest-remainder residual")
    fractions = scaled - floors
    # Real cells use stable admission ordinal for exact ties; reserve is last.
    tie_keys = [(0, int(ordinal)) for ordinal in real_ordinals] + [(1, 0)]
    order = sorted(range(len(probabilities)), key=lambda i: (-float(fractions[i]), tie_keys[i]))
    for index in order[:remaining]:
        floors[index] += 1
    result = tuple(int(x) for x in floors)
    if sum(result) != count:
        raise AssertionError("largest-remainder projection violated conservation")
    return result


def project_quotas(
    snapshot: PolicySnapshot,
    theta: Sequence[float],
    config: PAOSConfig = DEFAULT_CONFIG,
) -> ProjectedQuota:
    """Project reserve-inclusive softmax distributions to conserved integers."""

    catalogue = tuple(sorted(item.admission_ordinal for item in snapshot.targets))
    catalogue_index = {ordinal: index for index, ordinal in enumerate(catalogue)}
    quotas: list[KindQuota] = []
    for kind_index, kind in enumerate(KIND_ORDER):
        distribution = distribution_for_kind(snapshot, kind, theta, config=config)
        count = snapshot.decision_pool_counts[kind_index]
        local = _largest_remainder(count, distribution.probabilities, distribution.admission_ordinals)
        full_counts = [0] * len(catalogue)
        for ordinal, quota in zip(distribution.admission_ordinals, local[:-1]):
            full_counts[catalogue_index[ordinal]] = int(quota)
        quotas.append(
            KindQuota(
                kind=kind,
                decision_pool_count=count,
                target_counts=tuple(full_counts),
                reserve_count=int(local[-1]),
            )
        )
    return ProjectedQuota(catalogue_ordinals=catalogue, kinds=tuple(quotas))


def hadamard_directions() -> np.ndarray:
    """Return the fixed orthonormal Sylvester H8 rows."""

    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < THETA_DIMENSION:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    matrix = matrix / math.sqrt(THETA_DIMENSION)
    matrix.setflags(write=False)
    return matrix


def quota_tv(first: ProjectedQuota, second: ProjectedQuota) -> float:
    """Mean per-nonempty-kind total variation between two quota vectors."""

    _assert_aligned(first, second)
    values: list[float] = []
    for left, right in zip(first.kinds, second.kinds):
        count = left.decision_pool_count
        if count == 0:
            continue
        difference = sum(abs(a - b) for a, b in zip(left.target_counts, right.target_counts))
        difference += abs(left.reserve_count - right.reserve_count)
        values.append(float(difference) / (2.0 * count))
    return float(np.mean(values)) if values else 0.0


def _assert_aligned(first: ProjectedQuota, second: ProjectedQuota) -> None:
    if first.catalogue_ordinals != second.catalogue_ordinals:
        raise ValueError("quota projections use different catalogue ordinals")
    for left, right in zip(first.kinds, second.kinds):
        if left.kind != right.kind or left.decision_pool_count != right.decision_pool_count:
            raise ValueError("quota projections use different decision pools")


@dataclass(frozen=True, slots=True)
class PreviewChord:
    episode_key: str
    event_class: str
    snapshot_key: str
    plus: ProjectedQuota
    minus: ProjectedQuota

    def __post_init__(self) -> None:
        if not self.episode_key or not self.event_class or not self.snapshot_key:
            raise ValueError("preview chord keys cannot be empty")
        _assert_aligned(self.plus, self.minus)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.episode_key, self.event_class, self.snapshot_key


def _hierarchical_snapshot_weights(keys: Sequence[tuple[str, str, str]]) -> dict[tuple[str, str, str], float]:
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("snapshot keys must be non-empty and unique")
    episodes = sorted({episode for episode, _, _ in keys})
    result: dict[tuple[str, str, str], float] = {}
    for episode in episodes:
        episode_keys = [key for key in keys if key[0] == episode]
        classes = sorted({event_class for _, event_class, _ in episode_keys})
        for event_class in classes:
            class_keys = [key for key in episode_keys if key[1] == event_class]
            weight = 1.0 / len(episodes) / len(classes) / len(class_keys)
            for key in class_keys:
                result[key] = weight
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-12):
        raise AssertionError("hierarchical snapshot weights do not sum to one")
    return result


def weighted_displacement_matrix(
    direction_chords: Sequence[Sequence[PreviewChord]],
) -> np.ndarray:
    """Create the normalized executable-chord matrix used only for diagnosis."""

    if not direction_chords:
        raise ValueError("at least one direction is required")
    ordered: list[list[PreviewChord]] = []
    reference_keys: tuple[tuple[str, str, str], ...] | None = None
    for chords in direction_chords:
        current = sorted(chords, key=lambda item: item.key)
        keys = tuple(item.key for item in current)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("every direction must be cross-previewed on the same snapshot bank")
        ordered.append(current)
    assert reference_keys is not None
    weights = _hierarchical_snapshot_weights(reference_keys)
    columns: list[np.ndarray] = []
    for chords in ordered:
        cells: list[float] = []
        for chord in chords:
            snapshot_weight = weights[chord.key]
            nonempty = [
                index for index, item in enumerate(chord.plus.kinds) if item.decision_pool_count > 0
            ]
            if not nonempty:
                continue
            kind_weight = snapshot_weight / len(nonempty)
            for index in nonempty:
                plus = chord.plus.kinds[index]
                minus = chord.minus.kinds[index]
                denominator = 2.0 * plus.decision_pool_count
                vector = tuple(a - b for a, b in zip(plus.target_counts, minus.target_counts))
                vector += (plus.reserve_count - minus.reserve_count,)
                scale = math.sqrt(kind_weight) / denominator
                cells.extend(scale * value for value in vector)
        column = np.asarray(cells, dtype=np.float64)
        norm = float(np.linalg.norm(column))
        if norm > 0.0:
            column = column / norm
        columns.append(column)
    row_counts = {column.shape[0] for column in columns}
    if len(row_counts) != 1 or not row_counts or next(iter(row_counts)) == 0:
        raise ValueError("snapshot bank must yield an aligned non-empty displacement vector")
    return np.column_stack(columns)


@dataclass(frozen=True, slots=True)
class ActionGeometry:
    rank: int
    singular_values: tuple[float, ...]
    rank_ratio: float
    passes: bool
    matrix: np.ndarray = field(repr=False, compare=False)


def action_geometry(
    direction_chords: Sequence[Sequence[PreviewChord]],
    *,
    config: PAOSConfig = DEFAULT_CONFIG,
) -> ActionGeometry:
    matrix = weighted_displacement_matrix(direction_chords)
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix, tol=config.rank_tolerance))
    ratio = 0.0
    if singular.size and float(singular[0]) > 0.0:
        ratio = float(singular[-1] / singular[0])
    return ActionGeometry(
        rank=rank,
        singular_values=tuple(float(x) for x in singular),
        rank_ratio=ratio,
        passes=rank == config.dimension and ratio >= config.minimum_rank_ratio,
        matrix=matrix,
    )


@dataclass(frozen=True, slots=True)
class SecantResult:
    return_differences: tuple[float, ...]
    rms_return_difference: float
    secants: tuple[float, ...]
    gradient: tuple[float, ...]
    gradient_norm: float


def paired_secants(
    returns_plus: Sequence[float],
    returns_minus: Sequence[float],
    radii: Sequence[float],
    directions: np.ndarray | None = None,
    *,
    config: PAOSConfig = DEFAULT_CONFIG,
) -> SecantResult:
    """Compute the preregistered full-basis return secants, without scaling."""

    expected = config.dimension
    if not (len(returns_plus) == len(returns_minus) == len(radii) == expected):
        raise ValueError("paired secants require exactly eight complete antithetic pairs")
    plus = np.asarray(returns_plus, dtype=np.float64)
    minus = np.asarray(returns_minus, dtype=np.float64)
    radius = np.asarray(radii, dtype=np.float64)
    if not np.isfinite(plus).all() or not np.isfinite(minus).all():
        raise ValueError("official returns must be finite")
    if not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise ValueError("radii must be finite and positive")
    basis = hadamard_directions() if directions is None else np.asarray(directions, dtype=np.float64)
    if basis.shape != (expected, expected) or not np.isfinite(basis).all():
        raise ValueError("directions must be a finite 8x8 matrix")
    gram = basis @ basis.T
    if not np.allclose(gram, np.eye(expected), atol=1e-12, rtol=0.0):
        raise ValueError("directions must form an orthonormal basis")
    differences = plus - minus
    secants = differences / (2.0 * radius)
    gradient = np.sum(secants[:, None] * basis, axis=0)
    return SecantResult(
        return_differences=tuple(float(x) for x in differences),
        rms_return_difference=float(np.sqrt(np.mean(np.square(differences)))),
        secants=tuple(float(x) for x in secants),
        gradient=tuple(float(x) for x in gradient),
        gradient_norm=float(np.linalg.norm(gradient)),
    )


Projector = Callable[[PolicySnapshot, Sequence[float]], ProjectedQuota]


@dataclass(frozen=True, slots=True)
class TrustSnapshot:
    episode_key: str
    event_class: str
    snapshot_key: str
    state: PolicySnapshot

    @property
    def key(self) -> tuple[str, str, str]:
        return self.episode_key, self.event_class, self.snapshot_key


def _bank_tv(
    first_theta: Sequence[float],
    second_theta: Sequence[float],
    bank: Sequence[TrustSnapshot],
    projector: Projector,
) -> tuple[float, float]:
    if not bank:
        raise ValueError("trust bank cannot be empty")
    keys = [item.key for item in bank]
    weights = _hierarchical_snapshot_weights(keys)
    values: list[tuple[float, float]] = []
    for item in bank:
        tv = quota_tv(projector(item.state, first_theta), projector(item.state, second_theta))
        values.append((weights[item.key], tv))
    return sum(weight * tv for weight, tv in values), max(tv for _, tv in values)


@dataclass(frozen=True, slots=True)
class RadiusSelection:
    radius: float
    probe_tv: float
    bank_mean_tv: float | None


def select_preview_radius(
    center: Sequence[float],
    direction: Sequence[float],
    probe_state: PolicySnapshot,
    *,
    bank: Sequence[TrustSnapshot] = (),
    projector: Projector | None = None,
    config: PAOSConfig = DEFAULT_CONFIG,
) -> RadiusSelection | None:
    """Select the smallest feasible fixed-grid antithetic action radius."""

    center_values = np.asarray(_theta_tuple(center, config=config, bounds="center"))
    vector = np.asarray(_theta_tuple(direction, config=config), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise ValueError("radius direction must be unit length")
    project = projector or (lambda state, theta: project_quotas(state, theta, config))
    low, high = config.radius_tv_bounds
    for radius in config.radius_grid:
        plus = center_values + radius * vector
        minus = center_values - radius * vector
        try:
            plus_tuple = _theta_tuple(plus, config=config, bounds="candidate")
            minus_tuple = _theta_tuple(minus, config=config, bounds="candidate")
        except ValueError:
            continue
        probe_tv = quota_tv(project(probe_state, plus_tuple), project(probe_state, minus_tuple))
        if not low <= probe_tv <= high:
            continue
        bank_mean: float | None = None
        if bank:
            bank_mean, _ = _bank_tv(plus_tuple, minus_tuple, bank, project)
            if not low <= bank_mean <= high:
                continue
        return RadiusSelection(float(radius), float(probe_tv), bank_mean)
    return None


@dataclass(frozen=True, slots=True)
class TrustedUpdate:
    beta: float
    theta: tuple[float, ...]
    mean_tv: float
    max_tv: float


def select_trusted_beta(
    center: Sequence[float],
    direction: Sequence[float],
    bank: Sequence[TrustSnapshot],
    *,
    projector: Projector | None = None,
    config: PAOSConfig = DEFAULT_CONFIG,
) -> TrustedUpdate | None:
    """Choose the largest preregistered center-box-feasible quota-trusted step."""

    center_tuple = _theta_tuple(center, config=config, bounds="center")
    center_values = np.asarray(center_tuple, dtype=np.float64)
    vector = np.asarray(_theta_tuple(direction, config=config), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("update direction must be finite and nonzero")
    vector /= norm
    project = projector or (lambda state, theta: project_quotas(state, theta, config))
    low, high = config.update_mean_tv_bounds
    for beta in config.beta_grid:
        candidate = center_values + beta * vector
        try:
            theta = _theta_tuple(candidate, config=config, bounds="center")
        except ValueError:
            continue
        mean_tv, max_tv = _bank_tv(center_tuple, theta, bank, project)
        if low <= mean_tv <= high and max_tv <= config.update_max_tv:
            return TrustedUpdate(float(beta), theta, float(mean_tv), float(max_tv))
    return None


@dataclass(frozen=True, slots=True)
class PlanLatch:
    name: str
    value: bool


@dataclass(frozen=True, slots=True)
class LegalTargetState:
    admission_ordinal: int
    entity_type: int
    position: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class LegalPlatformState:
    platform_id: int
    kind: str
    position: tuple[float, float, float]
    lifecycle: str
    plan_id: int | None = None
    target_ordinal: int | None = None
    due_step: int | None = None
    leader: bool = False
    reason: str | None = None
    version: int = 0


@dataclass(frozen=True, slots=True)
class LegalStateSnapshot:
    """Allowlisted public/controller state for fresh-process identity checks."""

    step: int
    time_seconds: float
    targets: tuple[LegalTargetState, ...]
    platforms: tuple[LegalPlatformState, ...]
    plan_latches: tuple[PlanLatch, ...] = ()


def _quantized_float(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("canonical state cannot contain NaN or infinity")
    return format(number, ".9f")


def _position_payload(position: Sequence[float]) -> list[str]:
    if len(position) != 3:
        raise ValueError("positions must contain lon, lat, alt")
    return [_quantized_float(value) for value in position]


def canonical_legal_state_payload(snapshot: LegalStateSnapshot) -> dict[str, Any]:
    targets = []
    ordinals: set[int] = set()
    for item in sorted(snapshot.targets, key=lambda value: value.admission_ordinal):
        if item.admission_ordinal in ordinals or item.admission_ordinal < 0:
            raise ValueError("legal target ordinals must be unique and non-negative")
        if item.entity_type not in TARGET_TYPES:
            raise ValueError("legal-state hash contains unsupported target type")
        ordinals.add(item.admission_ordinal)
        targets.append(
            {
                "admission_ordinal": int(item.admission_ordinal),
                "entity_type": int(item.entity_type),
                "position": _position_payload(item.position),
            }
        )
    platforms = []
    platform_ids: set[int] = set()
    allowed_lifecycle = {"free", "active_pending", "issued", "unavailable_unissued"}
    for item in sorted(snapshot.platforms, key=lambda value: value.platform_id):
        if item.platform_id in platform_ids:
            raise ValueError("platform IDs must be unique")
        if item.kind not in KIND_ORDER or item.lifecycle not in allowed_lifecycle:
            raise ValueError("invalid public platform kind/lifecycle")
        if item.target_ordinal is not None and item.target_ordinal not in ordinals:
            raise ValueError("platform target ordinal is absent from legal catalogue")
        platform_ids.add(item.platform_id)
        platforms.append(
            {
                "due_step": item.due_step,
                "kind": item.kind,
                "leader": bool(item.leader),
                "lifecycle": item.lifecycle,
                "plan_id": item.plan_id,
                "platform_id": int(item.platform_id),
                "position": _position_payload(item.position),
                "reason": item.reason,
                "target_ordinal": item.target_ordinal,
                "version": int(item.version),
            }
        )
    latch_names = [item.name for item in snapshot.plan_latches]
    if len(latch_names) != len(set(latch_names)):
        raise ValueError("plan latch names must be unique")
    return {
        "plan_latches": {
            item.name: bool(item.value) for item in sorted(snapshot.plan_latches, key=lambda x: x.name)
        },
        "platforms": platforms,
        "step": int(snapshot.step),
        "targets": targets,
        "time_seconds": _quantized_float(snapshot.time_seconds),
    }


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_legal_state_json(snapshot: LegalStateSnapshot) -> str:
    return _canonical_json_bytes(canonical_legal_state_payload(snapshot)).decode("utf-8")


def canonical_legal_state_hash(snapshot: LegalStateSnapshot) -> str:
    return hashlib.sha256(_canonical_json_bytes(canonical_legal_state_payload(snapshot))).hexdigest()


def canonical_vector_hash(vector: Sequence[int]) -> str:
    values: list[int] = []
    for item in vector:
        value = int(item)
        if value != item or value < 0:
            raise ValueError("preview/seal vector must contain non-negative integers")
        values.append(value)
    return hashlib.sha256(_canonical_json_bytes(values)).hexdigest()


CHECKPOINT_SCHEMA = "r11_paos_theta_v1"
_FORBIDDEN_METADATA_FRAGMENTS = ("health", "destroy", "case_info", "outcome")


@dataclass(frozen=True, slots=True)
class ThetaCheckpoint:
    theta: tuple[float, ...]
    theta_sha256: str
    metadata: Mapping[str, Any]


def _check_metadata_boundary(value: Any, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            normalized = key.lower().replace("-", "_")
            if any(fragment in normalized for fragment in _FORBIDDEN_METADATA_FRAGMENTS):
                raise ValueError(f"forbidden hidden-outcome metadata key: {path}.{key}")
            _check_metadata_boundary(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_metadata_boundary(child, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{path} contains a non-JSON value")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains NaN or infinity")


def _config_checkpoint_payload(config: PAOSConfig) -> dict[str, Any]:
    return {
        "beta_grid": list(config.beta_grid),
        "candidate_bounds": list(config.candidate_bounds),
        "center_bounds": list(config.center_bounds),
        "radius_grid": list(config.radius_grid),
    }


def save_theta_checkpoint(
    path: str | os.PathLike[str],
    theta: Sequence[float],
    *,
    metadata: Mapping[str, Any] | None = None,
    config: PAOSConfig = DEFAULT_CONFIG,
    bounds: str = "center",
) -> ThetaCheckpoint:
    values = _theta_tuple(theta, config=config, bounds=bounds)
    clean_metadata = dict(metadata or {})
    _check_metadata_boundary(clean_metadata)
    digest = theta_hash(values, config=config)
    payload = {
        "config": _config_checkpoint_payload(config),
        "dimension": config.dimension,
        "metadata": clean_metadata,
        "method": "R11-PAOS",
        "schema_version": CHECKPOINT_SCHEMA,
        "theta": list(values),
        "theta_sha256": digest,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return ThetaCheckpoint(values, digest, clean_metadata)


def load_theta_checkpoint(
    path: str | os.PathLike[str],
    *,
    config: PAOSConfig = DEFAULT_CONFIG,
    bounds: str = "center",
) -> ThetaCheckpoint:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("PAOS checkpoint root must be an object")
    expected_keys = {
        "config",
        "dimension",
        "metadata",
        "method",
        "schema_version",
        "theta",
        "theta_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("PAOS checkpoint has an unexpected schema")
    if payload["schema_version"] != CHECKPOINT_SCHEMA or payload["method"] != "R11-PAOS":
        raise ValueError("unsupported PAOS checkpoint schema/method")
    if payload["dimension"] != config.dimension:
        raise ValueError("PAOS checkpoint dimension mismatch")
    if payload["config"] != _config_checkpoint_payload(config):
        raise ValueError("PAOS checkpoint frozen config mismatch")
    values = _theta_tuple(payload["theta"], config=config, bounds=bounds)
    digest = theta_hash(values, config=config)
    if payload["theta_sha256"] != digest:
        raise ValueError("PAOS checkpoint theta digest mismatch")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("PAOS checkpoint metadata must be an object")
    _check_metadata_boundary(metadata)
    return ThetaCheckpoint(values, digest, dict(metadata))


__all__ = [
    "ActionGeometry",
    "CandidateDistribution",
    "CHECKPOINT_SCHEMA",
    "DEFAULT_CONFIG",
    "KIND_ORDER",
    "KindQuota",
    "LegalPlatformState",
    "LegalStateSnapshot",
    "LegalTargetState",
    "OFFICIAL_WEIGHTS",
    "PAOSConfig",
    "PlanLatch",
    "PolicySnapshot",
    "PreviewChord",
    "ProjectedQuota",
    "RadiusSelection",
    "SecantResult",
    "TARGET_TYPES",
    "THETA_DIMENSION",
    "TargetCandidate",
    "ThetaCheckpoint",
    "TrustSnapshot",
    "TrustedUpdate",
    "action_geometry",
    "canonical_legal_state_hash",
    "canonical_legal_state_json",
    "canonical_legal_state_payload",
    "canonical_vector_hash",
    "compatible",
    "distribution_for_kind",
    "features_for_kind",
    "hadamard_directions",
    "load_theta_checkpoint",
    "paired_secants",
    "project_quotas",
    "quota_tv",
    "save_theta_checkpoint",
    "select_preview_radius",
    "select_trusted_beta",
    "theta_hash",
    "weighted_displacement_matrix",
]

def theta_sha256(theta: Sequence[float], *, config: PAOSConfig = DEFAULT_CONFIG) -> str:
    """Explicit alias used by probe/rollout identity records."""

    return theta_hash(theta, config=config)


def policy_snapshot_payload(snapshot: PolicySnapshot) -> dict[str, Any]:
    """Return the strict, outcome-free JSON schema used for frozen banks."""

    return {
        "decision_pool_counts": list(snapshot.decision_pool_counts),
        "initial_inventory": list(snapshot.initial_inventory),
        "targets": [
            {
                "admission_ordinal": item.admission_ordinal,
                "committed_counts": list(item.committed_counts),
                "distance": _quantized_float(item.distance),
                "entity_type": item.entity_type,
            }
            for item in sorted(snapshot.targets, key=lambda value: value.admission_ordinal)
        ],
        "time_fraction": _quantized_float(snapshot.time_fraction),
    }


def policy_snapshot_from_payload(payload: Mapping[str, Any]) -> PolicySnapshot:
    """Reconstruct a bank snapshot while rejecting extra/hidden fields."""

    required = {"decision_pool_counts", "initial_inventory", "targets", "time_fraction"}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("policy snapshot has an unexpected schema")
    raw_targets = payload["targets"]
    if not isinstance(raw_targets, list):
        raise ValueError("policy snapshot targets must be a list")
    target_keys = {"admission_ordinal", "committed_counts", "distance", "entity_type"}
    targets: list[TargetCandidate] = []
    for raw in raw_targets:
        if not isinstance(raw, Mapping) or set(raw) != target_keys:
            raise ValueError("policy snapshot target has an unexpected schema")
        targets.append(
            TargetCandidate(
                admission_ordinal=int(raw["admission_ordinal"]),
                entity_type=int(raw["entity_type"]),
                distance=float(raw["distance"]),
                committed_counts=tuple(raw["committed_counts"]),
            )
        )
    return PolicySnapshot(
        time_fraction=float(payload["time_fraction"]),
        initial_inventory=tuple(payload["initial_inventory"]),
        decision_pool_counts=tuple(payload["decision_pool_counts"]),
        targets=tuple(targets),
    )


def policy_snapshot_json(snapshot: PolicySnapshot) -> str:
    return _canonical_json_bytes(policy_snapshot_payload(snapshot)).decode("utf-8")


def policy_snapshot_from_json(serialized: str) -> PolicySnapshot:
    payload = json.loads(serialized)
    if not isinstance(payload, dict):
        raise ValueError("policy snapshot JSON root must be an object")
    return policy_snapshot_from_payload(payload)


__all__ += [
    "policy_snapshot_from_json",
    "policy_snapshot_from_payload",
    "policy_snapshot_json",
    "policy_snapshot_payload",
    "theta_sha256",
]
