"""Differential tests against the unmodified frozen B9 policy.

Run with unittest discovery. The reference fixture preserves the pre-optimization
policy; no experiment directories or native simulator are required.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from importlib import import_module
import importlib.util
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT.parent / "blue_defense"
REFERENCE_FILE = ROOT / "fixtures" / "b9_reference_5d473a91.py"


@dataclass(frozen=True)
class LegacyInterceptorAssignment:
    """The frozen algorithm constructs the pre-simplification output shape."""

    interceptor_id: int
    target_id: int
    target_pos_ecf: object
    target_vel_ecf: object
    priority: float = 0.0


@dataclass(frozen=True)
class AlgorithmObservation:
    """Prepared policy input, independent of the public own-entity contract."""

    step: int
    targets: object
    interceptors: object
    assets: object
    launched_map: object = field(default_factory=dict)
    # At 1 ms per step, the frozen reference's numeric times equal step counts.
    clock: object = field(default_factory=lambda: BASE.BlueClock(step_duration_ms=1),
                          kw_only=True)


@dataclass(frozen=True)
class FrozenAlgorithmObservation(AlgorithmObservation):
    """Expose the old time name only to the unchanged reference fixture."""

    @property
    def sim_time(self):
        return self.step


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_policies():
    # A private package keeps the frozen reference isolated from runtime classes.
    package = types.ModuleType("_b9_equivalence")
    package.__path__ = [str(CANDIDATE)]
    sys.modules[package.__name__] = package
    # Preserve the frozen reference's original package imports without editing it.
    sys.modules['_b9_equivalence.config'] = import_module('_b9_equivalence.shared.config')
    base = load_module("_b9_equivalence.base", CANDIDATE / "base" / "__init__.py")
    old = load_module("_b9_equivalence.reference", REFERENCE_FILE)
    old.InterceptorAssignment = LegacyInterceptorAssignment

    def legacy_planned_assignment(assignment, due_time, wave_index=0):
        # The frozen algorithm supplies milliseconds; its shared scheduling base
        # now reads steps. This test's explicit 1 ms clock makes them identical.
        return base.PlannedInterceptorAssignment(assignment=assignment,
                                                 due_step=due_time,
                                                 wave_index=wave_index)

    old.PlannedInterceptorAssignment = legacy_planned_assignment
    new = import_module('_b9_equivalence.scheduled.b9_min_cost_planned_multi_wave')
    return base, old.B9MinCostPlannedMultiWavePolicy, new.B9MinCostPlannedMultiWavePolicy


BASE, REFERENCE, OPTIMIZED = load_policies()
V = BASE.Vector3d


def target(identifier, pos, velocity=V(300, 0, 0)):
    return BASE.ThreatTrack(identifier, f"target_{identifier}", 1, pos, velocity, 0)


def interceptor(identifier, pos=V(), available=True):
    return BASE.InterceptorState(identifier, pos, V(), available)


def asset(identifier, pos, value=1.0):
    return BASE.DefendedAsset(identifier, f"asset_{identifier}", pos, 24.0, value, 9400)


@dataclass(frozen=True)
class FrozenCoordinateView:
    """Adapt current records only for the unchanged private reference fixture."""

    record: object

    @property
    def pos_ecf(self):
        return self.record.pos

    @property
    def vel_ecf(self):
        return self.record.vel

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "record"), name)


def reference_observation(observation):
    return FrozenAlgorithmObservation(
        step=observation.step, clock=observation.clock,
        targets=[FrozenCoordinateView(item) for item in observation.targets],
        interceptors=[FrozenCoordinateView(item) for item in observation.interceptors],
        assets=[FrozenCoordinateView(item) for item in observation.assets],
        launched_map=observation.launched_map,
    )


def serialized(items):
    # Removed transport fields do not affect allocation or wave scheduling.
    # Keep order, IDs, due steps and wave numbers in the independent comparison.
    result = []
    for item in items:
        if hasattr(item, "assignment"):
            result.append({
                "assignment": {
                    "interceptor_id": item.assignment.interceptor_id,
                    "target_id": item.assignment.target_id,
                },
                "due_step": item.due_step,
                "wave_index": item.wave_index,
            })
        else:
            result.append({"interceptor_id": item.interceptor_id, "target_id": item.target_id})
    return result


class B9EquivalenceTests(unittest.TestCase):
    def compare_step(self, reference, optimized, observation, context=""):
        before = asdict(observation)
        old_observation = reference_observation(copy.deepcopy(observation))
        old_before = asdict(old_observation)
        new_observation = copy.deepcopy(observation)
        old_released = reference.decide(old_observation)
        new_released = optimized.decide(new_observation)
        self.assertEqual(serialized(old_released), serialized(new_released), context)
        self.assertEqual(serialized(reference._pending), serialized(optimized._pending), context)
        self.assertEqual(reference._last_step, optimized._last_step, context)
        self.assertEqual(old_before, asdict(old_observation), "reference mutated the observation")
        self.assertEqual(before, asdict(new_observation), "optimized mutated the observation")
        return old_released

    def run_steps(self, observations, **kwargs):
        # Compare the algorithms with the same explicit settings, independent
        # of default configuration changes in the current public API.
        kwargs.setdefault("max_shots_per_target", 2)
        reference, optimized = REFERENCE(**kwargs), OPTIMIZED(**kwargs)
        for index, observation in enumerate(observations):
            self.compare_step(reference, optimized, observation, f"step={index}, params={kwargs}")

    def test_empty_inputs_and_assets(self):
        for targets, interceptors, assets in [
            ([], [], []),
            ([target(1, V(1000, 0, 0))], [], []),
            ([], [interceptor(10)], []),
            ([target(1, V(1000, 0, 0))], [interceptor(10), interceptor(11)], []),
        ]:
            with self.subTest(targets=len(targets), interceptors=len(interceptors)):
                self.run_steps([AlgorithmObservation(0, targets, interceptors, assets)], planning_delay=0)

    def test_equal_costs_preserve_observation_order(self):
        # Deliberately non-sorted IDs distinguish encounter order from numeric ID order.
        targets = [target(9, V(1000, 0, 0)), target(2, V(1000, 0, 0))]
        interceptors = [interceptor(i) for i in [19, 3, 17, 1, 12, 10, 4, 5, 6, 7]]
        for shots in [1, 3, 5]:
            for wave_size in [1, 2, 3]:
                with self.subTest(shots=shots, wave_size=wave_size):
                    obs = AlgorithmObservation(0, targets, interceptors, [])
                    self.run_steps([obs, replace(obs, step=5000), replace(obs, step=20000)],
                                   max_shots_per_target=shots, wave_size=wave_size,
                                   planning_delay=5000, wave_gap=5000)

    def test_same_distance_assets_have_different_values(self):
        targets = [target(1, V(), V()), target(2, V(20, 1, 0))]
        interceptors = [interceptor(i, V(i * 120, 0, 0)) for i in range(10, 28)]
        assets = [asset(7, V(-1000, 0, 0), 8), asset(2, V(1000, 0, 0), 1)]
        for asset_order in [assets, assets[::-1]]:
            self.run_steps([AlgorithmObservation(0, targets, interceptors, asset_order)],
                           planning_delay=0, max_shots_per_target=5)

    def test_duplicate_target_and_interceptor_ids(self):
        targets = [target(3, V(500, 0, 0)), target(3, V(900, 50, 0)), target(1, V(100, 0, 0))]
        interceptors = [interceptor(20, V(-100, 0, 0)), interceptor(2, V()),
                        interceptor(20, V(200, 0, 0)), interceptor(5, V(300, 0, 0)),
                        interceptor(9, V(50, 0, 0)), interceptor(11, V(800, 0, 0))]
        obs = AlgorithmObservation(0, targets, interceptors, [asset(1, V(1000, 0, 0), 3)])
        self.run_steps([obs, replace(obs, step=6000), replace(obs, step=11000)],
                       planning_delay=5000, wave_gap=5000, max_shots_per_target=3)

    def test_visibility_loss_reappearance_launch_feedback_losses_and_round_reset(self):
        targets = [target(4, V(1000, 500, 0)), target(2, V(900, -100, 0))]
        interceptors = [interceptor(i, V(i * 150, 0, 0)) for i in range(10, 28)]
        assets = [asset(1, V(2000, 0, 0), 5)]
        obs = AlgorithmObservation(0, targets, interceptors, assets)
        observations = [
            obs,
            replace(obs, step=3000, targets=[]),
            replace(obs, step=5000, targets=[], interceptors=interceptors[1:]),
            replace(obs, step=5500, targets=[replace(targets[0], pos=V(200, 40, 0))],
                    launched_map={4: [10]}, interceptors=interceptors[2:]),
            replace(obs, step=10000, launched_map={4: [10], 2: [11]}, interceptors=interceptors[2:]),
            replace(obs, step=21000, launched_map={4: [10, 12, 14, 16, 18]}),
            replace(obs, step=0),
            replace(obs, step=20000),
        ]
        self.run_steps(observations, planning_delay=5000, wave_gap=5000, max_shots_per_target=5)

    def test_randomized_multistep_scenes(self):
        for seed in range(160):
            rng = random.Random(seed)
            count = rng.randrange(1, 7)
            targets = [target(i + 1, V(rng.uniform(-80000, 80000), rng.uniform(-80000, 80000),
                                     rng.uniform(-1000, 1000)),
                              V(rng.uniform(-400, 400), rng.uniform(-400, 400), 0))
                       for i in range(count)]
            if seed % 7 == 0:
                targets.append(replace(targets[0], pos=V(50, 10, 1)))
            interceptors = [interceptor(100 + i, V(rng.uniform(-60000, 60000),
                                                rng.uniform(-60000, 60000), 0))
                            for i in range(rng.randrange(1, 20))]
            assets = [asset(i, V(rng.uniform(-90000, 90000), rng.uniform(-90000, 90000), 0),
                            rng.choice([1, 3, 8, 12])) for i in range(rng.randrange(9))]
            kwargs = dict(max_shots_per_target=rng.choice([1, 2, 5, 7]),
                          planning_delay=rng.choice([0.0, 1000.0, 5000.0]),
                          wave_gap=rng.choice([0.0, 700.0, 5000.0]),
                          wave_size=rng.choice([1, 2, 3]))
            reference, optimized = REFERENCE(**kwargs), OPTIMIZED(**kwargs)
            launched = {}
            with self.subTest(seed=seed, **kwargs):
                for step, now in enumerate([0, 1000, 4999, 5000, 11000, 16000, 0, 16000]):
                    if step == 6:
                        launched = {}
                    visible = [replace(item, pos=V(item.pos.x + step * 37,
                                                      item.pos.y - step * 23, item.pos.z))
                               for item in targets if rng.random() > 0.22]
                    rng.shuffle(visible)
                    observed_interceptors = [replace(item, available=rng.random() > 0.12)
                                             for item in interceptors]
                    rng.shuffle(observed_interceptors)
                    observation = AlgorithmObservation(now, visible, observed_interceptors, assets,
                                                       copy.deepcopy(launched))
                    released = self.compare_step(reference, optimized, observation, f"seed={seed}, step={step}")
                    for assignment in released:
                        # Some accepted plans receive delayed launch feedback.
                        if rng.random() > 0.2:
                            launched.setdefault(assignment.target_id, []).append(assignment.interceptor_id)

    def test_distance_work_does_not_repeat_for_each_allocation(self):
        observation = synthetic_observation(target_count=6, interceptor_count=48, asset_count=12)
        original_distance = BASE.distance_m

        def decide_and_count(policy_class, shots):
            calls = 0

            def counted_distance(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original_distance(*args, **kwargs)

            policy = policy_class(max_shots_per_target=shots, planning_delay=5000,
                                  wave_gap=5000, wave_size=1)
            policy_module = sys.modules[policy_class.__module__]
            with patch.object(BASE, "distance_m", counted_distance), \
                    patch.object(policy_module, "distance_m", counted_distance):
                input_observation = copy.deepcopy(observation)
                if policy_class is REFERENCE:
                    input_observation = reference_observation(input_observation)
                released = policy.decide(input_observation)
            return calls, (serialized(released), serialized(policy._pending), policy._last_step)

        reference_counts, optimized_counts = [], []
        for shots in [1, 5]:
            with self.subTest(max_shots_per_target=shots):
                reference_count, reference_output = decide_and_count(REFERENCE, shots)
                optimized_count, optimized_output = decide_and_count(OPTIMIZED, shots)
                self.assertEqual(reference_output, optimized_output)
                # Each target performs two asset scans plus one distance to its
                # nearest asset, and one distance for every available interceptor.
                distance_budget = len(observation.targets) * (
                    2 * len(observation.assets) + 1 + len(observation.interceptors)
                )
                self.assertEqual(optimized_count, distance_budget)
                self.assertGreater(reference_count, optimized_count)
                reference_counts.append(reference_count)
                optimized_counts.append(optimized_count)
        # More assigned waves must not multiply the geometric work.
        self.assertEqual(optimized_counts[0], optimized_counts[1])
        self.assertGreater(reference_counts[1], reference_counts[0])

    def test_large_coordinates_and_small_distance_perturbations(self):
        center = 6_371_000.0
        targets = [target(i, V(center + (i % 2) * 1e-8, i * 1e-9, 0), V(300, 0, 0))
                   for i in [8, 2, 9, 1]]
        interceptors = [interceptor(i, V(center + 1000 + (i % 3) * 1e-8, 0, 0))
                        for i in [17, 11, 19, 12, 20, 14, 15, 16]]
        self.run_steps([AlgorithmObservation(0, targets, interceptors,
                                            [asset(1, V(center + 100, 0, 0), 5)])],
                       planning_delay=0, wave_gap=100, max_shots_per_target=3)


def synthetic_observation(target_count, interceptor_count, asset_count):
    rng = random.Random(42)
    targets = [target(i, V(rng.uniform(-50000, 50000), rng.uniform(-50000, 50000), 100),
                      V(300, -20, 0)) for i in range(target_count)]
    interceptors = [interceptor(1000 + i, V(rng.uniform(-60000, 60000), rng.uniform(-60000, 60000), 0))
                    for i in range(interceptor_count)]
    assets = [asset(2000 + i, V(rng.uniform(-50000, 50000), rng.uniform(-50000, 50000), 0), 3)
              for i in range(asset_count)]
    return AlgorithmObservation(0, targets, interceptors, assets)



if __name__ == "__main__":
    unittest.main()
