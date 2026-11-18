"""Blue file entry points, independent instances and the engine boundary."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import Mock, patch

from envengine.common.BaseStruct import Vector3d
from blue_defense import (
    BLUE_POLICY_CHOICES, BaseBlueDefense, BlueClock, BlueDefenseConfig, BlueDefenseController,
    BlueDecision, BlueObservation, OwnEntityState, InterceptorAssignment, InterceptorState,
    ScheduledInterceptorAssignment, ThreatTrack
)
from blue_defense.blue_checker import check_blue_policy
from scenarios import load_scenario


ROOT = Path(__file__).resolve().parents[1]
E01 = ROOT / 'scenarios/E01.json'
CUSTOM = """
from blue_defense import BaseBlueDefense, BlueDecision

class BlueDefense(BaseBlueDefense):
    name = 'custom'

    def __init__(self, *, seed=None, asset_values=None):
        super().__init__(seed=seed, asset_values=asset_values)
        self.observations = []

    def decide(self, observation):
        self.observations.append(observation)
        return BlueDecision()
"""


def builtin_policy_path(name):
    number = int(name.split('_', 1)[0][1:])
    family = 'immediate' if number < 4 else 'coordination' if number < 7 else 'scheduled'
    return ROOT / 'blue_defense' / family / (name + '.py')


class BluePluginTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write_policy(self, source=CUSTOM, name='policy.py'):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding='utf-8')
        return path

    def test_check_loads_once_without_constructing_or_deciding(self):
        path = self.write_policy(r"""
            from pathlib import Path
            from blue_defense import BaseBlueDefense
            marker = Path(__file__).with_suffix('.loads')
            marker.write_text(marker.read_text() + 'load\n' if marker.exists() else 'load\n')

            class BlueDefense(BaseBlueDefense):
                name = 'load-only'

                def __init__(self, *, seed=None, asset_values=None):
                    raise AssertionError('checker must not construct')

                def decide(self, observation):
                    raise AssertionError('checker must not decide')
        """)
        first = check_blue_policy(path)
        self.assertIs(check_blue_policy(path), first)
        self.assertTrue(issubclass(first, BaseBlueDefense))
        self.assertEqual(path.with_suffix('.loads').read_text(), 'load\n')

    def test_checked_class_creates_independent_instances_and_copies_asset_values(self):
        implementation = check_blue_policy(self.write_policy())
        values = {201: 8.0}
        first = implementation(seed=11, asset_values=values)
        second = implementation(seed=11, asset_values=values)
        values[201] = 99
        first.decide('first frame')
        self.assertIsNot(first, second)
        self.assertEqual(first.observations, ['first frame'])
        self.assertEqual(second.observations, [])
        self.assertEqual(first.seed, 11)
        self.assertEqual(first.asset_value(201, 9400), 8.0)
        first.asset_values[201] = 1
        self.assertEqual(second.asset_value(201, 9400), 8.0)
        first.reset()
        first.close()
        second.close()

    def test_invalid_export_inheritance_abstract_signature_and_metadata_are_rejected(self):
        sources = {
            'missing': 'value = 1',
            'instance': CUSTOM + '\nBlueDefense = BlueDefense()',
            'unrelated': 'class BlueDefense:\n    pass\n',
            'abstract': "from blue_defense import BaseBlueDefense\nclass BlueDefense(BaseBlueDefense):\n    name = 'abstract'\n",
            'async': CUSTOM.replace('def decide(self, observation):', 'async def decide(self, observation):'),
            'decide_signature': CUSTOM.replace('def decide(self, observation):', 'def decide(self):'),
            'generator': CUSTOM.replace('return BlueDecision()', 'yield BlueDecision()'),
            'static_method': CUSTOM.replace('    def decide(', '    @staticmethod\n    def decide('),
            'constructor': CUSTOM.replace('*, seed=None, asset_values=None', 'required_argument'),
            'empty_name': CUSTOM.replace("name = 'custom'", "name = ''"),
            'non_bool_mode': CUSTOM.replace("name = 'custom'", "name = 'custom'\n    plans_launch_time = 1"),
            'non_bool_preservation': CUSTOM.replace("name = 'custom'", "name = 'custom'\n    preserves_pending_launches = 'yes'"),
        }
        for name, source in sources.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                check_blue_policy(self.write_policy(source, name + '.py'))

    def test_export_selects_explicit_entry_even_when_another_subclass_is_defined(self):
        implementation = check_blue_policy(self.write_policy(CUSTOM + """

class AnotherDefense(BlueDefense):
    name = 'helper'
"""))
        self.assertEqual(implementation.name, 'custom')

    def test_builtin_package_relative_imports_preserve_public_base_identity(self):
        implementation = check_blue_policy(builtin_policy_path('b5_successive_depth_coordination'))
        self.assertTrue(issubclass(implementation, BaseBlueDefense))
        self.assertEqual(implementation.__module__, 'blue_defense.coordination.b5_successive_depth_coordination')
        policy = implementation(seed=7, asset_values={201: 2.0})
        self.assertEqual(policy.name, 'b5_successive_depth_coordination')
        policy.close()

    def test_same_filename_in_different_directories_does_not_share_module_or_class(self):
        first = check_blue_policy(self.write_policy(CUSTOM, 'first/policy.py'))
        second = check_blue_policy(self.write_policy(CUSTOM.replace("'custom'", "'second'"), 'second/policy.py'))
        self.assertIsNot(first, second)
        self.assertNotEqual(first.__module__, second.__module__)
        self.assertEqual((first.name, second.name), ('custom', 'second'))

    def test_failed_module_execution_is_removed_and_can_be_retried(self):
        path = self.write_policy("raise RuntimeError('broken plugin import')\n")
        with self.assertRaisesRegex(ValueError, 'broken plugin import'):
            check_blue_policy(path)
        self.assertFalse(any(getattr(module, '__file__', None) == str(path)
                             for module in tuple(sys.modules.values()) if module is not None))
        self.write_policy()
        self.assertTrue(issubclass(check_blue_policy(path), BaseBlueDefense))

    def test_missing_or_non_python_file_is_rejected(self):
        for path in (self.directory / 'missing.py', self.write_policy(CUSTOM, 'policy.txt')):
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_blue_policy(path)


def observations():
    zero = Vector3d()
    initial = BlueObservation(
        step=0,
        targets=(
            ThreatTrack(101, 'threat-a', 21000, Vector3d(10_000, 0, 0), Vector3d(-300, 0, 0), 0, source_id=301),
            ThreatTrack(102, 'threat-b', 21001, Vector3d(0, 20_000, 0), Vector3d(0, -200, 0), 0, source_id=301),
        ),
        interceptors=tuple(InterceptorState(identifier, zero, zero, True)
                           for identifier in range(1, 7)),
        own_entities=(OwnEntityState(201, 'asset', 9400, zero, zero, 100.0),),
        clock=BlueClock(step_duration_ms=1000),
    )
    yield replace(initial, targets=())
    for step in (0, 0, 1, 5, 9, 10, 20, 50, 100, 200):
        yield replace(initial, step=step)
    yield replace(initial, step=201, launched_map={101: (1, 2)})
    yield replace(initial, step=601, targets=())


class BuiltinBluePluginTests(unittest.TestCase):
    def test_invalid_decision_batches_preserve_pending_plans_and_execute_no_prefix(self):
        from envengine.simulator.simlulator_impl.DefendCommanderModelSimulator import DefendCommanderModelSimulator

        valid = InterceptorAssignment(1, 101)
        second = replace(valid, interceptor_id=2, target_id=102)
        scheduled = ScheduledInterceptorAssignment(
            interceptor_id=2, target_id=102, decision_step=0,
            planned_launch_step=1, estimated_intercept_step=2,
        )
        invalid_batches = {
            'wrong_result_type': [valid],
            'wrong_assignment_type': BlueDecision((valid, object())),
            'bool_interceptor': BlueDecision((valid, replace(second, interceptor_id=True))),
            'bool_target': BlueDecision((valid, replace(second, target_id=False))),
            'float_interceptor': BlueDecision((valid, replace(second, interceptor_id=2.5))),
            'string_target': BlueDecision((valid, replace(second, target_id='102'))),
            'duplicate_interceptor': BlueDecision((valid, replace(second, interceptor_id=valid.interceptor_id))),
            'mixed_immediate_and_scheduled': BlueDecision((valid, scheduled)),
            'nan_decision_step': BlueDecision((scheduled, replace(scheduled, interceptor_id=3, decision_step=float('nan')))),
            'inf_launch_step': BlueDecision((scheduled, replace(scheduled, interceptor_id=3, planned_launch_step=float('inf')))),
            'nan_intercept_step': BlueDecision((scheduled, replace(scheduled, interceptor_id=3, estimated_intercept_step=float('nan')))),
        }
        # Bad batches begin with a valid action and must fail before any prefix
        # executes. Scheduled actions must also fail when planning is disabled.
        for plans_launch_time in (False, True):
            for name, decision in invalid_batches.items():
                with self.subTest(planning=plans_launch_time, invalid=name):
                    commander = object.__new__(DefendCommanderModelSimulator)
                    commander.defense_controller = Mock(
                        plans_launch_time=plans_launch_time, preserves_pending_launches=True,
                    )
                    commander.defense_controller.decide.return_value = decision
                    pending = {99: replace(scheduled, interceptor_id=99)}
                    commander.pending_launches = pending
                    commander.launched_list = {100: [98]}
                    for method in ('_build_blue_observation', '_execute_immediate_assignments',
                                   '_execute_due_launches', '_send_commands', '_record_solve_stats'):
                        setattr(commander, method, Mock())
                    with self.assertRaises((TypeError, ValueError)):
                        commander._decide_from_information()
                    commander.defense_controller.decide.assert_called_once_with(
                        commander._build_blue_observation.return_value,
                    )
                    self.assertIs(commander.pending_launches, pending)
                    self.assertEqual(pending, {99: replace(scheduled, interceptor_id=99)})
                    self.assertEqual(commander.launched_list, {100: [98]})
                    for method in ('_execute_immediate_assignments', '_execute_due_launches',
                                   '_send_commands', '_record_solve_stats'):
                        getattr(commander, method).assert_not_called()

    def test_all_file_entries_preserve_legacy_decisions_timing_and_reset_behavior(self):
        # This checks the file adapters and controller defaults; the frozen B9
        # algorithm differential suite remains the independent algorithm oracle.
        for name in BLUE_POLICY_CHOICES:
            with self.subTest(policy=name):
                implementation = check_blue_policy(builtin_policy_path(name))
                current = implementation(seed=17, asset_values={201: 8.0})
                legacy = BlueDefenseController(BlueDefenseConfig(policy_name=name, seed=17,
                                                                asset_values={201: 8.0}))
                self.addCleanup(current.close)
                self.assertEqual(current.name, name)
                self.assertIs(current.plans_launch_time, legacy.plans_launch_time)
                self.assertIs(current.preserves_pending_launches, legacy.preserves_pending_launches)
                self.assertEqual(current.asset_value(201, 9400), 8.0)
                for episode in range(2):
                    with patch('time.perf_counter', return_value=0):
                        for observation in observations():
                            expected = legacy.decide(deepcopy(observation))
                            actual = current.decide(deepcopy(observation))
                            self.assertEqual(actual, expected, (name, episode, observation.step))
                    current.reset()
                    legacy.reset()

    def test_real_environment_registers_resets_and_calls_checked_blue_policy(self):
        from envengine.environment.training_env import TrainingEnv
        from envengine.sdk.base_struct.Basic import DetectInfo
        from envengine.common.BaseStruct import Vector3d

        scene = load_scenario(E01)
        env = TrainingEnv(scene, record_logs=False)
        self.addCleanup(env.close)
        implementation = check_blue_policy(builtin_policy_path('b0_fixed_ratio_random'))
        policy = implementation(seed=42, asset_values=scene.asset_values)
        self.addCleanup(policy.close)
        env.register_blue_policy(policy)
        state = env.reset()
        self.assertEqual(state['step'], 0)
        factory = env.engine.simulator_factory
        commander = factory.get_simulators_by_type(35000)[0]
        self.assertIs(commander.defense_controller, policy)
        radar = factory.get_simulators_by_type(44000)[0].entity_ext.entity
        target = factory.get_simulators_by_type(21000)[0].entity_ext.entity
        record = DetectInfo(detect_from=radar.id, step=env.engine.current_step,
                            entity_id=target.id, entity_type=target.entityType,
                            nameChn=target.nameChn, pos=deepcopy(target.posEcf), vel=Vector3d(-300, 0, 0))
        commander.handel_detect_info({target.id: record})
        with patch.object(policy, 'decide', wraps=policy.decide) as decide:
            commander.update()
        observation = decide.call_args.args[0]
        self.assertEqual(observation.step, env.engine.current_step)
        self.assertEqual(observation.clock.start_time_ms, scene.profile.imagineProfile.simTime)
        self.assertEqual(observation.clock.step_duration_ms, scene.profile.imagineProfile.simStep)
        self.assertFalse(hasattr(observation, 'sim_time'))
        self.assertEqual(observation.clock.time_at(observation.step), env.engine.sim_time)
        self.assertEqual(tuple(track.target_id for track in observation.targets), (target.id,))
        self.assertTrue(observation.interceptors)
        self.assertIs(commander.defense_controller, policy)
        env.reset()
        self.assertEqual(policy.get_tracks(env.engine.simulator_factory.current_step), ())
        self.assertEqual(commander.pending_launches, {})


if __name__ == '__main__':
    unittest.main()
