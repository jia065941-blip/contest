"""Standalone replay parsing uses the public metre coordinates and step clock."""
from copy import deepcopy
import json
import gzip
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/replay_video'))
from replay_tool.log_input import load, save_compact

ORIGIN = 1783391450000
AREA = {'xMin': -60000, 'xMax': 60000, 'yMin': -45000, 'yMax': 45000}
RING = [[-45000, -35000], [-20000, -35000], [-20000, 35000], [-45000, 35000]]


def entity(identifier, kind, side, x, y, hp=24):
    return {'id': identifier, 'type': kind, 'side': side, 'nameChn': str(identifier),
            'pos': {'x': x, 'y': y, 'z': 100}, 'vel': {'x': 100, 'y': 0, 'z': 0},
            'health': hp, 'isVisible': True, 'stage': 3, 'detectInfo': {}}


def state(step, *, clock=True):
    row = {'data_type': 'state', 'step': step, 'map_area': deepcopy(AREA),
           'deployment_area': {'coordinates': [deepcopy(RING)]},
           'entities': {
               '1': entity(1, 9400, 1, 37000, 10000),
               '2': entity(2, 9500, 1, 40000, -5000, 1.2),
               '101': entity(101, 21000, 0, -25000 + step*5, -30000),
               '301': entity(301, 44000, 1, 36000, 5000),
           }}
    if clock:
        row['sim_time_ms'] = ORIGIN + step*50
    return row


def discovery(step):
    return {'data_type': 'event', 'type': 21001, 'logicTime': ORIGIN + 500,
            'content': {'event': {'sensor': {'id': 101, 'side': 0, 'type': 21000},
                                  'step': 10, 'detectTarget': [{
                                      'entityInfo': {'id': 2, 'type': 9500, 'side': 1},
                                      'step': step, 'pos': {'x': 42000, 'y': -5100, 'z': 100},
                                      'vel': {'x': 0, 'y': 0, 'z': 0},
                                  }]}}}


def satellite_state(step):
    row = state(step)
    row['entities']['401'] = entity(401, 9202, 0, 0, 0)
    return row


def satellite_discovery(step):
    row = discovery(step)
    row['content']['event']['sensor'] = {'id': 401, 'side': 0, 'type': 9202}
    row['content']['event']['detectTarget'][0]['radarId'] = 401
    return row


class ReplayLogInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'run.json'

    def write(self, rows):
        self.path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)+'\n', encoding='utf-8')
        return self.path

    def test_native_metre_positions_and_nested_deployment_are_preserved(self):
        meta, times, states, index, _, _ = load(self.write([state(0), state(10)]))
        self.assertEqual(meta['map_area'], AREA)
        self.assertEqual(meta['red_area'], [RING])
        self.assertEqual(meta['coordinate_system'], 'local_xyz_m')
        self.assertEqual(meta['step_seconds'], .05)
        np.testing.assert_equal(times, [0, .5])
        np.testing.assert_equal(states[0, index[101], :3], [-25000, -30000, 100])

    def test_first_discovery_uses_recorded_step_and_keeps_observed_position(self):
        final = state(10)
        final['entities']['101']['detectInfo']['2'] = {'pos': {'x': 40000, 'y': -5000, 'z': 100}, 'detected_step': 10}
        meta, _, _, _, events, _ = load(self.write([state(0), discovery(9), discovery(7), final]))
        self.assertEqual(meta['audit']['native_discovery_events'], 2)
        event = next(row for row in events if row['kind'] == 'dark_discovered')
        self.assertAlmostEqual(event['t'], .35)
        self.assertEqual(event['position'], [42000, -5100])
        self.assertEqual(event['precision'], 'native_event')

    def test_discovery_step_does_not_need_an_absolute_clock(self):
        rows = [{'data_type': 'config', 'simStep': 50}, state(0, clock=False), discovery(7), state(10, clock=False)]
        data = load(self.write(rows))
        self.assertAlmostEqual(data[4][0]['t'], .35)
        self.assertEqual(data[0]['step_seconds'], .05)

    def test_satellite_native_detection_precedes_sampled_confirmation(self):
        final = satellite_state(10)
        final['entities']['401']['detectInfo']['2'] = {'pos': {'x': 40000, 'y': -5000, 'z': 100}}
        data = load(self.write([satellite_state(0), satellite_discovery(9), satellite_discovery(7), final]))
        event = next(row for row in data[4] if row['kind'] == 'satellite')
        self.assertAlmostEqual(event['t'], .35)
        self.assertEqual(event['confirmation'], 'native_satellite_detection_records')
        self.assertEqual(event['precision'], 'native_event')
        self.assertIsNone(event['duration'])
        self.assertEqual(data[0]['audit']['satellite_time_basis'],
                         'first_successful_detection_not_exact_activation')

    def test_satellite_request_and_empty_detection_do_not_confirm_success(self):
        request = {'data_type': 'aiAction', 'commandType_id': 3013, 'executor_id': 101, 'requested': True}
        empty = satellite_discovery(7)
        empty['content']['event']['detectTarget'] = []
        data = load(self.write([satellite_state(0), request, empty, discovery(7), satellite_state(10)]))
        self.assertFalse(any(row['kind'] == 'satellite' for row in data[4]))
        self.assertTrue(any('没有天眼成功记录' in warning for warning in data[0]['audit']['warnings']))

    def test_satellite_native_detection_preserves_known_duration_window(self):
        final = satellite_state(10)
        final['entities']['401']['detectInfo']['2'] = {'pos': {'x': 40000, 'y': -5000, 'z': 100}}
        final.update(is_using_satellite=True, satellite_remaining_seconds=.25)
        data = load(self.write([satellite_state(0), satellite_discovery(7), final]))
        event = next(row for row in data[4] if row['kind'] == 'satellite')
        self.assertAlmostEqual(event['t'], .35)
        self.assertAlmostEqual(event['duration'], .4)
        self.assertAlmostEqual(event['t'] + event['duration'], .75)

    def test_satellite_explicit_success_state_remains_supported(self):
        for native_detection in (False, True):
            first = satellite_state(0)
            first.update(is_using_satellite=True, satellite_remaining_seconds=.75)
            rows = [first] + ([satellite_discovery(7)] if native_detection else []) + [satellite_state(10)]
            with self.subTest(native_detection=native_detection):
                data = load(self.write(rows))
                event = next(row for row in data[4] if row['kind'] == 'satellite')
                self.assertEqual(event['t'], 0)
                self.assertEqual(event['duration'], .75)
                self.assertNotEqual(event.get('confirmation'), 'native_satellite_detection_records')

    def test_missing_clock_requires_explicit_duration_and_checks_config_consistency(self):
        path = self.write([state(0, clock=False), state(10, clock=False)])
        with self.assertRaisesRegex(ValueError, 'step-seconds'):
            load(path)
        self.assertEqual(load(path, .25)[0]['step_seconds'], .25)
        path = self.write([{'data_type': 'config', 'simStep': 50}, state(0), state(10)])
        with self.assertRaisesRegex(ValueError, '场景步长不一致'):
            load(path, 1)

    def test_policy_caption_uses_short_id_but_preserves_reproduction_metadata(self):
        for original, label in (('b8_threat_multi_wave', 'B08'), ('b09_custom', 'B09'),
                                ('B08', 'B08'), ('custom_policy', 'custom_policy')):
            rows = [{'data_type': 'config', 'map': 'E06', 'blue': original, 'seed': 42},
                    state(0), state(10)]
            data = load(self.write(rows))
            self.assertEqual(data[0]['blue'], original)
            self.assertEqual(data[0]['audit']['display_metadata']['blue'], original)
            self.assertEqual(data[0]['subtitle'], f'场景 E06  /  虫族策略 {label}  /  种子 42')

    def test_map_requires_ordered_finite_xy_bounds(self):
        for area in ({'xMin': 1, 'xMax': 0, 'yMin': 0, 'yMax': 1},
                     {'xMin': 0, 'xMax': 1, 'yMin': 0},
                     {**AREA, 'yMax': float('inf')}):
            rows = [state(0), state(10)]
            for row in rows:
                row['map_area'] = area
            with self.subTest(area=area), self.assertRaises(ValueError):
                load(self.write(rows))

    def test_state_positions_require_pos_xyz_and_finite_values(self):
        for changed in ({'x': True, 'y': 0, 'z': 0}, {'x': 0, 'y': 0},
                        {'x': 0, 'y': float('nan'), 'z': 0}):
            row = state(10)
            row['entities']['1']['pos'] = changed
            with self.subTest(pos=changed), self.assertRaises(ValueError):
                load(self.write([state(0), row]))
        row = state(10)
        row['entities']['1']['position'] = row['entities']['1'].pop('pos')
        with self.assertRaisesRegex(ValueError, 'pos'):
            load(self.write([state(0), row]))

    def test_destroy_event_clock_preserves_true_score_and_damage_source(self):
        final = state(10)
        final['entities']['1']['health'] = 0
        event = {'data_type': 'event', 'type': 40000, 'entityId': 1, 'logicTime': ORIGIN+400,
                 'content': {'event': {'destroySrc': {'id': 101}}}}
        meta, _, _, _, events, summary = load(self.write([state(0), event, final]))
        self.assertAlmostEqual(summary['score'], 100*5/6)
        self.assertAlmostEqual(events[0]['t'], .4)
        self.assertEqual(meta['damage'], [{'source': 101, 'target': 1, 't': .4}])

    def test_track_waypoint_switches_preserve_coordinates_and_clear_previous_mode(self):
        track = {'data_type': 'aiAction', 'commandType_id': 3014, 'executor_id': 101,
                 'intent': 'track', 'target_id': 1}
        waypoint = {'data_type': 'aiAction', 'commandType_id': 3014, 'executor_id': 101,
                    'intent': 'waypoint', 'target': {'x': 0, 'y': 0, 'z': 0}}
        data = load(self.write([state(0), track, state(10), waypoint, state(20), track, state(30)]))
        meta = data[0]
        self.assertEqual([(row['t'], row['target']) for row in meta['assignments']],
                         [(.5, 1), (1, None), (1.5, 1)])
        self.assertEqual([(row['t'], row['position']) for row in meta['guidance']],
                         [(1, [0, 0]), (1.5, None)])
        self.assertEqual(meta['guidance'][0]['altitude'], 0)
        self.assertTrue(all(row['precision'] == 'state_sample' for row in meta['guidance']))
        self.assertEqual(meta['audit']['guidance_source_records'], 2)

    def test_waypoints_deduplicate_xyz_but_keep_position_and_altitude_changes(self):
        waypoint = {'data_type': 'aiAction', 'commandType_id': 3014, 'executor_id': 101,
                    'intent': 'waypoint', 'target': {'x': 7000, 'y': -8000, 'z': 3000}}
        moved = deepcopy(waypoint)
        moved['target']['x'] = 7100
        climbed = deepcopy(moved)
        climbed['target']['z'] = 3500
        meta = load(self.write([state(0), waypoint, state(10), deepcopy(waypoint), state(20),
                                moved, state(30), climbed, state(40), deepcopy(climbed), state(50)]))[0]
        self.assertEqual([(row['t'], row['position'], row['altitude']) for row in meta['guidance']],
                         [(.5, [7000, -8000], 3000), (1.5, [7100, -8000], 3000),
                          (2, [7100, -8000], 3500)])
        self.assertEqual(meta['assignments'], [])

    def test_waypoint_rejects_missing_or_nonfinite_xyz(self):
        for position in (None, {}, {'x': 1, 'y': 2}, {'x': True, 'y': 2, 'z': 3},
                         {'x': 1, 'y': float('inf'), 'z': 3},
                         {'x': 1, 'y': 2, 'z': float('nan')}):
            action = {'data_type': 'aiAction', 'commandType_id': 3014, 'executor_id': 101,
                      'intent': 'waypoint', 'target': position}
            with self.subTest(position=position), self.assertRaises(ValueError):
                load(self.write([state(0), action, state(10)]))

    def test_compact_round_trip_preserves_guidance_and_mode_clears(self):
        waypoint = {'data_type': 'aiAction', 'commandType_id': 3014, 'executor_id': 101,
                    'intent': 'waypoint', 'target': {'x': 37000, 'y': 10000, 'z': 3000}}
        track = {'data_type': 'aiAction', 'commandType_id': 3014, 'executor_id': 101,
                 'intent': 'track', 'target_id': 1}
        original = load(self.write([state(0), waypoint, discovery(7), state(10), track, state(20)]))
        directory = Path(self.directory.name)/'compact-guidance'
        save_compact(original, directory)
        replay = load(directory)
        self.assertEqual(original[0]['guidance'], replay[0]['guidance'])
        self.assertEqual(original[0]['assignments'], replay[0]['assignments'])
        self.assertEqual(original[4:], replay[4:])
        np.testing.assert_array_equal(original[1], replay[1])
        self.assertEqual(replay[0]['guidance'][0]['position'], [37000, 10000])
        self.assertEqual(replay[0]['guidance'][0]['altitude'], 3000)
        self.assertIsNone(replay[0]['guidance'][1]['position'])

    def test_compact_guidance_validates_coordinates_source_and_clock(self):
        directory = Path(self.directory.name)/'compact-invalid-guidance'
        save_compact(load(self.write([state(0), state(10)])), directory)
        path = directory/'replay.jsonl.gz'
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        valid = {'source': 101, 'position': [1, 2], 'altitude': 3, 't': .25}
        for changed in ({'position': [float('nan'), 2]}, {'position': [True, 2]},
                        {'position': [1]}, {'altitude': float('inf')},
                        {'source': 999}, {'source': True}, {'t': float('nan')}, {'t': 1}):
            rows[0]['guidance'] = [{**valid, **changed}]
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                stream.write('\n'.join(json.dumps(row) for row in rows)+'\n')
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                load(directory)

    def test_compact_round_trip_keeps_metre_coordinates_clock_and_events(self):
        original = load(self.write([state(0), discovery(7), state(10)]))
        directory = Path(self.directory.name)/'compact'
        save_compact(original, directory)
        replay = load(directory)
        np.testing.assert_array_equal(original[1], replay[1])
        np.testing.assert_array_equal(original[2], replay[2])
        self.assertEqual(original[3:], replay[3:])
        self.assertEqual(replay[0]['step_seconds'], .05)
        self.assertEqual(replay[0]['red_area'], [RING])

    def compact_without_clock(self, *, keep_steps=True):
        original = load(self.write([state(0), state(10)]))
        directory = Path(self.directory.name)/'compact'
        save_compact(original, directory)
        path = directory/'replay.jsonl.gz'
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            lines = [json.loads(line) for line in stream]
        lines[0].pop('step_seconds')
        lines[0]['audit'].pop('step_seconds')
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            stream.write('\n'.join(json.dumps(row) for row in lines)+'\n')
        if not keep_steps:
            summary_path = directory/'summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            summary.pop('steps')
            summary_path.write_text(json.dumps(summary), encoding='utf-8')
        return directory

    def test_compact_clock_is_derived_for_step_based_frame_export(self):
        directory = self.compact_without_clock()
        data = load(directory)
        self.assertEqual(data[0]['step_seconds'], .05)
        self.assertEqual(data[0]['audit']['step_seconds'], .05)
        # The same explicit duration is accepted for both directory and gzip input.
        explicit = load(directory, .05)
        self.assertEqual(explicit[0]['step_seconds'], .05)
        self.assertEqual(explicit[5]['steps'], 10)
        self.assertEqual(load(directory/'replay.jsonl.gz', .05)[0]['step_seconds'], .05)
        with self.assertRaisesRegex(ValueError, '不一致'):
            load(directory, 1)

    def test_compact_without_timing_requires_explicit_duration(self):
        directory = self.compact_without_clock(keep_steps=False)
        with self.assertRaisesRegex(ValueError, 'step-seconds'):
            load(directory)
        explicit = load(directory, .05)
        self.assertEqual(explicit[0]['step_seconds'], .05)
        self.assertEqual(explicit[5]['steps'], 10)
        for invalid in (0, -1, .3, float('nan'), True):
            with self.subTest(step_seconds=invalid), self.assertRaises(ValueError):
                load(directory, invalid)


if __name__ == '__main__':
    unittest.main()
