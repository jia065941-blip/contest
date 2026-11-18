"""Metric replay geometry and exact, unmasked static-frame export."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pygame

from tools.replay_video.replay_tool.video import ReplayPainter, Timeline, export_frame


FONT = str(Path(pygame.__file__).parent / pygame.font.get_default_font())


def replay_data(area=None, movement=(1000, 1000)):
    area = area or {'xMin': -20000, 'xMax': 20000, 'yMin': -20000, 'yMax': 20000}
    meta = {
        'map_area': area, 'red_area': [[[-18000, -15000], [-10000, -15000],
                                      [-10000, 15000], [-18000, 15000]]],
        'entities': {'1': {'type': 21000, 'side': 0}, '2': {'type': 9400, 'side': 1}},
        'title': 'Replay', 'subtitle': '', 'audit': {'source': 'test-records'}, 'step_seconds': 1.0,
        'scenario_name': '', 'blue_name': '',
    }
    times = np.array([0.0, 10.0])
    states = np.array([
        [[0, 0, 100, 10, 1, 3], [5000, 0, 0, 24, 1, 1]],
        [[movement[0], movement[1], 100, 10, 1, 3], [5000, 0, 0, 0, 0, 1]],
    ], dtype=np.float64)
    events = [{'kind': 'objective_destroyed', 'target': 2, 'type': 9400,
               'position': [5000, 0], 't': 10.0}]
    summary = {'objectives': [{'id': 2, 'type': 9400, 'initial_hp': 24,
                              'final_hp': 0, 'weight': 5}],
               'score': 100, 'natural_done': True, 'steps': 10}
    return meta, times, states, {1: 0, 2: 1}, events, summary


def painter(data):
    return ReplayPainter(*data, Timeline(data[1][-1], data[4], 30),
                         font_file=FONT, latin_font_file=FONT)


class ReplayMetricGeometryTests(unittest.TestCase):
    def test_square_metric_area_preserves_shape_and_axis_directions(self):
        view = painter(replay_data())
        self.assertLessEqual(abs(view.map_rect.width - view.map_rect.height), 1)
        center = view.point(0, 0)
        east, north = view.point(1000, 0), view.point(0, 1000)
        self.assertGreater(east[0], center[0])
        self.assertLess(north[1], center[1])
        self.assertEqual(east[1], center[1])
        self.assertEqual(north[0], center[0])
        self.assertLessEqual(abs((east[0] - center[0]) - (center[1] - north[1])), 1)

    def test_offset_asymmetric_area_uses_one_scale_in_metres(self):
        area = {'xMin': 10000, 'xMax': 110000, 'yMin': -50000, 'yMax': 30000}
        view = painter(replay_data(area))
        center = view.point(60000, -10000)
        self.assertEqual(center, view.map_box.center)
        east = view.point(70000, -10000)
        north = view.point(60000, 0)
        self.assertLessEqual(abs((east[0] - center[0]) - (center[1] - north[1])), 1)
        self.assertAlmostEqual(view.map_w / view.map_h, 1.25)
        self.assertEqual(view.point(area['xMin'], area['yMax']),
                         (round(view.map_left), round(view.map_top)))

    def test_grid_cells_represent_equal_distances_on_both_axes(self):
        view = painter(replay_data({'xMin': -55000, 'xMax': 55000,
                                    'yMin': -46000, 'yMax': 46000}))
        x_values, y_values = view.grid_coordinates()
        self.assertGreaterEqual(len(x_values), 2)
        self.assertGreaterEqual(len(y_values), 2)
        self.assertEqual(x_values[1] - x_values[0], y_values[1] - y_values[0])
        horizontal = view.point(x_values[1], y_values[0])[0] - view.point(x_values[0], y_values[0])[0]
        vertical = view.point(x_values[0], y_values[0])[1] - view.point(x_values[0], y_values[1])[1]
        self.assertLessEqual(abs(horizontal - vertical), 1)

    def test_movement_heading_follows_recorded_xy_displacement(self):
        for movement, angle in (((1000, 0), -90), ((0, 1000), 0), ((1000, 1000), -45)):
            with self.subTest(movement=movement):
                view = painter(replay_data(movement=movement))
                self.assertAlmostEqual(view.movement_headings(0)[0], angle)

    def test_invalid_metric_ranges_raise_a_clear_error(self):
        for area in ({'xMin': 1, 'xMax': 1, 'yMin': -1, 'yMax': 1},
                     {'xMin': -1, 'xMax': 1, 'yMin': 2, 'yMax': 1}):
            with self.subTest(area=area), self.assertRaises(ValueError):
                painter(replay_data(area))


class ReplayStaticFrameTests(unittest.TestCase):
    def test_static_export_at_final_record_has_no_outro_and_keeps_true_score(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'final.png'
            with patch.object(ReplayPainter, 'banner', side_effect=AssertionError('unexpected banner')):
                receipt = export_frame(replay_data(), output, 10.0, FONT, FONT)
            self.assertEqual(receipt['image_size'], [1920, 1080])
            self.assertEqual(receipt['simulation_seconds'], 10.0)
            self.assertEqual(receipt['score_at_frame'], 100)
            self.assertEqual(pygame.image.load(str(output)).get_size(), (1920, 1080))
            self.assertTrue(output.with_suffix('.validation.json').is_file())
            with self.assertRaises(FileExistsError):
                export_frame(replay_data(), output, 10.0, FONT, FONT)

    def test_map_frame_contains_legend_and_map_at_native_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = export_frame(replay_data(), Path(folder)/'map.png', 0.0, FONT, FONT, map_only=True)
            self.assertEqual(receipt['image_size'], [1374, 924])
            self.assertEqual(receipt['crop_pixels'], [24, 15, 1374, 924])
            self.assertEqual(receipt['coordinate_units'], 'metres')
            self.assertEqual(receipt['score_at_frame'], 0)

    def test_static_clock_preserves_fractional_steps_and_video_clock_is_unchanged(self):
        data = replay_data()
        data[0]['step_seconds'] = 0.5
        data[0]['map'] = 'E01'
        data[0]['subtitle'] = 'Scene E01 / B8 / seed 42'
        view = painter(data)
        with patch.object(view, 'text', wraps=view.text) as draw_text:
            view.render(view.timeline.video_time(1.25), simulation_seconds=1.25, static=True)
            values = [call.args[0] for call in draw_text.call_args_list]
            self.assertIn('第2.5步', values)
            self.assertIn(data[0]['subtitle'], values)
            self.assertFalse(any(value.startswith('仿真时间 /') for value in values))
        with patch.object(view, 'text', wraps=view.text) as draw_text:
            view.render(view.timeline.video_time(1.25), simulation_seconds=1.25)
            values = [call.args[0] for call in draw_text.call_args_list]
            self.assertIn('00:01', values)
            self.assertIn('仿真时间 / 10 秒', values)

    def test_static_units_can_use_panel_padding_without_moving_the_map(self):
        data = replay_data()
        data[2][:, 1, :2] = (0, 19900)
        data[4][0]['position'] = [0, 19900]
        view = painter(data)
        marker_x = view.point(0, 19900)[0]
        padding = pygame.Rect(marker_x - 35, view.map_rect.top - 24, 70, 18)
        baseline = pygame.image.tobytes(view.static.subsurface(padding), 'RGB')
        point_before = view.point(0, 19900)
        rect_before = view.map_rect.copy()
        static_image = view.render(view.timeline.video_time(5), simulation_seconds=5, static=True).copy()
        self.assertNotEqual(pygame.image.tobytes(static_image.subsurface(padding), 'RGB'), baseline)
        video_image = view.render(view.timeline.video_time(5), simulation_seconds=5)
        self.assertEqual(pygame.image.tobytes(video_image.subsurface(padding), 'RGB'), baseline)
        self.assertEqual(view.point(0, 19900), point_before)
        self.assertEqual(view.map_rect, rect_before)

    def test_static_event_and_timeline_use_recorded_steps_instead_of_video_speed(self):
        data = replay_data()
        data[0]['step_seconds'] = .5
        data[4].insert(0, {'kind': 'satellite', 'target': 0, 'position': [],
                          't': 1.25, 'duration': None,
                          'confirmation': 'native_satellite_detection_records'})
        view = painter(data)
        video_t = view.timeline.video_time(5)
        with patch.object(view, 'text', wraps=view.text) as draw_text:
            view.render(video_t, simulation_seconds=5, static=True)
            values = [call.args[0] for call in draw_text.call_args_list]
        self.assertIn('第10步', values)
        self.assertIn('第2.5步', values)
        self.assertIn('10 / 20 步', values)
        self.assertIn('天眼探测', values)
        self.assertFalse(any('秒' in value or ':' in value for value in values))
        # Half of the recorded simulation belongs at the track midpoint even
        # when the video deliberately allocates a different amount of time.
        with patch.object(pygame.draw, 'circle', wraps=pygame.draw.circle) as circle:
            view.draw_timeline(video_t, 5, use_steps=True)
        circle.assert_any_call(view.screen, (237, 243, 250), (959, view.TIMELINE_TRACK_Y), 8)
        markers = view.make_timeline_markers(use_steps=True)
        self.assertTrue(all(event['kind'] != 'satellite' for event, _ in markers))
        self.assertEqual(markers[0][1][0], 1864)
        with patch.object(view, 'text', wraps=view.text) as draw_text:
            view.draw_timeline(video_t, 5)
        self.assertTrue(any(call.args[0].startswith('播放 ') and call.args[0].endswith(' 秒')
                            for call in draw_text.call_args_list))

    def test_event_timestamp_icon_and_gain_keep_separate_columns_for_large_steps(self):
        data = replay_data()
        data[0]['step_seconds'] = 10 / 3000
        view = painter(data)
        with patch.object(view, 'text', wraps=view.text) as draw_text, \
             patch.object(view.renderer, 'event_marker', wraps=view.renderer.event_marker) as marker:
            view.event_feed(view.timeline.video_time(10), use_steps=True)
        calls = {call.args[0]: call for call in draw_text.call_args_list}
        stamp = calls['第3000步']
        icon_x = marker.call_args.args[1][0]
        stamp_right = stamp.args[1][0] + view.fonts[19].size(stamp.args[0])[0]
        self.assertGreaterEqual(icon_x - 11 - stamp_right, 10)
        body = calls['母巢摧毁']
        self.assertGreaterEqual(body.args[1][0] - (icon_x + 11), 12)
        body_right = body.args[1][0] + view.fonts[23].size(body.args[0])[0]
        self.assertLess(body_right, calls['+100.00分'].args[1][0])

    def test_timeline_includes_discoveries_without_changing_source_events(self):
        view = painter(replay_data())
        view.events = [
            {'kind': 'dark_discovered', 'target': 3, 't': 1.25},
            {'kind': 'dark_discovered', 'target': 4, 't': 1.25},
            {'kind': 'dark_discovered', 'target': 5, 't': 1.5},
            {'kind': 'high_red_discovered', 'target': 1, 't': 2},
            {'kind': 'satellite', 'target': 0, 't': 1.25},
            {'kind': 'objective_destroyed', 'target': 2, 'type': 9400, 't': 10},
        ]
        before = deepcopy(view.events)
        markers = view.make_timeline_markers(use_steps=True)
        discovered = [(event, point) for event, point in markers if event['kind'] == 'dark_discovered']
        self.assertEqual([(event['t'], event['target']) for event, _ in discovered], [(1.25, 3), (1.25, 4), (1.5, 5)])
        self.assertEqual(discovered[0][1][0], round(54 + 1810 * 1.25 / 10))
        self.assertEqual(len(markers), 4)
        self.assertEqual(view.events, before)
        video_markers = view.make_timeline_markers()
        self.assertEqual(video_markers[0][1][0],
                         round(54 + 1810 * view.timeline.video_time(1.25) / view.timeline.duration))

    def test_progress_cursor_does_not_cover_a_destroyed_event_marker(self):
        view = painter(replay_data())
        view.draw_timeline(view.timeline.video_time(10), 10, use_steps=True)
        center = (1864, view.TIMELINE_TRACK_Y)
        self.assertEqual(view.screen.get_at(center)[:3], view.renderer.TYPE_COLORS[9400])

    def test_static_units_do_not_paint_over_the_footer(self):
        data = replay_data()
        data[2][:, 0, :2] = (15000, -19900)
        view = painter(data)
        x, _ = view.point(15000, -19900)
        footer = pygame.Rect(x - 45, view.MAP_FOOTER_Y, 90, view.map_box.bottom - view.MAP_FOOTER_Y)
        image = view.render(view.timeline.video_time(5), simulation_seconds=5, static=True)
        self.assertEqual(pygame.image.tobytes(image.subsurface(footer), 'RGB'),
                         pygame.image.tobytes(view.static.subsurface(footer), 'RGB'))

    def test_timeline_has_four_separate_key_event_legend_items(self):
        view = painter(replay_data())
        view.events = [{'kind': 'dark_discovered', 'target': 3, 't': 1}]
        with patch.object(view, 'text', wraps=view.text) as draw_text:
            view.draw_timeline(view.timeline.video_time(5), 5, use_steps=True)
        calls = {call.args[0]: call for call in draw_text.call_args_list}
        labels = ('母巢摧毁', '哨卫巢摧毁', '暗巢摧毁', '暗巢发现')
        self.assertNotIn('天眼使用', calls)
        rectangles = [pygame.Rect(calls[label].args[1], view.fonts[20].size(label)) for label in labels]
        self.assertTrue(all(left.right + 30 < right.left for left, right in zip(rectangles, rectangles[1:])))
        self.assertLess(rectangles[-1].right, view.timeline_box.right)

    def test_static_export_rejects_times_outside_records_and_wrong_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            for value in (-1, 11, float('nan'), True):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    export_frame(replay_data(), Path(folder)/'frame.png', value, FONT, FONT)
            with self.assertRaises(ValueError):
                export_frame(replay_data(), Path(folder)/'frame.jpg', 1.0, FONT, FONT)


class ReplayGuidanceTests(unittest.TestCase):
    def test_waypoint_and_entity_target_modes_clear_each_other(self):
        data = replay_data()
        first, second = [12000, 10000], [-14000, 6000]
        data[0]['guidance'] = [{'t': 0, 'source': 1, 'position': first},
                               {'t': 2, 'source': 1, 'position': None},
                               {'t': 4, 'source': 1, 'position': second}]
        data[0]['assignments'] = [{'t': 2, 'source': 1, 'target': 2},
                                  {'t': 4, 'source': 1, 'target': None}]
        view = painter(data)
        current = data[2][0]
        self.assertEqual(view.active_guidance(1, current), [(1, first)])
        self.assertEqual(view.active_assignments(1, current), [])
        self.assertEqual(view.active_guidance(3, current), [])
        self.assertEqual(view.active_assignments(3, current), [(1, 2)])
        self.assertEqual(view.active_guidance(5, current), [(1, second)])
        self.assertEqual(view.active_assignments(5, current), [])

    def test_dead_and_invisible_sources_have_no_guidance(self):
        data = replay_data()
        data[0]['guidance'] = [{'t': 1, 'source': 1, 'position': [15000, 10000]}]
        view = painter(data)
        current = data[2][0].copy()
        self.assertEqual(view.active_guidance(0, current), [])
        current[0, 3] = 0
        self.assertEqual(view.active_guidance(5, current), [])
        current[0, 3], current[0, 4] = 10, 0
        self.assertEqual(view.active_guidance(5, current), [])

    def test_guidance_endpoint_uses_recorded_xy_and_is_visible_on_static_frame(self):
        data = replay_data()
        target = [-14000, 12000]
        data[0]['guidance'] = [{'t': 0, 'source': 1, 'position': target, 'altitude': 3000}]
        view = painter(data)
        positions, _ = view.sampled_positions(5)
        endpoint = view.point(*target)
        with patch.object(view, 'point', wraps=view.point) as project:
            endpoints = view.draw_guidance(5, positions, data[2][0])
            project.assert_any_call(*positions[0])
            project.assert_any_call(*target)
        self.assertEqual(endpoints, {endpoint})
        self.assertNotEqual(endpoint, view.point(*positions[1]))
        image = view.render(view.timeline.video_time(5), simulation_seconds=5, static=True)
        endpoint_pixels = {image.get_at((endpoint[0] + dx, endpoint[1] + dy))[:3]
                           for dx in range(-6, 7) for dy in range(-6, 7)}
        self.assertIn(view.ASSIGNMENT_COLOR, endpoint_pixels)

    def test_shared_waypoint_has_one_endpoint_and_does_not_bind_to_a_dead_entity(self):
        data = replay_data()
        target = [5000, 0]
        data[2][0, 1, :2] = [-5000, 5000]
        data[0]['guidance'] = [{'t': 0, 'source': 1, 'position': target},
                               {'t': 0, 'source': 2, 'position': target}]
        view = painter(data)
        positions, _ = view.sampled_positions(5)
        self.assertEqual(len(view.active_guidance(5, data[2][0])), 2)
        self.assertEqual(view.draw_guidance(5, positions, data[2][0]), {view.point(*target)})
        self.assertEqual(view.active_guidance(10, data[2][1]), [(1, target)])

    def test_missing_guidance_metadata_draws_nothing(self):
        data = replay_data()
        view = painter(data)
        positions, _ = view.sampled_positions(5)
        before = pygame.image.tobytes(view.screen, 'RGB')
        self.assertEqual(view.active_guidance(5, data[2][0]), [])
        self.assertEqual(view.draw_guidance(5, positions, data[2][0]), set())
        self.assertEqual(pygame.image.tobytes(view.screen, 'RGB'), before)


if __name__ == '__main__':
    unittest.main()
