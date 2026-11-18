import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from replay_tool.log_input import load
from replay_tool.video import ReplayPainter, Timeline, WEIGHTS, NAMES, pygame


def main():
    parser = argparse.ArgumentParser(description='用真实日志核对实时兵力、平滑插值、六条事件及文字尺寸')
    parser.add_argument('log', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = load(args.log)
    meta, times, states, index, events, summary = data
    before = hashlib.sha256(states.tobytes()).hexdigest()
    painter = ReplayPainter(*data, Timeline(times[-1], events, 112))
    panels = (painter.map_box, painter.timeline_box, painter.legend_box, painter.score_box, painter.stats_box)
    for i, panel in enumerate(panels):
        assert painter.screen.get_rect().contains(panel)
        assert not any(panel.colliderect(other) for other in panels[:i])
    assert painter.score_box.top < painter.map_box.top < painter.stats_box.top
    clock = pygame.Rect(painter.clock_position, painter.bold[48].size('30:01'))
    title = pygame.Rect((40, 15), painter.bold[52].size(meta['title']))
    clock_label = pygame.Rect((painter.clock_position[0], 96), painter.fonts[20].size(f'仿真时间 / {int(times[-1])} 秒'))
    assert clock.left > title.right and clock.right < painter.legend_box.left
    assert clock_label.right < painter.legend_box.left and clock_label.bottom < painter.map_box.top
    assert painter.stats_box.contains(painter.feed_box)
    assert painter.feed_box.bottom <= painter.map_box.bottom
    for event in events:
        if event['kind'] != 'objective_destroyed':
            continue
        outline = pygame.Surface((48, 48), pygame.SRCALPHA)
        filled = pygame.Surface((48, 48), pygame.SRCALPHA)
        painter.renderer.event_marker(outline, (24, 24), 32, event, filled=False)
        painter.renderer.event_marker(filled, (24, 24), 32, event)
        alpha = pygame.surfarray.array_alpha(outline)
        assert alpha[24, 16] == 0
        assert outline.get_at((24, 24))[:3] == painter.renderer.TYPE_COLORS[event['type']]
        assert np.count_nonzero(alpha) < np.count_nonzero(pygame.surfarray.array_alpha(filled))
    counts = np.zeros_like(painter.troop_counts)
    for group, (side, types) in enumerate(((0, (21000, 21001, 21002)), (1, (24000,)))):
        rows = [index[int(key)] for key, entity in meta['entities'].items() if entity['side'] == side and entity['type'] in types]
        deployed = set()
        for frame, state in enumerate(states):
            alive = {row for row in rows if state[row, 3] > 0}
            deployed.update(alive)
            counts[frame, group] = len(alive), len(deployed - alive), len(deployed)
    assert np.array_equal(counts, painter.troop_counts)
    seam_crossings = 0
    for i in range(len(times) - 1):
        middle = (times[i] + times[i + 1]) / 2
        positions, _ = painter.sampled_positions(middle)
        assert np.allclose(positions, (states[i, :, :2] + states[i + 1, :, :2]) / 2, atol=1e-12)
        assert np.array_equal(painter.troop_statistics(middle), counts[i])
        difference = (painter.headings[i + 1] - painter.headings[i] + 180) % 360 - 180
        expected = painter.headings[i] + difference / 2
        assert np.allclose(painter.movement_headings(middle), expected, atol=1e-9)
        seam_crossings += int(np.count_nonzero(np.abs(painter.headings[i + 1] - painter.headings[i]) > 180))
        epsilon = min(1e-6, (times[i + 1] - times[i]) * 1e-5)
        first = painter.movement_headings(times[i + 1] - epsilon)
        last = painter.movement_headings(times[i + 1] + epsilon)
        assert np.max(np.abs((last - first + 180) % 360 - 180)) < .01
    for event, shown_at in zip(painter.feed, painter.feed_at):
        assert shown_at >= painter.timeline.video_time(event['t'])
        assert event['count'] == len(event['members'])
        assert event['t'] == max(member[2] for member in event['members'])
        if event['kind'] == 'objective_destroyed':
            assert event['count'] == 1
        if event['count'] > 1:
            assert event['kind'] == 'dark_discovered'
            assert len({int(member[2] / painter.EVENT_FOLD_SECONDS) for member in event['members']}) == 1
    assert np.all(np.diff(painter.feed_at) >= 0)
    assert painter.feed_at[-1] + painter.EVENT_SLIDE_SECONDS < painter.timeline.duration
    represented = {member for event in painter.feed for member in event['members']}
    assert represented == {(event['kind'], event['target'], event['t']) for event in events if event['kind'] != 'high_red_discovered'}
    observed, full_maximum, entering_frames = set(), 0, 0
    video_frames = round(painter.timeline.duration * 30)
    for video_t in np.arange(video_frames) / 30:
        layout = painter.event_feed_layout(video_t)
        full_maximum = max(full_maximum, sum(painter.feed_box.contains(rect) for _, rect in layout))
        assert len(layout) <= painter.EVENT_ROWS
        assert all(first['t'] >= last['t'] for (first, _), (last, _) in zip(layout, layout[1:]))
        for row, (event, rect) in enumerate(layout):
            assert event['t'] <= painter.timeline.simulation_time(video_t) + 1e-9
            assert not any(rect.colliderect(other) for _, other in layout[:row])
            assert painter.feed_box.contains(rect)
            observed.add((event['kind'], event['target']))
            entering_frames += int(row == 0 and 0 < video_t - painter.timeline.video_time(event['t']) < painter.EVENT_SLIDE_SECONDS)
    assert full_maximum == 6
    assert observed == {(event['kind'], event['target']) for event in painter.feed}
    assert entering_frames > 0
    clipping_samples = {68.0, painter.timeline.video_time(450)}
    for event in events:
        if event['kind'] == 'objective_destroyed' and event['type'] == 9400:
            clipping_samples.update((painter.timeline.video_time(event['t']) + .03,
                                    painter.timeline.video_time(event['t']) + .2))
    for video_t in sorted(clipping_samples):
        painter.render(video_t)
        before_pixels = pygame.surfarray.array3d(painter.screen)
        painter.event_feed(video_t)
        after_pixels = pygame.surfarray.array3d(painter.screen)
        box = painter.feed_box
        assert np.array_equal(before_pixels[:, :box.top], after_pixels[:, :box.top])
        assert np.array_equal(before_pixels[:, box.bottom:], after_pixels[:, box.bottom:])
        assert np.array_equal(before_pixels[:box.left], after_pixels[:box.left])
        assert np.array_equal(before_pixels[box.right:], after_pixels[box.right:])
    for use_steps in (False, True):
        stamp_right, icon_center, text_left = painter.event_feed_columns(use_steps)
        assert icon_center - 11 - stamp_right >= 10
        assert text_left - icon_center - 11 >= 12
        for event in painter.feed:
            assert 10 + painter.fonts[19].size(painter.event_stamp(event['t'], use_steps))[0] <= stamp_right
            if event['kind'] == 'objective_destroyed':
                gain = 100 * WEIGHTS[event['type']] / painter.total_weight
                line = NAMES[event['type']] + '摧毁'
                right = 438 - 10 - painter.fonts[21].size(f'+{gain:.2f}分')[0]
                assert text_left + 12 + painter.fonts[23].size(line)[0] <= right - 12
    assert 953 + painter.fonts[21].get_height() <= painter.TIMELINE_TRACK_Y - painter.TIMELINE_MARKER_SIZE / 2
    assert painter.TIMELINE_TRACK_Y + painter.TIMELINE_MARKER_SIZE / 2 <= 1040 - painter.TIMELINE_MARKER_SIZE / 2
    expected_discoveries = Counter(event['t'] for event in events if event['kind'] == 'dark_discovered')
    expected_destructions = {(event['target'], event['t']) for event in events
                             if event['kind'] == 'objective_destroyed'}
    expected_timeline = Counter((event['kind'], event['target'], event['t']) for event in events
                                if event['kind'] in ('objective_destroyed', 'dark_discovered'))
    for use_steps in (False, True):
        markers = painter.make_timeline_markers(use_steps)
        assert all(event['kind'] in ('objective_destroyed', 'dark_discovered') for event, _ in markers)
        assert {(event['target'], event['t']) for event, _ in markers
                if event['kind'] == 'objective_destroyed'} == expected_destructions
        assert Counter((event['kind'], event['target'], event['t']) for event, _ in markers) == expected_timeline
        end = times[-1] if use_steps else painter.timeline.duration
        for event, center in markers:
            stamp = event['t'] if use_steps else painter.timeline.video_time(event['t'])
            assert center == (round(54 + 1810 * stamp / end), painter.TIMELINE_TRACK_Y)
    timeline_labels = ('母巢摧毁', '哨卫巢摧毁', '暗巢摧毁', '暗巢发现')
    for i, label in enumerate(timeline_labels):
        right = 104 + i * 440 + painter.fonts[20].size(label)[0]
        assert right < (60 + (i + 1) * 440 if i < 3 else painter.timeline_box.right)
    for alive, lost, _ in counts.reshape(-1, 3):
        assert painter.fonts[21].size(f'存活 {alive}')[0] < 140
        assert painter.fonts[21].size(f'损失 {lost}')[0] < 144
    assert painter.fonts[23].get_height() + 5 < painter.EVENT_PITCH - 2
    assert painter.fonts[21].size('移动单位')[0] < 414
    for path in (Path(__file__).parent / 'replay_tool/video.py', Path(__file__).parent / 'replay_tool/log_input.py'):
        constants = [node.value for node in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
                     if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        assert not any(word in text for text in constants for word in ('红方', '蓝方', '人类'))
    assert before == hashlib.sha256(states.tobytes()).hexdigest()
    report = {'real_log_only': True, 'score': summary['score'], 'statistics_frames_checked': len(times),
              'initial_alive_loss_deployed': counts[0].tolist(), 'final_alive_loss_deployed': counts[-1].tolist(),
              'loss_includes_self_destruction': True, 'buildings_excluded_from_mobile_counts': True,
              'position_midpoints_checked': len(times) - 1, 'shortest_angle_seam_crossings_checked': seam_crossings,
              'event_feed_frames_checked': video_frames, 'maximum_full_event_rows': full_maximum,
              'all_feed_entries_seen': len(observed), 'all_raw_feed_events_represented': len(represented),
              'entry_animation_samples': entering_frames, 'feed_is_newest_first': True,
              'feed_rows_fully_inside_container': True, 'clipping_samples_checked': len(clipping_samples),
              'outside_container_pixels_unchanged': True,
              'folded_discovery_groups': [event['count'] for event in painter.feed if event['count'] > 1],
              'terminology_and_text_fit_passed': True, 'state_unchanged': True, 'visual_inspection': False}
    report['timeline_key_event_types'] = list(timeline_labels)
    report['timeline_discovery_markers'] = sum(event['kind'] == 'dark_discovered'
                                               for event, _ in painter.make_timeline_markers(True))
    report['timeline_discovery_event_steps'] = len(expected_discoveries)
    report['timeline_raw_discoveries_represented'] = sum(expected_discoveries.values())
    report['timeline_destructions_represented'] = len(expected_destructions)
    report['header_panels_do_not_overlap'] = True
    report['clock_position'] = list(painter.clock_position)
    report['legend_panel'] = list(painter.legend_box)
    report['score_panel'] = list(painter.score_box)
    report['statistics_panel'] = list(painter.stats_box)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
