import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from replay_tool.log_input import load
from replay_tool.video import ReplayPainter, Timeline, pygame


def luminance(rgb):
    channels = [value / 255 for value in rgb]
    linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055)**2.4 for value in channels]
    return sum(value * weight for value, weight in zip(linear, (.2126, .7152, .0722)))


def contrast(first, second):
    values = sorted((luminance(first), luminance(second)))
    return (values[1] + .05) / (values[0] + .05)


def composite(rgb, alpha, background):
    return tuple(round(value * alpha / 255 + base * (1 - alpha / 255)) for value, base in zip(rgb, background))


def main():
    parser = argparse.ArgumentParser(description='使用真实对局检查投影配色数值、关键事件和重复渲染一致性')
    parser.add_argument('log', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = load(args.log)
    meta, times, states, index, events, summary = data
    before = hashlib.sha256(states.tobytes()).hexdigest()
    event_snapshot = json.dumps(events, sort_keys=True)
    painter = ReplayPainter(*data, Timeline(times[-1], events, 112))
    palette = ReplayPainter
    measurements = {
        'map_to_outer_luminance_ratio': luminance(palette.MAP_COLOR) / luminance(palette.BACKGROUND_COLOR),
        'grid_to_map_contrast': contrast(palette.GRID_COLOR, palette.MAP_COLOR),
        'secondary_to_map_contrast': contrast(palette.SECONDARY_TEXT, palette.MAP_COLOR),
        'secondary_to_panel_contrast': contrast(palette.SECONDARY_TEXT, palette.PANEL_COLOR),
        'muted_to_map_contrast': contrast(palette.MUTED_TEXT, palette.MAP_COLOR),
        'gain_capsule_contrast': contrast(palette.GAIN_TEXT, composite(palette.GAIN_BACKGROUND[:3], palette.GAIN_BACKGROUND[3], palette.MAP_COLOR)),
        'remnant_cross_contrast': contrast(palette.REMNANT_COLOR, palette.MAP_COLOR),
    }
    assert measurements['map_to_outer_luminance_ratio'] >= 1.6
    assert measurements['grid_to_map_contrast'] >= 1.5
    assert all(measurements[key] >= 4.5 for key in ('secondary_to_map_contrast', 'secondary_to_panel_contrast', 'muted_to_map_contrast', 'gain_capsule_contrast'))
    assert 1 < measurements['remnant_cross_contrast'] < 2
    assert palette.SATELLITE_FILL_ALPHA / 255 <= .04
    assert palette.RED_TRAIL_ALPHA / 255 <= .35
    assert palette.RED_TRAIL_SECONDS <= .3
    red_sizes = {}
    observed_red = {row['type'] for row in meta['entities'].values()} & set(palette.RED_TRAIL_WIDTHS)
    for kind in sorted(observed_red):
        alpha = pygame.surfarray.array_alpha(painter.sprites[kind])
        assert set(np.unique(alpha)) == {0, palette.RED_RING_ALPHA, 255}
        xs, ys = np.nonzero(alpha == 255)
        diameter = min(int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1)
        assert palette.RED_TRAIL_WIDTHS[kind] <= diameter * .5
        red_sizes[str(kind)] = {'opaque_core_min_diameter': diameter, 'trail_max_width': palette.RED_TRAIL_WIDTHS[kind],
                                'soft_alpha_pixels': int(np.count_nonzero((alpha > 0) & (alpha < palette.RED_RING_ALPHA)))}
    for kind, sprite in painter.remnants.items():
        alpha = pygame.surfarray.array_alpha(sprite)
        colors = pygame.surfarray.array3d(sprite)[alpha > 0]
        assert np.all(colors == palette.REMNANT_COLOR)
        assert np.count_nonzero(alpha) < alpha.size / 2
        expected = round((48 if kind == 9400 else 42) * .64 * .6)
        assert sprite.get_size() == (expected, expected)
    samples = {0.0, 3.0, 5.0, 8.0, 12.0, 15.0, 20.0, 25.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
               44.0, 48.0, 55.0, 67.0, 68.0, 75.0, 100.0, 104.0, 105.0, 106.0, 111.0}
    destroys = [event for event in events if event['kind'] == 'objective_destroyed']
    for event in destroys:
        at = painter.timeline.video_time(event['t'])
        samples.update((at + .2, at + 1.25, at + 1.52))
    checked_segments = 0
    for video_t in sorted(samples):
        first = hashlib.sha256(pygame.image.tobytes(painter.render(video_t), 'RGB')).hexdigest()
        second = hashlib.sha256(pygame.image.tobytes(painter.render(video_t), 'RGB')).hexdigest()
        assert first == second
        for start, end, color, widths, alpha, red in painter.trail_segments(video_t):
            if red:
                assert start != end and 0 <= widths[0] < widths[1] <= max(palette.RED_TRAIL_WIDTHS.values())
                assert 0 <= alpha <= palette.RED_TRAIL_ALPHA
                checked_segments += 1
    assert checked_segments > 0
    for sprite in painter.heading_sprites.values():
        assert set(np.unique(pygame.surfarray.array_alpha(sprite))) <= {0, palette.RED_RING_ALPHA, 255}
    notice_widths = [245]
    for video_t in np.linspace(painter.timeline.video_time(painter.timeline.fast_start), 106.9, 20):
        sim_t = painter.timeline.simulation_time(video_t)
        speed = (painter.timeline.simulation_time(video_t + .1) - sim_t) / .1
        value = f'后续无新增关键事件 · 快进至日志终点  {speed:.1f}x'
        notice_widths.append(painter.fonts[23].size(value)[0] + 116)
    for width in notice_widths:
        notice = painter.map_notice_rectangle(width, 48)
        assert notice.centerx == painter.map_box.centerx and painter.map_box.contains(notice)
    assert before == hashlib.sha256(states.tobytes()).hexdigest()
    assert event_snapshot == json.dumps(events, sort_keys=True)
    report = {'measurements': measurements, 'real_frame_samples': len(samples),
              'red_sprites': red_sizes, 'red_trail_segments_checked': checked_segments,
              'red_sprites_have_no_soft_halo': True, 'rotated_icons_have_no_interpolated_alpha': True,
              'rubble_has_only_small_gray_cross': True, 'satellite_fill_alpha_max': palette.SATELLITE_FILL_ALPHA / 255,
              'satellite_and_fast_forward_share_map_center': True,
              'deterministic_rendering': True, 'state_and_events_unchanged': True,
              'score': summary['score'], 'source_event_counts': meta['audit']['event_counts'],
              'manual_visual_inspection': False, 'projection_hardware_tested': False}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
