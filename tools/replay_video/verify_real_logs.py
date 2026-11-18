import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from replay_tool.log_input import load, save_compact
from replay_tool.video import ReplayPainter, Timeline, pygame


def main():
    parser = argparse.ArgumentParser(description='只用传入的真实日志检查读取、计分、时间映射和渲染')
    parser.add_argument('log', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--step-seconds', type=float)
    args = parser.parse_args()
    data = load(args.log, args.step_seconds)
    meta, times, states, index, events, summary = data
    timeline = Timeline(times[-1], events, 112)
    mapped = np.asarray([timeline.simulation_time(t) for t in np.arange(3360) / 30])
    assert np.all(np.diff(mapped) >= 0) and mapped[0] == 0 and mapped[-1] == times[-1]
    for event in events:
        assert abs(timeline.simulation_time(timeline.video_time(event['t'])) - event['t']) < 1e-6
    if timeline.fast_start < times[-1]:
        assert timeline.video_time(times[-1]) - timeline.video_time(timeline.fast_start) <= 3.00001
    painter = ReplayPainter(*data, timeline)
    sample_times = {0.0, 111.0}
    for kind in ('dark_discovered', 'high_red_discovered', 'objective_destroyed', 'satellite'):
        matches = [event for event in events if event['kind'] == kind]
        if matches:
            sample_times.add(timeline.video_time(matches[0]['t']) + .3)
    for t in sample_times:
        frame = painter.render(t)
        first = hashlib.sha256(pygame.image.tobytes(frame, 'RGB')).hexdigest()
        frame_again = painter.render(t)
        assert first == hashlib.sha256(pygame.image.tobytes(frame_again, 'RGB')).hexdigest()
        assert frame.get_size() == (1920, 1080)
    capsule_frames = set()
    for event, at in zip(events, painter.event_at):
        if event['kind'] == 'objective_destroyed':
            capsule_frames.update(range(int(np.ceil(at * 30)), int(np.ceil((at + 1.1) * 30))))
    maximum_capsules = 0
    for frame_index in sorted(capsule_frames):
        painter.render(frame_index / 30)
        annotations = painter.frame_annotations
        rectangles = annotations['capsules']
        maximum_capsules = max(maximum_capsules, len(rectangles))
        for i, rectangle in enumerate(rectangles):
            assert painter.map_rect.inflate(-8, -8).contains(rectangle)
            assert not any(rectangle.inflate(6, 6).colliderect(other) for other in rectangles[:i])
            assert not any(rectangle.inflate(6, 6).colliderect(label) for label in annotations['labels'])
    heading_samples, maximum_heading_error = 0, 0.0
    for key, entity in meta['entities'].items():
        if entity['type'] not in (21000, 21001):
            continue
        row = index[int(key)]
        delta = np.diff(states[:, row, :2], axis=0)
        moving = np.any(np.abs(delta) >= 1e-10, axis=1)
        dx = delta[moving, 0] * painter.map_w / (painter.area['xMax'] - painter.area['xMin'])
        dy = -delta[moving, 1] * painter.map_h / (painter.area['yMax'] - painter.area['yMin'])
        angles = np.degrees(np.arctan2(-dx, -dy))
        actual = painter.headings[:-1, row][moving]
        errors = np.abs((actual - angles + 180) % 360 - 180)
        if len(errors):
            maximum_heading_error = max(maximum_heading_error, float(errors.max()))
            heading_samples += len(errors)
        assert np.all(errors <= 2.50001)
    assert not any(key == 'envengine' or key.startswith('envengine.') for key in sys.modules)
    compact = args.output.with_suffix('.compact')
    save_compact(data, compact)
    roundtrip = load(compact)
    assert np.array_equal(states, roundtrip[2]) and np.array_equal(times, roundtrip[1])
    assert events == roundtrip[4] and summary['score'] == roundtrip[5]['score']
    report = {'real_log_only': True, 'source': str(args.log), 'score': summary['score'],
              'frames': len(times), 'entities': len(index), 'event_counts': meta['audit']['event_counts'],
              'damage_source_records': meta['audit']['damage_source_records'],
              'rendered_deterministic_samples': len(sample_times), 'engine_imported': False,
              'real_capsule_frames_checked': len(capsule_frames), 'maximum_simultaneous_capsules': maximum_capsules,
              'capsule_and_building_labels_do_not_overlap': True,
              'recorded_movement_heading_samples': heading_samples, 'maximum_heading_error_degrees': maximum_heading_error,
              'compact_roundtrip_exact': True, 'idle_tail_at_most_3_seconds': True,
              'visual_inspection': False}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
