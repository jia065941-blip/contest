import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from replay_tool.log_input import blue_display_id, load, records, save_compact
from replay_tool.video import ReplayPainter, Timeline, pygame


def main():
    parser = argparse.ArgumentParser(description='从真实日志核验展示信息与连续两次紧凑转换')
    parser.add_argument('output', type=Path, help='尚不存在的检查目录')
    parser.add_argument('logs', type=Path, nargs='+', help='真实原生或紧凑JSONL日志文件')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    report = {'real_logs_only': True, 'cases': [], 'visual_inspection': False}
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    for case, source in enumerate(args.logs):
        original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        raw = {}
        for row in records(source):
            if row.get('data_type', row.get('kind')) in ('config', 'meta'):
                raw.update({key: row[key] for key in ('map', 'blue', 'seed', 'variant') if key in row})
        data = load(source)
        meta, times, states, _, events, summary = data
        assert meta['red_name'] == '示例策略' and meta['title'] == '对局回放'
        expected = []
        if raw.get('map') is not None:
            expected.append(f"场景 {raw['map']}")
        if raw.get('blue') is not None:
            expected.append(f"虫族策略 {blue_display_id(raw['blue'])}")
        if raw.get('seed') is not None:
            expected.append(f"种子 {raw['seed']}")
        assert meta['subtitle'] == '  /  '.join(expected)
        assert {field: value for field, value in meta['audit']['display_metadata'].items()
                if field in ('map', 'blue', 'seed')} == {
                    field: raw[field] for field in ('map', 'blue', 'seed') if raw.get(field) is not None}
        state_hash = hashlib.sha256(states.tobytes()).hexdigest()
        painter = ReplayPainter(*data, Timeline(times[-1], events, 112))
        for video_t in (0, 5, painter.timeline.duration - 1):
            assert painter.render(video_t).get_size() == (1920, 1080)
        assert state_hash == hashlib.sha256(states.tobytes()).hexdigest()
        # 真实日志连续转换两次，名称不应重复添加前缀，也不覆盖原策略标识。
        for round_number in range(2):
            destination = output / f'case_{case}_round_{round_number}.compact'
            save_compact(data, destination)
            data = load(destination)
            assert data[0]['subtitle'] == meta['subtitle']
            assert data[0].get('map') == raw.get('map')
            assert data[0].get('blue') == raw.get('blue')
            assert data[0].get('variant') == raw.get('variant')
            assert np.array_equal(states, data[2]) and events == data[4]
            assert summary['score'] == data[5]['score']
        assert original_hash == hashlib.sha256(source.read_bytes()).hexdigest()
        report['cases'].append({'source': str(source), 'title': meta['title'],
                                'subtitle': meta['subtitle'], 'score': summary['score'],
                                'roundtrips': 2, 'input_sha256_unchanged': original_hash})
        pygame.quit()
    literals = [node.value for node in ast.walk(ast.parse((here / 'replay_tool/video.py').read_text(encoding='utf-8')))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not any('组织方全局回放' in value or '灰色暗巢' in value for value in literals)
    cli = (here / 'replay_video.py').read_text(encoding='utf-8')
    assert all(option not in cli for option in ('--red-name', '--blue-name', '--scenario-name'))
    assert not any(key == 'envengine' or key.startswith('envengine.') for key in sys.modules)
    report.update(unrecorded_fields_omitted=True, manual_name_options_removed=True,
                  map_note_lines_removed=True, engine_imported=False)
    (output / 'metadata_display_verification.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
