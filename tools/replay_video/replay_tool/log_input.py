import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
from .rules import WEIGHTS

MAX_LINE = 32 * 1024**2
MAX_BYTES = 4 * 1024**3


def blue_display_id(value):
    """Short public policy ID; keep the original identifier in replay metadata."""
    match = re.fullmatch(r'b(\d{1,2})(?:_.*)?', value, flags=re.IGNORECASE)
    return f'B{int(match.group(1)):02d}' if match else value


def records(path):
    opener = gzip.open if path.suffix == '.gz' else open
    size = 0
    with opener(path, 'rt', encoding='utf-8-sig') as stream:
        number = 0
        while line := stream.readline(MAX_LINE + 1):
            number += 1
            line_bytes = len(line.encode('utf-8'))
            size += line_bytes
            if line_bytes > MAX_LINE or size > MAX_BYTES:
                raise ValueError('日志超过单条32MiB或解压后4GiB的读取限制')
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}第{number}条JSON记录无效：{error.msg}') from error
            if not isinstance(row, dict):
                raise ValueError(f'第{number}条记录应为JSON对象；此工具读取逐行JSON日志')
            yield row


def number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{field}必须是有限数值')
    return float(value)


def position_xy(position, field):
    if not isinstance(position, dict) or not {'x', 'y'} <= position.keys():
        raise ValueError(f'{field}需要pos中的x和y坐标，单位为米')
    return [number(position['x'], f'{field}x'), number(position['y'], f'{field}y')]


def deployment_rings(coordinates):
    """Normalize the deployment area to rings of planar metre coordinates."""
    if not coordinates:
        return []
    if not isinstance(coordinates, (list, tuple)):
        raise ValueError('部署区域必须是坐标列表')
    if (isinstance(coordinates[0], (list, tuple)) and coordinates[0]
            and isinstance(coordinates[0][0], (int, float))):
        coordinates = [coordinates]
    result = []
    for ring in coordinates:
        if not isinstance(ring, (list, tuple)) or len(ring) < 3:
            raise ValueError('部署多边形至少需要三个顶点')
        result.append([])
        for point in ring:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError('部署顶点必须是[x,y]米制坐标')
            result[-1].append([number(point[0], '部署x'), number(point[1], '部署y')])
    return result


def validate_data(meta, times, states, ids, events, summary, source):
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('日志必须含第0步及至少两个时间递增的完整状态帧')
    if not np.isfinite(states).all():
        raise ValueError('实体状态包含无效数值')
    if len(set(map(int, ids))) != len(ids) or np.any(ids != np.asarray(ids, dtype=np.int64)):
        raise ValueError('实体编号应为不重复的整数')
    index = {int(key): i for i, key in enumerate(ids)}
    area = meta.get('map_area')
    keys = ('xMin', 'xMax', 'yMin', 'yMax')
    if not isinstance(area, dict) or not set(keys) <= area.keys():
        raise ValueError('地图边界需要xMin/xMax/yMin/yMax米制坐标')
    area = {key: number(area[key], f'地图{key}') for key in keys}
    if area['xMin'] >= area['xMax'] or area['yMin'] >= area['yMax']:
        raise ValueError('地图边界范围无效')
    meta['map_area'] = area
    meta['red_area'] = deployment_rings(meta.get('red_area', []))
    if meta.get('coordinate_system', 'local_xyz_m') != 'local_xyz_m':
        raise ValueError('日志坐标标识必须为local_xyz_m')
    meta['coordinate_system'] = 'local_xyz_m'
    for objective in summary['objectives']:
        key = objective['id']
        if objective['initial_hp'] <= 0 or key not in index or objective['weight'] != WEIGHTS[objective['type']]:
            raise ValueError('计分目标初始血量和实体目录不完整')
        if abs(states[-1, index[key], 3] - objective['final_hp']) > 1e-7:
            raise ValueError(f'目标{key}最终血量与成绩摘要不符')
    objectives = {row['id']: row for row in summary['objectives']}
    if not objectives:
        raise ValueError('日志缺少初始计分目标，不能计算真实得分')
    total = sum(row['weight'] for row in objectives.values())
    score = 100 * sum(row['weight'] for row in objectives.values() if row['final_hp'] <= 0) / total
    if summary['score'] is not None and abs(summary['score'] - score) > 1e-7:
        raise ValueError('摘要得分与完整摧毁5/2/1规则重算结果不符')
    summary['score'] = score
    seen, destruction = set(), set()
    for event in events:
        if event['kind'] not in ('satellite', 'dark_discovered', 'high_red_discovered', 'objective_destroyed'):
            raise ValueError('关键事件类型不受支持')
        marker = (event['kind'], event['target'])
        if marker in seen or not 0 <= event['t'] <= times[-1]:
            raise ValueError('事件重复或超出日志时间范围')
        seen.add(marker)
        if event['kind'] == 'satellite':
            if event['duration'] is not None and event['duration'] <= 0:
                raise ValueError('天眼成功事件缺少有效持续时间')
            continue
        if event['target'] not in index or len(event['position']) != 2 or not all(math.isfinite(v) for v in event['position']):
            raise ValueError('事件的目标或坐标无效')
        kind = meta['entities'][str(event['target'])]['type']
        if event['kind'] == 'dark_discovered' and kind != 9500:
            raise ValueError('暗巢发现事件目标类型不符')
        if event['kind'] == 'high_red_discovered' and kind != 21000:
            raise ValueError('高等级个体发现事件目标类型不符')
        if event['kind'] == 'objective_destroyed':
            if event['target'] not in objectives or event['type'] != kind:
                raise ValueError('摧毁事件的计分目标类型不符')
            destruction.add(event['target'])
    if destruction != {key for key, row in objectives.items() if row['final_hp'] <= 0}:
        raise ValueError('摧毁事件与目标最终血量不一致')
    events.sort(key=lambda row: (row['t'], row['kind'], row['target']))
    for hit in meta.get('damage', []):
        if hit['source'] not in index or hit['target'] not in index or not 0 <= hit['t'] <= times[-1]:
            raise ValueError('伤害来源、目标或时刻不在本局记录中')
    for assignment in meta.get('assignments', []):
        if (type(assignment['source']) is not int or assignment['source'] not in index
                or assignment['target'] is not None and (type(assignment['target']) is not int or assignment['target'] not in index)
                or not 0 <= number(assignment['t'], '任务分配时刻') <= times[-1]):
            raise ValueError('任务分配的单位、目标或时刻不在本局记录中')
    meta['assignments'] = sorted(meta.get('assignments', []), key=lambda row: row['t'])
    guidance = meta.get('guidance', [])
    if not isinstance(guidance, list):
        raise ValueError('航点引导记录必须为列表')
    for row in guidance:
        if (not isinstance(row, dict) or type(row.get('source')) is not int
                or row['source'] not in index or 'position' not in row
                or not 0 <= number(row.get('t'), '航点引导时刻') <= times[-1]):
            raise ValueError('航点引导的单位、坐标或时刻不在本局记录中')
        if row['position'] is not None:
            if not isinstance(row['position'], (list, tuple)) or len(row['position']) != 2:
                raise ValueError('引导航点需要[x,y]米制坐标')
            for value in row['position']:
                number(value, '引导航点坐标')
            number(row.get('altitude'), '引导航点高度')
    meta['guidance'] = sorted(guidance, key=lambda row: row['t'])
    meta['audit']['source'] = str(source.resolve())
    meta['audit']['frames'] = len(times)
    meta['audit']['entities'] = len(index)
    meta['audit']['score_recomputed'] = score
    meta['audit']['damage_source_records'] = len(meta.get('damage', []))
    meta['audit']['assignment_source_records'] = len(meta['assignments'])
    meta['audit']['guidance_source_records'] = len(meta['guidance'])
    meta['audit']['event_counts'] = {kind: sum(event['kind'] == kind for event in events)
                                    for kind in ('dark_discovered', 'high_red_discovered', 'objective_destroyed', 'satellite')}
    return meta, times, states, index, events, summary


def compact_data(directory, source_log=None, step_seconds=None):
    required = [source_log or directory / 'replay.jsonl.gz', directory / 'events.json', directory / 'summary.json']
    if not all(path.is_file() for path in required):
        raise ValueError('紧凑回放目录需要replay.jsonl.gz、events.json和summary.json三个文件')
    iterator = iter(records(required[0]))
    meta = next(iterator)
    if meta.get('kind') != 'meta':
        raise ValueError('紧凑日志首条记录缺少元数据')
    frames, times, ids = [], [], None
    for row in iterator:
        values = np.asarray(row['entities'], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 7 or row.get('kind') != 'state':
            raise ValueError('紧凑日志状态帧形状无效')
        if ids is None:
            ids = values[:, 0].astype(np.int64)
        if not np.array_equal(values[:, 0], ids):
            raise ValueError('紧凑日志各帧实体目录不一致')
        times.append(number(row['t'], '状态时间'))
        frames.append(values[:, 1:])
    if len(times) < 2:
        raise ValueError('紧凑日志至少需要两个完整状态帧')
    meta['audit'] = dict(meta.get('audit', {'warnings': [], 'event_precision': 'native_event'}))
    meta['audit']['format'] = 'compact'
    events = json.loads(required[1].read_text(encoding='utf-8'))
    meta['damage'] = list(meta.get('damage', []))
    if (directory / 'damage.json').is_file():
        for row in json.loads((directory / 'damage.json').read_text(encoding='utf-8')):
            if row['source'] != row['target'] and row['before'] > row['after']:
                meta['damage'].append({'t': (row['time_ms'] - meta['origin_ms']) / 1000,
                                       'source': row['source'], 'target': row['target']})
    summary = json.loads(required[2].read_text(encoding='utf-8'))
    durations = []
    for value, field in ((step_seconds, '指定每步秒数'),
                         (meta.get('step_seconds'), '回放每步秒数'),
                         (meta['audit'].get('step_seconds'), '审计每步秒数')):
        if value is not None:
            value = number(value, field)
            if value <= 0:
                raise ValueError(f'{field}必须大于0')
            durations.append(value)
    if summary.get('steps') is not None:
        end_step = number(summary['steps'], '对局步骤数')
        if end_step <= 0 or not end_step.is_integer():
            raise ValueError('紧凑回放摘要steps必须为正整数')
        durations.append(times[-1] / end_step)
    if not durations:
        raise ValueError('紧凑回放缺少每步秒数和摘要steps；请指定--step-seconds')
    step_seconds = durations[0]
    if step_seconds <= 0 or any(not math.isclose(value, step_seconds, rel_tol=1e-9, abs_tol=1e-9)
                                for value in durations[1:]):
        raise ValueError('紧凑回放的每步秒数、摘要steps和记录时间不一致')
    if summary.get('steps') is None:
        end_step = times[-1] / step_seconds
        if end_step <= 0 or not math.isclose(end_step, round(end_step), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('记录末尾时间与指定步长不能确定整数结束步号')
        summary['steps'] = round(end_step)
    meta['step_seconds'] = step_seconds
    meta['audit']['step_seconds'] = step_seconds
    task_file = directory / 'task_changes.json'
    if task_file.is_file():
        if meta.get('assignments'):
            raise ValueError('回放元数据与task_changes.json重复提供任务分配')
        meta['assignments'] = [{'t': number(row['step'], '任务变更步骤') * step_seconds,
                                'source': row['unit'], 'target': row['to'], 'precision': 'recorded_step'}
                               for row in json.loads(task_file.read_text(encoding='utf-8'))]
    return validate_data(meta, np.asarray(times), np.stack(frames), ids, events, summary, required[0])


def sdk_data(path, step_seconds):
    meta = {'red_area': [], 'map': None, 'blue': None, 'variant': None, 'seed': None}
    frames, clock, events, seen = {}, {}, [], set()
    ids, catalog, origin = None, {}, None
    detection_present, done = False, False
    raw_destructions, raw_damage, pending_damage = [], [], []
    raw_discoveries = []
    native_satellite_steps, satellite_duration_sample = [], None
    pending_assignments, assignments, recorded_targets = [], [], {}
    pending_guidance, guidance, recorded_guidance = [], [], {}
    initial_hp = {}
    memory_bytes = 0
    configured_step_seconds = None

    def add_event(kind, target, step, position, **extra):
        marker = (kind, target)
        if marker not in seen:
            seen.add(marker)
            events.append({'kind': kind, 'target': target, 'step': step, 'position': position, **extra})

    for row in records(path):
        kind = row.get('data_type', row.get('kind'))
        if kind in ('config', 'meta'):
            if 'mapArea' in row:
                meta['map_area'] = row['mapArea']
            if 'redArea' in row:
                meta['red_area'] = row['redArea']['coordinates']
            for field in ('map', 'blue', 'variant', 'seed'):
                if field in row:
                    meta[field] = row[field]
            timing = row.get('imagineProfile', row)
            if 'simStep' in timing:
                configured_step_seconds = number(timing['simStep'], '场景步长') / 1000
            continue
        if kind in ('aiAction', 'ai_action'):
            if row.get('commandType_id') == 3014 and 'intent' in row:
                source = row['executor_id']
                target = row.get('target_id') if row['intent'] == 'track' else None
                if row['intent'] == 'waypoint':
                    position = row.get('target')
                    point = position_xy(position, '引导航点')
                    altitude = number(position.get('z'), '引导航点高度')
                    waypoint = (*point, altitude)
                    if waypoint != recorded_guidance.get(source):
                        pending_guidance.append({'source': source, 'position': point, 'altitude': altitude})
                        recorded_guidance[source] = waypoint
                elif row['intent'] == 'track' and recorded_guidance.get(source) is not None:
                    pending_guidance.append({'source': source, 'position': None})
                    recorded_guidance[source] = None
                if target != recorded_targets.get(source):
                    pending_assignments.append({'source': source, 'target': target})
                    recorded_targets[source] = target
            continue
        if kind == 'event':
            if row.get('type') == 21001:
                body = row['content']['event']
                sensor = body['sensor']
                for target in body['detectTarget']:
                    if sensor['type'] == 9202:
                        native_satellite_steps.append(number(target.get('step', body.get('step')), '天眼探测步号'))
                    target_info = target['entityInfo']
                    if sensor['side'] == 0 and target_info['type'] == 9500:
                        event_kind = 'dark_discovered'
                    elif sensor['type'] == 44000 and target_info['type'] == 21000:
                        event_kind = 'high_red_discovered'
                    else:
                        continue
                    point = target['pos']
                    discovery = {'kind': event_kind, 'target': int(target_info['id']),
                                 'step': number(target.get('step', body.get('step')), '首次探测步号'),
                                 'position': position_xy(point, '探测')}
                    raw_discoveries.append(discovery)
            if row.get('type') == 40000 and 'logicTime' in row:
                raw_destructions.append((int(row['entityId']), number(row['logicTime'], '摧毁时刻')))
            if row.get('type') in (40000, 60002):
                body = row['content']['event']
                source = body.get('destroySrc') if row['type'] == 40000 else body.get('source')
                if source is not None and int(source['id']) != int(row['entityId']):
                    hit = {'source': int(source['id']), 'target': int(row['entityId']),
                           'time_ms': number(row['logicTime'], '伤害时刻'), 'sample_step': None}
                    raw_damage.append(hit)
                    pending_damage.append(hit)
            continue
        if kind in ('summary', 'final_summary'):
            done = row.get('status') == 'complete' or row.get('natural_done') is True
            continue
        if kind != 'state':
            raise ValueError(f'不支持的日志记录类型：{kind}')
        step = number(row['step'], 'step')
        if step < 0 or not step.is_integer():
            raise ValueError('状态步号必须是非负整数')
        assignments.extend({**item, 'step': step} for item in pending_assignments)
        pending_assignments.clear()
        guidance.extend({**item, 'step': step} for item in pending_guidance)
        pending_guidance.clear()
        for hit in pending_damage:
            hit['sample_step'] = step
        pending_damage.clear()
        entities = row['entities']
        if not isinstance(entities, dict) or not entities:
            raise ValueError('SDK状态必须包含完整entities字典')
        current_ids = sorted(map(int, entities))
        if ids is None:
            if step != 0:
                raise ValueError('SDK日志必须从第0步开始，避免漏掉初始计分目标')
            ids = np.asarray(current_ids, dtype=np.int64)
            catalog = {str(key): {'type': int(value['type']), 'side': int(value['side']),
                       'name': str(value.get('nameChn', key)), 'initial_hp': number(value['health'], '初始血量')}
                       for key, value in entities.items()}
            initial_hp = {int(key): value['initial_hp'] for key, value in catalog.items()
                          if value['side'] == 1 and value['type'] in WEIGHTS}
            meta['entities'] = catalog
            origin = row.get('sim_time_ms')
        elif not np.array_equal(ids, current_ids):
            raise ValueError('SDK各帧必须包含相同的完整实体目录')
        if 'map_area' in row:
            meta['map_area'] = row['map_area']
        if 'deployment_area' in row:
            meta['red_area'] = row['deployment_area']['coordinates']
        values = []
        for key in ids:
            entity = entities[str(key)]
            position = entity.get('pos')
            point = position_xy(position, '实体')
            if 'z' not in position:
                raise ValueError('实体pos需要z高度坐标，单位为米')
            if type(entity['isVisible']) is not bool:
                raise ValueError('isVisible必须为布尔值')
            values.append([*point, number(position['z'], '高度'), number(entity['health'], '血量'),
                           bool(entity['isVisible']), number(entity['stage'], '阶段')])
            if 'detectInfo' in entity:
                detection_present = True
                tracks = entity['detectInfo']
                if not isinstance(tracks, dict):
                    raise ValueError('detectInfo必须为字典')
                if entity['type'] == 9202 and tracks:
                    add_event('satellite', 0, step, [], duration=None,
                              precision='state_sample', confirmation='satellite_detection_records')
                for target, track in tracks.items():
                    target = int(target)
                    if str(target) not in catalog:
                        raise ValueError('探测目标不在完整实体目录中')
                    target_type = catalog[str(target)]['type']
                    target_position = (track.get('pos') if isinstance(track, dict) else None)
                    if target_position is None:
                        target_position = entities[str(target)]['pos']
                    point = position_xy(target_position, '目标')
                    if entity['side'] == 0 and target_type == 9500:
                        add_event('dark_discovered', target, step, point, precision='state_sample')
                    elif entity['type'] == 44000 and target_type == 21000:
                        add_event('high_red_discovered', target, step, point, precision='state_sample')
        array = np.asarray(values, dtype=np.float64)
        if step not in frames:
            memory_bytes += array.nbytes
        if memory_bytes > 512 * 1024**2:
            raise ValueError('状态帧超过512MiB内存限制，请使用保留事件的紧凑日志')
        frames[step] = array
        if 'sim_time_ms' in row:
            clock[step] = number(row['sim_time_ms'], '仿真时刻')
        if row.get('is_using_satellite') is True:
            if 'satellite_remaining_seconds' not in row:
                raise ValueError('天眼成功状态需要satellite_remaining_seconds才能显示真实持续时间')
            remaining = number(row['satellite_remaining_seconds'], '天眼剩余时间')
            if remaining <= 0:
                raise ValueError('天眼成功事件缺少有效持续时间')
            if satellite_duration_sample is None:
                satellite_duration_sample = (step, remaining)
            add_event('satellite', 0, step, [], duration=remaining)
    steps = sorted(frames)
    if len(steps) < 2:
        raise ValueError('日志至少需要两个完整状态帧')
    if step_seconds is None and configured_step_seconds is not None:
        step_seconds = configured_step_seconds
    if step_seconds is None:
        if len(clock) != len(steps):
            raise ValueError('日志缺少仿真时钟；请按场景实际步长指定--step-seconds，例如1')
        step_seconds = (clock[steps[-1]] - clock[steps[0]]) / (steps[-1] - steps[0]) / 1000
    step_seconds = number(step_seconds, '每步秒数')
    if step_seconds <= 0:
        raise ValueError('仿真步长必须大于0')
    if configured_step_seconds is not None and abs(step_seconds - configured_step_seconds) > 1e-9:
        raise ValueError('指定每步秒数与场景步长不一致')
    times = np.asarray(steps) * step_seconds
    if clock:
        if origin is None:
            origin = clock[steps[0]]
        if any(abs((value - origin) / 1000 - step * step_seconds) > 1e-6 for step, value in clock.items()):
            raise ValueError('记录的仿真时钟与指定步长不一致')
    states = np.stack([frames[step] for step in steps])
    index = {int(key): i for i, key in enumerate(ids)}
    exact_destructions = {key: (stamp - origin) / 1000 for key, stamp in raw_destructions
                          if origin is not None and key in initial_hp}
    objectives = []
    for key, hp in initial_hp.items():
        series = states[:, index[key], 3]
        dead_frames = np.flatnonzero(series <= 0)
        destroyed_step = steps[int(dead_frames[0])] if len(dead_frames) else None
        entity_type = catalog[str(key)]['type']
        if destroyed_step is not None:
            position = states[int(dead_frames[0]), index[key], :2].tolist()
            add_event('objective_destroyed', key, destroyed_step, position, type=entity_type, precision='state_sample')
        objectives.append({'id': key, 'type': entity_type, 'initial_hp': hp, 'final_hp': float(series[-1]),
                           'weight': WEIGHTS[entity_type], 'destroyed_step': destroyed_step})
    for event in events:
        event['t'] = event.pop('step') * step_seconds
        if event['kind'] == 'objective_destroyed' and event['target'] in exact_destructions:
            event['t'] = exact_destructions[event['target']]
            event['precision'] = 'native_event'
    satellite = next((event for event in events if event['kind'] == 'satellite'), None)
    if satellite is not None and satellite_duration_sample is not None:
        sample_step, remaining = satellite_duration_sample
        if satellite['t'] == sample_step * step_seconds:
            satellite['duration'] = remaining
    if native_satellite_steps:
        stamp = min(native_satellite_steps) * step_seconds
        confirmation = 'native_satellite_detection_records'
        if satellite is None:
            satellite = {'kind': 'satellite', 'target': 0, 'position': [], 'duration': None,
                         't': stamp, 'precision': 'native_event', 'confirmation': confirmation}
            events.append(satellite)
        elif stamp < satellite['t'] or (stamp == satellite['t'] and satellite_duration_sample is None):
            # Keep the observed end of a known active window when its first detection precedes the sample.
            known_end = satellite['t'] + satellite['duration'] if satellite['duration'] is not None else None
            satellite.update(t=stamp, precision='native_event', confirmation=confirmation)
            if known_end is not None:
                satellite['duration'] = known_end - stamp
    first_discoveries = {}
    for discovery in raw_discoveries:
        marker = discovery['kind'], discovery['target']
        previous = first_discoveries.get(marker)
        if previous is None or discovery['step'] < previous['step']:
            first_discoveries[marker] = discovery
    for marker, discovery in first_discoveries.items():
        events = [event for event in events if (event['kind'], event['target']) != marker]
        stamp = discovery['step'] * step_seconds
        events.append({'kind': marker[0], 'target': marker[1], 't': stamp,
                       'position': discovery['position'],
                       'precision': 'native_event'})
        detection_present = True
    warnings = []
    if not detection_present:
        warnings.append('缺少探测记录，不能显示首次发现效果；灰色暗巢只表示发现状态未知')
    if not any(row['kind'] == 'satellite' for row in events):
        warnings.append('没有天眼成功记录，本片不生成天眼启动效果；动作请求不作为成功确认')
    if any(row.get('precision') == 'state_sample' for row in events):
        warnings.append('部分事件按首次记录到的状态帧显示，时间精度受日志采样间隔限制')
    if assignments or guidance:
        warnings.append('AI指令未提供独立时钟，目标分配与航点引导按其后首个采样帧显示')
    if pending_assignments or pending_guidance:
        warnings.append('日志末尾的引导指令缺少后续状态帧，回放未推测其生效时刻')
    if not done:
        warnings.append('日志未给出正常结束确认，片尾标为日志结束')
    if 'map_area' not in meta:
        raise ValueError('日志缺少地图边界，无法准确绘制场景')
    if any(row['kind'] == 'satellite' and row['duration'] is None for row in events):
        warnings.append('天眼实际探测记录已确认启用；未记录持续时间，仅显示确认时刻的提示效果')
    meta['damage'] = [{'source': row['source'], 'target': row['target'],
                       't': (row['time_ms'] - origin) / 1000 if origin is not None else row['sample_step'] * step_seconds}
                      for row in raw_damage if origin is not None or row['sample_step'] is not None]
    meta['assignments'] = [{**item, 't': item['step'] * step_seconds, 'precision': 'state_sample'}
                           for item in assignments]
    meta['guidance'] = [{**item, 't': item['step'] * step_seconds, 'precision': 'state_sample'}
                       for item in guidance]
    if pending_damage:
        warnings.append('日志末尾的伤害事件缺少后续状态帧，回放未推测这些事件的位置')
    meta['step_seconds'] = step_seconds
    if origin is not None:
        meta['origin_ms'] = origin
    meta['audit'] = {'format': 'sdk_jsonl', 'warnings': warnings, 'step_seconds': step_seconds,
                     'max_sample_interval_seconds': float(np.max(np.diff(times))),
                     'detection_records_present': detection_present, 'event_precision': 'reported_per_event',
                     'native_discovery_events': len(raw_discoveries)}
    if satellite is not None and satellite.get('confirmation') == 'native_satellite_detection_records':
        meta['audit']['satellite_time_basis'] = 'first_successful_detection_not_exact_activation'
    summary = {'score': None, 'steps': steps[-1], 'natural_done': done, 'objectives': objectives}
    return validate_data(meta, times, states, ids, events, summary, path)


def load(path, step_seconds=None):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        data = compact_data(path, step_seconds=step_seconds)
    else:
        with_iterator = iter(records(path))
        first = next(with_iterator)
        with_iterator.close()
        compact = first.get('kind') == 'meta' and {'map', 'red_area', 'entities'} <= first.keys()
        data = compact_data(path.parent, path, step_seconds=step_seconds) if compact else sdk_data(path, step_seconds)
    meta = data[0]
    for field in ('map', 'blue'):
        value = meta.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()
                                  or len(value) > 60 or any(ord(char) < 32 for char in value)):
            raise ValueError(f'日志中的{field}须为60字以内的非空单行文字')
    seed = meta.get('seed')
    if seed is not None and type(seed) is not int:
        raise ValueError('日志中的seed须为整数')
    meta['red_name'] = '示例策略'
    meta['scenario_name'] = f"场景 {meta['map']}" if meta.get('map') is not None else ''
    meta['blue_name'] = f"虫族策略 {blue_display_id(meta['blue'])}" if meta.get('blue') is not None else ''
    meta['title'] = '对局回放'
    details = [value for value in (meta['scenario_name'], meta['blue_name']) if value]
    if seed is not None:
        details.append(f'种子 {seed}')
    meta['subtitle'] = '  /  '.join(details)
    meta['audit']['display_metadata'] = {'source': 'replay_metadata',
                                       **{field: meta[field] for field in ('map', 'blue', 'seed')
                                          if meta.get(field) is not None},
                                       'human_strategy_label': '示例策略'}
    return data


def save_compact(data, directory):
    meta, times, states, index, events, summary = data
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {**meta, 'kind': 'meta'}
    with gzip.open(directory / 'replay.jsonl.gz', 'wt', encoding='utf-8', compresslevel=6) as stream:
        stream.write(json.dumps(metadata, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
        identifiers = np.array(list(index), dtype=np.float64).reshape(-1, 1)
        for t, frame in zip(times, states):
            row = {'kind': 'state', 't': float(t), 'entities': np.concatenate((identifiers, frame), axis=1).tolist()}
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    for name, value in (('events', events), ('summary', summary), ('audit', meta['audit'])):
        (directory / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
