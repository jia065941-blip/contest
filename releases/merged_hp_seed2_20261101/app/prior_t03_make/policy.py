"""Trusted compiler runtime. SPEC is embedded at export; no offline evidence is read."""
import copy
import math


def step_skill(state, memory, parameters):
    spec = specification()
    memory = copy.deepcopy(memory)
    transitions = []
    feedback = state.get('feedback', {})
    for event, task in list(memory.items()):
        info = feedback.get(event, {})
        if not info:
            continue  # Missing evidence is not an acknowledgement or successful exit.
        guards = {'all_acknowledged': bool(info.get('all_acknowledged')),
                  'any_unavailable': bool(info.get('any_unavailable')),
                  'all_exited': bool(info.get('all_exited')),
                  'ack_timeout': float(info.get('elapsed_s', 0)) >= parameters['ack_timeout_s']}
        for edge in spec['transitions']:
            if edge['from_phase'] == task['phase'] and guards.get(edge['condition'], False):
                task['phase'] = edge['to_phase']
                if edge['effect'] == 'release_observer':
                    task['observer_roles'] = []
                if edge['effect'] == 'release_all':
                    task['observer_roles'] = []
                    task['receiver_roles'] = []
                transitions.append({'event_ref': event, 'phase': task['phase'], 'condition': edge['condition']})
                break  # One causal transition per observation, never an invented sequence.
    held = {r for task in memory.values() if task['phase'] != 'exit'
            for r in task['observer_roles'] + task['receiver_roles']}
    used = set()
    replacements = []
    deferred = []
    events = list(state.get('events', []))
    if spec['binding']['event_schedule'] == 'acknowledged_waves':
        urgency = {}
        for event in events:
            feasible = [r for r in state.get('roles', []) if r.get('eligible') and event['ref'] in r.get('distances_m', {}) and not any(e != event['ref'] for e in r.get('original_event_refs', []))]
            if spec['binding']['event_priority'] == 'least_slack_first':
                urgency[event['ref']] = min((r['remaining_time_s'] - r['distances_m'][event['ref']] / 100 for r in feasible), default=0)
            else:
                urgency[event['ref']] = min((r['distances_m'][event['ref']] for r in feasible), default=0)
        events.sort(key=lambda e: urgency[e['ref']])
    for event in events:
        ref = event['ref']
        if ref in memory:  # A delivery is processed once; task memory is episode-local.
            continue
        if (spec['binding']['event_schedule'] == 'acknowledged_waves' and
            any(t['phase'] == 'await_ack' for t in memory.values()) and
            event.get('queue_age_s', 0) < parameters['ack_timeout_s']):
            deferred.append(ref)
            continue
        required = len(set(event['original_roles']))
        eligible = []
        for role in state.get('roles', []):
            r = role['ref']; distance = role.get('distances_m', {}).get(ref)
            remaining = role.get('remaining_time_s', 0)
            if (r in used or r in held or not role.get('eligible') or
                role.get('task_kind') not in ('original', 'search') or distance is None or
                not math.isfinite(distance) or not math.isfinite(remaining) or distance < 0 or
                remaining < 60 or distance > min(90000, 100 * remaining) or
                any(other != ref for other in role.get('original_event_refs', []))):
                continue
            eligible.append(role)
        if not required or len(eligible) < required:
            continue
        searchers = [r for r in eligible if r['task_kind'] == 'search' and not r['original_event_refs']]
        observers = []
        if spec['binding']['observer'] == 'preserve_searcher' and searchers:
            # Reserve a currently searching participant. It is NOT claimed to be the detector.
            observers = [max(searchers, key=lambda r: r['remaining_time_s'])['ref']]
        pool = [r for r in eligible if r['ref'] not in observers]
        if len(pool) < required:
            continue  # Full V0 group size is frozen; no implicit reduction.
        mode = spec['binding']['receivers']
        if mode == 'task_complement':
            pool.sort(key=lambda r: (r['task_kind'] != 'original', r['distances_m'][ref]))
        elif mode == 'retain_cohort':
            pool.sort(key=lambda r: (r['ref'] not in event['original_roles'], r['distances_m'][ref]))
        elif mode == 'search_first':
            pool.sort(key=lambda r: (r['task_kind'] != 'search', r['distances_m'][ref]))
        else:
            pool.sort(key=lambda r: r['distances_m'][ref])
        selected = [r['ref'] for r in pool[:required]]
        memory[ref] = dict(phase='await_ack', receiver_roles=selected, observer_roles=observers)
        replacements.append(dict(event_ref=ref, roles=selected))
        used.update(selected); held.update(selected + observers)
        transitions.append(dict(event_ref=ref, phase='await_ack', condition='legal_discovery'))
    return dict(replacements=replacements, deferred_events=deferred, memory=memory, transitions=transitions)


def select_handoff(state, parameters):
    return {'replacements': step_skill(state, {}, parameters)['replacements']}


def specification():
    return {'name': 'ack_wave_task_complement_handoff', 'hypothesis': '按原生确认逐组启动交接、保留一个当前搜索者、接收者优先补充原任务成员，可能在不放弃搜索覆盖的前提下改善地图等权官方总分；该配合设计为待评估假设，录像未证明其收益。', 'relations': [{'id': 'division', 'predicate': 'prior_task_partition', 'kind': 'role_division', 'statement': '接收者在交接前已有合法的原任务或搜索任务记录。', 'status': 'observed', 'evidence': ['J001:prior_task_partition', 'J003:prior_task_partition', 'J005:prior_task_partition', 'J006:prior_task_partition'], 'counterevidence': [], 'uncertainty': '观察到任务划分不证明保留搜索者会提高得分；J004 片段显示存在无接收者的反例，但该片段无对应 fact id。'}, {'id': 'order', 'predicate': 'legal_discovery_before_retarget', 'kind': 'precedence', 'statement': '合法发现与生成的重定向同帧或更早发生。', 'status': 'observed', 'evidence': ['J001:legal_discovery_before_retarget', 'J003:legal_discovery_before_retarget', 'J005:legal_discovery_before_retarget', 'J006:legal_discovery_before_retarget'], 'counterevidence': [], 'uncertainty': '同帧事件不能确立严格子步顺序。'}, {'id': 'ack', 'predicate': 'native_acknowledgement', 'kind': 'handoff_condition', 'statement': '原生重定向确认可作为执行反馈使用。', 'status': 'observed', 'evidence': ['J001:native_acknowledgement', 'J003:native_acknowledgement', 'J005:native_acknowledgement', 'J006:native_acknowledgement'], 'counterevidence': [], 'uncertainty': '确认仅代表命令返回，不意味着目标受损或任务成功。'}, {'id': 'exit', 'predicate': 'task_or_observation_exit', 'kind': 'exit_condition', 'statement': '观察到的任务重分配或自身观察不可用会终止本地跟踪。', 'status': 'observed', 'evidence': ['J001:task_or_observation_exit', 'J003:task_or_observation_exit', 'J005:task_or_observation_exit', 'J006:task_or_observation_exit'], 'counterevidence': [], 'uncertainty': '观察丢失不等于目标被摧毁；J002 还包含保守寿命或回合上限退出。'}, {'id': 'detector', 'predicate': 'unknown_relation', 'kind': 'role_division', 'statement': '上报的检测者是一个可控的 L 搜索参与者。', 'status': 'unknown', 'evidence': [], 'counterevidence': [], 'uncertainty': '上报者可能不在自身观察中；不得把检测者身份绑定到接收者。'}, {'id': 'wait_dep', 'predicate': 'unknown_relation', 'kind': 'handoff_condition', 'statement': '收到确认意味着另一组必须等待。', 'status': 'unknown', 'evidence': [], 'counterevidence': [], 'uncertainty': '录像未证实该因果等待；新的等待依赖仅写入执行设计与假设，不冒充已证实关系。'}], 'binding': {'event_schedule': 'acknowledged_waves', 'event_priority': 'least_slack_first', 'observer': 'preserve_searcher', 'receivers': 'task_complement', 'relation_ids': ['division', 'order', 'ack', 'exit']}, 'transitions': [{'from_phase': 'await_ack', 'to_phase': 'executing', 'condition': 'all_acknowledged', 'effect': 'keep_bindings', 'relation_ids': ['ack']}, {'from_phase': 'await_ack', 'to_phase': 'exit', 'condition': 'ack_timeout', 'effect': 'release_all', 'relation_ids': ['ack']}, {'from_phase': 'executing', 'to_phase': 'exit', 'condition': 'all_exited', 'effect': 'release_all', 'relation_ids': ['exit']}], 'parameters': {'ack_timeout_s': 5}}
