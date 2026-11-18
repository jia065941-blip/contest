"""All skills see one legal snapshot; proposals own nothing until atomic selection."""
import copy
import math
from harness.skill_discovery.common import digest
from .program import validate, family_id


def validate_state(state):
    if not isinstance(state,dict) or set(state)-{'step','dt_s','roles','targets','current_arrangement'}:
        raise ValueError('Only whitelisted legal state can enter a skill or selector')
    if type(state.get('step')) is not int or state['step']<0 or not isinstance(state.get('roles'),list) or not isinstance(state.get('targets'),list):
        raise ValueError('Invalid legal state schema')
    if type(state.get('dt_s')) not in (int,float) or not math.isfinite(state['dt_s']) or state['dt_s']<=0:raise ValueError('Invalid step duration')
    role_fields={'ref','type','position','eligible','remaining_s','nominal_speed_mps','current_target','native_ack_step','native_ack_target'}
    target_fields={'ref','type','position','reporter_refs'}
    refs=set()
    for role in state['roles']:
        if not role_fields<=set(role) or set(role)-role_fields-{'current_goal','task_kind'} or role['type'] not in ('H','M','L') or role['ref'] in refs or type(role['eligible']) is not bool:
            raise ValueError('Invalid or duplicated role projection')
        refs.add(role['ref'])
        if any(type(role[k]) not in (int,float) or not math.isfinite(role[k]) or role[k]<0 for k in ('remaining_s','nominal_speed_mps')) or role['nominal_speed_mps']==0:
            raise ValueError('Invalid role timing or nominal speed')
    refs=set()
    for target in state['targets']:
        if set(target)!=target_fields or target['type'] not in (9400,9500,9600) or target['ref'] in refs:
            raise ValueError('Illegal or duplicated target projection')
        refs.add(target['ref'])
    for item in state['roles']+state['targets']:
        xy=item['position']
        if not isinstance(xy,list) or len(xy)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in xy):
            raise ValueError('Invalid legal coordinates')


def describe(memory):
    return dict(current_wave=memory.get('current_wave',0),waves=[[
        dict(skill_id=p['proposal']['skill_id'],target_ref=p['proposal']['target_ref'],bindings=p['bindings'],
             stages={k:v['phase'] for k,v in p['stages'].items()}) for p in wave] for wave in memory.get('waves',[])])


def distance(a,b):
    lat1,lat2=map(math.radians,[a[1],b[1]])
    q=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(math.radians(b[0]-a[0])/2)**2
    return 12742000*math.asin(min(1.,math.sqrt(max(0.,q))))


def propose(program,state,event_kinds=None):
    """Pure proposal construction: no assignment, leases, queues, calls or task mutation."""
    validate_state(state);p=validate(program);fingerprint=digest(state);offers=[]
    if event_kinds is not None and not set(p['start_events'])&set(event_kinds):return []
    for target in state['targets']:
        if target['type']!=p['target']['type']:continue
        options={}
        for slot,binding in p['bindings'].items():
            pool=[]
            for role in state['roles']:
                if not role['eligible'] or role['type']!=binding['type']:continue
                if binding['source']=='delivered_reporter' and role['ref'] not in target.get('reporter_refs',[]):continue
                eta=distance(role['position'],target['position'])/role['nominal_speed_mps']
                if role['remaining_s']<p['parameters']['min_remaining_s'] or eta>min(role['remaining_s'],p['parameters']['max_eta_s']):continue
                pool.append((role['current_target']!=target['ref'] if binding['rank']=='preserve_current' else False,eta,role['ref']))
            pool.sort();options[slot]=[r for _,_,r in pool[:max(16,binding['count'])]]
            if len(options[slot])<binding['count']:break
        else:
            role_rows={r['ref']:r for r in state['roles']}
            etas={ref:distance(role_rows[ref]['position'],target['position'])/role_rows[ref]['nominal_speed_mps'] for values in options.values() for ref in values}
            centers=sorted(set(etas.values())) if any(b['rank']=='arrival_compact' for b in p['bindings'].values()) else [None]
            alternatives=[]
            for center in centers:
                bound={};used=set()
                for slot,values in options.items():
                    order=[r for r in values if r not in used]
                    if center is not None and p['bindings'][slot]['rank']=='arrival_compact':order.sort(key=lambda r:(abs(etas[r]-center),etas[r],r))
                    count=p['bindings'][slot]['count']
                    if len(order)<count:break
                    bound[slot]=order[:count];used.update(bound[slot])
                else:
                    times=[etas[r] for values in bound.values() for r in values]
                    alternatives.append((sum(times)+max(times)-min(times),bound))
            if not alternatives:continue
            refs=min(alternatives,key=lambda x:x[0])[1]
            if len({r for values in refs.values() for r in values})!=sum(map(len,refs.values())):continue
            offer=dict(skill_id='skill_'+digest(p)[:20],family_id=family_id(p),
                state_hash=fingerprint,target_ref=target['ref'],binding_options=options,default_bindings=refs,program=p)
            offer['proposal_id']='proposal_'+digest(offer)[:20];offers.append(offer)
    if digest(state)!=fingerprint:raise AssertionError('A proposer mutated the common legal snapshot')
    return offers


def validate_selection(choice,offers,state):
    validate_state(state)
    if set(choice)!={'waves','reason'} or not isinstance(choice['reason'],str) or not isinstance(choice['waves'],list):
        raise ValueError('Selector must return waves and reason')
    by_id={p['proposal_id']:p for p in offers};selected=[];seen=set();state_hash=digest(state)
    if len(choice['waves'])>12:raise ValueError('Too many selected phases')
    for wave in choice['waves']:
        if not isinstance(wave,list) or not wave:raise ValueError('Empty or invalid execution wave')
        owners=set();items=[]
        for item in wave:
            if set(item)!={'proposal_id','bindings'} or item['proposal_id'] not in by_id or item['proposal_id'] in seen:
                raise ValueError('Unknown or repeated proposal')
            proposal=by_id[item['proposal_id']]
            if proposal['state_hash']!=state_hash:raise ValueError('Stale legal snapshot')
            bindings=item['bindings']
            if set(bindings)!=set(proposal['program']['bindings']):raise ValueError('Incomplete role binding')
            participants=[]
            for slot,roles in bindings.items():
                if not isinstance(roles,list) or len(roles)!=proposal['program']['bindings'][slot]['count']:
                    raise ValueError('Participant count differs from selected skill')
                if any(r not in proposal['binding_options'][slot] for r in roles):raise ValueError('Illegal participant binding')
                participants.extend(roles)
            if len(set(participants))!=len(participants) or set(participants)&owners:
                raise ValueError('Parallel skills cannot own the same participant')
            owners.update(participants);seen.add(item['proposal_id'])
            items.append(dict(proposal=copy.deepcopy(proposal),bindings=copy.deepcopy(bindings),
                stages={s['id']:dict(phase='ready',issued_step=None,issued_targets={}) for s in proposal['program']['stages']}))
        selected.append(items)
    return selected


def commit(choice,offers,state,current):
    """Validate complete binding/conflicts before replacing any current arrangement."""
    selected=validate_selection(choice,offers,state)
    if not selected:return copy.deepcopy(current)  # Explicit keep preserves the active plan.
    return dict(waves=selected,current_wave=0,committed_state_hash=digest(state))


def advance(memory,state):
    """Only selected stages can issue commands; future waves do not occupy members."""
    memory=copy.deepcopy(memory);actions=[];events=[];owners=set()
    wave_index=memory.get('current_wave',0)
    if wave_index>=len(memory.get('waves',[])):return memory,actions,events,owners
    targets={t['ref']:t for t in state['targets']};roles={r['ref']:r for r in state['roles']}
    for plan in memory['waves'][wave_index]:
        p=plan['proposal']['program'];target=targets.get(plan['proposal']['target_ref'])
        if target is None:
            for task in plan['stages'].values():
                if task['phase'] not in ('exit','failed'):task['phase']='failed'
            events.append(dict(kind='legal_target_unavailable',target_ref=plan['proposal']['target_ref']))
            continue
        for stage in p['stages']:
            task=plan['stages'][stage['id']]
            if task['phase'] in ('exit','failed'):continue
            if any(plan['stages'][before]['phase']=='failed' for before in stage['after']):
                task['phase']='failed';continue
            if any(plan['stages'][before]['phase']!='exit' for before in stage['after']):continue
            participants=[r for a in stage['actions'] for r in plan['bindings'][a['slot']]]
            if any(r not in roles or not roles[r]['eligible'] for r in participants):
                if task['phase']!='ready' and stage['complete_when']=='all_task_exit' and all(r not in roles or not roles[r]['eligible'] for r in participants):
                    task['phase']='exit';events.append(dict(kind='observed_task_exit',stage=stage['id'],objective_success='unknown'));continue
                task['phase']='failed';events.append(dict(kind='own_unavailable',stage=stage['id']));continue
            if task['phase']!='ready':
                elapsed=(state['step']-task['issued_step'])*state['dt_s']
                changed=any(roles[r]['current_target']!=task['issued_targets'].get(r) for r in participants if r in task['issued_targets'])
                retarget_roles=[r for a in stage['actions'] if a['operator']=='retarget' for r in plan['bindings'][a['slot']]]
                acknowledged=bool(retarget_roles) and all(roles[r].get('native_ack_step',-1)>=task['issued_step'] and
                       roles[r].get('native_ack_target')==task['issued_targets'][r] for r in retarget_roles)
                all_changed=bool(task['issued_targets']) and all(roles[r]['current_target']!=task['issued_targets'].get(r) for r in participants)
                finished=(stage['complete_when']=='all_native_ack' and acknowledged) or \
                         (stage['complete_when']=='all_task_exit' and all_changed) or \
                         (stage['complete_when']=='legal_report_delivered' and bool(target.get('reporter_refs')))
                timed_out=elapsed>=p['parameters']['phase_timeout_s'] or (task['phase']=='await_ack' and elapsed>=p['parameters']['ack_timeout_s'])
                if finished or changed or timed_out:
                    task['phase']='exit' if finished else 'failed'
                    events.append(dict(kind='observed_transition' if finished else 'own_binding_changed' if changed else 'phase_timeout',stage=stage['id']))
                    continue
            if task['phase']=='ready':
                proposed=[]
                for action in stage['actions']:
                    for ref in plan['bindings'][action['slot']]:
                        if ref in owners:raise ValueError('Selected active stages conflict')
                        if action['operator']=='retarget':
                            eta=distance(roles[ref]['position'],target['position'])/roles[ref]['nominal_speed_mps']
                            if eta>min(roles[ref]['remaining_s'],p['parameters']['max_eta_s']):break
                            proposed.append(dict(role_ref=ref,operator='retarget',target_ref=target['ref'],stage=stage['id'],skill_id=plan['proposal']['skill_id']))
                    else:continue
                    break
                else:
                    actions.extend(proposed);task.update(phase='await_ack' if proposed else 'executing',issued_step=state['step'],
                       issued_targets={a['role_ref']:a['target_ref'] for a in proposed})
                    owners.update(participants);continue
                task['phase']='failed';events.append(dict(kind='selected_task_infeasible',stage=stage['id']))
                continue  # Failed feasibility leaves the current real tasks unchanged.
            owners.update(participants)
    if all(all(s['phase'] in ('exit','failed') for s in plan['stages'].values()) for plan in memory['waves'][wave_index]):
        failed=any(s['phase']=='failed' for plan in memory['waves'][wave_index] for s in plan['stages'].values())
        memory['current_wave']=len(memory['waves']) if failed else wave_index+1
        events.append(dict(kind='phase_failed' if failed else 'stage_transition',wave=wave_index))
    return memory,actions,events,owners
