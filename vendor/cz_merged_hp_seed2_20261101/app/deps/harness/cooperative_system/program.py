"""Evidence-backed typed stage graphs; no inherited L-handoff implementation template."""
import copy
from harness.skill_discovery.common import digest, finite

TYPES={'H':21000,'M':21001,'L':21002}
INTERFACES={'H':['retarget'],'M':['retarget'],'L':['retarget']}
OPERATORS={'retarget','hold'}
EVENTS={'new_legal_target','became_executable','task_exit','task_changed','stage_transition','selection_review'}


def validate(program,facts=None,interfaces=None):
    p=copy.deepcopy(program);interfaces=INTERFACES if interfaces is None else interfaces
    fields={'schema_version','name','objective','start_events','bindings','target','stages','exit','parameters','evidence_refs','hypothesis'}
    if not isinstance(p,dict) or set(p)!=fields or p['schema_version']!=1:raise ValueError('Unsupported typed program schema')
    if not all(isinstance(p[k],str) and p[k].strip() for k in ('name','objective','hypothesis')):
        raise ValueError('Explicit objective and design hypothesis required')
    if not isinstance(p['start_events'],list) or not p['start_events'] or not set(p['start_events'])<=EVENTS:raise ValueError('Unsupported start events')
    if set(p['target'])!={'scope','type'} or p['target']['scope']!='legally_known' or p['target']['type'] not in (9400,9500,9600):
        raise ValueError('Only currently legal target selectors are executable')
    if not isinstance(p['bindings'],dict) or not 1<=len(p['bindings'])<=12:raise ValueError('Invalid role slots')
    for slot,binding in p['bindings'].items():
        if (not isinstance(slot,str) or set(binding)!={'type','count','source','rank'} or binding['type'] not in TYPES or
            type(binding['count']) is not int or not 1<=binding['count']<=64 or
            binding['source'] not in ('eligible','delivered_reporter') or binding['rank'] not in ('nearest','arrival_compact','preserve_current')):
            raise ValueError('Invalid typed role binding')
    if not isinstance(p['stages'],list) or not 1<=len(p['stages'])<=12:raise ValueError('Invalid stages')
    previous=set()
    for stage in p['stages']:
        if (not isinstance(stage,dict) or set(stage)!={'id','after','actions','complete_when'} or stage['id'] in previous or
            not isinstance(stage['after'],list) or not isinstance(stage['actions'],list) or
            not isinstance(stage['id'],str) or not set(stage['after'])<=previous or
            stage['complete_when'] not in ('all_native_ack','all_task_exit','legal_report_delivered')):
            raise ValueError('Stage graph must be acyclic and use observable transition guards')
        if not stage['actions']:raise ValueError('Empty executable stage')
        used=set()
        for action in stage['actions']:
            if (set(action)!={'slot','operator'} or action['slot'] not in p['bindings'] or
                action['slot'] in used or action['operator'] not in OPERATORS):raise ValueError('Invalid stage action')
            used.add(action['slot']);kind=p['bindings'][action['slot']]['type']
            if action['operator']!='hold' and action['operator'] not in interfaces.get(kind,[]):
                raise ValueError('High-level operator is not authorized for this role type')
        if stage['complete_when']=='all_native_ack' and all(a['operator']=='hold' for a in stage['actions']):
            raise ValueError('Hold cannot fabricate a native acknowledgement')
        previous.add(stage['id'])
    ancestors={}
    for stage in p['stages']:
        ancestors[stage['id']]=set(stage['after'])|set().union(*(ancestors[x] for x in stage['after']))
    for i,left in enumerate(p['stages']):
        for right in p['stages'][i+1:]:
            overlap={a['slot'] for a in left['actions']}&{a['slot'] for a in right['actions']}
            if overlap and left['id'] not in ancestors[right['id']]:
                raise ValueError('A participant slot cannot execute independent parallel stages')
    if p['exit']!=['own_unavailable','phase_timeout','observed_task_exit','all_stages_complete']:
        raise ValueError('Unbounded or hidden-state exit')
    if (set(p['parameters'])!={'ack_timeout_s','phase_timeout_s','min_remaining_s','max_eta_s'} or
        any(not finite(v) for v in p['parameters'].values()) or
        not 0<p['parameters']['ack_timeout_s']<=60 or not 0<p['parameters']['phase_timeout_s']<=3600 or
        not 0<=p['parameters']['min_remaining_s']<=3600 or not 0<p['parameters']['max_eta_s']<=3600):
        raise ValueError('Finite bounded phase parameters required')
    if not isinstance(p['evidence_refs'],list) or len(p['evidence_refs'])!=len(set(p['evidence_refs'])):
        raise ValueError('Invalid source references')
    if facts is not None:
        if any(r not in facts for r in p['evidence_refs']):raise ValueError('Invented source relation')
        selected=[facts[r] for r in p['evidence_refs']]
        if len({r['episode_id'] for r in selected})<2:raise ValueError('A discovery needs two independent source episodes')
        declared={v['type'] for v in p['bindings'].values()}
        if not any(declared<=set(f['role_types']) for f in selected):raise ValueError('Typed cooperation is not represented in source evidence')
        for kind in declared:
            if len({f['episode_id'] for f in selected if kind in f['role_types'] and f['target_type']==p['target']['type']})<2:
                raise ValueError('Every bound role type needs repeated source evidence')
        for binding in p['bindings'].values():
            if binding['source']=='delivered_reporter' and len({f['episode_id'] for f in selected if
                f['relation']=='identified_reporter_before_receiver_assignment' and f.get('reporter_type')==binding['type']})<2:
                raise ValueError('Reporter identity is unknown; cannot fabricate an observer role')
    return p


def family_id(program):
    p=validate(program)
    # Ranking, cardinality, time thresholds, names and evidence aliases do not create mechanisms.
    roles={k:digest({n:v[n] for n in ('type','source')}) for k,v in p['bindings'].items()}
    phases={s['id']:digest(s['complete_when']) for s in p['stages']}
    # Label-independent graph refinement also normalizes reordered independent stages.
    for _ in range(len(roles)+len(phases)):
        new_roles={k:digest([roles[k],sorted((phases[s['id']],a['operator']) for s in p['stages'] for a in s['actions'] if a['slot']==k)]) for k in roles}
        new_phases={s['id']:digest([phases[s['id']],sorted(phases[x] for x in s['after']),
            sorted(phases[t['id']] for t in p['stages'] if s['id'] in t['after']),
            sorted((roles[a['slot']],a['operator']) for a in s['actions'])]) for s in p['stages']}
        roles,phases=new_roles,new_phases
    graph=dict(target=p['target'],bindings=sorted(roles.values()),stages=sorted(phases.values()),exit=p['exit'])
    return 'cross_family_'+digest(graph)[:16]


def joint_program(role_types,target_type,evidence_refs):
    bindings={kind.lower():dict(type=kind,count=1,source='eligible',rank='arrival_compact') for kind in sorted(role_types)}
    return dict(schema_version=1,name='joint_'+''.join(sorted(role_types)),objective='joint_legal_target_assignment',
        start_events=['became_executable','new_legal_target','task_exit','task_changed','selection_review'],bindings=bindings,
        target=dict(scope='legally_known',type=target_type),
        stages=[dict(id='joint_dispatch',after=[],actions=[dict(slot=k,operator='retarget') for k in bindings],complete_when='all_native_ack')],
        exit=['own_unavailable','phase_timeout','observed_task_exit','all_stages_complete'],
        parameters=dict(ack_timeout_s=5.,phase_timeout_s=120.,min_remaining_s=60.,max_eta_s=900.),
        evidence_refs=evidence_refs,hypothesis='Repeated explicit shared-target assignment supplies roles and objective; regrouping and native-ack phase control are testable execution designs, not established causal benefit.')
