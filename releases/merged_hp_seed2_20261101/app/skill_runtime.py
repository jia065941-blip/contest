"""Frozen strategic selectors composed with the measured chronological HP tree.

HP on/off policies share one native prefix; only the winning merged selector is deployed. Future strategic opportunities
following HP divergence are endogenous whole-policy mediators, not fixed-role
A/B interventions. HP pools always derive from original public assignments.
"""
import copy
import hashlib
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
from types import SimpleNamespace
from skills import cooperation as compiler, proposal, feature_projector, merged_policy
from skills.alternatives import alternatives
prep=SimpleNamespace(compiler=compiler, project=feature_projector, alternatives=alternatives)
hp=load('winner_hp_chronological_tree',HERE/'skills/hp_tree/runtime.py')
EVENT=None
STRATEGY=None
RECORDS=[]
COMMANDS=[]
PENDING={}
MEMORY={}
STATS=Counter()
DIRECTORY=None
BINDINGS={}
MASK={'KEEP':0,'A':1,'B':2,'AB':3}


def field(x,k):return x.get(k) if isinstance(x,dict) else getattr(x,k)


def legal_context(agent,binding,clock):
    obs=agent.latest_observation;p=obs['self']['position'];poly=agent.directory.polygon
    tracks=obs['self'].get('detectInfo') or {};target=binding['target_id']
    raw=tracks.get(target,tracks.get(str(target)));track=None
    if raw is not None:
        pos=field(raw,'lla');stamp=field(raw,'time')
        if pos is not None and stamp is not None:
            track=dict(position=[float(field(pos,k)) for k in ('x','y','z')],timestamp_s=float(stamp)/1000.)
    return dict(role=binding['role'],binding_matches=agent.task.get('target_id')==target,
        time_s=int(obs['step'])+clock,position=[p['lon'],p['lat'],p['alt']],own_track=track,
        public_track=dict(position=binding['public_position']),
        map_bounds=[min(x.lon for x in poly),max(x.lon for x in poly),min(x.lat for x in poly),max(x.lat for x in poly)])


def begin(reason,step,extra=None):
    global EVENT
    import matched
    assert EVENT is None
    EVENT=dict(time_s=step,reason=reason,binding=dict(fixed_before_fork=True,participants=list(BINDINGS.values()),
        rule='Original public H-sentry/M-mother assignments, sealed before branching'),
        strategic_prefix=copy.deepcopy(STRATEGY),details=extra,
        estimand='HP on/off with the same frozen merged strategic selector; later opportunities are endogenous mediators')
    matched.begin_event(EVENT)


def validate_execution(execution,manifest,arm,event,trace=None):
    e=execution['evidence_skill'];assert e['arm']==arm and e['event']==event
    assert not e['pending_commands'] and not e['actors_removed'] and not e['HP_mission_bindings_changed']
    bound={x['unit_id']:x for x in e['fixed_HP_participants']};seen=set()
    for row in e['commands']:
        key=(row['step'],row['unit_id'],row['action_type']);assert key not in seen;seen.add(key)
        assert arm.endswith('_hp') and row['unit_id'] in bound
        assert row['target_id']==bound[row['unit_id']]['target_id']
        assert row['legal_context']['role']==bound[row['unit_id']]['role']
        decision,_=hp.step(row['legal_context'],row['memory_before']);assert decision==row['decision']
        assert row['status']=='native_return' and row['receipt_step']==row['step']
        assert row['legal_context']['binding_matches']
        if row['action_type']==0:
            assert decision['clear_maneuver'] and row['value']==row['receipt_value']==0 and row['baseline_value']!=0
        elif row['action_type']==2:
            assert decision['waypoint']['action']=='WAYPOINT' and row['point']==decision['waypoint']['point']
            assert max(abs(a-b) for a,b in zip(row['point'],row['receipt_point']))<1e-7
        else:raise AssertionError('Unsupported HP command')
    strategy=e['strategy']
    if strategy:
        choice=strategy['applied_choice'];assert choice in MASK
        assert strategy['actors_removed']==0 and strategy['applied_mask']==MASK[choice]
        assert choice==strategy['choices']['merged'] and choice in ('KEEP','A')
    if event is None:assert not e['commands']
    assert trace is not None and trace['event']==event and trace['arm']==arm and not trace['error']
    history={};command_index={(r['step'],r['unit_id'],r['action_type']):r for r in e['commands']}
    checked_commands=set();last_steps={}
    for record in trace['records']:
        pid=record['unit'];assert pid in bound and record['target']==bound[pid]['target_id']
        assert record['step']>last_steps.get(pid,-1);last_steps[pid]=record['step']
        prior=history.get(pid,{})
        decision,next_memory=hp.step(record['context'],prior)
        assert decision==record['decision']
        original=record['original_actions']
        maneuvers=[a for a in original if a[0]==0]
        targets=[a for a in original if a[0]==4 and int(a[2])==bound[pid]['target_id']]
        assert len(maneuvers)==1
        expected=set()
        if record['enabled']:
            if decision['clear_maneuver'] and maneuvers[0][2]!=0:expected.add(0)
            if decision['waypoint']['action']=='WAYPOINT' and len(targets)==1:expected.add(2)
        actual={kind for kind in (0,2) if (record['step'],pid,kind) in command_index}
        assert actual==expected, 'Missing or extra HP intervention relative to frozen decision'
        for kind in (0,2):
            key=(record['step'],pid,kind)
            if key in command_index:
                row=command_index[key]
                assert record['enabled'] and row['memory_before']==prior and row['legal_context']==record['context']
                assert row['decision']==decision
                checked_commands.add(key)
        history[pid]=next_memory
    assert checked_commands==set(command_index)
    return dict(status='passed',issued=len(e['commands']),receipts=len(seen),fixed_HP_participants=len(bound),
        chronological_HP_records=len(trace['records']),
        strategy_choice=strategy['applied_choice'] if strategy else None,
        HP_replayed_from_legal_history=True,whole_policy_estimand=True)


def finish(error=None):
    with (Path(os.environ['MIGRATED_EPISODE'])/'legal_terminal_trace.json').open('w') as f:
        json.dump(dict(arm=os.environ.get('RSI_ARM','merged'),event=EVENT,records=RECORDS,error=error),f,separators=(',',':'),allow_nan=False)


def install():
    import agents,matched
    from common import episode_inputs
    from envengine.common import SimmerCommandType
    data=episode_inputs();manifest=data['skill']['manifest'];compiled=data['compiled']
    targets={t['ref']:t for t in compiled['plan']['public_targets']}
    for unit in compiled['plan']['units']:
        task=unit['task'];ref=task.get('target_ref');role=unit['role']
        if task['kind']=='attack' and ((role=='H' and targets.get(ref,{}).get('type')==9600) or
                                     (role=='M' and targets.get(ref,{}).get('type')==9400)):
            pid=int(compiled['unit_ids'][unit['unit_key']]);BINDINGS[pid]=dict(unit_id=pid,unit=unit['unit_key'],
                target_id=int(compiled['target_ids'][ref]),target=ref,role=role,public_position=list(targets[ref]['position']))
    anchor,provenance=compiler.load_anchor();binder=compiler.EventBinder(anchor)
    original_step=agents.step_group;original_action=agents.PlanAgent.get_action;persist=agents.PlanDirectory.persist

    def strategic(plan,current,targets,time_s,memory):
        global STRATEGY
        response=original_step(plan,current,targets,time_s,memory)
        if binder.event is not None or time_s>900 or time_s%15:return response
        ctx=compiler.project_context(compiler._legacy().context(plan,current,targets,time_s,data['own_initial_clock_s'],response))
        before=len(binder.attempts)
        try:event=binder.consider(ctx,response)
        except (ValueError,KeyError,TypeError) as exc:
            STATS['proposal_rejected:'+str(exc)[:150]]+=1;return response
        if len(binder.attempts)!=before:STATS['proposal_calls']+=1
        if event is None:return response
        roles=copy.deepcopy(event['roles']);features=prep.project.features(ctx,roles)
        candidates=prep.alternatives(ctx,roles)
        public=[dict(candidate_id='C'+str(i+1),features=c['features']) for i,c in enumerate(candidates)]
        binding_decision=proposal.choose(features,public)
        tree_binding=dict(status='no_alternate_at_first_old_opportunity',roles=None,features=None,prediction=None)
        choices=dict(merged='KEEP')
        if binding_decision['candidate_id'] is not None:
            index=next(i for i,p in enumerate(public) if p['candidate_id']==binding_decision['candidate_id'])
            alt=candidates[index];alt_roles=alt['roles']
            assert alt_roles[0]==roles[0] and alt_roles[1]['unit']!=roles[1]['unit']
            compiler.validate_roles(alt_roles,ctx)
            for mask in (0,1):
                assert json.dumps(compiler.merge(response,roles,mask,ctx))==json.dumps(compiler.merge(response,alt_roles,mask,ctx))
            prediction=merged_policy.predict(alt['features']);choices['merged']=prediction['choice']
            assert choices['merged'] in ('KEEP','A')
            tree_binding=dict(status='valid_alternate',roles=copy.deepcopy(alt_roles),features=alt['features'],prediction=prediction)
        event=copy.deepcopy(event)
        event['binding']['fixed_before_fork']=EVENT is None
        event['binding']['fixed_before_local_execution']=True
        event['binding']['whole_policy_fork_preceded_binding']=EVENT is not None
        STRATEGY=dict(time_s=time_s,original_roles=roles,legal_features=features,tree_binding=tree_binding,
            binding_decision=binding_decision,choices=choices,anchor_provenance=provenance,
            event=copy.deepcopy(event),actors_removed=0,bindings_fixed_before_local_execution=True,
            opportunity_after_HP_fork=EVENT is not None)
        selector='merged'
        choice=choices[selector];mask=MASK[choice]
        STRATEGY.update(applied_choice=choice,applied_mask=mask,quality=compiler.coverage_audit(response,roles,mask,ctx))
        STATS['strategic_events']+=1
        return compiler.merge(response,roles,mask,ctx)

    def action(agent):
        global DIRECTORY
        actions=original_action(agent);pid=agent.entity_id
        if pid not in BINDINGS:return actions
        DIRECTORY=agent.directory;step=int(agent.latest_observation['step']);binding=BINDINGS[pid]
        ctx=legal_context(agent,binding,data['own_initial_clock_s']);prior=copy.deepcopy(MEMORY.get(pid,{}))
        decision,MEMORY[pid]=hp.step(ctx,prior)
        maneuver=[i for i,a in enumerate(actions) if a[0]==0]
        track=[i for i,a in enumerate(actions) if a[0]==4 and int(a[2])==binding['target_id']]
        assert len(maneuver)==1
        differs=(decision['clear_maneuver'] and actions[maneuver[0]][2]!=0) or (decision['waypoint']['action']=='WAYPOINT' and len(track)==1)
        if EVENT is None and differs:begin('HP_command_divergence',step,dict(unit=pid,legal_context=ctx,decision=decision))
        arm=os.environ.get('RSI_ARM','merged');enabled=arm.endswith('_hp')
        RECORDS.append(dict(step=step,unit=pid,target=binding['target_id'],context=ctx,decision=decision,
            enabled=enabled,original_actions=actions.tolist()))
        STATS['observed_'+binding['role']]+=1
        if not ctx['binding_matches']:STATS['original_binding_conflict_'+binding['role']]+=1
        if decision['waypoint']['action']=='WAYPOINT' and len(track)!=1:STATS['missing_matching_action4']+=1
        if not enabled:return actions
        new=actions.copy()
        def issue(kind,extra):
            key=(pid,kind)
            if key in PENDING:raise RuntimeError('Unacknowledged prior HP command')
            row=dict(step=step,unit_id=pid,target_id=binding['target_id'],action_type=kind,legal_context=ctx,
                memory_before=prior,decision=decision,status='emitted',receipt_step=None,**extra)
            PENDING[key]=row;COMMANDS.append(row);STATS['emitted_type_'+str(kind)]+=1
        if decision['waypoint']['action']=='WAYPOINT' and len(track)==1:
            point=decision['waypoint']['point'];new[track[0]]=[2,pid,*point];issue(2,dict(point=point))
        if decision['clear_maneuver'] and new[maneuver[0]][2]!=0:
            previous=float(new[maneuver[0]][2]);new[maneuver[0]][2]=0.;agent.set_acc_z_z=0
            issue(0,dict(value=0.,baseline_value=previous))
        return new

    import importlib
    for role in ('H','M'):
        name='CompCruiseMissile'+role+'Simulator'
        cls=getattr(importlib.import_module('envengine.simulator.simlulator_impl.'+name),name);native=cls.command_received
        def received(simulator,command,_native=native):
            result=_native(simulator,command);pid=int(command.executorId)
            kind=0 if command.commandTypeId==SimmerCommandType.SET_DESIRED_ACC_Z else 2 if command.commandTypeId==SimmerCommandType.CHANGE_MISSILE_TARGET else None
            row=PENDING.get((pid,kind))
            if row is not None:
                if kind==0:
                    value=command.commandAttributes['acc_z'];assert value==row['value']==0;row['receipt_value']=float(value)
                else:
                    p=command.commandAttributes['target'];point=[p['x'],p['y']]
                    assert max(abs(a-b) for a,b in zip(point,row['point']))<1e-7;row['receipt_point']=point
                row.update(status='native_return',receipt_step=int(DIRECTORY.step));del PENDING[(pid,kind)]
                STATS['native_receipt_type_'+str(kind)]+=1
            return result
        cls.command_received=received

    def save(directory):
        result=persist(directory)
        result['evidence_skill']=dict(name=manifest['skill_name'],arm=os.environ.get('RSI_ARM','merged'),
            event=copy.deepcopy(EVENT),strategy=copy.deepcopy(STRATEGY),fixed_HP_participants=list(BINDINGS.values()),
            stats=dict(STATS),commands=copy.deepcopy(COMMANDS),pending_commands=len(PENDING),actors_removed=0,
            HP_mission_bindings_changed=0,baseline_model='Frozen V12-R3 computed unchanged',
            interpretation='Whole-policy composition; HP fixed pools, future A/B opportunities endogenous')
        return result
    agents.step_group=strategic;agents.PlanAgent.get_action=action;agents.PlanDirectory.persist=save
