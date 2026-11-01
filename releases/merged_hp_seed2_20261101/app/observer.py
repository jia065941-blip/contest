"""Passive evaluator hooks. No fields are added to agent observations."""
from pathlib import Path
from collections import Counter
import importlib,json,os,hashlib
from metrics import Ledger,ROLES,TARGETS
STATE={}


def keys(env):
    return tuple(sorted((int(s.entity_ext.entity.id),id(s),id(s.entity_ext.entity)) for s in env.engine.simulator_factory.get_all_simulators()))


def key_hash(value):return hashlib.sha256(repr(value).encode()).hexdigest()


def install():
    from envengine.environment.training_env import TrainingEnv
    from envengine.simulator.interfaces import ISimulator
    old_reset=TrainingEnv.reset
    def reset(env,*a,**kw):
        result=old_reset(env,*a,**kw)
        if STATE:raise RuntimeError('Diagnostics expects exactly one native episode')
        sims=env.engine.simulator_factory.get_all_simulators()
        entities=[dict(id=int(s.entity_ext.entity.id),type=int(s.entity_ext.entity.entityType),health=float(s.entity_ext.entity.survivePoints)) for s in sims]
        STATE.update(env=env,ledger=Ledger(entities),origin=float(env.engine.sim_time),registry=keys(env),launch_time={},terminal={},interceptions=[],steps=0)
        return result
    TrainingEnv.reset=reset
    old_damage=ISimulator._damage
    def damage(sim,amount,source):
        entity=sim.entity_ext.entity;before=float(entity.survivePoints)
        value=old_damage(sim,amount,source)
        if STATE:
            after=float(entity.survivePoints);src=int(source.id);dst=int(entity.id)
            when=(float(sim.sim_time)-STATE['origin'])/1000
            ledger=STATE['ledger'];ledger.damage(src,dst,before,after,float(amount),when)
            if int(entity.entityType) in ROLES:
                if src==dst and dst not in STATE['terminal']:
                    ret=getattr(sim,'ret',None);elapsed=when-STATE['launch_time'].get(dst,when)
                    cause='self_detonation_model_return' if ret is not None and ret>=0 else 'self_detonation_lifetime' if int(entity.entityType)==21002 and elapsed>=1800 else 'self_damage_other'
                    STATE['terminal'][dst]=dict(cause=cause,time_s=when,model_return=ret)
                elif before>0 and after<=0:
                    st=ledger.entities.get(src,{}).get('type');cause='intercepted' if st==24000 else 'other_damage'
                    STATE['terminal'].setdefault(dst,dict(cause=cause,time_s=when,source_type=st))
                    if st==24000:STATE['interceptions'].append(dict(unit=dst,interceptor=src,time_s=when))
        return value
    ISimulator._damage=damage
    for role in ROLES.values():
        name=f'CompCruiseMissile{role}Simulator';cls=getattr(importlib.import_module('envengine.simulator.simlulator_impl.'+name),name)
        original=cls.start_mission
        def start(sim,*a,_original=original,_role=role,**kw):
            value=_original(sim,*a,**kw)
            launched=sim.launch>=0 if _role=='L' else sim.launch!=0
            if STATE and launched:
                uid=int(sim.entity_ext.entity.id);STATE['ledger'].launch(uid)
                STATE['launch_time'].setdefault(uid,(float(sim.sim_time)-STATE['origin'])/1000)
            return value
        cls.start_mission=start
    old_step=TrainingEnv.step
    def step(env,*a,**kw):
        result=old_step(env,*a,**kw);STATE['steps']+=1
        # Address audit is evaluator-only; never substitute official tie keys.
        if keys(env)!=STATE['registry']:raise RuntimeError('Scheduler object registry changed during event collection')
        return result
    TrainingEnv.step=step


def finish(error=None):
    from common import save
    folder=Path(os.environ['MIGRATED_EPISODE'])
    if error or not STATE:
        save(folder/'combat_metrics_failed.json',dict(error=error or 'no initialized episode'));return
    env=STATE['env'];ledger=STATE['ledger'];sims=env.engine.simulator_factory.get_all_simulators()
    final={int(s.entity_ext.entity.id):float(s.entity_ext.entity.survivePoints) for s in sims}
    summaries=list((folder/'native').glob('*/summary.json'))
    if len(summaries)!=1:raise RuntimeError('Missing unique native summary')
    summary=json.loads(summaries[0].read_text());objectives=summary['objectives']
    wanted={u for u,e in ledger.entities.items() if e['type'] in TARGETS}
    assert wanted=={int(o['id']) for o in objectives}
    for o in objectives:
        u=int(o['id']);assert abs(final[u]-o['final_health'])<1e-8
        assert abs(ledger.entities[u]['health']-o['initial_health'])<1e-8
        ledger.entities[u]['weight']=o['weight']
    result=ledger.finalize(final)
    total_weight=sum(o['weight'] for o in objectives)
    for o in result['objectives']:
        o['lost_score']=0 if o['destroyed'] else 100*o['weight']/total_weight
        o['unattributed_health_loss']=o['initial_health']-o['final_health']-sum(o['damage_by_role'].values())-o['other_source_damage']
    result['target_summary']={}
    for typ in TARGETS.values():
        group=[o for o in result['objectives'] if o['type']==typ]
        result['target_summary'][typ]=dict(count=len(group),destroyed=sum(o['destroyed'] for o in group),
            damaged_surviving=sum(0<o['final_health']<o['initial_health'] for o in group),
            undamaged_surviving=sum(o['final_health']>=o['initial_health'] for o in group),
            remaining_health=sum(o['final_health'] for o in group),lost_score=sum(o['lost_score'] for o in group))
    assert abs(100-sum(o['lost_score'] for o in result['objectives'])-summary['score']['score'])<1e-8
    result['units']=[]
    for uid,e in ledger.entities.items():
        if e['type'] not in ROLES:continue
        targets=[t for u,t in ledger.pairs if u==uid]
        terminal=STATE['terminal'].get(uid,dict(cause='alive_at_end' if final[uid]>0 else 'unclassified_death',time_s=None))
        result['units'].append(dict(unit=uid,role=ROLES[e['type']],launched=uid in ledger.launched,
            launch_s=STATE['launch_time'].get(uid),final_health=final[uid],terminal=terminal,
            hit_targets=targets,scoring_targets=[t for t in targets if final[t]<=0],surviving_hit_targets=[t for t in targets if final[t]>0]))
    for role in ROLES.values():
        result['roles'][role]['terminal_causes']=dict(Counter(u['terminal']['cause'] for u in result['units'] if u['role']==role))
    result['damage_pairs']=[dict(unit=u,target=t,**row) for (u,t),row in sorted(ledger.pairs.items())]
    result['interception_events']=STATE['interceptions']
    result['audit']=dict(steps=STATE['steps'],start_registry_hash=key_hash(STATE['registry']),end_registry_hash=key_hash(keys(env)),
        official_scheduler_unchanged=True,policy_has_no_access_to_metrics=True,score=summary['score']['score'],
        launched=sum(r['launched'] for r in result['roles'].values()),purpose='descriptive event accounting, no skill-effect comparison')
    assert result['unresolved_damage_sources']==0
    save(folder/'combat_metrics.json',result)
