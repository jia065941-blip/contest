"""Same-memory negative controls without replacing the official event scheduler."""
from pathlib import Path
import hashlib
import json
import os
import pickle
import random
import signal
import struct
import sys
import time

BRANCHES=('baseline_a','baseline_b','allocation_stress')
STATE={}


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,separators=(',',':'))+'\n');tmp.replace(path)


def digest(value):return hashlib.sha256(pickle.dumps(value,protocol=4)).hexdigest()


def registry(env):
    # Both kinds of address are used by the unchanged Engine.step scheduler.
    sims=env.engine.simulator_factory.get_all_simulators()
    return tuple(sorted((int(s.entity_ext.entity.id),id(s),id(s.entity_ext.entity)) for s in sims))


def rng_state():
    import numpy as np
    import torch
    return random.getstate(),np.random.get_state(),torch.get_rng_state().clone()


def restore_rng(value):
    import numpy as np
    import torch
    # CPython random reseeds itself after fork; restore the actual snapshot.
    random.setstate(value[0]);np.random.set_state(value[1]);torch.set_rng_state(value[2])


def rng_digest():
    p,n,t=rng_state()
    return digest((p,(n[0],n[1].tobytes(),*n[2:]),t.numpy().tobytes()))


def canonical(value):
    import numpy as np
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {str(k):canonical(v) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
    if hasattr(value,'value'):return canonical(value.value)
    if hasattr(value,'__dict__'):return canonical(vars(value))
    raise TypeError('Unsupported audit value '+type(value).__name__)


def physics_digest(env,rewards,done):
    # Only hashes leave the evaluator. Hidden physical state is never provided
    # to a policy or an LLM. No large full-observation serialization/trajectory.
    rows=[]
    for s in env.engine.simulator_factory.get_all_simulators():
        e=s.entity_ext.entity
        fields=[int(e.id),int(e.entityType),bool(e.isVisible),float(e.survivePoints),getattr(e,'stage',None)]
        for name in ('lla','posEcf','velEcf'):
            v=getattr(e,name);fields.extend((float(v.x),float(v.y),float(v.z)))
        rows.append(fields)
    evidence=dict(step=env.current_step,engine_time=env.engine.sim_time,entities=sorted(rows),
                  actions=canonical(env._last_actions),rewards=canonical(rewards),done=bool(done),rng=rng_digest())
    return hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def isolate_output(env,label,root):
    from envengine.sdk.writer import get_writer
    folder=root/'branches'/label;folder.mkdir(parents=True,exist_ok=False)
    os.environ['MIGRATED_EPISODE']=str(folder)
    writer=get_writer()
    assert not writer.buffer, 'Cannot fork with pending asynchronous writes'
    assert not writer.executor._threads, 'Cannot fork after writer thread startup'
    writer.output_dir=folder/'native'/'episode';writer.output_dir.mkdir(parents=True)
    writer.config.output_dir=str(writer.output_dir)
    directories={}
    for agent in env.agent_manager.get_all_agents():
        d=getattr(agent,'directory',None)
        if d is not None:directories[id(d)]=d
    for d in directories.values():
        d.folder=folder
        if hasattr(d,'selector'):d.selector.folder=folder
    return folder


def validate_children(root,snapshot,children):
    rows={};errors=[]
    for label,pid,code in children:
        path=root/'branches'/label/'branch_audit.json'
        if code!=0 or not path.exists():errors.append(dict(branch=label,pid=pid,exit_status=code,missing_audit=not path.exists()));continue
        r=json.loads(path.read_text());rows[label]=r
        if not r.get('address_registry_unchanged') or r.get('start_registry_hash')!=snapshot['registry_hash'] or r.get('start_rng_hash')!=snapshot['rng_hash']:
            errors.append(dict(branch=label,error='snapshot or address drift'))
        if r.get('error'):errors.append(dict(branch=label,error=r['error']))
    equal=False
    if len(rows)==len(BRANCHES) and not errors:
        for label,r in rows.items():
            if not r.get('frame_hashes') or r.get('final_score') is None or r.get('steps') is None:
                errors.append(dict(branch=label,error='Incomplete continuation evidence'))
    if len(rows)==len(BRANCHES) and not errors:
        ref=rows[BRANCHES[0]]
        equal=all(rows[b]['frame_hashes']==ref['frame_hashes'] and rows[b]['final_score']==ref['final_score']
                  and rows[b]['termination']==ref['termination'] and rows[b]['steps']==ref['steps'] for b in BRANCHES[1:])
        if not equal:errors.append(dict(error='Identical policies diverged across matched branches'))
    result=dict(status='passed' if equal and not errors else 'failed',snapshot=snapshot,branches=rows,errors=errors,
                official_event_order_unchanged=True,new_skill_enabled=False,
                interpretation='Baseline/baseline and asymmetric-allocation negative controls from same process snapshot; no skill-effect claim.')
    save(root/'comparison.json',result)
    return equal and not errors


def split(env):
    from envengine.sdk.writer import get_writer
    if len(list(Path('/proc/self/task').iterdir()))!=1:raise RuntimeError('Fork requires a single-threaded native process')
    writer=get_writer()
    if writer.buffer or writer.executor._threads:raise RuntimeError('Fork with asynchronous writer is unsafe')
    root=Path(os.environ['PAIR_ROOT'])
    keys=registry(env);rng=rng_state()
    snapshot=dict(step=env.current_step,registry_hash=digest(keys),rng_hash=rng_digest(),entity_count=len(keys),
                  parent_pid=os.getpid(),time_unix=time.time())
    save(root/'snapshot.json',snapshot)
    sys.stdout.flush();sys.stderr.flush()
    children=[]
    def initialize(label,coordinator=False):
        restore_rng(rng)
        folder=isolate_output(env,label,root)
        STATE.update(child=True,coordinator=coordinator,label=label,folder=folder,env=env,keys=keys,frames=[],error=None,
                     start_registry_hash=digest(registry(env)),start_rng_hash=rng_digest(),snapshot=snapshot,
                     stress=[],started=time.monotonic(),root=root)
        assert STATE['start_registry_hash']==snapshot['registry_hash']
        assert STATE['start_rng_hash']==snapshot['rng_hash']
    for label in BRANCHES[1:]:
        pid=os.fork()
        if pid==0:
            initialize(label)
            return
        children.append((label,pid))
    save(root/'children.json',dict(children=children))
    # The original process becomes baseline_a after all forks, so every process
    # performs useful work; no idle parent retains a separate full snapshot.
    initialize(BRANCHES[0],coordinator=True)
    STATE['children']=children
    def stop(signum,frame):
        for _,pid in children:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def finish(error=None):
    if not STATE.get('child'):return
    folder=STATE['folder'];env=STATE['env']
    try:
        summaries=list((folder/'native').glob('*/summary.json'))
        summary=json.loads(summaries[0].read_text()) if len(summaries)==1 else {}
        save(folder/'branch_audit.json',dict(label=STATE['label'],pid=os.getpid(),
             start_registry_hash=STATE['start_registry_hash'],start_rng_hash=STATE['start_rng_hash'],
             address_registry_unchanged=registry(env)==STATE['keys'],frame_hashes=STATE['frames'],
             final_score=summary.get('score',{}).get('score'),termination=summary.get('termination_reason'),
             steps=summary.get('steps_executed'),elapsed_s=time.monotonic()-STATE['started'],
             error=STATE.get('error') or error or (None if summary else 'Missing final native summary')))
    except BaseException as exc:
        save(folder/'branch_audit.json',dict(error=str(exc)))
    if STATE.get('coordinator'):
        results=[(BRANCHES[0],os.getpid(),0 if not error else 1)]
        for label,pid in STATE['children']:
            _,code=os.waitpid(pid,0);results.append((label,pid,code))
        if not validate_children(STATE['root'],STATE['snapshot'],results):
            raise SystemExit('Matched branch negative control failed')


def install():
    from envengine.environment.training_env import TrainingEnv
    from envengine.engine.engine import Engine
    original_step=TrainingEnv.step
    engine_step=Engine.step
    fork_step=int(os.environ['PAIR_FORK_STEP'])
    def guarded_engine(self,*args,**kwargs):
        if STATE.get('child') and registry(STATE['env'])!=STATE['keys']:
            STATE['error']='New/replaced scheduler-key object before step';raise RuntimeError(STATE['error'])
        value=engine_step(self,*args,**kwargs)
        if STATE.get('child') and registry(STATE['env'])!=STATE['keys']:
            STATE['error']='New/replaced scheduler-key object after step';raise RuntimeError(STATE['error'])
        return value
    def step(self):
        if not STATE.get('child') and self.current_step==fork_step:split(self)
        if STATE.get('label')=='allocation_stress':
            # Exercise asymmetric allocator traffic without changing actions/RNG.
            STATE['stress']=[bytearray(97+i%11) for i in range(1024)]
        value=original_step(self)
        if STATE.get('child'):STATE['frames'].append(physics_digest(self,value[1],value[2]))
        return value
    Engine.step=guarded_engine
    TrainingEnv.step=step
