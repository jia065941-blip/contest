"""Own telemetry and legally delivered registry only; no simulator-global observation."""
from .program import TYPES
from .runtime import distance,describe

SPEEDS={'H':200.,'M':120.,'L':100.}

def field(value,key,default=None):return value.get(key,default) if isinstance(value,dict) else getattr(value,key,default)


class TypedTasks:
    def __init__(self,dt_s,max_steps):
        self.dt_s,self.max_steps=dt_s,max_steps
        self.roles={};self.latest={};self.events=[];self.reporters={};self.target_aliases={};self.role_aliases={}
    def observe(self,observations,step):
        self.latest={int(o['entity_id']):o.get('self') or {} for o in observations}
        events=[]
        for pid,role in self.roles.items():
            if pid not in self.latest and role['alive']:
                role['alive']=False;events.append(('task_exit',pid))
        for pid,own in self.latest.items():
            kind=next((k for k,v in TYPES.items() if own.get('type')==v),None)
            if kind is None:continue
            role=self.roles.setdefault(pid,dict(type=kind,target_id=None,goal=None,launch_step=None,flight_observed=False,
                native_ack_step=-1,native_ack_target=None,previous=None,alive=False,command=None))
            if pid not in self.role_aliases:self.role_aliases[pid]=kind+str(sum(r.startswith(kind) for r in self.role_aliases.values()))
            alive=bool(own.get('isVisible') and own.get('health',0)>0)
            position=own.get('position') or {};xy=[field(position,'lon'),field(position,'lat')]
            if alive and role['launch_step'] is not None and step>role['launch_step'] and own.get('stage',-1)>=0:
                if not role['flight_observed'] and role['previous'] and all(isinstance(v,(int,float)) for v in xy):
                    if distance(xy,role['previous'])>1.:
                        role['flight_observed']=True;events.append(('became_executable',pid))
            if role['alive'] and not alive:events.append(('task_exit',pid))
            role['alive']=alive
            role['previous']=xy if all(isinstance(v,(int,float)) for v in xy) else None
        self.events=events;return events
    def generated(self,command,row,step):
        role=self.roles.get(command.platform_id)
        if role is None or command.kind not in ('launch','retarget'):return
        if command.kind=='launch':role['launch_step']=step
        ref=command.target
        tid=ref.entity_id if ref.mode=='entity' else command.bound_entity_id if ref.mode=='event_entity' else None
        role.update(target_id=tid,goal=list(row[2:]),command=dict(step=step,kind=command.kind,goal=list(row[2:]),target_id=tid))
    def acknowledged(self,pid,step,goal):
        role=self.roles.get(pid);command=role.get('command') if role else None
        if command and command['kind']=='retarget' and command['step']==step and command['goal']==list(goal):
            role.update(native_ack_step=step,native_ack_target=command['target_id'])


def snapshot(commander,observations):
    step=commander._observation_step;tracker=commander.cooperative_tasks
    known={}
    entities=commander.init_ship_observation.get('entities') or {}
    for raw,entity in entities.items():
        kind=int(field(entity,'type',-1))
        if kind in (9400,9600):known[int(raw)]=kind
    for obs in observations:
        own=obs.get('self') or {}
        for key,report in (own.get('detectInfo') or {}).items():
            kind=field(report,'entity_type')
            if kind in (9400,9500,9600):known[int(field(report,'entity_id',key))]=kind
    known.update({int(tid):9500 for tid in getattr(commander,'tracks',{})})
    target_ids=[]
    for tid in sorted(known):
        if commander._executor.registry.resolve(tid) is not None:target_ids.append(tid)
    for tid in target_ids:
        if tid not in tracker.target_aliases:tracker.target_aliases[tid]='T'+str(len(tracker.target_aliases))
    target_refs={tid:tracker.target_aliases[tid] for tid in target_ids}
    # Role aliases persist through the episode, including unavailable roles, avoiding rebinding by disappearance.
    role_refs=tracker.role_aliases
    roles=[]
    for pid,role in sorted(tracker.roles.items()):
        own=tracker.latest.get(pid,{})
        if role['previous'] is None:continue
        remaining=max(0.,(tracker.max_steps-step)*tracker.dt_s)
        if role['type']=='L' and role['launch_step'] is not None:
            remaining=min(remaining,max(0.,1800-(step-role['launch_step'])*tracker.dt_s))
        eligible=bool(role['alive'] and role['flight_observed'] and own.get('stage',-1)>=0 and remaining>20.)
        target_id=role['target_id'];goal=role['goal'];kind='original'
        task=commander.tasks.tasks.get(pid)
        if role['type']=='L' and task:
            target_id=(task.target_ref or {}).get('id');goal=task.goal;kind=task.kind
        roles.append(dict(ref=role_refs[pid],type=role['type'],position=role['previous'],eligible=eligible,
            remaining_s=remaining,nominal_speed_mps=SPEEDS[role['type']],current_target=target_refs.get(target_id),
            current_goal=goal,task_kind=kind,
            native_ack_step=role['native_ack_step'],native_ack_target=target_refs.get(role['native_ack_target'])))
    targets=[]
    for tid in target_ids:
        point=commander._executor.registry.resolve(tid)
        reporters=set()
        for obs in observations:
            for report in ((obs.get('self') or {}).get('detectInfo') or {}).values():
                reporter=field(report,'detect_from')
                if field(report,'entity_id')==tid and reporter in role_refs:reporters.add(role_refs[reporter])
        targets.append(dict(ref=target_refs[tid],type=known[tid],position=[point.lon,point.lat],reporter_refs=sorted(reporters)))
    return dict(step=step,dt_s=tracker.dt_s,roles=roles,targets=targets,
        current_arrangement=describe(getattr(commander,'cooperative_memory',{}))),role_refs,target_refs
