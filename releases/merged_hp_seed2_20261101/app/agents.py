"""Native adapter for Astra's executable plan; all dynamic state is group-local."""
from collections import Counter
import numpy as np
import base_agents as base
from common import read,save,HERE,episode_inputs
from design.planner_v2 import step_group

copy=base.copy
KIND=base.KIND


class PlanDirectory(base.Directory):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        compiled=episode_inputs()['compiled']
        self.plan=compiled['plan'];self.compiled=compiled
        self.key_by_id={int(v):k for k,v in compiled['unit_ids'].items()}
        self.id_by_key={v:k for k,v in self.key_by_id.items()}
        self.units={u['unit_key']:u for u in self.plan['units']}
        self.initial_ref_by_id={int(v):k for k,v in compiled['target_ids'].items()}
        self.initial_id_by_ref={v:k for k,v in self.initial_ref_by_id.items()}
        self.plan_events=[]

    def persist(self):
        result=super().persist()
        result['plan_events']=dict(events=self.plan_events,metrics=dict(self.metrics),
            evidence='native command receipts and legal group observations; no target damage inferred')
        return result


class PlanGroup(base.Group):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.plan_memory={};self.prior_memory={};self.prior_started={};self.wave=0;self.satellite_requests=set()

    def prepare(self,step):
        if self.prepared==step:return self
        self.prepared=step
        current={pid:(obs,task) for pid,(obs,task) in self.reports.items() if obs['step']==step}
        assert set(current)<=self.members
        incoming={pid:copy.deepcopy(task) for pid,(obs,task) in current.items()}
        self.baseline.begin_step(tuple(obs for obs,task in current.values()))
        self.targets={t.entity_id:t for t in self.baseline.targets}
        d=self.directory
        refs={tid:d.initial_ref_by_id.get(tid,'D'+str(tid)) for tid in self.targets}
        ids={ref:tid for tid,ref in refs.items()}
        legal_targets={refs[tid]:dict(type=t.entity_type,position=[t.position.lon,t.position.lat,t.position.alt],
            timestamp_s=self.baseline.track_fusion._observed_at.get(tid,0)/1000.) for tid,t in self.targets.items()}
        reports={}
        for pid,(obs,task) in current.items():
            own=obs['self'];position=own['position'];key=d.key_by_id[pid]
            life=max(0,(1800 if own['type']==21002 else d.max_steps)-step)
            reports[key]=dict(position=[position['lon'],position['lat'],position['alt']],task=copy.deepcopy(task),
                remaining_life_s=life,stage=own.get('stage',0),visible=own.get('isVisible',True),health=own.get('health',0),
                satellite_active=bool(obs.get('is_using_satellite',False)))
            peers=own.get('commRangeInfo',{})
            velocity=peers.get(pid,peers.get(str(pid),{})).get('velEcf')
            if velocity is not None:
                reports[key]['velocity']=[velocity[a] for a in ('x','y','z')]
        response=step_group(d.plan,reports,legal_targets,step,self.plan_memory)
        self.plan_memory=response['memory']
        self.satellite_requests={d.id_by_key[k] for k in response.get('request_satellite',[]) if k in reports}
        proposals={};self.actions={}
        for key,order in response.get('orders',{}).items():
            if key not in reports:raise ValueError('Cross-group plan command')
            pid=d.id_by_key[key];task=current[pid][1]
            if order['kind']=='attack':
                ref=order['target_ref']
                if ref not in ids:raise ValueError('Plan attempted unknown target')
                tid=ids[ref];typ=self.targets[tid].entity_type
                role=KIND[current[pid][0]['self']['type']]
                if (role=='L')!=(typ==9500):raise ValueError('Illegal target type')
                if role=='L' and task.get('target_id')!=tid:
                    if task.get('target_id') is not None:raise ValueError('Prior-bound L cannot abandon its target')
                    proposals.setdefault(tid,[]).append(pid)
                    # Commit only after frozen prior accepts the actual binding.
                    continue
                if task.get('target_id') not in (None,tid):
                    d.metrics['hm_reassignments']+=1
                task.update(copy.deepcopy(order));task.update(target_id=tid,last_point=[self.targets[tid].position.lon,self.targets[tid].position.lat])
            elif order['kind']=='search':
                if task.get('target_id') is not None:raise ValueError('Cannot silently discard committed attack task')
                task.update(copy.deepcopy(order));task['target_id']=None
            else:raise ValueError('Unsupported Astra task')
        self.dispatch_prior(step,current,proposals)
        for pid,(obs,task) in current.items():
            tid=task.get('target_id')
            if tid is not None and tid in self.targets:
                p=self.targets[tid].position;task['last_point']=[p.lon,p.lat];task['kind']='attack';task['target_ref']=refs[tid]
                self.actions[pid]=[4,pid,tid,0]
                d.metrics['attack_coordinate_updates']+=1
            elif tid is not None:
                # Preserve old commitment; missing registry is not death evidence.
                self.actions[pid]=None
            else:
                points=task.get('waypoints') or d.units[d.key_by_id[pid]]['task'].get('waypoints',[])
                if not points:raise ValueError('Searching unit without executable route')
                index=min(int(task.get('waypoint_index',0)),len(points)-1)
                task.update(last_point=list(points[index]),waypoint_index=index)
                self.actions[pid]=[2,pid,*points[index]]
            self.reports[pid]=(obs,task)
        d.metrics['group_preparations']+=1
        d.metrics['runtime_plan_calls']+=1
        return self

    def dispatch_prior(self,step,current,proposals):
        d=self.directory
        lroles={d.key_by_id[pid]:(pid,obs,task) for pid,(obs,task) in current.items() if obs['self']['type']==21002}
        if not lroles:return
        feedback={}
        for event,mem in self.prior_memory.items():
            tid=int(event.split(':')[0][1:]);start=self.prior_started[event]
            receivers=mem['receiver_roles']
            feedback[event]=dict(all_acknowledged=bool(receivers) and all(
                k in lroles and d.receipts.get(lroles[k][0],{}).get('ack_target')==tid and
                d.receipts.get(lroles[k][0],{}).get('ack_step',-1)>=start for k in receivers),
                any_unavailable=any(k not in lroles for k in receivers),
                all_exited=bool(receivers) and all(k not in lroles or lroles[k][2].get('target_id')!=tid for k in receivers),
                elapsed_s=step-start)
        events=[];event_targets={}
        for tid,pids in sorted(proposals.items()):
            self.wave+=1;event=f'D{tid}:W{self.wave}';event_targets[event]=tid
            events.append(dict(ref=event,original_roles=[d.key_by_id[p] for p in pids],queue_age_s=0))
        roles=[]
        for key,(pid,obs,task) in lroles.items():
            own=obs['self'];point=own['position'];available=task.get('target_id') is None
            roles.append(dict(ref=key,eligible=available,remaining_time_s=max(0,1800-step),task_kind='search' if available else 'original',
                original_event_refs=[] if available else ['committed:'+str(task['target_id'])],
                distances_m={event:base.runtime.distance([point['lon'],point['lat']],[self.targets[tid].position.lon,self.targets[tid].position.lat]) for event,tid in event_targets.items()}))
        result=d.prior_step(dict(events=events,roles=roles,feedback=feedback),self.prior_memory,d.prior_parameters)
        self.prior_memory=result['memory']
        for event in self.prior_memory:self.prior_started.setdefault(event,step)
        for item in result['replacements']:
            event=item['event_ref'];tid=event_targets[event];target=self.targets[tid]
            for key in item['roles']:
                pid,obs,task=lroles[key]
                assert task.get('target_id') is None
                task.update(kind='attack',target_id=tid,target_ref='D'+str(tid),binding='hidden_attack',
                    last_point=[target.position.lon,target.position.lat],handoff_step=step,handoff_event=event)
                row=dict(step=step,group=self.key,entity_id=pid,role_ref=key,skill_id='prior_t03_make',controller='prior_t03_make',
                    previous_target=None,target_id=tid,target_ref='D'+str(tid),point=list(task['last_point']),status='prepared',same_target=False)
                d.ledger.append(row);d.pending[pid]=row;d.metrics['prior_dispatches']+=1;d.metrics['commands_prepared']+=1
            d.plan_events.append(dict(step=step,group=self.key,event='hidden_handoff',target_id=tid,receivers=item['roles']))
        d.metrics['prior_transitions']+=len(result['transitions'])
        d.metrics['prior_deferred_proposals']+=len(events)-len(result['replacements'])


class PlanAgent(base.MigratedAgent):
    def configure(self,directory,motion):
        super().configure(directory,motion)
        unit=directory.units[directory.key_by_id[self.entity_id]]
        self.task=copy.deepcopy(unit['task']);self.task.setdefault('binding',unit.get('binding','primary'))
        if self.task['kind']=='attack':self.task['target_id']=directory.initial_id_by_ref[self.task['target_ref']]
        else:self.task.update(target_id=None,waypoint_index=0)

    def get_action(self):
        obs=self.latest_observation;step=int(obs['step']);group=self._group.prepare(step)
        if self.launch_step<0:self.launch_step=step
        _,self.task=group.reports[self.entity_id];self.task=copy.deepcopy(self.task)
        actions=[];command=group.actions.get(self.entity_id)
        if command:
            actions.append(command)
            pending=self.directory.pending.get(self.entity_id)
            if pending and command[0]==4 and int(command[2])==pending['target_id']:pending['status']='emitted'
        if self.entity_id in group.satellite_requests:
            actions.append([3,self.entity_id,0,0]);self.task['satellite_requested_step']=step
            self.directory.metrics['satellite_requests']+=1
            self.directory.plan_events.append(dict(step=step,entity_id=self.entity_id,event='satellite_requested'))
        pos=obs['self']['position'];goal=self.task.get('last_point') or [pos['lon'],pos['lat']]
        actions.append(self.motion.action(self,goal))
        return np.asarray(actions,dtype=np.float64).reshape(-1,4)


base.Group=PlanGroup
base.Directory=PlanDirectory
base.MigratedAgent=PlanAgent
Factory=base.Factory
