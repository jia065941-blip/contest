"""Agent-owned planning from engine-authorized communication groups only."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib,importlib.util,json,os,sys
import numpy as np
from common import HERE,HARNESS,OLD,ENV,MODEL,read,save,digest,episode_inputs

sys.path[:0]=[str(ENV),str(HARNESS),str(OLD)]
from envengine import fastcopy as copy
from user_agents.red.attack_missile_agent import AttackMissileAgent
from user_agents.red.factory import RedAgentFactory
from user_agents.red.commander import RedBaselineCommander
from user_agents.red.priors import initial_targets_from_observation
from user_agents.red.contracts import Position
from harness.cooperative_system import runtime
from harness.cooperative_system.state import SPEEDS
from harness.skill_discovery.common import digest as state_digest

KIND={21000:'H',21001:'M',21002:'L'}


class NoOnlineSelector:
    """Fail closed: this evaluation has zero online LLM or seven-skill choices."""
    def __init__(self,folder):
        self.folder=Path(folder);self.stats=dict(requests=0,accepted=0,keep=0,failed=0,budget_holds=0)
    def ready(self,*args):return False
    def choose(self,*args):raise RuntimeError('Online seven-skill selector is disabled')
    def persist(self):
        save(self.folder/'cost.json',dict(stats=self.stats,requests=[],upper_cny=0.,limit_cny=0.,online_enabled=False))


class Motion:
    def __init__(self,max_steps):
        os.environ['LOCAL_AVOID_GATE_MODE']='mission_v12'
        from policies.red.skills.evasion_ppo.local_runtime import LocalAvoidRuntimePolicy
        import torch
        torch.set_num_threads(1)
        self.policy=LocalAvoidRuntimePolicy(max_steps=max_steps);self.policy.load(str(MODEL))
        self.calls=Counter();self.nonzero=Counter()

    def action(self,agent,goal):
        obs=agent.latest_observation
        encoded=self.policy.encode_motion_observation(obs,agent_id=agent.agent_id,entity_id=agent.entity_id,
            launch_step=agent.launch_step,maneuver_state=agent.set_acc_z_z,
            goal_position=SimpleNamespace(lon=goal[0],lat=goal[1]))
        mask=self.policy.prepare_action_mask(encoded,np.ones(3,dtype=np.bool_))
        selected=self.policy.select_action(encoded,mask)
        acc=(-1.,0.,1.)[selected];agent.set_acc_z_z=int(acc)
        kind=KIND[obs['self']['type']];self.calls[kind]+=1;self.nonzero[kind]+=int(acc!=0)
        return [0,agent.entity_id,acc,0]


class Group:
    def __init__(self,directory,members):
        self.directory=directory;self.members=frozenset(members);self.reports={};self.prepared=-1
        self.key=digest(sorted(members))[:12]
        self.baseline=RedBaselineCommander(directory.public,'r8_satellite_packages',seed=directory.seed,search_polygon=directory.polygon)
        for pid in self.members:self.baseline.register_platform(pid)
        self.memory={};self.aliases={};self.target_aliases={};self.targets={};self.event=0;self.last_review=-45
        self.prior_memory={};self.prior_started={}
        self.last_signature=None;self.actions={};self.followups=[]

    def receive(self,observation,task):
        pid=int(observation['entity_id'])
        if pid not in self.members:raise ValueError('Cross-group report')
        self.reports[pid]=(copy.deepcopy(observation),copy.deepcopy(task))


class Directory:
    def __init__(self,public,config,polygon,folder,shared,test_mode):
        self.public=public;self.seed=config.seed;self.max_steps=config.max_steps;self.polygon=polygon
        self.folder=folder;self.groups={};self.step=-1;self.registered=set();self.metrics=Counter();self.coverage=[]
        self.online_enabled=False
        self.programs=[];self.selector=NoOnlineSelector(folder)
        self.test_mode=test_mode;self.test_used=False;self.ledger=[];self.pending={};self.receipts={}
        self.allocation=episode_inputs()['opening']['allocation']
        spec=importlib.util.spec_from_file_location('frozen_group_prior',HERE/'prior_t03_make/policy.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.prior_step=module.step_skill;self.prior_parameters=read(HERE/'prior_t03_make/parameters.json')

    def receive(self,pid,observation,task):
        if observation['entity_id']!=pid or pid not in self.registered:raise ValueError('Forged own report')
        peers={int(x) for x in observation['self'].get('commRangeInfo',[])}|{pid}
        if not peers<=self.registered:raise ValueError('Unknown member in engine communication group')
        key=tuple(sorted(peers));step=observation['step']
        if step!=self.step:
            self.previous=self.groups;self.groups={};self.step=step
        group=self.groups.get(key)
        if group is None:
            group=getattr(self,'previous',{}).get(key) or Group(self,key);self.groups[key]=group
        group.receive(observation,task);return group

    def acknowledge(self,pid,point):
        row=self.pending.get(pid)
        if row and row['status']=='emitted' and max(abs(a-b) for a,b in zip(point,row['point']))<1e-7:
            row['status']='native_return';self.metrics['native_returns']+=1
            self.receipts[pid]=dict(ack_step=row['step'],ack_target=row['target_id'])
            del self.pending[pid]
            # Ack belongs to the issuing group only; do not route another
            # group's state into a new planner after membership changes.
            for group in self.groups.values():
                if group.key==row['group'] and pid in group.reports:
                    group.reports[pid][1].update(ack_step=row['step'],ack_target=row['target_id'])
        self.metrics['legal_target_coordinate_returns']+=1

    def persist(self):
        return dict(online_actions=dict(metrics=dict(self.metrics),commands=self.ledger,pending=len(self.pending),test_only=bool(self.test_mode)),
            coverage_events=dict(events=self.coverage,scope='single engine-authorized communication group',future_unlaunched=0),
            cost=dict(stats=self.selector.stats,requests=[],upper_cny=0.,limit_cny=0.,online_enabled=False))


class MigratedAgent(AttackMissileAgent):
    def configure(self,directory,motion):
        if self._actor is not None or self._learning_transition is not None:
            raise RuntimeError("History opt-out requires frozen evaluation without a learning actor")
        self.record_history = False
        self.directory=directory;self.motion=motion;self.task={};self._group=None
        directory.registered.add(self.entity_id)

    def set_observation(self,observation):
        self.latest_observation=observation
        self.task.update(self.directory.receipts.get(self.entity_id,{}))
        if observation and observation.get('self'):self._group=self.directory.receive(self.entity_id,observation,self.task)


class Factory(RedAgentFactory):
    def __init__(self,config):
        # A short compatibility run truncates observation time, not the skill's
        # real 3000-second mission horizon or its ETA feasibility conditions.
        config=replace(config,max_steps=3000)
        super().__init__(config)
        self.directory=None;self.motion=Motion(config.max_steps)

    def create_agents(self,public_observation,platforms,*,search_polygon=None):
        if self._created:raise RuntimeError('Factory already used')
        self._created=True
        folder=Path(os.environ['MIGRATED_EPISODE']);shared=Path(os.environ['MIGRATED_SHARED'])
        polygon=tuple(Position(*p) for p in search_polygon)
        self.directory=Directory(initial_targets_from_observation(public_observation),self.config,polygon,folder,shared,os.getenv('MIGRATED_TEST_MODE'))
        agents=[]
        for aid,pid in platforms:
            agent=MigratedAgent(aid,pid,copy.deepcopy(public_observation));agent.configure(self.directory,self.motion);agents.append(agent)
        return agents

    def create_deploy_agent(self,entities,coordinates,coordinates_hm,*,plan_path=None,launch_altitude=None):
        from user_agents.base_agent import BaseAgent,AgentType
        rows=episode_inputs()['compiled']['deployment_rows']
        class Deployment(BaseAgent):
            def __init__(self):super().__init__(-1,-1,AgentType.DEPLOY,{});self.index=0
            def set_observation(self,observation):pass
            def get_action(self):
                batch=rows[self.index:self.index+10];self.index+=len(batch)
                return np.asarray(batch or [[1,-1,0,0,0]],dtype=np.float64)
            def reset(self):super().reset();self.index=0
        return Deployment()
