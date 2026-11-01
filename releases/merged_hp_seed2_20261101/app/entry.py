"""Install user agents and passive own-command receipts; execute official main."""
from pathlib import Path
import importlib,importlib.util,json,os,sys
from common import HERE,HARNESS,ENV,OLD,read,save,sha,MODEL,episode_inputs

from r3.adapter import install
install()
inputs=episode_inputs()

sys.path[:0]=[str(ENV),str(HARNESS),str(OLD),str(HERE)]
import skill_runtime
skill_runtime.install()
from agents import Factory
from envengine.common import SimmerCommandType
instance=[]
class RegisteredFactory(Factory):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);instance.append(self)

# Passive command-return observers. Original native handlers and data are
# invoked unchanged; no environment lifecycle, physics or validation is replaced.
for role in ('H','M','L'):
    name=f'CompCruiseMissile{role}Simulator'
    cls=getattr(importlib.import_module('envengine.simulator.simlulator_impl.'+name),name)
    original=cls.command_received
    def received(simulator,command,_original=original):
        result=_original(simulator,command)
        if instance and instance[0].directory and command.commandTypeId==SimmerCommandType.CHANGE_MISSILE_TARGET:
            point=command.commandAttributes['target']
            instance[0].directory.acknowledge(int(simulator.entity_ext.entity.id),[point['x'],point['y']])
        return result
    cls.command_received=received

import socket,requests
def blocked(*args,**kw):raise RuntimeError('Native process network and external model API calls are disabled')
socket.socket.connect=blocked;socket.socket.connect_ex=blocked;requests.sessions.Session.request=blocked

spec=importlib.util.spec_from_file_location('official_pku_main',ENV/'main.py')
main=importlib.util.module_from_spec(spec);spec.loader.exec_module(main)
main.RedAgentFactory=RegisteredFactory
import user_agents.red
import user_agents.red.factory
user_agents.red.RedAgentFactory=RegisteredFactory
user_agents.red.factory.RedAgentFactory=RegisteredFactory
code=1
try:
    code=main.main(sys.argv[1:])
finally:
    if instance and instance[0].directory:
        obj=instance[0];execution=obj.directory.persist()
        execution['integration']=dict(
            native_seed=obj.directory.seed,model_sha256=sha(obj.motion.model_path),motion='V12-R3',gate=obj.motion.policy.gate_mode,
            actor_obs_dim=obj.motion.policy.model.config.actor_obs_dim,
            algorithm=obj.motion.policy.model.algorithm_name,
            motion_audit=dict(obj.motion.audit),
            motion_calls=dict(obj.motion.calls),
            nonzero_motion_calls=dict(obj.motion.nonzero),group_metrics=dict(obj.directory.metrics),
            pending=len(obj.directory.pending),envengine_module=str(Path(sys.modules['envengine'].__file__).resolve()),
            test_only=bool(obj.directory.test_mode),
            future_unlaunched=0,training=False,main_return_code=code,
            seven_skills_enabled=False,online_requests=0,
            user_factory='Frozen RSI plans + V12-R3 + frozen prior_t03_make',
            candidate_version=inputs['opening']['candidate_version'])
        save(Path(os.environ['MIGRATED_EPISODE'])/'execution.json',execution)
raise SystemExit(code)
