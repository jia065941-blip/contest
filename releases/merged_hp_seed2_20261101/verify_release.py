"""Validate recorded winning-policy behavior and imports, without a simulator game."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from support import ROOT, APP, ENV, read, sha, verify, payload

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def contracts():
    from skills import cooperation, feature_projector, proposal, merged_policy
    from skills.alternatives import alternatives
    hp=load('contract_hp_tree',APP/'skills/hp_tree/runtime.py')
    reference=read(ROOT/'evidence/reference.json')
    archive=ROOT/'evidence/winner_contracts.tar.gz'
    assert sha(archive)==reference['contracts_archive_sha256']
    lookup={r['map']:r for r in reference['rows']}
    assert len(lookup)==20
    records=commands=strategies=0
    seen=set()
    with tarfile.open(archive) as tar:
        for member in tar:
            raw=tar.extractfile(member).read();case=json.loads(raw);m=case['map']
            assert m not in seen and m in lookup;seen.add(m)
            assert hashlib.sha256(raw).hexdigest()==lookup[m]['contract_fixture_sha256']
            history={};last={};expected_commands=set()
            command_index={(c['step'],c['unit_id'],c['action_type']):c for c in case['commands']}
            assert len(command_index)==len(case['commands'])
            bindings={b['unit_id']:b for b in case['fixed_HP_participants']}
            for row in case['records']:
                unit=row['unit'];assert unit in bindings and row['target']==bindings[unit]['target_id']
                assert row['step']>last.get(unit,-1);last[unit]=row['step']
                prior=history.get(unit,{})
                decision,history[unit]=hp.step(row['context'],prior)
                assert decision==row['decision'],(m,unit,row['step'])
                original=row['original_actions'];expected=set()
                maneuvers=[a for a in original if a[0]==0]
                track=[a for a in original if a[0]==4 and int(a[2])==row['target']]
                assert len(maneuvers)==1
                if row['enabled']:
                    if decision['clear_maneuver'] and maneuvers[0][2]!=0:expected.add(0)
                    if decision['waypoint']['action']=='WAYPOINT' and len(track)==1:expected.add(2)
                actual={kind for kind in (0,2) if (row['step'],unit,kind) in command_index}
                assert expected==actual,(m,unit,row['step'])
                for kind in expected:
                    key=row['step'],unit,kind;c=command_index[key]
                    assert c['legal_context']==row['context'] and c['memory_before']==prior
                    assert c['decision']==decision and c['status']=='native_return'
                    assert c['receipt_step']==row['step']
                    if kind==0:
                        assert c['value']==c['receipt_value']==0
                    else:
                        assert c['point']==decision['waypoint']['point']
                        assert max(abs(a-b) for a,b in zip(c['point'],c['receipt_point']))<1e-7
                    expected_commands.add(key)
                records+=1
            assert expected_commands==set(command_index)
            commands+=len(command_index)
            strategy=case['strategy']
            if strategy:
                ctx=strategy['event']['legal_context'];roles=strategy['original_roles']
                assert cooperation.propose(ctx)['roles']==roles
                features=feature_projector.features(ctx,roles)
                assert features==strategy['legal_features']
                candidates=alternatives(ctx,roles)
                public=[dict(candidate_id='C'+str(i+1),features=c['features']) for i,c in enumerate(candidates)]
                decision=proposal.choose(features,public)
                assert decision==strategy['binding_decision']
                choice='KEEP'
                if decision['candidate_id'] is not None:
                    i=next(i for i,p in enumerate(public) if p['candidate_id']==decision['candidate_id'])
                    selected=candidates[i]
                    assert selected['features']==strategy['tree_binding']['features']
                    assert selected['roles']==strategy['tree_binding']['roles']
                    prediction=merged_policy.predict(selected['features'])
                    assert prediction==strategy['tree_binding']['prediction'];choice=prediction['choice']
                assert choice==strategy['applied_choice'] and choice in ('KEEP','A')
                assert {'KEEP':0,'A':1}[choice]==strategy['applied_mask']
                # Map identity must be rejected, even when the other features are valid.
                try:
                    merged_policy.predict(dict(features,map=m))
                except ValueError:pass
                else:raise AssertionError('Identity feature accepted')
                strategies+=1
            summary=case['native_summary'];objects=summary['objectives']
            score=100*sum(o['weight'] for o in objects if o['final_health']<=0)/sum(o['weight'] for o in objects)
            assert abs(score-lookup[m]['score'])<1e-8 and abs(score-summary['score']['score'])<1e-8
    assert seen==set(lookup)
    assert abs(sum(r['score'] for r in lookup.values())/20-reference['mean'])<1e-10
    from compiler import compile_case
    jobs=read(APP/'jobs.json');plans=0
    for m,choice in read(ROOT/'opening/frozen_choices.json')['choices'].items():
        actual,_=compile_case(ENV/'scenarios/cases'/jobs[m]['scenario']/'scenario.json',choice)
        assert actual==payload(m)['compiled'],m
        plans+=1
    return dict(recorded_maps=20,HP_state_records=records,HP_native_commands=commands,
                strategic_decisions=strategies,frozen_opening_plans_recompiled=plans,
                historical_reference_mean=reference['mean'],new_games=0,external_model_API_calls=0)

def native_imports():
    sys.path[:0]=list(map(str,[APP/'performance_v2',APP/'performance_v3',APP/'r3',ENV,APP/'deps/engine',APP/'deps']))
    from envengine.sdk.Util import _UtilsPy
    import compiled_copy,ownership,communication_copy
    from copy_location import configure
    configure()
    compiled_copy.install();ownership.install();communication_copy.install()
    from r3.adapter import install
    install()
    import agents,skill_runtime,matched
    controls=matched.controls()
    import torch
    torch.set_num_threads(1)
    import base_agents
    motion=base_agents.Motion(max_steps=3000)
    policy=motion.policy
    assert policy.model.algorithm_name=='local_avoid_set_attention_ppo_v1'
    assert policy.model.config.actor_obs_dim==85 and policy.gate_mode=='mission_v12'
    assert not policy.training and not policy._rollout_mode
    assert policy._trace_path is None
    return dict(native_imports='passed',R3_model_loaded=True,actor_observation_dim=85,
                historical_negative_control_maps=controls['completed'],new_games=0)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-imports',action='store_true')
    args=parser.parse_args()
    verify()
    report=native_imports() if args.native_imports else contracts()
    print(json.dumps(dict(status='passed',**report)))

if __name__=='__main__':main()
