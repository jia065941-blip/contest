#!/usr/bin/env python3
"""Build 64 full-legal b10 states by verified merge and missing-branch simulation."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import torch
from train_c0a_goal_b9 import validate_item, subprocess_environment, SIM_ROOT, TEACHER_MODEL, write_json
from train_c0a_goal_b8_advantage import file_sha256,model_sha256


def load_groups(source):
    groups=defaultdict(list)
    for path in sorted(source.glob('round_*/states/*/item.pt')):
        item=torch.load(path,map_location='cpu',weights_only=True)
        if int(item['seed'])>=1000: raise ValueError('formal eval seed in training pool')
        validate_item(item,SimpleNamespace(k=10,rho=.2))
        for i,b in enumerate(item['branch_results']):
            bp=path.parent/f'branches/branch_{i:02d}.json'
            if json.loads(bp.read_text())!=b: raise ValueError(f'branch file/item mismatch: {bp}')
        groups[item['start_state_id']].append((path,item))
    merged=[]
    for state,records in sorted(groups.items()):
        base=copy.deepcopy(records[0][1]); base['boundary_source']='damage_anchors' if any(d['boundary_source']=='damage_anchors' for _,d in records) else 'all_launches'; labels=defaultdict(list); branch_by_slot={}; coords={}; provenance=[]
        for path,item in records:
            for key in ['physical_state_sha256','public_rng_sha256','seed','timestep','executor_id']:
                if item[key]!=base[key]: raise ValueError(f'{state}: inconsistent {key}')
            for key in ['observation','target_features','target_valid_mask']:
                if not torch.equal(item[key],base[key]): raise ValueError(f'{state}: inconsistent {key}')
            for i,slot in enumerate(item['candidate_target_indices'].tolist()):
                b=item['branch_results'][i]
                signature={k:b[k] for k in ['target_id','target_index','official_joint_return','score','end_step','initial_objective_health','final_objective_health','objective_weights','suffix_semantics']}
                if slot in branch_by_slot:
                    previous={k:branch_by_slot[slot][k] for k in signature}
                    if signature!=previous:raise ValueError(f'{state}: repeated branch differs for {slot}')
                    if not torch.equal(coords[slot],item['candidate_coordinates'][i]):raise ValueError('coordinate mismatch')
                branch_by_slot[slot]=b;coords[slot]=item['candidate_coordinates'][i]
                labels[slot].append(float(item['official_joint_returns'][i]))
                bp=path.parent/f'branches/branch_{i:02d}.json'
                provenance.append({'slot':slot,'item':str(path.resolve()),'item_sha256':file_sha256(path),'branch':str(bp.resolve()),'branch_sha256':file_sha256(bp)})
        legal=base['target_valid_mask'][0].nonzero().flatten().tolist()
        missing=sorted(set(legal)-set(labels))
        merged.append({'state':state,'base':base,'labels':dict(labels),'branches':branch_by_slot,'coordinates':coords,'provenance':provenance,'legal':legal,'missing':missing})
    return merged


def select_states(groups):
    targets={'damage_anchors':48,'all_launches':16}
    chosen=[]
    for name,count in targets.items():
        group=[g for g in groups if g['base']['boundary_source']==name]
        group.sort(key=lambda g:(len(g['missing']),g['state']))
        if len(group)<count:raise ValueError(f'insufficient {name}: {len(group)}')
        chosen.extend(group[:count])
    complete={g['state'] for g in groups if not g['missing']}
    if not complete.issubset({g['state'] for g in chosen}):raise ValueError('selection omitted existing complete state')
    return chosen


def make_complete(g):
    item=copy.deepcopy(g['base']); slots=sorted(g['legal'])
    if set(g['labels'])!=set(slots):raise ValueError('incomplete full-legal item')
    item.update({
        'candidate_target_indices':torch.tensor(slots),
        'candidate_target_ids':torch.tensor([g['branches'][s]['target_id'] for s in slots]),
        'candidate_coordinates':torch.stack([g['coordinates'][s] for s in slots]),
        'official_joint_returns':torch.tensor([sum(g['labels'][s])/len(g['labels'][s]) for s in slots],dtype=torch.float64),
        'candidate_roles':['full_legal_evaluated']*len(slots),
        'branch_results':[g['branches'][s] for s in slots],
        'b10_provenance':g['provenance'],'label_estimator':'mean of matched deterministic observations; repeats are not independent RNG samples',
        'all_legal_evaluated':True,'coverage':len(slots),'b10_bridge_verified':g.get('bridge_verified',False),
    })
    return item


def run_sim(args,manifest,out,request,mode):
    request['mode']=mode
    rp=out/f'{mode}_request.json';write_json(rp,request)
    env=subprocess_environment(request,mode,args.checkpoint);env['RED_C0A_B9_REQUEST']=str(rp)
    command=[sys.executable,str(SIM_ROOT/'core/main.py'),'--scenario',manifest['scenario'],'--output-dir',str(out/f'sim_{mode}'),'--max-steps','3000','--total-rounds','1','--render-mode','none','--disable-log-color']
    with (out/f'{mode}.log').open('w') as log:
        result=subprocess.run(command,cwd=SIM_ROOT/'core',env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{mode} failed at {out}; exit={result.returncode}')


def complete_state(args,manifest,g,index):
    out=args.output_dir/'states'/f'state_{index:03d}';out.mkdir(parents=True,exist_ok=True)
    final=out/'complete.pt'
    # Queue-level resume is bound to the immutable protocol and input hashes.
    if final.exists():
        item=torch.load(final,map_location='cpu',weights_only=True)
        if item['start_state_id']!=g['state'] or not item.get('all_legal_evaluated'):raise ValueError('resume identity mismatch')
        return {'state':g['state'],'path':str(final),'resumed':True,'missing_count':len(g['missing'])}
    start=time.monotonic()
    if not g['missing']:
        torch.save(make_complete(g),final)
        return {'state':g['state'],'path':str(final),'missing_count':0,'bridge':False,'seconds':time.monotonic()-start}
    base=g['base']
    # Read original capture request for trace provenance, not reconstructed metadata.
    first=Path(g['provenance'][0]['item']).parent/'capture_request.json'
    original=json.loads(first.read_text())
    request={key:original[key] for key in ['trace','seed','start_state_id','boundary_source','timestep','executor_id']}
    request.update({'k':len(g['legal']),'rho':.2,'sampling_seed':args.seed+index,
                    'snapshot_model_sha256':args.model_sha,'capture_path':str(out/'capture.pt'),
                    'item_path':str(out/'new_branches.pt'),'result_dir':str(out/'branches'),
                    'branch_workers':args.branch_workers})
    run_sim(args,manifest,out,request,'capture')
    captured=torch.load(out/'capture.pt',map_location='cpu',weights_only=True)
    for key in ['physical_state_sha256']:
        if captured[key]!=base[key]:raise ValueError(f'{g["state"]}: fresh capture {key} mismatch')
    for key in ['observation','target_features','target_valid_mask']:
        if not torch.equal(captured[key],base[key]):raise ValueError(f'{g["state"]}: fresh capture {key} mismatch')
    if set(captured['candidate_target_indices'].tolist())!=set(g['legal']):raise ValueError('capture did not enumerate all legal targets')
    reference=min(g['labels']); selected=[reference]+g['missing']
    mapping={slot:i for i,slot in enumerate(captured['candidate_target_indices'].tolist())}
    take=torch.tensor([mapping[s] for s in selected])
    for key in ['candidate_target_indices','candidate_target_ids','candidate_coordinates']:
        captured[key]=captured[key][take]
    captured['candidate_roles']=['bridge_reference']+['missing_legal']*len(g['missing'])
    torch.save(captured,out/'branch_input.pt');request['capture_path']=str(out/'branch_input.pt')
    run_sim(args,manifest,out,request,'branches')
    payload=torch.load(out/'new_branches.pt',map_location='cpu',weights_only=True)
    if payload['public_rng_sha256']!=base['public_rng_sha256']:raise ValueError('new/cached RNG mismatch')
    old=g['branches'][reference]; new=payload['branch_results'][0]
    keys=['target_id','target_index','official_joint_return','score','end_step','initial_objective_health','final_objective_health','objective_weights','suffix_semantics','physical_state_sha256','public_rng_sha256','suppressed_target_change_commands','option_termination_step','teacher_closed_loop_steps']
    mismatch=[k for k in keys if new[k]!=old[k]]
    if mismatch:raise ValueError(f'{g["state"]}: bridge mismatch {mismatch}')
    write_json(out/'bridge_check.json',{'status':'passed','reference_slot':reference,'compared_fields':keys,'new':new,'cached':old})
    for i,slot in enumerate(selected):
        branch=payload['branch_results'][i]
        if branch['physical_state_sha256']!=base['physical_state_sha256'] or branch['public_rng_sha256']!=base['public_rng_sha256'] or branch['non_target_boundary_changes']!=0 or branch['suffix_semantics']!='frozen_teacher_closed_loop_target_locked':raise ValueError('new branch contract mismatch')
        g['labels'].setdefault(slot,[]).append(float(payload['official_joint_returns'][i]))
        g['branches'][slot]=branch;g['coordinates'][slot]=payload['candidate_coordinates'][i]
        bp=out/f'branches/branch_{i:02d}.json'
        if json.loads(bp.read_text())!=branch:raise ValueError('new branch file mismatch')
        g['provenance'].append({'slot':slot,'item':str(out/'new_branches.pt'),'item_sha256':file_sha256(out/'new_branches.pt'),'branch':str(bp),'branch_sha256':file_sha256(bp)})
    g['bridge_verified']=True
    torch.save(make_complete(g),final)
    return {'state':g['state'],'path':str(final),'missing_count':len(g['missing']),'bridge':True,'seconds':time.monotonic()-start}


def fresh_state(args,manifest,g,index):
    out=args.output_dir/'states'/f'state_{index:03d}';out.mkdir(parents=True,exist_ok=True)
    final=out/'complete.pt'
    if final.exists():
        item=torch.load(final,map_location='cpu',weights_only=True)
        if item['start_state_id']!=g['state'] or not item.get('fresh_all_legal') or item.get('old_return_labels_reused') is not False:
            raise ValueError('fresh resume identity/label provenance mismatch')
        return {'state':g['state'],'path':str(final),'resumed':True,'new_target_count':len(g['legal'])}
    start=time.monotonic()
    original=json.loads((Path(g['provenance'][0]['item']).parent/'capture_request.json').read_text())
    request={key:original[key] for key in ['trace','seed','start_state_id','timestep','executor_id']}
    request['boundary_source']=g['base']['boundary_source']
    request.update({'k':len(g['legal']),'rho':.2,'sampling_seed':args.seed+index,
                    'snapshot_model_sha256':args.model_sha,'capture_path':str(out/'capture.pt'),
                    'item_path':str(out/'new_branches.pt'),'result_dir':str(out/'branches'),
                    'branch_workers':args.branch_workers})
    run_sim(args,manifest,out,request,'capture')
    captured=torch.load(out/'capture.pt',map_location='cpu',weights_only=True)
    for key in ['observation','target_features','target_valid_mask']:
        if not torch.equal(captured[key],g['base'][key]):raise ValueError(f'{g["state"]}: selected training input changed: {key}')
    if set(captured['candidate_target_indices'].tolist())!=set(g['legal']):raise ValueError('full legal enumeration failed')
    write_json(out/'historical_state_diagnostic.json',{
        'historical_physical_state_sha256':g['base']['physical_state_sha256'],
        'current_physical_state_sha256':captured['physical_state_sha256'],
        'matches_historical':captured['physical_state_sha256']==g['base']['physical_state_sha256'],
        'policy_inputs_exactly_equal':True,'old_return_labels_reused':False})
    run_sim(args,manifest,out,request,'branches')
    item=torch.load(out/'new_branches.pt',map_location='cpu',weights_only=True)
    validate_item(item,SimpleNamespace(k=len(g['legal']),rho=.2))
    provenance=[]
    for i,slot in enumerate(item['candidate_target_indices'].tolist()):
        bp=out/f'branches/branch_{i:02d}.json'
        if json.loads(bp.read_text())!=item['branch_results'][i]:raise ValueError('fresh branch artifact mismatch')
        provenance.append({'slot':slot,'item':str(out/'new_branches.pt'),'item_sha256':file_sha256(out/'new_branches.pt'),'branch':str(bp),'branch_sha256':file_sha256(bp)})
    item.update({'b10_provenance':provenance,'all_legal_evaluated':True,'coverage':len(g['legal']),
                 'fresh_all_legal':True,'old_return_labels_reused':False,'b10_bridge_verified':False,
                 'label_estimator':'one deterministic common-RNG frozen-teacher continuation per legal target'})
    torch.save(item,final)
    return {'state':g['state'],'path':str(final),'new_target_count':len(g['legal']),'fresh_all_legal':True,'seconds':time.monotonic()-start}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--branch-workers',type=int,default=6);p.add_argument('--seed',type=int,default=20260920)
    p.add_argument('--fresh-all',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    for name in ['source_run','checkpoint','output_dir']:setattr(args,name,getattr(args,name).resolve())
    torch.set_num_threads(1)
    if args.output_dir.exists() and not args.resume:raise ValueError('refusing existing output directory')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    source_protocol=json.loads((args.source_run/'protocol.json').read_text())
    manifest_path=Path(source_protocol['manifest']);manifest=json.loads(manifest_path.read_text())
    if file_sha256(TEACHER_MODEL)!=source_protocol['teacher_sha256']:raise ValueError('teacher changed since cached labels')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True);args.model_sha=model_sha256(payload['model'])
    chosen=select_states(load_groups(args.source_run))
    jobs=[{'index':i,'state':g['state'],'group':g['base']['boundary_source'],'seed':g['base']['seed'],'existing':len(g['labels']),'legal':len(g['legal']),'missing_slots':g['missing'],'new_branch_count':len(g['legal']) if args.fresh_all else len(g['missing'])+bool(g['missing'])} for i,g in enumerate(chosen)]
    paths=[Path(__file__),Path(__file__).with_name('train_c0a_goal_b9.py'),Path(__file__).with_name('train_c0a_goal_b8_advantage.py'),Path(__file__).with_name('c0a_goal_b9_runtime.py'),Path(__file__).with_name('c0a_goal_b9_objective.py'),SIM_ROOT/'core/main.py']
    protocol={'stage':'b10_full_legal_collection','fresh_all_legal':args.fresh_all,'old_return_labels_reused':not args.fresh_all,'state_count':64,'groups':{'damage_anchors':48,'all_launches':16},'retained_complete_states':sum(not g['missing'] for g in chosen),'new_target_evaluations':sum(len(g['legal']) if args.fresh_all else len(g['missing']) for g in chosen),'bridge_evaluations':0 if args.fresh_all else sum(bool(g['missing']) for g in chosen),'source_run':str(args.source_run),'checkpoint':str(args.checkpoint),'checkpoint_sha256':file_sha256(args.checkpoint),'teacher_checkpoint':str(TEACHER_MODEL),'teacher_sha256':file_sha256(TEACHER_MODEL),'manifest':str(manifest_path),'manifest_sha256':file_sha256(manifest_path),'selection':'within each source group, fewest missing labels then lexical state ID; no reward-based selection','seed':args.seed,'code_sha256':{str(x.resolve()):file_sha256(x) for x in paths},'jobs':jobs}
    protocol_path=args.output_dir/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError('resume protocol changed')
    else:write_json(protocol_path,protocol)
    write_json(args.output_dir/'cached_input_hashes.json',{row['item']:row['item_sha256'] for g in chosen for row in g['provenance']})
    print(json.dumps({k:v for k,v in protocol.items() if k not in ['jobs','code_sha256']},ensure_ascii=False),flush=True)
    if args.prepare_only:return
    completed=[]
    write_json(args.output_dir/'progress.json',{'status':'collecting','completed':0,'total':64})
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(fresh_state if args.fresh_all else complete_state,args,manifest,g,i):i for i,g in enumerate(chosen)}
            for future in as_completed(futures):
                row=future.result();completed.append(row)
                write_json(args.output_dir/'progress.json',{'status':'collecting','completed':len(completed),'total':64,'states':completed})
                print(json.dumps({'event':'complete_state',**row},ensure_ascii=False),flush=True)
    except BaseException as error:
        write_json(args.output_dir/'failure.json',{'status':'failed','error':repr(error)})
        raise
    items=[torch.load(args.output_dir/'states'/f'state_{i:03d}'/'complete.pt',map_location='cpu',weights_only=True) for i in range(64)]
    torch.save({'stage':'C0a_goal_b10_full_legal','items':items,'protocol':protocol},args.output_dir/'candidate_batch.pt')
    write_json(args.output_dir/'progress.json',{'status':'complete','completed':64,'total':64,'candidate_batch':str(args.output_dir/'candidate_batch.pt'),'states':completed})
    print('B10_COLLECTION_COMPLETE',flush=True)

if __name__=='__main__':main()
