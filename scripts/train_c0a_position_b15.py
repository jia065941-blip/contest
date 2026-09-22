#!/usr/bin/env python3
"""One isolated C0a_position batch: 64 distinct seeds, paired live-fork returns."""
from __future__ import annotations
import argparse,copy,json,math,os,sys,time,traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import torch
from torch.distributions import Normal,kl_divergence
from train_c0a_goal_b9 import subprocess_environment,scenario_entity_types,SIM_ROOT
from train_c0a_goal_b8_advantage import load_model,file_sha256,write_json,model_sha256
from tools.build_native_guidance_manifest import scenario_targets,nearest_target
from evaluate_c0a_goal_b14_e01 import validate_branch
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'competition-platform-refine-logs/target_transformer_main_20260914'
RUNTIME=ROOT/'scripts/c0a_position_runtime/c0a_goal_b9_runtime.py'
SOURCE=BASE/'runs/c0a_goal_b14_continuation_20260921/training/checkpoints/working.pt'
SOURCE_SHA='4609add14e70a5e25df87ddb3377756c2e45b0d3fa136d4de66406f26ca2be6e'
MANIFEST=SIM_ROOT/'refine-logs/target_transformer_main_20260914/inputs/final20_guidance_manifest64.json'
PREFIXES=('initial_mean_head.','initial_log_std')

def require(value,message):
    if not value:raise RuntimeError(message)

def catalogue(manifest):
    types=scenario_entity_types(Path(manifest['scenario']));jobs=[]
    rows=sorted(manifest['trajectories'],key=lambda r:int(r['seed']))
    for i,row in enumerate(rows):
        trace=json.loads(Path(row['trace']).read_text());kind=(21000,21001,21002)[i%3]
        launches=[(int(step['step']),int(a['executor_id']),a) for step in trace['steps'] for a in step['actions']
                  if int(a.get('commandType_id',-1))==200 and types.get(int(a['executor_id']))==kind]
        require(bool(launches),f'no launch type {kind}, seed {row["seed"]}')
        launches.sort(key=lambda x:(x[1],x[0]));step,executor,command=launches[(i//3)%len(launches)]
        targets=scenario_targets(Path(manifest['scenario']),set(map(int,trace['summary']['score']['objective_weights'])))
        target=nearest_target(command,targets)
        jobs.append({'order':i+1,'seed':int(row['seed']),'trace':row['trace'],'timestep':step,'executor_id':executor,
                     'entity_type':kind,'teacher_target_id_hint':target, 
                     'start_state_id':f'seed{row["seed"]}:step{step}:entity{executor}'})
    require(len(jobs)==len({x['seed'] for x in jobs})==64,'expected64 unique training seeds')
    require(all(0<=x['seed']<1000 for x in jobs),'evaluation seeds in training')
    return jobs

def validate_item(item,job,smoke):
    for key in ('seed','timestep','executor_id','start_state_id'):require(item[key]==job[key],f'identity {key}')
    require(item['snapshot_model_sha256']==SOURCE_SHA,'checkpoint mismatch')
    branches=item['branch_results'];require(len(branches)==(3 if smoke else 2),'branch count')
    for i,b in enumerate(branches):
        validate_branch_position(b,item,i)
    if smoke:
        left,right=branches[1:]
        for k in ('score','official_joint_return','end_step','summary','identity_trace','common_snapshot_python_sha256'):
            # StepTrace receipt path differs; compare its semantic digest below.
            if k=='identity_trace':continue
            require(left[k]==right[k],f'duplicate teacher mismatch {k}')
        require(left['exact_boundary_actions_equal'] and right['exact_boundary_actions_equal'],'teacher boundary altered')
        require(left['identity_trace']['sha256']==right['identity_trace']['sha256'],'teacher step traces differ')
    bounds=item['deploy_bounds'];pos=item['candidate_position_commands'][0]['lla']
    require(bounds[0]<=pos['x']<=bounds[1] and bounds[2]<=pos['y']<=bounds[3],'position out of bounds')
    require(all(math.isfinite(b['official_joint_return']) for b in branches),'nonfinite returns')

def validate_branch_position(branch,item,index):
    # Position stage returns all target decisions to the same frozen teacher.
    temporary=dict(branch);temporary['suffix_semantics']=('frozen_teacher_closed_loop_search_then_release' if branch['target_id']==-100 else 'frozen_teacher_closed_loop_target_locked')
    validate_branch(temporary,item,index)
    require(branch['suffix_semantics']=='frozen_teacher_closed_loop_native_unlocked','position continuation mismatch')
    require(branch['non_position_boundary_changes']==0,'non-position change')
    require(branch['position_command']==item['candidate_position_commands'][index],'position receipt mismatch')

def collect(job,out,manifest,smoke=False):
    started=time.monotonic();out.mkdir(parents=True)
    request={**job,'sampling_seed':20260921+job['order'],'snapshot_model_sha256':SOURCE_SHA,
             'capture_path':str(out/'capture.pt'),'item_path':str(out/'item.pt'),'result_dir':str(out/'branches'),
             'branch_workers':2,'identity_audit':smoke,'smoke_duplicate_teacher':smoke}
    for mode in ('capture','branches'):
        request['mode']=mode;rp=out/f'{mode}_request.json';write_json(rp,request)
        env=subprocess_environment(request,mode,SOURCE)
        env['PYTHONPATH']=str(RUNTIME.parent)+os.pathsep+env['PYTHONPATH'];env['RED_C0A_B9_REQUEST']=str(rp)
        cmd=[sys.executable,str(SIM_ROOT/'core/main.py'),'--scenario',manifest['scenario'],'--output-dir',str(out/f'sim_{mode}'),
             '--max-steps','3000','--total-rounds','1','--render-mode','none','--disable-log-color']
        with (out/f'{mode}.log').open('w') as f:
            result=subprocess.run(cmd,cwd=SIM_ROOT/'core',env=env,stdout=f,stderr=subprocess.STDOUT)
        require(result.returncode==0,f'{mode} failed seed{job["seed"]}, see {out}')
    item=torch.load(out/'item.pt',map_location='cpu',weights_only=True);validate_item(item,job,smoke)
    files=[out/'capture.pt',out/'item.pt',*[out/f'branches/branch_{i:02d}.json' for i in range(len(item['branch_results']))]]
    student,teacher=item['branch_results'][:2]
    row={**job,'item_path':str(out/'item.pt'),'smoke':smoke,'student_score':student['score'],'teacher_score':teacher['score'],
         'advantage':student['official_joint_return']-teacher['official_joint_return'],'seconds':time.monotonic()-started,
         'artifact_sha256':{str(p):file_sha256(p) for p in files}}
    write_json(out/'result.json',row);return row

def norm(gs):return float(sum(g.detach().double().square().sum() for g in gs).sqrt())

def update(items,out,steps=4):
    source,config,model=load_model(SOURCE,'cpu');model.eval()
    initial={k:v.detach().clone() for k,v in model.state_dict().items()};params=[];names=[]
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(PREFIXES))
        if p.requires_grad:params.append(p);names.append(name)
    obs=torch.cat([x['observation'] for x in items]);features=torch.cat([x['target_features'] for x in items]);valid=torch.cat([x['target_valid_mask'] for x in items])
    raw=torch.stack([x['sampled_initial_raw'] for x in items]);mean=torch.stack([x['old_initial_mean'] for x in items]);logs=torch.stack([x['old_initial_log_std'] for x in items])
    old=Normal(mean,logs.exp());oldlp=old.log_prob(raw).sum(-1).detach()
    advantage=torch.tensor([float(x['official_joint_returns'][0]-x['official_joint_returns'][1]) for x in items]).detach()
    def distribution():
        m=model.distribution_parameters(obs,target_features=features,target_valid_mask=valid)['initial_mean'];return Normal(m,model.initial_log_std.exp().expand_as(m))
    with torch.no_grad():
        d=distribution();require(torch.allclose(d.loc,mean,atol=1e-7,rtol=1e-6),'old policy means do not reconstruct')
        require(torch.equal(d.scale,logs.exp()),'old policy std mismatch')
    opt=torch.optim.Adam(params,lr=2e-5);history=[]
    for step in range(1,steps+1):
        opt.zero_grad(set_to_none=True);d=distribution();ratio=(d.log_prob(raw).sum(-1)-oldlp).exp()
        policy_loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
        kl=kl_divergence(old,d).sum(-1).mean();loss=policy_loss+.01*kl
        pg=torch.autograd.grad(policy_loss,params,retain_graph=True);kg=torch.autograd.grad(.01*kl,params,retain_graph=True)
        loss.backward();require(torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),'nonfinite update')
        premean=d.loc.detach().clone();prestd=d.scale.detach().clone();opt.step()
        with torch.no_grad():
            after=distribution();stagekl=kl_divergence(old,after).sum(-1);stepkl=kl_divergence(Normal(premean,prestd),after).sum(-1)
            afterratio=(after.log_prob(raw).sum(-1)-oldlp).exp()
        row={'attempt':step,'accepted':True,'accepted_updates':step,'rollback_reason':None,'learning_rate':2e-5,'optimizer_step':step,
             'policy_loss':float(policy_loss.detach()),'kl_term':float(kl.detach()),'policy_gradient_norm':norm(pg),'weighted_kl_gradient_norm':norm(kg),
             'gradient_cosine':float(sum((a*b).sum() for a,b in zip(pg,kg)))/(norm(pg)*norm(kg)) if norm(pg)*norm(kg)>0 else None,
             'stage_kl_mean':float(stagekl.mean()),'stage_kl_max':float(stagekl.max()),'step_kl_mean':float(stepkl.mean()),
             'positive_advantage_density_ratio':float(afterratio[advantage>0].mean()) if bool((advantage>0).any()) else None,
             'before_initial_mean':premean.tolist(),'after_initial_mean':after.loc.tolist(),'after_initial_std':after.scale.tolist()}
        history.append(row);write_json(out/'training_history.json',history)
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v,initial[k])]
    require(all(k.startswith(PREFIXES) for k in changed),'frozen parameter changed')
    require(bool(changed) or not bool((advantage!=0).any()),'nonzero advantages but no update')
    output={'algorithm':source['algorithm'],'config':asdict(config),'model':model.state_dict(),'optimizer':opt.state_dict(),
            'update_count':steps,'transition_count':len(items),'continuation':{'stage':'C0a_position','source_checkpoint':str(SOURCE),'source_sha256':SOURCE_SHA,
            'trainable_names':names,'position_only':True,'source_optimizer_loaded':False},'metrics':history[-1]}
    cp=out/'checkpoints/working.pt';cp.parent.mkdir(parents=True);torch.save(output,cp)
    summary={'status':'complete','stage':'C0a_position','attempted_updates':steps,'accepted_updates':steps,'rollback_counts':{'step_kl':0,'stage_kl':0,'score_probe':0},
             'hard_acceptance_gates_enabled':False,'learning_rate':2e-5,'optimizer_inherited':False,'changed_tensors':changed,'changed_non_position_tensors':[],
             'training_states':len(items),'positive_advantage_states':int((advantage>0).sum()),'negative_advantage_states':int((advantage<0).sum()),
             'output_checkpoint':str(cp),'output_sha256':file_sha256(cp),'model_sha256':model_sha256(model.state_dict()),'independent_evaluation_performed':False}
    write_json(out/'training_summary.json',summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    out=args.output_dir.resolve();require(not out.exists(),'output exists');out.mkdir(parents=True)
    torch.set_num_threads(1);torch.manual_seed(20260921)
    require(file_sha256(SOURCE)==SOURCE_SHA,'unexpected source checkpoint')
    promotion=json.loads((BASE/'promotions/b14_c0a_goal_user_override_20260921.json').read_text())
    require(promotion['promoted'] and promotion['checkpoint_sha256']==SOURCE_SHA,'promotion prerequisite')
    manifest=json.loads(MANIFEST.read_text());jobs=catalogue(manifest)
    ep=json.loads((BASE/'validations/b14_e01_new32_20260921/protocol.json').read_text())
    heldout=set(ep['seeds'])|set(ep['excluded_previous_evaluation_seeds'])
    require(not {j['seed'] for j in jobs}&heldout,'training evaluation overlap')
    code=[Path(__file__),RUNTIME,ROOT/'scripts/c0a_goal_b9_identity.py',SIM_ROOT/'core/main.py',SIM_ROOT/'core/envengine/environment/training_env.py',SIM_ROOT/'experiments/unified_mappo/model.py',SIM_ROOT/'policies/red/learning/unified_mappo_policy.py']
    protocol={'stage':'C0a_position','candidate':'b15','source_checkpoint':str(SOURCE),'source_sha256':SOURCE_SHA,
              'manifest':str(MANIFEST),'manifest_sha256':file_sha256(MANIFEST),'jobs':jobs,'seeds':[j['seed'] for j in jobs],
              'entity_type_counts':dict(Counter(j['entity_type'] for j in jobs)),'workers':args.workers,'branch_workers':2,
              'selection':'64 existing distinct training seeds; rotate H/M/L by sorted seed order, then rotate launch entity; no damage or return selection within traces',
              'continuation':'same live fork; original teacher boundary target and all other actions; only sampled initial DEPLOY differs; both branches then native unlocked frozen teacher closed loop',
              'objective':'clipped PPO position-only surrogate with detached paired official suffix return advantage, plus 0.01 KL(batch source || current); continuous action density, not categorical regret',
              'steps':4,'learning_rate':2e-5,'optimizer':'fresh Adam','hard_acceptance_gates_enabled':False,'trainable_prefixes':list(PREFIXES),
              'scope':'one launch position per seed; not whole-team deployment; b14 target head frozen',
              'smoke':'first3 distinct training seeds H/M/L each include extra identical teacher branch; passed episodes reused in same64-state batch',
              'code_sha256':{str(x):file_sha256(x) for x in code},'trace_sha256':{j['trace']:file_sha256(Path(j['trace'])) for j in jobs}}
    write_json(out/'protocol.json',protocol);rows=[]
    def progress(status):write_json(out/'progress.json',{'status':status,'completed':len(rows),'total':64,'pid':os.getpid(),'updated_unix':time.time()})
    try:
        progress('smoke_running')
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(collect,j,out/'episodes'/f'i{j["order"]:02d}_s{j["seed"]}',manifest,True) for j in jobs[:3]]
            for f in as_completed(futures):rows.append(f.result());progress('smoke_running')
        rows.sort(key=lambda r:r['order']);smoke_items=[torch.load(r['item_path'],map_location='cpu',weights_only=True) for r in rows]
        update(smoke_items,out/'smoke_update_check')
        write_json(out/'smoke_checks.json',{'status':'PASS','entity_types':[r['entity_type'] for r in rows],'teacher_identity_pairs':3,'full_python_state_fingerprint':True,'exact_teacher_step_trace_match':True,'position_only_optimizer_check':True,'smoke_checkpoint_not_used_for_training':True})
        print('SMOKE_PASS; collecting remaining61 unique states',flush=True);progress('collecting')
        write_json(out/'episodes.json',rows)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(collect,j,out/'episodes'/f'i{j["order"]:02d}_s{j["seed"]}',manifest) for j in jobs[3:]]
            for f in as_completed(futures):
                rows.append(f.result());rows.sort(key=lambda r:r['order']);write_json(out/'episodes.json',rows);progress('collecting');print(json.dumps({'completed':len(rows),'total':64}),flush=True)
        for path,sha in protocol['code_sha256'].items():require(file_sha256(Path(path))==sha,'code changed during collection')
        require(file_sha256(SOURCE)==SOURCE_SHA,'source changed')
        for r in rows:
            for path,sha in r['artifact_sha256'].items():require(file_sha256(Path(path))==sha,'artifact changed')
        items=[torch.load(r['item_path'],map_location='cpu',weights_only=True) for r in rows]
        require(len(items)==len({x['seed'] for x in items})==64,'64-state uniqueness failed')
        torch.save({'stage':'C0a_position','protocol':protocol,'items':items},out/'position_batch.pt')
        progress('training');summary=update(items,out/'training');progress('complete')
        write_json(out/'completion.json',{'status':'complete','training_summary':summary,'course_promoted':False,'evaluation_pending':True})
    except BaseException as exc:
        write_json(out/'progress.json',{'status':'failed','completed':len(rows),'total':64,'error':str(exc),'pid':os.getpid()});traceback.print_exc();raise

if __name__=='__main__':main()
