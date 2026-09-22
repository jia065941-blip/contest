#!/usr/bin/env python3
"""b10: target-only continuation from b9 on 64 fully evaluated legal action sets."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from c0a_goal_b10_objective import regret_loss
from train_c0a_goal_b8_advantage import TARGET_PREFIXES,load_model,target_logits,file_sha256,model_sha256,write_json
from train_c0a_goal_b9 import stack_items


def verify_dataset(payload):
    if payload['stage']!='C0a_goal_b10_full_legal' or len(payload['items'])!=64:raise ValueError('expected 64 complete b10 states')
    if not payload['protocol'].get('fresh_all_legal') or payload['protocol'].get('old_return_labels_reused') is not False:raise ValueError('b10 requires fresh complete labels after historical state mismatch')
    items=payload['items'];seen=set();files={}
    for item in items:
        if item['start_state_id'] in seen:raise ValueError('duplicate training state')
        seen.add(item['start_state_id'])
        if not item.get('fresh_all_legal') or item.get('old_return_labels_reused') is not False:raise ValueError('historical labels may not enter recovered b10 dataset')
        if item['seed']>=1000 or not item['all_legal_evaluated']:raise ValueError('eval leakage or incomplete state')
        indices=item['candidate_target_indices'].tolist()
        count=len(indices)
        if not count or any(len(item[k])!=count for k in ['candidate_target_ids','official_joint_returns','branch_results','b10_provenance']):raise ValueError('candidate/return/branch/provenance cardinality mismatch')
        if sorted(row['slot'] for row in item['b10_provenance'])!=sorted(indices):raise ValueError('provenance must cover each legal slot exactly once')
        if len(indices)!=len(set(indices)) or set(indices)!=set(item['target_valid_mask'][0].nonzero().flatten().tolist()):raise ValueError('incomplete legal coverage')
        for i,b in enumerate(item['branch_results']):
            if b['target_index']!=indices[i] or b['target_id']!=int(item['candidate_target_ids'][i]):raise ValueError('branch label alignment error')
            if b['physical_state_sha256']!=item['physical_state_sha256'] or b['public_rng_sha256']!=item['public_rng_sha256'] or b['non_target_boundary_changes'] or b['suffix_semantics']!='frozen_teacher_closed_loop_target_locked':raise ValueError('branch semantics mismatch')
            initial={int(r['id']):float(r['initial_health']) for r in b['summary']['objectives']}
            weights={int(k):float(v) for k,v in b['objective_weights'].items()}
            before={int(k):float(v) for k,v in b['initial_objective_health'].items()};after={int(k):float(v) for k,v in b['final_objective_health'].items()}
            expected=sum(w/sum(weights.values())*max(0.,before[k]-after[k])/initial[k] for k,w in weights.items() if initial[k]>0)
            if abs(expected-b['official_joint_return'])>1e-9 or abs(expected-float(item['official_joint_returns'][i]))>1e-9:raise ValueError('formal label reconstruction failed')
        source_paths={row['item'] for row in item['b10_provenance']}
        if len(source_paths)!=1:raise ValueError('fresh state must cite one complete branch item')
        recorded=torch.load(next(iter(source_paths)),map_location='cpu',weights_only=True)
        for key in ['observation','target_features','target_valid_mask','candidate_target_indices','candidate_target_ids','candidate_coordinates','official_joint_returns']:
            if not torch.equal(item[key],recorded[key]):raise ValueError(f'training tensor differs from evaluated source item: {key}')
        for key in ['start_state_id','seed','timestep','executor_id','boundary_source','physical_state_sha256','public_rng_sha256','snapshot_model_sha256']:
            if item[key]!=recorded[key]:raise ValueError(f'training state differs from source item: {key}')
        for row in item['b10_provenance']:
            actual=json.loads(Path(row['branch']).read_text())
            expected=item['branch_results'][indices.index(row['slot'])]
            if actual!=expected or actual['target_index']!=row['slot']:raise ValueError('training branch content differs from cited artifact')
            for name in ['item','branch']:
                if row[name] in files and files[row[name]]!=row[name+'_sha256']:raise ValueError('inconsistent provenance hash')
                files[row[name]]=row[name+'_sha256']
    for path,expected in files.items():
        if file_sha256(Path(path))!=expected:raise ValueError(f'changed input: {path}')
    groups={name:sum(x['boundary_source']==name for x in items) for name in ['damage_anchors','all_launches']}
    if groups!={'damage_anchors':48,'all_launches':16}:raise ValueError('state distribution mismatch')
    return {'fresh_all_legal':True,'old_return_labels_reused':False,'states':len(items),'legal_branches':sum(len(x['candidate_target_indices']) for x in items),'groups':groups,'verified_provenance_files':len(files),'training_seeds':sorted({x['seed'] for x in items}),'all_return_labels_reconstructed':True}


def measure(model,old,data,coef):
    with torch.no_grad():
        logits=target_logits(model,data['observation'],data['features'],data['valid'])
        loss,d=regret_loss(logits,old,data['valid'],data['indices'],data['returns'],data['candidate_mask'],coef)
        return {'loss':float(loss),'expected_regret':float(d['regret']),'expected_regret_score_points':100*float(d['regret']),'kl_b9_to_current':float(d['kl']),'expected_suffix_return':float(d['expected_return']),'greedy_suffix_return':float(d['top1_return']),'mean_best_suffix_return':float(d['maximum_returns'].mean()),'mean_total_variation_from_b9':float((.5*(d['probabilities']-old.softmax(-1)).abs().sum(-1)).mean())}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source-checkpoint',type=Path,required=True);p.add_argument('--candidate-batch',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=48)
    p.add_argument('--learning-rate',type=float,default=2e-5);p.add_argument('--kl-coef',type=float,default=.01)
    p.add_argument('--seed',type=int,default=20260920);p.add_argument('--device',default='cpu')
    args=p.parse_args()
    for name in ['source_checkpoint','candidate_batch','output_dir']:setattr(args,name,getattr(args,name).resolve())
    if args.output_dir.exists():raise ValueError('refusing existing training output')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    payload=torch.load(args.candidate_batch,map_location='cpu',weights_only=True)
    validation=verify_dataset(payload)
    if file_sha256(args.source_checkpoint)!=payload['protocol']['checkpoint_sha256']:raise ValueError('training source differs from collection b9')
    source,config,model=load_model(args.source_checkpoint,args.device);model.eval()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    params=[];names=[]
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TARGET_PREFIXES))
        if parameter.requires_grad:params.append(parameter);names.append(name)
    if not all(any(n.startswith(prefix) for n in names) for prefix in TARGET_PREFIXES):raise ValueError('missing target module')
    optimizer=torch.optim.Adam(params,lr=args.learning_rate)
    data=stack_items(payload['items'],args.device)
    with torch.no_grad():old=target_logits(model,data['observation'],data['features'],data['valid']).detach().clone()
    code=[Path(__file__),Path(__file__).with_name('c0a_goal_b10_objective.py'),Path(__file__).with_name('train_c0a_goal_b9.py'),Path(__file__).with_name('train_c0a_goal_b8_advantage.py')]
    protocol={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    protocol.update({'stage':'C0a_goal_b10_regret','objective':'mean_k sum_legal pi(a|o_k) * stopgrad(max_G_k - G_k(a)) + lambda * KL(pi_b9 || pi_current)','return_units':'raw formal weighted-health suffix return in [0,1]; score-point diagnostics multiply by 100; no per-state normalization','reference_policy':'fixed frozen b9 for all 48 updates','state_weighting':'uniform over all 64 unique states, including ties','source_sha256':file_sha256(args.source_checkpoint),'data_sha256':file_sha256(args.candidate_batch),'code_sha256':{str(x.resolve()):file_sha256(x) for x in code},'dataset_validation':validation,'trainable_names':names,'fresh_optimizer_state_count':len(optimizer.state),'source_optimizer_loaded':False,'gradient_clipping':False,'elite_cross_entropy':False,'promotion_evaluation_performed':False})
    write_json(args.output_dir/'protocol.json',protocol);write_json(args.output_dir/'data_validation.json',validation)
    before=measure(model,old,data,args.kl_coef);history=[];start=time.monotonic()
    for step in range(1,args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        logits=target_logits(model,data['observation'],data['features'],data['valid'])
        loss,_=regret_loss(logits,old,data['valid'],data['indices'],data['returns'],data['candidate_mask'],args.kl_coef)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('invalid target gradient')
        grad_norm=float(torch.sqrt(sum(p.grad.detach().double().square().sum() for p in params)))
        optimizer.step()
        row={'step':step,'gradient_norm':grad_norm,**measure(model,old,data,args.kl_coef)};history.append(row)
        write_json(args.output_dir/'training_history.json',history)
        write_json(args.output_dir/'progress.json',{'status':'training','completed_steps':step,'total_steps':args.steps,'latest':row})
        print(json.dumps(row),flush=True)
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v.detach().cpu(),initial[k])]
    frozen_changed=[k for k in changed if not k.startswith(TARGET_PREFIXES)]
    if frozen_changed or not changed:raise ValueError('parameter isolation/update failed')
    if sorted({int(s['step']) for s in optimizer.state.values()})!=[args.steps]:raise ValueError('Adam lifecycle error')
    with torch.no_grad():
        logits=target_logits(model,data['observation'],data['features'],data['valid'])
        _,detail=regret_loss(logits,old,data['valid'],data['indices'],data['returns'],data['candidate_mask'],args.kl_coef)
        per_state=[]
        for i,item in enumerate(payload['items']):
            full=detail['full_returns'][i];valid=data['valid'][i]
            b9_slot=int(old[i].argmax());b10_slot=int(logits[i].argmax())
            per_state.append({'state':item['start_state_id'],'group':item['boundary_source'],'b9_slot':b9_slot,'b10_slot':b10_slot,'b9_greedy_return':float(full[b9_slot]),'b10_greedy_return':float(full[b10_slot]),'b9_expected_return':float((old[i].softmax(-1)*full).sum()),'b10_expected_return':float((logits[i].softmax(-1)*full).sum()),'best_return':float(full[valid].max()),'b10_regret':float(detail['state_regret'][i])})
    write_json(args.output_dir/'per_state_training_metrics.json',per_state)
    cp=args.output_dir/'checkpoints/working.pt';cp.parent.mkdir()
    checkpoint={'algorithm':source['algorithm'],'config':asdict(config),'model':model.state_dict(),'optimizer':optimizer.state_dict(),'update_count':args.steps,'transition_count':64,'continuation':{'stage':'C0a_goal_b10_regret','source_checkpoint':str(args.source_checkpoint),'source_sha256':protocol['source_sha256'],'objective':'full_legal_expected_regret_plus_fixed_b9_kl','source_optimizer_loaded':False,'target_only':True},'metrics':history[-1]}
    torch.save(checkpoint,cp)
    summary={'status':'complete','model':'b10','states':64,'legal_labels':validation['legal_branches'],'adam_steps':args.steps,'before':before,'after':history[-1],'changed_target_tensors':len(changed),'changed_frozen_tensors':frozen_changed,'output_checkpoint':str(cp),'output_sha256':file_sha256(cp),'output_model_sha256':model_sha256(model.state_dict()),'elapsed_seconds':time.monotonic()-start,'promotion_evaluation_performed':False,'metrics_scope':'training states only; no held-out performance claim'}
    write_json(args.output_dir/'training_summary.json',summary);write_json(args.output_dir/'progress.json',summary)
    after=history[-1]
    report=['# b10 全合法集 regret 训练完成', '',
            '64 个独立训练状态、960 个新仿真合法目标标签，48 次目标 Transformer Adam 更新。', '',
            '| 训练集指标 | b9 初始化 | b10 最终 |', '|---|---:|---:|']
    for key in ['expected_regret','expected_regret_score_points','expected_suffix_return','greedy_suffix_return','kl_b9_to_current']:
        report.append(f'| {key} | {before[key]:.12g} | {after[key]:.12g} |')
    report.extend(['',f'检查点：`{cp}`',f'SHA256：`{summary["output_sha256"]}`',
                   f'变化的目标张量：{len(changed)}；非目标张量变化：0。', '',
                   '回报标签停止梯度；不使用 elite CE，不强制优秀候选均匀化，不按状态回报跨度归一化。KL 参考全程固定为 b9。', '',
                   '全部指标来自这 64 个训练状态的完整候选回报表，不是独立 E01 晋级评估；不能据此宣称泛化改善。', '',
                   '原缓存物理指纹无法严格复现，故最终 960 个回报全部重新仿真，旧回报未用于训练。', ''])
    (args.output_dir/'TRAINING_REPORT.md').write_text('\n'.join(report))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
