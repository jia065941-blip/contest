#!/usr/bin/env python3
"""Same-budget regret training with explicit gradient, acceptance and probability traces."""
from __future__ import annotations
import argparse,json,math,time
from dataclasses import asdict
from pathlib import Path
import torch
from train_c0a_goal_b10_regret import verify_dataset
from c0a_goal_b10_objective import regret_loss
from train_c0a_goal_b9 import stack_items
from train_c0a_goal_b8_advantage import TARGET_PREFIXES,load_model,target_logits,file_sha256,model_sha256,write_json


def norm(gs):
    return float(torch.sqrt(sum(g.detach().double().square().sum() for g in gs)))


def dot(left,right):
    return float(sum((a.detach().double()*b.detach().double()).sum() for a,b in zip(left,right)))


def optimizer_steps(optimizer):
    values=[int(s['step']) for s in optimizer.state.values() if 'step' in s]
    return {'parameter_states':len(values),'min':min(values) if values else 0,'max':max(values) if values else 0}


def probability_kl(old,new,valid):
    lold=old.masked_fill(~valid,-torch.inf).log_softmax(-1)
    lnew=new.masked_fill(~valid,-torch.inf).log_softmax(-1)
    return (lold.exp()*(lold.masked_fill(~valid,0)-lnew.masked_fill(~valid,0))).sum(-1)


def describe(logits,anchor,data,coef,masks):
    loss,d=regret_loss(logits,anchor,data['valid'],data['indices'],data['returns'],data['candidate_mask'],coef)
    probs=d['probabilities'];best=(probs*masks['optimal']).sum(-1);good=(probs*masks['near_optimal']).sum(-1)
    better=(probs*masks['better_than_initial_greedy']).sum(-1);info=masks['informative']
    return {'loss':float(loss),'expected_regret':float(d['regret']),'expected_regret_score_points':100*float(d['regret']),
            'kl_anchor_to_current':float(d['kl']),'weighted_kl_term':coef*float(d['kl']),
            'stage_kl_max':float(d['state_kl'].max()),'stage_kl_p90':float(torch.quantile(d['state_kl'].double(),.9)),
            'expected_suffix_return':float(d['expected_return']),'greedy_suffix_return':float(d['top1_return']),
            'optimal_probability_mean':float(best.mean()),'near_optimal_probability_mean':float(good.mean()),
            'optimal_probability_informative_mean':float(best[info].mean()) if bool(info.any()) else None,
            'near_optimal_probability_informative_mean':float(good[info].mean()) if bool(info.any()) else None,
            'better_than_initial_greedy_probability_mean':float(better.mean())},d


def make_masks(logits,anchor,data,coef):
    _,d=regret_loss(logits,anchor,data['valid'],data['indices'],data['returns'],data['candidate_mask'],coef)
    full=d['full_returns'];valid=data['valid'];best=d['maximum_returns'];minimum=full.masked_fill(~valid,torch.inf).min(-1).values
    initial_slots=logits.masked_fill(~valid,-torch.inf).argmax(-1)
    initial_returns=full.gather(1,initial_slots[:,None])
    return {'optimal':valid & ((best[:,None]-full)<=1e-9),
            'near_optimal':valid & ((best[:,None]-full)<=.001),
            'better_than_initial_greedy':valid & (full>initial_returns+1e-9),
            'informative':(best-minimum)>1e-9,'initial_slots':initial_slots}


def state_records(logits,anchor,data,items,masks,coef):
    _,d=regret_loss(logits,anchor,data['valid'],data['indices'],data['returns'],data['candidate_mask'],coef)
    records=[]
    for i,item in enumerate(items):
        legal=data['valid'][i].nonzero().flatten();p=d['probabilities'][i];greedy=int(logits[i].masked_fill(~data['valid'][i],-torch.inf).argmax())
        id_by_slot=dict(zip(item['candidate_target_indices'].tolist(),item['candidate_target_ids'].tolist()))
        records.append({'state':item['start_state_id'],'seed':item['seed'],'informative':bool(masks['informative'][i]),
                        'legal_slots':legal.tolist(),'legal_target_ids':[id_by_slot[int(x)] for x in legal],
                        'probabilities':p[legal].tolist(),'optimal_probability':float(p[masks['optimal'][i]].sum()),
                        'near_optimal_probability':float(p[masks['near_optimal'][i]].sum()),
                        'better_than_initial_greedy_probability':float(p[masks['better_than_initial_greedy'][i]].sum()),
                        'greedy_slot':greedy,'greedy_target_id':id_by_slot[greedy],
                        'greedy_return':float(d['full_returns'][i,greedy]),'expected_regret':float(d['state_regret'][i]),
                        'stage_kl':float(d['state_kl'][i])})
    return records


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ['source-checkpoint','anchor-checkpoint','candidate-batch','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=48);p.add_argument('--learning-rate',type=float,default=2e-5);p.add_argument('--kl-coef',type=float,default=.01)
    p.add_argument('--seed',type=int,default=20260920);p.add_argument('--expected-final-checkpoint',type=Path);p.add_argument('--label',default='b11')
    args=p.parse_args()
    for key in ['source_checkpoint','anchor_checkpoint','candidate_batch','output_dir']:
        setattr(args,key,getattr(args,key).resolve())
    if args.output_dir.exists():raise ValueError('refusing existing output directory')
    if args.steps!=48:raise ValueError('diagnostic contract requires exactly 48 updates')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    payload=torch.load(args.candidate_batch,map_location='cpu',weights_only=True);validation=verify_dataset(payload)
    if file_sha256(args.anchor_checkpoint)!=payload['protocol']['checkpoint_sha256']:raise ValueError('anchor must remain collection b9')
    source,config,model=load_model(args.source_checkpoint,'cpu');anchor_source,anchor_config,anchor_model=load_model(args.anchor_checkpoint,'cpu')
    if asdict(config)!=asdict(anchor_config):raise ValueError('anchor/current config mismatch')
    model.eval();anchor_model.eval();initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    params=[];names=[]
    for name,value in model.named_parameters():
        value.requires_grad_(name.startswith(TARGET_PREFIXES))
        if value.requires_grad:params.append(value);names.append(name)
    for value in anchor_model.parameters():value.requires_grad_(False)
    for name,value in model.state_dict().items():
        if not name.startswith(TARGET_PREFIXES) and not torch.equal(value,anchor_model.state_dict()[name]):raise ValueError('non-target source/anchor mismatch')
    optimizer=torch.optim.Adam(params,lr=args.learning_rate);data=stack_items(payload['items'],'cpu')
    with torch.no_grad():
        anchor=target_logits(anchor_model,data['observation'],data['features'],data['valid']).detach().clone()
        start_logits=target_logits(model,data['observation'],data['features'],data['valid']).detach().clone()
        masks=make_masks(start_logits,anchor,data,args.kl_coef);before,_=describe(start_logits,anchor,data,args.kl_coef,masks)
        initial_states=state_records(start_logits,anchor,data,payload['items'],masks,args.kl_coef)
    args.output_dir.mkdir(parents=True)
    code=[Path(__file__),Path(__file__).with_name('c0a_goal_b10_objective.py'),Path(__file__).with_name('train_c0a_goal_b10_regret.py'),Path(__file__).with_name('train_c0a_goal_b9.py'),Path(__file__).with_name('train_c0a_goal_b8_advantage.py')]
    gates={'step_kl':{'enabled':False,'threshold':None},'stage_kl':{'enabled':False,'threshold':None},'score_probe':{'enabled':False,'threshold':None}}
    protocol={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    protocol.update({'source_sha256':file_sha256(args.source_checkpoint),'anchor_sha256':file_sha256(args.anchor_checkpoint),
                     'data_sha256':file_sha256(args.candidate_batch),'code_sha256':{str(x.resolve()):file_sha256(x) for x in code},
                     'states':64,'legal_labels':960,'state_presentations':64*48,'new_simulation_episodes':0,'validation':validation,
                     'objective':'mean full-legal regret + 0.01 KL(fixed b9 || current)','reference_reset':False,
                     'initial_checkpoint_update_count':source.get('update_count'),'optimizer_state_inherited':False,'optimizer_initial_state_count':0,
                     'optimizer':'fresh Adam','scheduler':None,'gradient_clipping':False,'acceptance_gates':gates,
                     'nonfinite_handling':'abort; no silent rollback or retry','trainable_names':names,
                     'optimal_definition':'return gap <=1e-9 raw units','near_optimal_definition':'return gap <=0.001 raw units (0.1 score point)',
                     'better_definition':'strictly above this run initial greedy return by >1e-9; mask fixed through updates',
                     'informative_states':int(masks['informative'].sum()),'metric_scope':'same training states only; no new E01 evaluation'})
    write_json(args.output_dir/'protocol.json',protocol);write_json(args.output_dir/'initial_states.json',initial_states)
    attempts=accepted=0;rollback_counts={k:0 for k in gates};history=[];start=time.monotonic()
    with (args.output_dir/'per_state_probability_history.jsonl').open('w') as states_log:
        states_log.write(json.dumps({'attempt':0,'phase':'initial','states':initial_states})+'\n');states_log.flush()
        for step in range(1,args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            logits=target_logits(model,data['observation'],data['features'],data['valid'])
            old_step_logits=logits.detach().clone()
            loss,d=regret_loss(logits,anchor,data['valid'],data['indices'],data['returns'],data['candidate_mask'],args.kl_coef)
            if not torch.isfinite(loss):raise ValueError('nonfinite objective')
            gr=torch.autograd.grad(d['state_regret'].mean(),params,retain_graph=True)
            gku=torch.autograd.grad(d['state_kl'].mean(),params,retain_graph=True)
            gkw=torch.autograd.grad((args.kl_coef*d['state_kl']).to(d['state_regret'].dtype).mean(),params,retain_graph=True)
            loss.backward();gt=[v.grad for v in params]
            if any(g is None or not torch.isfinite(g).all() for g in [*gr,*gku,*gkw,*gt]):raise ValueError('nonfinite gradient')
            nr,nku,nkw,nt=map(norm,[gr,gku,gkw,gt]);inner=dot(gr,gkw)
            with torch.no_grad():pre,_=describe(old_step_logits,anchor,data,args.kl_coef,masks)
            pre_parameters=[x.detach().clone() for x in params];before_steps=optimizer_steps(optimizer)
            attempts+=1;optimizer.step();accepted+=1
            with torch.no_grad():
                after_logits=target_logits(model,data['observation'],data['features'],data['valid'])
                post,_=describe(after_logits,anchor,data,args.kl_coef,masks)
                skl=probability_kl(old_step_logits,after_logits,data['valid'])
                delta=[x-old for x,old in zip(params,pre_parameters)]
                states=state_records(after_logits,anchor,data,payload['items'],masks,args.kl_coef)
            groups={prefix:{'regret':norm([g for n,g in zip(names,gr) if n.startswith(prefix)]),
                            'weighted_kl':norm([g for n,g in zip(names,gkw) if n.startswith(prefix)])} for prefix in TARGET_PREFIXES}
            row={'attempt':attempts,'accepted':True,'accepted_update_count':accepted,'rollback_reason':None,'rollback_counts':dict(rollback_counts),
                 'gates_evaluated':{k:False for k in gates},'learning_rates':[g['lr'] for g in optimizer.param_groups],
                 'optimizer_before':before_steps,'optimizer_after':optimizer_steps(optimizer),
                 'regret_gradient_norm':nr,'kl_gradient_norm_unweighted':nku,'kl_gradient_norm_weighted':nkw,'total_gradient_norm':nt,
                 'weighted_kl_to_regret_gradient_ratio':nkw/nr if nr else None,
                 'regret_kl_gradient_cosine':inner/(nr*nkw) if nr and nkw else None,
                 'total_projection_on_regret':dot(gr,gt)/(nr*nr) if nr else None,
                 'gradient_decomposition_relative_residual':norm([a-b-c for a,b,c in zip(gt,gr,gkw)])/nt if nt else 0.,
                 'gradient_norms_by_target_module':groups,'parameter_update_norm':norm(delta),'regret_first_order_change_under_adam':dot(gr,delta),
                 'step_kl_old_to_attempt_mean':float(skl.mean()),'step_kl_old_to_attempt_max':float(skl.max()),
                 'before':pre,'post_attempt':post,'post_accept':post,'post_accept_equals_post_attempt':True,
                 'accepted_model_sha256':model_sha256(model.state_dict())}
            history.append(row);write_json(args.output_dir/'training_history.json',history)
            states_log.write(json.dumps({'attempt':step,'phase':'post_accept_equals_post_attempt','states':states})+'\n');states_log.flush()
            write_json(args.output_dir/'progress.json',{'status':'training','attempted':attempts,'accepted':accepted,'total':48})
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v,initial[k])];frozen=[k for k in changed if not k.startswith(TARGET_PREFIXES)]
    if frozen or not changed or optimizer_steps(optimizer)['min']!=48:raise ValueError('update isolation/lifecycle failure')
    replay_match=None
    if args.expected_final_checkpoint:
        expected=torch.load(args.expected_final_checkpoint,map_location='cpu',weights_only=True)
        replay_match=all(torch.equal(value,expected['model'][name]) for name,value in model.state_dict().items())
        if not replay_match:raise ValueError('instrumentation changed original b10 model trajectory')
        for name,state in optimizer.state_dict()['state'].items():
            for k,v in state.items():
                other=expected['optimizer']['state'][name][k]
                if isinstance(v,torch.Tensor) and not torch.equal(v,other):raise ValueError('instrumentation changed original b10 optimizer')
    checkpoint={'algorithm':source['algorithm'],'config':asdict(config),'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'update_count':48,'transition_count':64,'continuation':{'stage':args.label,'source_checkpoint':str(args.source_checkpoint),
                'source_sha256':protocol['source_sha256'],'anchor_checkpoint':str(args.anchor_checkpoint),'anchor_sha256':protocol['anchor_sha256'],
                'target_only':True,'source_optimizer_loaded':False,'cumulative_updates_from_b9':(0 if protocol['source_sha256']==protocol['anchor_sha256'] else source.get('continuation',{}).get('cumulative_updates_from_b9',source.get('update_count',0)))+48},'metrics':history[-1]['post_accept']}
    cp=args.output_dir/'checkpoints/working.pt';cp.parent.mkdir();torch.save(checkpoint,cp)
    write_json(args.output_dir/'final_states.json',states)
    ratios=[r['weighted_kl_to_regret_gradient_ratio'] for r in history if r['weighted_kl_to_regret_gradient_ratio'] is not None]
    cosines=[r['regret_kl_gradient_cosine'] for r in history if r['regret_kl_gradient_cosine'] is not None]
    improved=sum(b['near_optimal_probability']>a['near_optimal_probability']+1e-9 for a,b in zip(initial_states,states))
    declined=sum(b['near_optimal_probability']<a['near_optimal_probability']-1e-9 for a,b in zip(initial_states,states))
    summary={'status':'complete','label':args.label,'attempted_updates':attempts,'accepted_updates':accepted,'rollback_counts':rollback_counts,
             'acceptance_gates':gates,'before':before,'after':history[-1]['post_accept'],
             'gradient_ratio_first':ratios[0],'gradient_ratio_last':ratios[-1],'gradient_ratio_min':min(ratios),'gradient_ratio_max':max(ratios),
             'gradient_cosine_last':cosines[-1] if cosines else None,'weighted_kl_norm_exceeds_regret_steps':sum(x>1 for x in ratios),
             'negative_gradient_cosine_steps':sum(x<0 for x in cosines),'near_optimal_probability_states_increased':improved,
             'near_optimal_probability_states_decreased':declined,'greedy_target_changed_states':sum(a['greedy_slot']!=b['greedy_slot'] for a,b in zip(initial_states,states)),
             'optimizer_initial_step':0,'optimizer_final':optimizer_steps(optimizer),'learning_rate_initial':args.learning_rate,
             'learning_rate_final':[g['lr'] for g in optimizer.param_groups],'changed_target_tensors':len(changed),'changed_frozen_tensors':frozen,
             'replay_model_and_optimizer_exact':replay_match,'output_checkpoint':str(cp),'output_sha256':file_sha256(cp),
             'output_model_sha256':model_sha256(model.state_dict()),'elapsed_seconds':time.monotonic()-start,'independent_evaluation_performed':False}
    write_json(args.output_dir/'training_summary.json',summary);write_json(args.output_dir/'progress.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
