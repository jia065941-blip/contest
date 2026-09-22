#!/usr/bin/env python3
"""Independent b10 C0a_goal E01 episodes with matched frozen-teacher references."""
from __future__ import annotations
import argparse, copy, csv, json, math, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch
from scipy.stats import t
from complete_c0a_goal_b10_data import run_sim
from train_c0a_goal_b9 import SIM_ROOT, TEACHER_MODEL, write_json
from train_c0a_goal_b8_advantage import file_sha256
from evaluate_c0a_goal_b9_closedloop import TARGET_SLOT_IDS, boundary_teacher_command, outcomes
from tools.train_start_state_option_curriculum import weighted_type_score

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'competition-platform-refine-logs/target_transformer_main_20260914'
TRAIN=BASE/'runs/c0a_goal_b10_regret_64_20260920'
SOURCE=BASE/'validations/c0a_goal_b9_64_20260919/training_progress.json'
MANIFEST=SIM_ROOT/'refine-logs/target_transformer_main_20260914/validations/c0a_lifecycle_paired128_20260915/guidance_manifest128.json'


def paired_gate(rows, student, teacher, threshold=-.5):
    delta=[r[student]-r[teacher] for r in rows]
    n=len(delta)
    if n<2:raise ValueError('at least two distinct pairs required')
    se=statistics.stdev(delta)/math.sqrt(n);critical=float(t.ppf(.95,n-1))
    lower=statistics.fmean(delta)-critical*se
    return {'student_mean':statistics.fmean(r[student] for r in rows),
            'teacher_mean':statistics.fmean(r[teacher] for r in rows),
            'mean_delta':statistics.fmean(delta),'standard_error':se,
            'one_sided_95_lower':lower,'required_lower':threshold,
            'degrees_of_freedom':n-1,'critical_value':critical,'pass':lower>=threshold}


def validate_branch(branch,item,index):
    if (branch['candidate_index']!=index or branch['target_index']!=int(item['candidate_target_indices'][index])
        or branch['target_id']!=int(item['candidate_target_ids'][index])
        or branch['physical_state_sha256']!=item['physical_state_sha256']
        or branch['public_rng_sha256']!=item['public_rng_sha256']
        or branch['non_target_boundary_changes']!=0
        or branch['suffix_semantics']!='frozen_teacher_closed_loop_target_locked'
        or branch['teacher_closed_loop_steps']!=branch['end_step']-item['timestep']):
        raise ValueError('branch identity/control contract failed')
    initial={int(r['id']):float(r['initial_health']) for r in branch['summary']['objectives']}
    weights={int(k):float(v) for k,v in branch['objective_weights'].items()}
    before={int(k):float(v) for k,v in branch['initial_objective_health'].items()}
    after={int(k):float(v) for k,v in branch['final_objective_health'].items()}
    reward=sum(w/sum(weights.values())*max(0.,before[k]-after[k])/initial[k] for k,w in weights.items() if initial[k]>0)
    if abs(reward-branch['official_joint_return'])>1e-9:raise ValueError('formal suffix reward mismatch')
    if abs(branch['score']-branch['summary']['score']['score'])>1e-9:raise ValueError('formal score mismatch')
    if not all(math.isfinite(branch[k]) for k in ['score','official_joint_return']):raise ValueError('nonfinite result')


def run_episode(args,manifest,job):
    start=time.monotonic();out=args.output_dir/'episodes'/f"i{job['order']:02d}_s{job['seed']}";out.mkdir(parents=True,exist_ok=True)
    result_path=out/'result.json'
    if args.resume and result_path.exists():
        row=json.loads(result_path.read_text())
        if row['checkpoint_sha256']!=args.checkpoint_sha or row['seed']!=job['seed']:raise ValueError('resume mismatch')
        for path,sha in row['artifact_sha256'].items():
            if file_sha256(Path(path))!=sha:raise ValueError('resume artifact mismatch')
        return row
    request={k:job[k] for k in ['seed','trace','start_state_id','timestep','executor_id']}
    request.update({'boundary_source':'formal_damage_anchor','k':25,'rho':.2,'sampling_seed':job['seed'],
                    'snapshot_model_sha256':args.checkpoint_sha,'capture_path':str(out/'capture.pt'),
                    'item_path':str(out/'branch_output.pt'),'result_dir':str(out/'branches'),'branch_workers':2})
    run_sim(args,manifest,out,request,'capture')
    captured=torch.load(out/'capture.pt',map_location='cpu',weights_only=True)
    valid=captured['target_valid_mask'][0]
    slot=int(captured['old_logits'].masked_fill(~valid,-torch.inf).argmax())
    pos=captured['candidate_target_indices'].tolist().index(slot)
    teacher_slot=TARGET_SLOT_IDS.index(job['teacher_target_id'])
    if not bool(valid[slot]) or not bool(valid[teacher_slot]):raise ValueError('student/teacher illegal target')
    item=copy.deepcopy(captured)
    item['candidate_target_indices']=torch.tensor([slot,teacher_slot])
    item['candidate_target_ids']=torch.tensor([int(captured['candidate_target_ids'][pos]),job['teacher_target_id']])
    item['candidate_coordinates']=torch.tensor([captured['candidate_coordinates'][pos].tolist(),job['teacher_coordinate']],dtype=torch.float64)
    item['candidate_roles']=['student_b10_greedy','teacher_reference']
    torch.save(item,out/'branch_input.pt');request['capture_path']=str(out/'branch_input.pt')
    write_json(out/'decision.json',{'checkpoint_sha256':args.checkpoint_sha,'student_slot':slot,'student_target_id':int(item['candidate_target_ids'][0]),'teacher_target_id':job['teacher_target_id'],'legal_slots':valid.nonzero().flatten().tolist(),'source':'fresh_b10_logits_argmax_before_any_branch_return'})
    run_sim(args,manifest,out,request,'branches')
    payload=torch.load(out/'branch_output.pt',map_location='cpu',weights_only=True)
    branches=payload['branch_results']
    if len(branches)!=2:raise ValueError('expected exactly one student and one reference')
    for i,b in enumerate(branches):
        validate_branch(b,payload,i)
        if json.loads((out/f'branches/branch_{i:02d}.json').read_text())!=b:raise ValueError('branch artifact mismatch')
    student,teacher=branches
    if student['physical_state_sha256']!=captured['physical_state_sha256']:raise ValueError('capture/fork state mismatch')
    row={'order':job['order'],'seed':job['seed'],'start_state_id':job['start_state_id'],'decision_step':job['timestep'],
         'executor_id':job['executor_id'],'checkpoint_sha256':args.checkpoint_sha,
         'student_target_id':student['target_id'],'teacher_target_id':teacher['target_id'],
         'student_score':student['score'],'teacher_score':teacher['score'],
         'student_fixed_target_score':weighted_type_score(student['summary'],{9400,9600}),
         'teacher_fixed_target_score':weighted_type_score(teacher['summary'],{9400,9600}),
         'student_suffix_return':student['official_joint_return'],'teacher_suffix_return':teacher['official_joint_return'],
         'student_end_step':student['end_step'],'teacher_end_step':teacher['end_step'],
         'physical_state_sha256':student['physical_state_sha256'],'public_rng_sha256':student['public_rng_sha256'],
         'illegal_samples':0,'elapsed_seconds':time.monotonic()-start,
         'artifact_sha256':{str(p):file_sha256(p) for p in [out/'capture.pt',out/'branch_input.pt',out/'branch_output.pt',out/'decision.json',out/'branches/branch_00.json',out/'branches/branch_01.json']}}
    write_json(result_path,row);return row


def summarize(rows,protocol):
    if len(rows)!=32 or len({r['seed'] for r in rows})!=32:raise ValueError('expected 32 unique completed episodes')
    score=paired_gate(rows,'student_score','teacher_score');fixed=paired_gate(rows,'student_fixed_target_score','teacher_fixed_target_score')
    sr=statistics.fmean(r['student_suffix_return'] for r in rows);tr=statistics.fmean(r['teacher_suffix_return'] for r in rows)
    suffix={'student_mean':sr,'teacher_mean':tr,'ratio':sr/tr if tr else None,'required_min':.8,'pass':bool(tr>0 and sr/tr>=.8)}
    gates={'e01_score_noninferiority':score,'fixed_target_score_noninferiority':fixed,'suffix_return_ratio':suffix,
           'target_legality':{'illegal_samples':sum(r['illegal_samples'] for r in rows),'controlled_decisions':32,'pass':all(r['illegal_samples']==0 for r in rows)},
           'integrity':{'unique_seeds':32,'training_seed_overlap':[],'common_state_pairs':32,'common_rng_pairs':32,'pass':True}}
    passed=all(g['pass'] for g in gates.values());scores=[r['student_score'] for r in rows]
    sem=statistics.stdev(scores)/math.sqrt(32);critical=float(t.ppf(.975,31))
    return {'status':'complete','candidate':'b10','stage':'C0a_goal','student_episodes':32,'teacher_reference_branches':32,
            'score_mean':statistics.fmean(scores),'score_std':statistics.stdev(scores),'score_min':min(scores),'score_max':max(scores),
            'score_mean_95_ci':[statistics.fmean(scores)-critical*sem,statistics.fmean(scores)+critical*sem],
            'course_numeric_gates_decision':'PASS' if passed else 'FAIL','gates':gates,
            'score_outcomes':outcomes([r['student_score']-r['teacher_score'] for r in rows]),
            'target_matches_teacher':sum(r['student_target_id']==r['teacher_target_id'] for r in rows),
            'formal_promotion':False,'formal_promotion_reason':'32-episode requested assessment; project plan requires 128 independent paired episodes for formal promotion',
            'checkpoint_sha256':protocol['checkpoint_sha256'],'scope':'one goal boundary per seed; frozen teacher controls all other actions; not full-team student deployment'}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--resume',action='store_true');args=p.parse_args()
    args.output_dir=args.output_dir.resolve();args.checkpoint=TRAIN/'training/checkpoints/working.pt';args.checkpoint_sha=file_sha256(args.checkpoint)
    if args.checkpoint_sha!='486bee78b1a75b5a802db0733e06ecc190338eb061418f4e660c7c53c98e6210':raise ValueError('unexpected b10 checkpoint')
    torch.set_num_threads(1)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:raise ValueError('output exists; explicit resume required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(MANIFEST.read_text());trajectories={int(r['seed']):r for r in manifest['trajectories']}
    source=json.loads(SOURCE.read_text());selected=source['episodes'][:32]
    training_seeds=set(json.loads((TRAIN/'training/data_validation.json').read_text())['training_seeds'])
    seeds=[int(r['seed']) for r in selected]
    if len(seeds)!=32 or len(set(seeds))!=32 or set(seeds)&training_seeds or min(seeds)<1000:raise ValueError('independent seed contract')
    jobs=[]
    for order,row in enumerate(selected,1):
        seed=int(row['seed']);trajectory=trajectories[seed];target=int(row['assigned_target_id']);anchor=trajectory['damage_anchors'][str(target)]
        boundary=int(anchor['decision_step']);executor=int(anchor['attacking_entity_id'])
        trace=json.loads(Path(trajectory['trace']).read_text());command=boundary_teacher_command(trace,boundary,executor)
        jobs.append({'order':order,'seed':seed,'trace':trajectory['trace'],'start_state_id':f'seed{seed}:step{boundary}:entity{executor}',
                     'timestep':boundary,'executor_id':executor,'teacher_target_id':target,
                     'teacher_coordinate':[float(command['target']['x']),float(command['target']['y'])]})
    code=[Path(__file__),Path(__file__).with_name('c0a_goal_b9_runtime.py'),Path(__file__).with_name('train_c0a_goal_b9.py'),Path(__file__).with_name('complete_c0a_goal_b10_data.py'),SIM_ROOT/'core/main.py']
    protocol={'candidate':'b10','student_episodes':32,'teacher_reference_branches':32,'checkpoint':str(args.checkpoint),'checkpoint_sha256':args.checkpoint_sha,
              'teacher_checkpoint':str(TEACHER_MODEL),'teacher_sha256':file_sha256(TEACHER_MODEL),'scenario':manifest['scenario'],
              'scenario_sha256':file_sha256(Path(manifest['scenario'])),'manifest':str(MANIFEST),'manifest_sha256':file_sha256(MANIFEST),
              'seed_boundary_source':str(SOURCE),'seed_boundary_source_sha256':file_sha256(SOURCE),
              'seed_selection':'first 32 in pre-existing b9 evaluation order; no b10 results used','seeds':seeds,'training_seeds':sorted(training_seeds),
              'workers':args.workers,'branch_workers':2,'max_steps':3000,'student_action':'fresh b10 legal argmax at goal boundary; exactly one decision per episode',
              'continuation':'common-fork frozen r9 closed-loop; identical target-lock semantics for student and reference',
              'score_threshold':-.5,'fixed_score_threshold':-.5,'suffix_ratio_min':.8,'confidence':'paired one-sided Student t 95%, df=31',
              'formal_promotion_min_episodes':128,'code_sha256':{str(x.resolve()):file_sha256(x) for x in code},
              'trace_sha256':{j['trace']:file_sha256(Path(j['trace'])) for j in jobs},'jobs':jobs}
    pp=args.output_dir/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('immutable protocol mismatch')
    write_json(pp,protocol);rows=[]
    write_json(args.output_dir/'progress.json',{'status':'running','completed':0,'total':32,'pid':os.getpid(),'started_unix':time.time()})
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run_episode,args,manifest,j):j for j in jobs}
            for future in as_completed(futures):
                row=future.result();rows.append(row);rows.sort(key=lambda r:r['order'])
                write_json(args.output_dir/'episodes.json',rows)
                write_json(args.output_dir/'progress.json',{'status':'running','completed':len(rows),'total':32,'pid':os.getpid(),'updated_unix':time.time(),'latest_seed':row['seed'],'partial_student_mean':statistics.fmean(r['student_score'] for r in rows)})
                print(json.dumps({'completed':len(rows),'seed':row['seed'],'student_score':row['student_score'],'teacher_score':row['teacher_score'],'seconds':row['elapsed_seconds']}),flush=True)
        for path,sha in protocol['code_sha256'].items():
            if file_sha256(Path(path))!=sha:raise ValueError('source code changed during evaluation')
        if file_sha256(args.checkpoint)!=args.checkpoint_sha:raise ValueError('checkpoint changed')
        result=summarize(rows,protocol);write_json(args.output_dir/'evaluation_summary.json',result)
        fields=['order','seed','student_target_id','teacher_target_id','student_score','teacher_score','student_fixed_target_score','teacher_fixed_target_score','student_suffix_return','teacher_suffix_return','student_end_step','teacher_end_step']
        with (args.output_dir/'episodes.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
        report=['# b10 E01 独立 32 种子评估','',f"b10 回合：32；同种子教师参考分支：32。平均分：{result['score_mean']:.6f}，标准差：{result['score_std']:.6f}。",'',f"本次 32 回合数值门槛：{result['course_numeric_gates_decision']}。正式晋级：未满足原计划 128 配对回合的样本量要求。",'','| 门槛 | 结果 |','|---|---|']
        report.extend(f"| {k} | {'PASS' if v['pass'] else 'FAIL'} |" for k,v in result['gates'].items())
        report.extend(['','全部 b10 动作由当前检查点重新推理，未读取历史 b9 动作或回报；教师参考也重新仿真。','每个种子只接管一个目标选择边界，其他动作及之后闭环控制由冻结教师执行。','种子与 b10/b9 本轮训练池分离，但来自既有课程验证池，不能视为从未用于历史模型评估的新测试集。','',f"检查点 SHA256：{args.checkpoint_sha}"])
        (args.output_dir/'REPORT.md').write_text('\n'.join(report)+'\n')
        write_json(args.output_dir/'progress.json',{'status':'complete','completed':32,'total':32,'finished_unix':time.time(),'summary':str(args.output_dir/'evaluation_summary.json')})
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        write_json(args.output_dir/'progress.json',{'status':'failed','completed':len(rows),'total':32,'error':repr(exc),'pid':os.getpid()});raise


if __name__=='__main__':main()
