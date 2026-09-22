#!/usr/bin/env python3
"""Independent b12 C0a_goal E01 episodes with matched frozen-teacher references."""
from __future__ import annotations
import argparse, copy, csv, json, math, os, statistics, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch
from scipy.stats import t
from train_c0a_goal_b9 import subprocess_environment
from train_c0a_goal_b9 import SIM_ROOT, TEACHER_MODEL, write_json
from train_c0a_goal_b8_advantage import file_sha256
from evaluate_c0a_goal_b9_closedloop import TARGET_SLOT_IDS, boundary_teacher_command, outcomes
from tools.train_start_state_option_curriculum import weighted_type_score

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'competition-platform-refine-logs/target_transformer_main_20260914'
TRAIN=BASE/'runs/c0a_goal_b10_regret_64_20260920'
SOURCE=BASE/'validations/c0a_goal_b9_64_20260919/training_progress.json'
EVAL_RUNTIME=Path(__file__).parent/'b10_evaluation_runtime/c0a_goal_b9_runtime.py'
MANIFEST=SIM_ROOT/'refine-logs/target_transformer_main_20260914/validations/c0a_lifecycle_paired128_20260915/guidance_manifest128.json'


def run_sim(args,manifest,out,request,mode):
    request['mode']=mode
    rp=out/f'{mode}_request.json';write_json(rp,request)
    env=subprocess_environment(request,mode,args.checkpoint)
    env['RED_C0A_B9_REQUEST']=str(rp)
    if mode=='capture':
        env['PYTHONPATH']=str(EVAL_RUNTIME.parent)+os.pathsep+env['PYTHONPATH']
    command=[sys.executable,str(SIM_ROOT/'core/main.py'),'--scenario',manifest['scenario'],'--output-dir',str(out/f'sim_{mode}'),'--max-steps','3000','--total-rounds','1','--render-mode','none','--disable-log-color']
    with (out/f'{mode}.log').open('w') as log:
        result=subprocess.run(command,cwd=SIM_ROOT/'core',env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{mode} failed at {out}; exit={result.returncode}')


def reuse_episode(args,job,out):
    if args.reuse_run is None:return None
    source=args.reuse_run/'episodes'/out.name/'result.json'
    if not source.exists():return None
    row=json.loads(source.read_text())
    if row['checkpoint_sha256']!=args.checkpoint_sha or row['seed']!=job['seed']:raise ValueError('reuse checkpoint/seed mismatch')
    required={str(source.parent/name) for name in ['capture.pt','branch_input.pt','branch_output.pt','decision.json','branches/branch_00.json','branches/branch_01.json']}
    if set(row['artifact_sha256'])!=required:raise ValueError('reuse provenance coverage mismatch')
    for path,sha in row['artifact_sha256'].items():
        if file_sha256(Path(path))!=sha:raise ValueError('reuse artifact hash mismatch')
    capture=torch.load(source.parent/'capture.pt',map_location='cpu',weights_only=True)
    valid=capture['target_valid_mask'][0]
    if bool(valid[TARGET_SLOT_IDS.index(-100)]):raise ValueError('old capture may only be reused where SEARCH was not legal')
    slot=int(capture['old_logits'].masked_fill(~valid,-torch.inf).argmax())
    payload=torch.load(source.parent/'branch_output.pt',map_location='cpu',weights_only=True)
    if slot!=int(payload['candidate_target_indices'][0]) or int(payload['candidate_target_ids'][0])!=row['student_target_id']:raise ValueError('reuse greedy action mismatch')
    if len(payload['branch_results'])!=2:raise ValueError('reuse branch count mismatch')
    for key in ['seed','timestep','executor_id','start_state_id']:
        if capture[key]!=job[key] or payload[key]!=job[key]:raise ValueError(f'reuse boundary mismatch: {key}')
    if capture['snapshot_model_sha256']!=args.checkpoint_sha or capture['physical_state_sha256']!=payload['physical_state_sha256']:raise ValueError('reuse capture provenance mismatch')
    for i,b in enumerate(payload['branch_results']):
        validate_branch(b,payload,i)
        if json.loads((source.parent/f'branches/branch_{i:02d}.json').read_text())!=b:raise ValueError('reuse branch JSON mismatch')
    student,teacher=payload['branch_results']
    expected={'order':job['order'],'seed':job['seed'],'start_state_id':job['start_state_id'],'decision_step':job['timestep'],'executor_id':job['executor_id'],
              'student_target_id':student['target_id'],'teacher_target_id':teacher['target_id'],
              'student_score':student['score'],'teacher_score':teacher['score'],
              'student_fixed_target_score':weighted_type_score(student['summary'],{9400,9600}),
              'teacher_fixed_target_score':weighted_type_score(teacher['summary'],{9400,9600}),
              'student_suffix_return':student['official_joint_return'],'teacher_suffix_return':teacher['official_joint_return'],
              'student_end_step':student['end_step'],'teacher_end_step':teacher['end_step'],
              'physical_state_sha256':student['physical_state_sha256'],'public_rng_sha256':student['public_rng_sha256'],'illegal_samples':0}
    if teacher['target_id']!=job['teacher_target_id']:raise ValueError('reuse teacher target mismatch')
    for key,value in expected.items():
        if row[key]!=value:raise ValueError(f'reuse result scalar mismatch: {key}')
    row.update(expected)
    row['source_result_sha256']=file_sha256(source)
    row['source_protocol_sha256']=file_sha256(args.reuse_run/'protocol.json')
    row['legal_target_count']=int(valid.sum())
    row['teacher_target_legal_in_student_mask']=bool(valid[TARGET_SLOT_IDS.index(row['teacher_target_id'])])
    row['reused_from']=str(source);row['reuse_reason']='capture runtime differs only in rejecting legal SEARCH; this state has no legal SEARCH; branch runtime unchanged'
    write_json(out/'result.json',row);return row


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
        or branch['suffix_semantics']!=('frozen_teacher_closed_loop_search_then_release' if branch['target_id']==-100 else 'frozen_teacher_closed_loop_target_locked')
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
    reused=reuse_episode(args,job,out)
    if reused is not None:return reused
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
    if not bool(valid[slot]):raise ValueError('student illegal target')
    item=copy.deepcopy(captured)
    item['candidate_target_indices']=torch.tensor([slot,teacher_slot])
    item['candidate_target_ids']=torch.tensor([int(captured['candidate_target_ids'][pos]),job['teacher_target_id']])
    item['candidate_coordinates']=torch.tensor([captured['candidate_coordinates'][pos].tolist(),job['teacher_coordinate']],dtype=torch.float64)
    item['candidate_roles']=['student_b12_greedy','teacher_reference']
    torch.save(item,out/'branch_input.pt');request['capture_path']=str(out/'branch_input.pt')
    write_json(out/'decision.json',{'checkpoint_sha256':args.checkpoint_sha,'student_slot':slot,'student_target_id':int(item['candidate_target_ids'][0]),'teacher_target_id':job['teacher_target_id'],'legal_slots':valid.nonzero().flatten().tolist(),'source':'fresh_b12_logits_argmax_before_any_branch_return'})
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
         'illegal_samples':0,'legal_target_count':int(valid.sum()),'teacher_target_legal_in_student_mask':bool(valid[teacher_slot]),'elapsed_seconds':time.monotonic()-start,
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
    return {'status':'complete','candidate':'b12','stage':'C0a_goal','student_episodes':32,'teacher_reference_branches':32,
            'score_mean':statistics.fmean(scores),'score_std':statistics.stdev(scores),'score_min':min(scores),'score_max':max(scores),
            'score_mean_95_ci':[statistics.fmean(scores)-critical*sem,statistics.fmean(scores)+critical*sem],
            'course_numeric_gates_decision':'PASS' if passed else 'FAIL','gates':gates,
            'score_outcomes':outcomes([r['student_score']-r['teacher_score'] for r in rows]),
            'target_matches_teacher':sum(r['student_target_id']==r['teacher_target_id'] for r in rows),'search_choices':sum(r['student_target_id']==-100 for r in rows),'singleton_decisions':sum(r['legal_target_count']==1 for r in rows),'teacher_targets_outside_student_mask':sum(r['teacher_target_legal_in_student_mask'] is False for r in rows),'reused_valid_episodes':sum('reused_from' in r for r in rows),
            'decision_groups':{label:{'episodes':len(group),'student_mean':statistics.fmean(r['student_score'] for r in group),'teacher_mean':statistics.fmean(r['teacher_score'] for r in group),'mean_delta':statistics.fmean(r['student_score']-r['teacher_score'] for r in group)} for label,group in [('search_only',[r for r in rows if r['legal_target_count']==1 and r['student_target_id']==-100]),('physical_target_choice',[r for r in rows if r['student_target_id']!=-100]),('other',[r for r in rows if r['student_target_id']==-100 and r['legal_target_count']>1])] if group},'formal_promotion':False,'formal_promotion_reason':'32-episode requested assessment; project plan requires 128 independent paired episodes for formal promotion',
            'checkpoint_sha256':protocol['checkpoint_sha256'],'scope':'one goal boundary per seed; frozen teacher controls all other actions; not full-team student deployment'}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--resume',action='store_true');p.add_argument('--reuse-run',type=Path);args=p.parse_args()
    if args.reuse_run is not None:args.reuse_run=args.reuse_run.resolve()
    args.output_dir=args.output_dir.resolve();args.checkpoint=BASE/'runs/c0a_goal_b12_continuation_20260921/training/checkpoints/working.pt';args.checkpoint_sha=file_sha256(args.checkpoint)
    if args.checkpoint_sha!='fed5de916c45c83aa8f0ef7e043c521693cfeb0288ccc05ce44723073d07eabd':raise ValueError('unexpected b12 checkpoint')
    torch.set_num_threads(1)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:raise ValueError('output exists; explicit resume required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(MANIFEST.read_text());trajectories={int(r['seed']):r for r in manifest['trajectories']}
    source=json.loads(SOURCE.read_text());selected=source['episodes'][32:64]
    train_protocol_path=BASE/'runs/c0a_goal_b12_continuation_20260921/training/protocol.json'
    train_protocol=json.loads(train_protocol_path.read_text())
    data_path=Path(train_protocol['candidate_batch'])
    if file_sha256(data_path)!=train_protocol['data_sha256']:raise ValueError('b12 actual training data hash mismatch')
    training_data=torch.load(data_path,map_location='cpu',weights_only=True)
    training_seeds={int(item['seed']) for item in training_data['items']}
    if sorted(training_seeds)!=train_protocol['validation']['training_seeds']:raise ValueError('b12 actual training seed list mismatch')
    training_summary_path=train_protocol_path.parent/'training_summary.json'
    if json.loads(training_summary_path.read_text())['output_sha256']!=args.checkpoint_sha:raise ValueError('b12 training/evaluation checkpoint mismatch')
    previous_protocol_path=BASE/'validations/b11_e01_32_20260921/protocol.json'
    previous_protocol=json.loads(previous_protocol_path.read_text())
    if file_sha256(Path(train_protocol['source_checkpoint']))!=train_protocol['source_sha256'] or train_protocol['source_sha256']!=previous_protocol['checkpoint_sha256']:raise ValueError('b12 must continue evaluated b11')
    seeds=[int(r['seed']) for r in selected]
    if len(seeds)!=32 or len(set(seeds))!=32 or set(seeds)&training_seeds or min(seeds)<1000:raise ValueError('independent seed contract')
    if set(seeds)&set(previous_protocol['seeds']):raise ValueError('b12 evaluation seeds overlap b11')
    jobs=[]
    for order,row in enumerate(selected,1):
        seed=int(row['seed']);trajectory=trajectories[seed];target=int(row['assigned_target_id']);anchor=trajectory['damage_anchors'][str(target)]
        boundary=int(anchor['decision_step']);executor=int(anchor['attacking_entity_id'])
        trace=json.loads(Path(trajectory['trace']).read_text());command=boundary_teacher_command(trace,boundary,executor)
        jobs.append({'order':order,'seed':seed,'trace':trajectory['trace'],'start_state_id':f'seed{seed}:step{boundary}:entity{executor}',
                     'timestep':boundary,'executor_id':executor,'teacher_target_id':target,
                     'teacher_coordinate':[float(command['target']['x']),float(command['target']['y'])]})
    code=[Path(__file__),EVAL_RUNTIME,Path(__file__).with_name('c0a_goal_b9_runtime.py'),Path(__file__).with_name('train_c0a_goal_b9.py'),Path(__file__).with_name('complete_c0a_goal_b10_data.py'),SIM_ROOT/'core/main.py']
    protocol={'capture_runtime':'evaluation-only, allows SEARCH exactly when original legal mask allows it; rejects empty slots; no mask modifications','reuse_run':str(args.reuse_run) if args.reuse_run else None,'candidate':'b12','student_episodes':32,'teacher_reference_branches':32,'checkpoint':str(args.checkpoint),'checkpoint_sha256':args.checkpoint_sha,
              'teacher_checkpoint':str(TEACHER_MODEL),'teacher_sha256':file_sha256(TEACHER_MODEL),'scenario':manifest['scenario'],
              'scenario_sha256':file_sha256(Path(manifest['scenario'])),'manifest':str(MANIFEST),'manifest_sha256':file_sha256(MANIFEST),
              'seed_boundary_source':str(SOURCE),'seed_boundary_source_sha256':file_sha256(SOURCE),
              'seed_selection':'positions 33-64 in pre-existing b9 evaluation order; disjoint from b10/b11 positions 1-32; no b12 results used','seeds':seeds,'training_seeds':sorted(training_seeds),
              'workers':args.workers,'branch_workers':2,'max_steps':3000,'student_action':'fresh b12 legal argmax at goal boundary; exactly one decision per episode',
              'continuation':'common-fork frozen r9 closed-loop; physical targets locked until termination; SEARCH boundary command then immediate teacher release',
              'teacher_reference_legality':'original native teacher boundary command; may target objectives outside student observation mask; legality gate applies to student only','score_threshold':-.5,'fixed_score_threshold':-.5,'suffix_ratio_min':.8,'confidence':'paired one-sided Student t 95%, df=31',
              'formal_promotion_min_episodes':128,'code_sha256':{str(x.resolve()):file_sha256(x) for x in code},
              'trace_sha256':{j['trace']:file_sha256(Path(j['trace'])) for j in jobs},'jobs':jobs}
    b10_protocol_path=BASE/'validations/b10_e01_32_20260921_v2/protocol.json'
    b10_protocol=json.loads(b10_protocol_path.read_text())
    comparison_keys=['teacher_sha256','scenario_sha256','manifest_sha256','seed_boundary_source_sha256','training_seeds','max_steps','continuation','score_threshold','fixed_score_threshold','suffix_ratio_min','confidence','formal_promotion_min_episodes']
    for key in comparison_keys:
        if protocol[key]!=b10_protocol[key]:raise ValueError(f'b10 comparison protocol differs: {key}')
    for runtime in [EVAL_RUNTIME,Path(__file__).with_name('c0a_goal_b9_runtime.py')]:
        if b10_protocol['code_sha256'][str(runtime.resolve())]!=file_sha256(runtime):raise ValueError('b10 runtime changed')
    protocol['b10_comparison_protocol']=str(b10_protocol_path)
    protocol['b10_comparison_protocol_sha256']=file_sha256(b10_protocol_path)
    protocol['matched_b10_protocol_fields']=comparison_keys
    protocol['training_protocol']=str(train_protocol_path)
    protocol['training_protocol_sha256']=file_sha256(train_protocol_path)
    protocol['training_summary_sha256']=file_sha256(training_summary_path)
    protocol['training_data_sha256']=file_sha256(data_path)
    protocol['previous_b11_protocol']=str(previous_protocol_path)
    protocol['previous_b11_protocol_sha256']=file_sha256(previous_protocol_path)
    protocol['excluded_b11_evaluation_seeds']=previous_protocol['seeds']
    protocol['training_and_previous_evaluation_seed_overlap']=[]
    protocol['cross_model_mean_comparison']='different evaluation seeds; do not interpret b12 minus prior b11 mean as paired model improvement'
    if args.reuse_run is not None:
        previous=json.loads((args.reuse_run/'protocol.json').read_text())
        for key in ['checkpoint_sha256','teacher_sha256','scenario_sha256','manifest_sha256','seed_boundary_source_sha256','seeds','jobs','trace_sha256','max_steps']:
            if previous[key]!=protocol[key]:raise ValueError(f'reuse protocol differs: {key}')
        old_runtime=str(Path(__file__).with_name('c0a_goal_b9_runtime.py').resolve())
        if previous['code_sha256'][old_runtime]!=file_sha256(Path(old_runtime)):raise ValueError('branch runtime changed; cannot reuse')
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
        report=['# b12 E01 独立 32 种子评估','',f"b12 回合：32；同种子教师参考分支：32。平均分：{result['score_mean']:.6f}，标准差：{result['score_std']:.6f}。",'',f"本次 32 回合数值门槛：{result['course_numeric_gates_decision']}。正式晋级：未满足原计划 128 配对回合的样本量要求。",'','| 门槛 | 结果 |','|---|---|']
        report.extend(f"| {k} | {'PASS' if v['pass'] else 'FAIL'} |" for k,v in result['gates'].items())
        report.extend(['','全部 b12 动作由当前检查点重新推理，未读取历史 b9 动作或回报；教师参考也重新仿真。','每个种子只接管一个目标选择边界，其他动作及之后闭环控制由冻结教师执行；原始合法集合包含 SEARCH 时保留该动作，选择 SEARCH 后立即交还教师。','本轮 32 个种子与 b12/b11/b10/b9 本轮训练池及上轮 b11 的 32 个评估种子分离，但来自既有课程验证池，不能视为从未用于历史模型评估的新测试集。','',f"检查点 SHA256：{args.checkpoint_sha}"])
        (args.output_dir/'REPORT.md').write_text('\n'.join(report)+'\n')
        write_json(args.output_dir/'progress.json',{'status':'complete','completed':32,'total':32,'finished_unix':time.time(),'summary':str(args.output_dir/'evaluation_summary.json')})
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        write_json(args.output_dir/'progress.json',{'status':'failed','completed':len(rows),'total':32,'error':repr(exc),'pid':os.getpid()});raise


if __name__=='__main__':main()
