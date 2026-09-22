#!/usr/bin/env python3
"""Durable local collection -> validated b10 training chain."""
import json,os,subprocess,sys
from pathlib import Path
from train_c0a_goal_b8_advantage import write_json
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'competition-platform-refine-logs/target_transformer_main_20260914/runs'
run=BASE/'c0a_goal_b10_regret_64_20260920'
source=BASE/'c0a_goal_b9_elite_closedloop_20260918'
commands=[('collection',[sys.executable,str(ROOT/'scripts/complete_c0a_goal_b10_data.py'),'--source-run',str(source),'--checkpoint',str(source/'checkpoints/working.pt'),'--output-dir',str(run/'data'),'--workers','24','--branch-workers','6','--fresh-all','--resume']),('training',[sys.executable,str(ROOT/'scripts/train_c0a_goal_b10_regret.py'),'--source-checkpoint',str(source/'checkpoints/working.pt'),'--candidate-batch',str(run/'data/candidate_batch.pt'),'--output-dir',str(run/'training'),'--steps','48','--learning-rate','2e-5','--kl-coef','0.01','--device','cpu'])]
write_json(run/'launch.json',{'pid':os.getpid(),'backend':'local_cpu','commands':commands,'cwd':str(ROOT)})
for phase,command in commands:
    write_json(run/'pipeline_status.json',{'status':'running','phase':phase,'pid':os.getpid()})
    with (run/f'{phase}.log').open('a') as log:
        process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:
        write_json(run/'pipeline_status.json',{'status':'failed','phase':phase,'returncode':process.returncode})
        sys.exit(process.returncode)
write_json(run/'pipeline_status.json',{'status':'complete','checkpoint':str(run/'training/checkpoints/working.pt')})
