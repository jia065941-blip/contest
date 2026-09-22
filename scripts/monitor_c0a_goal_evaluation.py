#!/usr/bin/env python3
"""Read-only progress and process-liveness check for a goal-boundary evaluation."""
import argparse,json,time
from pathlib import Path

p=argparse.ArgumentParser(__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args();directory=a.directory.resolve()
progress=json.loads((directory/'progress.json').read_text())
pid=progress.get('pid');process=Path('/proc')/str(pid)
try:
    command=(process/'cmdline').read_bytes()
    alive=b'evaluate_c0a_goal_' in command and str(directory).encode() in command
except OSError:
    alive=False
active=[d for d in (directory/'episodes').iterdir() if not (d/'result.json').exists()]
files=[f for d in active for f in (d/'branches').glob('*_progress.json')]
steps=[json.loads(f.read_text())['step'] for f in files]
result={'status':progress['status'],'completed':progress['completed'],'total':progress['total'],
        'active_episodes':len(active),'active_branch_steps':sorted(set(steps)),
        'finished_branch_results':len(list((directory/'episodes').glob('*/branches/branch_[0-9][0-9].json'))),
        'evaluation_process_alive':alive if progress['status']!='complete' else None,
        'seconds_since_active_branch_progress':round(time.time()-max(f.stat().st_mtime for f in files)) if files else None,
        'partial_student_mean':progress.get('partial_student_mean')}
print(json.dumps(result,ensure_ascii=False))
