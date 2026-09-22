#!/usr/bin/env python3
"""Admit live b10 compute processes without discarding simulation state.

A global compute budget fills slots across state tails. Parents with live
children only orchestrate fork/wait and do not consume a compute slot. Capture
and replay parents without children do consume a slot. Admission is sampled,
so new forks can briefly exceed the target between scans. Only this exact run
is managed; graceful exit resumes its paused processes.
"""
import json
import os
import re
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'competition-platform-refine-logs/target_transformer_main_20260914/runs/c0a_goal_b10_regret_64_20260920'
COMPUTE_SLOTS = 18


def identify(directory):
    argv = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    if str(RUN / 'data/states') not in argv or '/core/main.py ' not in argv:
        return None
    match = re.search(r'/states/state_(\d+)/', argv)
    return int(match.group(1)) if match else None


def processes():
    result = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            state = identify(directory)
            if state is None:
                continue
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            result[int(directory.name)] = {
                'state_index': state, 'ppid': int(fields[1]), 'status': fields[0]}
        except (OSError, ValueError):
            continue
    return result


def admission(current, slots=COMPUTE_SLOTS):
    if slots < 1:
        raise ValueError('positive compute budget required')
    parents = {row['ppid'] for row in current.values()} & current.keys()
    compute = sorted(current.keys() - parents,
                     key=lambda pid: (current[pid]['state_index'], pid))
    selected = set(compute[:slots])
    return parents | selected, selected, parents


def send(pid, sig):
    try:
        # Revalidate ownership before signalling, guarding against PID reuse.
        if identify(Path('/proc') / str(pid)) is not None:
            os.kill(pid, sig)
    except (OSError, ValueError):
        pass


def main():
    stopping = False
    paused = set()

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            status = json.loads((RUN / 'pipeline_status.json').read_text())
            if status.get('status') != 'running' or status.get('phase') != 'collection':
                break
            current = processes()
            allowed, compute, parents = admission(current)
            # Stop excess workers first, then fill slots (including stopped
            # workers inherited from a previous controller instance).
            for pid in current.keys() - allowed:
                if current[pid]['status'] != 'T':
                    send(pid, signal.SIGSTOP)
                paused.add(pid)
            for pid in allowed:
                if current[pid]['status'] == 'T' or pid in paused:
                    send(pid, signal.SIGCONT)
                paused.discard(pid)
            paused.intersection_update(current)
            info = {
                'status': 'running', 'pid': os.getpid(), 'version': 2,
                'active_state_indices': sorted({current[p]['state_index'] for p in compute}),
                'active_processes': len(allowed), 'active_compute_processes': len(compute),
                'orchestrator_processes': len(parents), 'paused_processes': len(paused),
                'compute_slot_target': COMPUTE_SLOTS,
                'reason': '16-core shared quota; fill branch slots across state tails',
                'updated_unix': time.time()}
            temp = RUN / 'cpu_admission.tmp'
            temp.write_text(json.dumps(info, indent=2))
            temp.replace(RUN / 'cpu_admission.json')
            time.sleep(2)
    finally:
        current = processes()
        for pid in paused & current.keys():
            send(pid, signal.SIGCONT)
        (RUN / 'cpu_admission.json').write_text(json.dumps({
            'status': 'stopped', 'all_managed_processes_resumed': True}, indent=2))


if __name__ == '__main__':
    main()
