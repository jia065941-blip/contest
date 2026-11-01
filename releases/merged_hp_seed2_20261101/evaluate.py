"""Seed2 all20: frozen merged tree with/without HP, matched in-memory branches."""
from pathlib import Path
import argparse
import csv
import fcntl
import json
import os
import signal
import statistics
import subprocess
import sys
import tarfile
import time
from support import ROOT, APP, ENV, read, sha, save, verify, native_environment, assert_tmpfs, payload
import host_resources as resources

ARMS = ('merged', 'merged_hp')
GIB = 1024**3

def memory():
    info = {line.split(':')[0]: int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemTotal:', 'MemAvailable:'))}
    layers = [dict(name='host', total=info['MemTotal'], available=info['MemAvailable'])]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            cgroup = Path('/sys/fs/cgroup')/line[3:].lstrip('/')
            for path in [cgroup, *cgroup.parents]:
                if path == Path('/sys/fs'):
                    break
                limit, current = path/'memory.max', path/'memory.current'
                if limit.is_file() and limit.read_text().strip() != 'max':
                    n = int(limit.read_text()); used = int(current.read_text())
                    layers.append(dict(name=str(path), total=n, available=max(0, n-used)))
    return dict(layers=layers, total=min(x['total'] for x in layers),
                available=min(x['available'] for x in layers),
                headroom=min(x['available']-.10*x['total'] for x in layers))

def cpu_capacity(requested):
    capacity = min(requested, 64, len(os.sched_getaffinity(0)))
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            group = Path('/sys/fs/cgroup')/line[3:].lstrip('/')
            for path in [group, *group.parents]:
                if path == Path('/sys/fs'):
                    break
                f = path/'cpu.max'
                if f.is_file():
                    quota, period = f.read_text().split()
                    if quota != 'max':
                        capacity = min(capacity, max(1, int(quota)//int(period)))
    if capacity < 2:
        raise ValueError('Matched process branches require at least two native CPU slots')
    return capacity

def sealed_input(value):
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    fd = os.memfd_create('merged-hp-frozen-input', os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
    try:
        with os.fdopen(os.dup(fd), 'wb') as stream:
            stream.write(raw)
        os.lseek(fd, 0, os.SEEK_SET)
        seals = 0x0008|0x0004|0x0002|0x0001
        fcntl.fcntl(fd, getattr(fcntl, 'F_ADD_SEALS', 1033), seals)
        if fcntl.fcntl(fd, getattr(fcntl, 'F_GET_SEALS', 1034)) != seals:
            raise RuntimeError('Cannot seal episode input')
        import hashlib
        return fd, hashlib.sha256(raw).hexdigest()
    except BaseException:
        os.close(fd)
        raise

def validate(folder, arm, event):
    import skill_runtime
    summaries = list((folder/'native').glob('*/summary.json'))
    if len(summaries) != 1:
        raise ValueError('Missing unique final native summary')
    summary = read(summaries[0]); execution = read(folder/'execution.json')
    i, ledger, cost = execution['integration'], execution['online_actions'], execution['cost']
    assert i['native_seed'] == 2 and i['candidate_version'] == 'pku_initial_astra_v2'
    assert i['motion'] == 'V12-R3' and i['gate'] == 'mission_v12'
    assert i['model_sha256'] == sha(APP/'r3/model.pt')
    assert i['actor_obs_dim'] == 85 and i['algorithm'] == 'local_avoid_set_attention_ppo_v1'
    assert not i['training'] and not i['test_only'] and i['main_return_code'] == 0
    assert i['envengine_module'] == str((ENV/'envengine/__init__.py').resolve())
    assert set(i['motion_calls']) == {'H','M','L'} and min(i['motion_calls'].values()) > 0
    assert i['motion_audit']['encoded_85'] == sum(i['motion_calls'].values())
    assert i['pending'] == 0 and all(c['status']=='native_return' for c in ledger['commands'])
    assert len(ledger['commands']) == ledger['metrics'].get('native_returns', 0)
    assert not cost['requests'] and cost['upper_cny'] == 0 and i['seven_skills_enabled'] is False
    assert summary['termination_reason'] in ('environment_done', 'time_limit')
    assert summary['steps_executed'] <= 3000
    if summary['termination_reason'] == 'time_limit':
        assert summary['steps_executed'] == 3000
    objectives = summary['objectives']
    score = 100*sum(x['weight'] for x in objectives if x['final_health']<=0)/sum(x['weight'] for x in objectives)
    assert abs(score-summary['score']['score']) < 1e-8
    audit = skill_runtime.validate_execution(execution, read(ROOT/'policy_manifest.json'), arm, event,
                                            trace=read(folder/'legal_terminal_trace.json'))
    metrics = read(folder/'combat_metrics.json')
    assert metrics['unresolved_damage_sources'] == 0
    assert metrics['audit']['start_registry_hash'] == metrics['audit']['end_registry_hash']
    assert abs(metrics['audit']['score']-score) < 1e-8
    return dict(score=score, steps=summary['steps_executed'], termination=summary['termination_reason'],
                native_summary_sha256=sha(summaries[0]), execution_sha256=sha(folder/'execution.json'), validation=audit)

def collect(folder, map_name):
    import matched
    comparison = read(folder/'comparison.json')
    if comparison['status'] != 'complete':
        raise ValueError('Failed native comparison: '+str(comparison.get('errors', comparison.get('error'))))
    if comparison['event_triggered']:
        errors = matched.validate_audits(comparison['snapshot'], comparison['branches'], ARMS)
        if errors:
            raise ValueError(errors)
        returns = {}
        for arm in ARMS:
            result = validate(folder/'branches'/arm, arm, comparison['event'])
            branch = comparison['branches'][arm]
            assert result['score'] == branch['final_score'] and result['steps'] == branch['steps']
            assert result['termination'] == branch['termination']
            returns[arm] = result
        actual = 2
    else:
        result = validate(folder, 'merged', None)
        assert result['score'] == comparison['score']
        returns = {arm: dict(result) for arm in ARMS}
        actual = 1
    return dict(map=map_name, seed=2, status='complete', policies=returns,
                actual_trajectories=actual, event_triggered=comparison['event_triggered'],
                hp_gain=returns['merged_hp']['score']-returns['merged']['score'],
                no_divergence_note=None if actual==2 else 'One actual command-equivalent trajectory, not two simulations')

def archive(folder, dest):
    names = [p for p in folder.rglob('*') if p.is_file() and
             (p.name in ('comparison.json','snapshot.json','children.json','execution.json','branch_audit.json',
                         'combat_metrics.json','combat_metrics_failed.json','legal_terminal_trace.json','console.log','root_console.log') or
              p.name=='summary.json')]
    files = {str(p.relative_to(folder)): sha(p) for p in sorted(names)}
    with tarfile.open(dest, 'w:gz', compresslevel=3) as tar:
        for name in files:
            tar.add(folder/name, arcname=name, recursive=False)
    import hashlib
    with tarfile.open(dest) as tar:
        assert len(tar.getmembers()) == len(files)
        for member in tar:
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == files[member.name]
    return dict(path=dest.name, sha256=sha(dest), files=files, validated_by_reopening=True)

def report(out, rows, failures, maps, status):
    groups = {'development17':[m for m in maps if m not in ('E05','M06','H04')],
              'adapted_M06':[m for m in maps if m=='M06'],
              'remaining_test2':[m for m in maps if m in ('E05','H04')], 'all_requested':maps}
    partitions = {}
    for name, members in groups.items():
        selected = [r for r in rows if r['map'] in members]
        complete = len(selected)==len(members) and bool(members)
        partitions[name] = dict(planned=len(members), completed=len(selected),
                               means={a:statistics.mean(r['policies'][a]['score'] for r in selected) for a in ARMS} if complete else None,
                               gains=sum(r['hp_gain']>1e-10 for r in selected),
                               harms=sum(r['hp_gain'] < -1e-10 for r in selected),
                               ties=sum(abs(r['hp_gain'])<=1e-10 for r in selected))
    save(out/'summary.json', dict(status=status, partitions=partitions, rows=rows, failures=failures,
                                 missing_maps=sorted(set(maps)-{r['map'] for r in rows}),
                                 external_model_API_calls=0, retries=0,
                                 limitation='Fresh roots on this host, official address tie ordering retained. Historical99.4278 is a reference, never substituted for current returns.'))
    with (out/'returns.csv').open('w') as stream:
        writer = csv.writer(stream); writer.writerow(['map',*ARMS,'HP_gain','actual_trajectories'])
        for r in sorted(rows, key=lambda x:x['map']):
            writer.writerow([r['map'],*[r['policies'][a]['score'] for a in ARMS],r['hp_gain'],r['actual_trajectories']])

def stop(process):
    for sig in (signal.SIGTERM, signal.SIGCONT):
        try: os.killpg(process.pid, sig)
        except ProcessLookupError: pass
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        records=resources.scan()
        if not any(r['native'] and resources.descendant(pid,process.pid,records) for pid,r in records.items()):
            break
        time.sleep(.2)
    try: os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait(timeout=15)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ram', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=64, help='Native processes including fork descendants, maximum64')
    parser.add_argument('--maps', nargs='+', help='Optional subset of the frozen20 seed2 maps')
    parser.add_argument('--prepare-only', action='store_true', help='Verify inputs without starting native games')
    args = parser.parse_args()
    if not 2 <= args.workers <= 64: parser.error('--workers must be2..64')
    verify()
    from bootstrap import check_platform
    check_platform()
    import matched
    controls=matched.controls()
    subprocess.run([sys.executable,'-B',str(ROOT/'verify_release.py'),'--native-imports'],
                   env=native_environment(Path('/tmp'),Path('/tmp')),check=True)
    choices=read(ROOT/'opening/frozen_choices.json')['choices']
    maps=sorted(args.maps or choices)
    if len(maps)!=len(set(maps)) or not set(maps)<=set(choices):
        parser.error('Unique frozen seed2 maps only')
    inputs={m:payload(m) for m in maps}
    capacity=cpu_capacity(args.workers)
    if args.prepare_only:
        print(json.dumps(dict(status='prepared', maps=maps, cpu_capacity=capacity, branches_per_root=2, games_started=0)))
        return
    out=args.out.resolve(); ram=args.ram.resolve()
    assert_tmpfs(ram)
    if out==ram or out in ram.parents or ram in out.parents:
        raise ValueError('Durable output and transient root must be separate')
    out.mkdir(parents=True,exist_ok=False);ram.mkdir(mode=0o700,parents=True,exist_ok=False)
    (out/'archives').mkdir();(out/'started').mkdir()
    stage='merged_hp_release_'+str(os.getpid())+'_'+str(time.time_ns())
    leases=resources.HostLeases('/tmp/launch20-native-host-leases-v1',stage)
    protocol=dict(seed=2,maps=maps,choices={m:choices[m] for m in maps},arms=ARMS,max_steps=3000,
                  release_manifest_sha256=sha(ROOT/'release_manifest.json'),controls=controls,
                  cpu_capacity=capacity,memory_fraction=.90,no_retries=True,
                  official_scheduler_unchanged=True,external_model_API_calls=0)
    save(out/'protocol.json',protocol)
    save(out/'controller.json',dict(pid=os.getpid(),start=resources.scan()[os.getpid()]['start'],ram=str(ram),stage=stage))
    todo=list(maps);active={};rows=[];failures=[]
    def interrupt(signum,frame): raise KeyboardInterrupt('Controller interrupted; no automatic retries')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        while todo or active:
            for m,item in list(active.items()):
                process=item['process']
                if process.poll() is None:continue
                item['log'].close();folder=ram/m
                if process.returncode:raise RuntimeError(f'{m}: native exit{process.returncode}; retained in {folder}')
                row=collect(folder,m)
                row['archive']=archive(folder,out/'archives'/(m+'.tar.gz'))
                save(out/(m+'.json'),row);rows.append(row)
                leases.release(item['lease']);del active[m]
                import shutil
                shutil.rmtree(folder)
                print(json.dumps(dict(completed=len(rows),map=m,returns={a:row['policies'][a]['score'] for a in ARMS})),flush=True)
            mem=memory();host=leases.snapshot()
            if mem['headroom'] < 0 or host['actual_native_threads']>capacity or host['reserved_cpu_slots']>capacity:
                raise RuntimeError('Host/cgroup memory or native CPU budget exceeded')
            while todo and host['reserved_cpu_slots']+2<=capacity:
                m=todo[0];reservation=2*{'E':4,'M':9,'H':23}[m[0]]*GIB
                if reservation>mem['total']*.90:raise RuntimeError(f'{m}: insufficient total memory for two branch reservations')
                token,host=leases.acquire(m+'_seed2',reservation,mem['total'],mem['available'],
                                         cpu_capacity=capacity,memory_headroom=mem['headroom'])
                if token is None:break
                folder=ram/m;folder.mkdir(mode=0o700)
                fd,input_hash=sealed_input(inputs[m]);log=(folder/'root_console.log').open('xb')
                save(out/'started'/(m+'.json'),dict(map=m,seed=2,lease_token=token,input_sha256=input_hash,sealed=True))
                job=read(APP/'jobs.json')[m]
                command=[sys.executable,'-B',str(APP/'native_entry.py'),'--scenario',job['scenario'],'--seed','2',
                         '--red-policy','r8_satellite_packages','--red-motion-policy','straight',
                         '--blue-policy','b0_fixed_ratio_random','--blue-interceptor-ratio','2',
                         '--max-steps','3000','--render-mode','none','--disable-log-color','--output-dir',str(folder/'native')]
                e=native_environment(folder,ram);e.update(MIGRATED_INPUT_FD=str(fd),PAIR_ROOT=str(folder),PAIR_FORK_STEP='-1',RSI_ARM='merged')
                try:
                    process=subprocess.Popen(command,cwd=ram,env=e,pass_fds=(fd,),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:
                    log.close();leases.release(token);raise
                finally:os.close(fd)
                active[m]=dict(process=process,log=log,lease=token)
                leases.bind(token,process.pid);todo.pop(0)
                mem=memory();host=leases.snapshot()
            report(out,rows,failures,maps,'running')
            save(ram/'status.json',dict(active=list(active),pending=todo,host=host,memory=mem))
            time.sleep(2)
        verify()
        report(out,rows,failures,maps,'complete')
    except BaseException as exc:
        failures.append(dict(error=repr(exc),active=list(active),pending=todo))
        for m,item in active.items():
            try:
                stop(item['process']);item['log'].close();leases.release(item['lease'])
            except BaseException as stop_error:
                failures.append(dict(map=m,stop_error=repr(stop_error)))
            try:
                evidence=archive(ram/m,out/'archives'/(m+'_failed.tar.gz'))
                save(out/(m+'_failed.json'),dict(map=m,status='failed',archive=evidence,
                                              native_exit_code=item['process'].returncode,no_retry=True))
            except BaseException as archive_error:
                failures.append(dict(map=m,archive_error=repr(archive_error),raw_tmpfs=str(ram/m)))
        report(out,rows,failures,maps,'failed')
        raise

if __name__=='__main__':
    main()
