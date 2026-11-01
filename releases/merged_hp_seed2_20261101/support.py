"""Portable paths, exact assets and frozen native launch contract; no games on import."""
from pathlib import Path
import hashlib
import json
import os
import sys
import sysconfig

if sys.flags.optimize:
    raise RuntimeError('Python -O/-OO disables required validation and is not permitted')

ROOT = Path(__file__).resolve().parent
APP = ROOT/'app'
ENV = APP/'.runtime/environment'
sys.path.insert(0, str(APP))

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def verify():
    for name, wanted in read(ROOT/'release_manifest.json')['files'].items():
        if sha(ROOT/name) != wanted:
            raise ValueError('Release file changed: '+name)
    for name, wanted in read(ROOT/'assets_manifest.json')['files'].items():
        if sha(APP/name) != wanted:
            raise ValueError('Asset changed: '+name)
    from common import verify_environment
    verify_environment()

def native_environment(folder, ram):
    e = os.environ.copy()
    for key in list(e):
        if key.startswith(('RED_', 'BLUE_', 'PLAN_', 'EVASION_', 'LOCAL_AVOID_',
                           'ONLINE_', 'MIGRATED_', 'PAIR_', 'RSI_')) or key.endswith('API_KEY'):
            e.pop(key)
    libs = sorted({str(p.parent) for p in ENV.rglob('*.so')})
    libdir = sysconfig.get_config_var('LIBDIR') or str(Path(sys.base_prefix)/'lib')
    e.update(PYTHONPATH=':'.join(map(str, [APP, ENV, APP/'deps', APP/'deps/engine', APP/'r3'])),
             PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
             NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
             LD_LIBRARY_PATH=':'.join([libdir, str(Path(sys.base_prefix)/'lib'), str(ENV), *libs]),
             MIGRATED_EPISODE=str(folder), MIGRATED_SHARED=str(ram), R3_GATE_MODE='mission_v12')
    e.pop('LD_PRELOAD', None)
    e.pop('PYTHONOPTIMIZE', None)
    e.pop('PYTHONHOME', None)
    e.pop('PYTHONUSERBASE', None)
    e['PYTHONNOUSERSITE'] = '1'
    return e

def assert_tmpfs(path, executable=False):
    path = Path(path).resolve()
    mounts = []
    for line in Path('/proc/mounts').read_text().splitlines():
        parts = line.split(); mount = Path(parts[1])
        if path == mount or mount in path.parents:
            mounts.append((len(str(mount)), parts[2], parts[3].split(',')))
    if not mounts or max(mounts)[1] != 'tmpfs':
        raise ValueError('Temporary output must be on tmpfs: '+str(path))
    if executable and 'noexec' in max(mounts)[2]:
        raise ValueError('Native copier requires an executable tmpfs mount')
    if len(Path('/proc/swaps').read_text().splitlines()) != 1:
        raise ValueError('Disable swap to keep transient output in RAM')

def payload(map_name):
    choices = read(ROOT/'opening/frozen_choices.json')['choices']
    compiled = read(APP/'compiled'/map_name/(choices[map_name]+'.json'))
    job = read(APP/'jobs.json')[map_name]
    scenario = read(ENV/'scenarios/cases'/job['scenario']/'scenario.json')
    wanted = {int(v) for v in compiled['unit_ids'].values()}
    units = [v['entity'] for v in scenario['imagineProfile']['entityList'] if int(v['entity']['id']) in wanted]
    clocks = {float(v['time'])/1000 for v in units}
    assert len(units) == len(wanted) and len(clocks) == 1
    return dict(compiled=compiled,
                opening=dict(choice=choices[map_name], candidate_version='pku_initial_astra_v2',
                             allocation='astra_v2', source='Previously frozen actual DeepSeek opening; not regenerated'),
                skill={'manifest': read(ROOT/'policy_manifest.json')}, own_initial_clock_s=clocks.pop())
