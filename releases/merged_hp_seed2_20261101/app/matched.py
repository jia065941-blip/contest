"""N-arm extension of the unchanged, previously validated process-fork core."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MATCH = HERE
sys.path.insert(0, str(MATCH))
import fork_probe as fork

CURRENT = None
EVENT = None
OBSERVER = None
MANIFEST = None
INSTALLED = False
FINISHED = False


def labels(manifest):
    arms = manifest['arms']
    values = [arm['label'] for arm in arms]
    if not 2 <= len(values) <= 64 or len(set(values)) != len(values):
        raise ValueError('Require 2..64 unique policy arms; first arm executes in the root')
    if any(not isinstance(x, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', x) for x in values):
        raise ValueError('Unsafe branch label')
    return tuple(values)


def controls():
    folder = ROOT / 'evidence/controls'
    result = json.loads((folder / 'results.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    from common import ENV, digest, sha
    assert result['status'] == 'passed' and result['completed'] == 20 and not result['errors']
    assert result['protocol_sha256'] == digest(protocol)
    assert len(result['rows']) == 20 and len({r['map'] for r in result['rows']}) == 20
    assert sha(MATCH / 'fork_probe.py') == protocol['branch_sources']['fork_probe.py']
    assert sha(ENV / 'envengine/engine/engine.py') == protocol['engine_sha256']
    for row in result['rows']:
        c = row['comparison']; snapshot = c['snapshot']
        assert row['status'] == c['status'] == 'passed' and not c['errors']
        assert set(c['branches']) == {'baseline_a', 'baseline_b', 'allocation_stress'}
        ref = c['branches']['baseline_a']
        for branch in c['branches'].values():
            assert not branch['error'] and branch['address_registry_unchanged']
            assert branch['start_registry_hash'] == snapshot['registry_hash']
            assert branch['start_rng_hash'] == snapshot['rng_hash']
            for key in ('final_score', 'steps', 'termination', 'frame_count', 'all_frames_sha256'):
                assert branch[key] == ref[key]
    return dict(path=str(folder), results_sha256=sha(folder / 'results.json'),
                protocol_sha256=sha(folder / 'protocol.json'), completed=20,
                reused=True, new_control_simulations=0)


def begin_event(event):
    global EVENT
    if EVENT is not None or fork.STATE or CURRENT is None:
        raise RuntimeError('Exactly one shared fork, inside an active native step, is required')
    if event.get('binding', {}).get('fixed_before_fork') is not True:
        raise ValueError('Actual participant bindings must be sealed before branching')
    EVENT = json.loads(json.dumps(event, allow_nan=False))
    fork.split(CURRENT)


def validate_audits(snapshot, rows, expected):
    errors = []
    if set(rows) != set(expected):
        errors.append(dict(error='Missing or extra branches', actual=sorted(rows), expected=list(expected)))
    for label, row in rows.items():
        if row.get('label') != label or row.get('error') or not row.get('address_registry_unchanged'):
            errors.append(dict(branch=label, error=row.get('error') or 'Branch/address mismatch'))
        if row.get('start_registry_hash') != snapshot['registry_hash'] or row.get('start_rng_hash') != snapshot['rng_hash']:
            errors.append(dict(branch=label, error='Snapshot/RNG mismatch'))
        if row.get('final_score') is None or not row.get('frame_hashes') or row.get('steps') is None:
            errors.append(dict(branch=label, error='Incomplete native continuation'))
        if row.get('termination') not in ('time_limit', 'environment_done'):
            errors.append(dict(branch=label, error='Invalid native termination'))
    return errors


def aggregate(root, snapshot, children):
    rows, errors = {}, []
    for label, pid, code in children:
        path = root / 'branches' / label / 'branch_audit.json'
        if code:
            errors.append(dict(branch=label, pid=pid, exit_status=code))
        if not path.exists():
            errors.append(dict(branch=label, error='Missing branch audit'))
            continue
        try:
            rows[label] = json.loads(path.read_text())
        except Exception as exc:
            errors.append(dict(branch=label, error='Unreadable audit: ' + repr(exc)))
    errors.extend(validate_audits(snapshot, rows, fork.BRANCHES))
    fork.save(root / 'comparison.json', dict(status='failed' if errors else 'complete',
        event_triggered=True, event=EVENT, snapshot=snapshot, branches=rows, errors=errors,
        official_event_order_unchanged=True, branch_count=len(fork.BRANCHES),
        root_is_arm=fork.BRANCHES[0], estimand=MANIFEST['scope']))
    return not errors


def finish(error=None):
    global FINISHED
    if FINISHED:
        return
    FINISHED = True
    failures = [error] if error else []
    runtime = sys.modules.get('skill_runtime')
    hook = getattr(runtime, 'finish', None)
    if hook:
        try:
            hook(error)
        except BaseException as exc:
            failures.append('Skill finalization: ' + repr(exc))
    if OBSERVER is not None:
        try:
            OBSERVER.finish('; '.join(failures) or None)
        except BaseException as exc:
            failures.append('Combat observer finalization: ' + repr(exc))
    failure = '; '.join(failures) or None
    if fork.STATE:
        # Always produce audits and reap children, including observer/hook errors.
        fork.finish(failure)
    else:
        root = Path(os.environ['PAIR_ROOT'])
        paths = list((root / 'native').glob('*/summary.json'))
        summary = json.loads(paths[0].read_text()) if len(paths) == 1 else {}
        fork.save(root / 'comparison.json', dict(status='complete' if summary and not failure else 'failed',
            event_triggered=False, score=summary.get('score', {}).get('score'), branches={},
            error=failure, root_is_arm=labels(MANIFEST)[0], note='One actual root policy episode; no divergence and no fabricated branch returns'))
    if failure:
        raise RuntimeError(failure)


def install():
    global OBSERVER, MANIFEST, INSTALLED
    if INSTALLED:
        raise RuntimeError('Matched instrumentation installed twice')
    INSTALLED = True
    controls()
    from common import episode_inputs
    MANIFEST = copy.deepcopy(episode_inputs()['skill']['manifest'])
    fork.BRANCHES = labels(MANIFEST)
    if os.environ.get('PAIR_FORK_STEP') != '-1':
        raise ValueError('Only explicit fixed-binding fork is permitted')
    diag = HERE
    sys.path.insert(0, str(diag))
    spec = importlib.util.spec_from_file_location('hp20_passive_combat_observer', diag / 'observer.py')
    OBSERVER = importlib.util.module_from_spec(spec); spec.loader.exec_module(OBSERVER)
    OBSERVER.install()
    original_isolate = fork.isolate_output

    def isolate(env, label, root):
        folder = original_isolate(env, label, root)
        os.environ['RSI_ARM'] = label
        # No shared writable stream offsets survive the branch transition.
        sys.stdout.flush(); sys.stderr.flush()
        fd = os.open(folder / 'console.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.dup2(fd, 1); os.dup2(fd, 2)
        finally:
            os.close(fd)
        return folder

    fork.isolate_output = isolate
    fork.validate_children = aggregate
    fork.install()
    from envengine.environment.training_env import TrainingEnv
    original_step = TrainingEnv.step

    def current(env, *args, **kwargs):
        global CURRENT
        CURRENT = env
        try:
            return original_step(env, *args, **kwargs)
        finally:
            CURRENT = None
    TrainingEnv.step = current
