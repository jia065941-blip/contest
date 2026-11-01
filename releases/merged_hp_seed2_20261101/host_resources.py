"""Host admission leases plus conservative reservations for unmodified controllers.

An unleased controller can start work without our lock. Therefore its remaining
work is reserved up to its full 64-slot capacity, not only its visible children.
This can deliberately leave us waiting until its tail has provable capacity.
The lock coordinates NEW controllers only; it cannot control arbitrary unrelated
future processes. Runtime pressure trips terminate only this stage's processes.
"""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import time

import stat
ROOT = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 2 * 1024 * 1024

class BridgeError(RuntimeError):
    pass

def encode(value, limit=MAX_REQUEST_BYTES):
    try:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BridgeError("payload is not finite JSON") from exc
    if len(data) > limit:
        raise BridgeError(f"JSON exceeds {limit} byte limit")
    return data

def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise BridgeError(f"duplicate JSON key: {key}")
        value[key] = item
    return value

def decode(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              BridgeError("non-finite JSON")))
    except (UnicodeError, ValueError) as exc:
        raise BridgeError("invalid JSON") from exc

def read_json(path, limit=MAX_REQUEST_BYTES):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise BridgeError("JSON input is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise BridgeError(f"JSON input exceeds {limit} byte limit")
        return decode(data)
    finally:
        os.close(fd)

def atomic_json(path, value, *, replace=False, limit=MAX_REQUEST_BYTES):
    """Publish complete JSON, exclusively unless explicit state replacement."""
    path = Path(path)
    data = encode(value, limit)
    temporary = path.parent / (".tmp-" + os.urandom(16).hex())
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

GIB = 1024 ** 3
CPU_LIMIT = 64
MEMORY_FRACTION = .90
PER_BRANCH = {'E': 4 * GIB, 'M': 9 * GIB, 'H': 23 * GIB}
KNOWN = {}


def scan(proc=Path('/proc')):
    records = {}
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = [a.decode(errors='replace') for a in (directory / 'cmdline').read_bytes().split(b'\0') if a]
            status = dict(line.split(':', 1) for line in (directory / 'status').read_text().splitlines() if ':' in line)
            if status.get('State', '').strip().startswith('Z'):
                continue
            stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            records[int(directory.name)] = {
                'args': args, 'command': ' '.join(args), 'ppid': int(status['PPid']),
                'start': stat[19], 'rss': int(status.get('VmRSS', '0 kB').split()[0]) * 1024,
                'peak': int(status.get('VmHWM', '0 kB').split()[0]) * 1024,
                'threads': int(status.get('Threads', '1')),
                'native': '--scenario' in args,
            }
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError('Cannot reliably account for host process ' + directory.name) from exc
    return records


def descendant(pid, ancestor, records):
    seen = set()
    while pid in records and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        pid = records[pid]['ppid']
    return False


def branch_count(command):
    for fragment, count in [('launch20_astra_cooperative_skills_', 4), ('launch20_astra_dynamic_', 2),
                            ('launch20_gx_v15_', 3), ('launch20_evidence_rsi_', 9),
                            ('launch20_matched_branches_', 3)]:
        if fragment in command:
            return count
    # Unknown native roots have unknown future fork fanout; block admission.
    return CPU_LIMIT


def native_groups(records):
    groups = {}
    for pid, row in records.items():
        if not row['native']:
            continue
        root = pid
        while records.get(records[root]['ppid'], {}).get('native'):
            root = records[root]['ppid']
        groups.setdefault(root, []).append(pid)
    return groups


def legacy_reservations(records, result_root=ROOT / 'results/launch20'):
    """Never trust stopped results alone to imply native descendants disappeared."""
    result = []
    for pid, row in records.items():
        command = row['command']
        if row['native'] or not any(Path(arg).name == 'evaluate.py' for arg in row['args']):
            continue
        name = next((n for n in KNOWN if n in command), None)
        if name:
            folder, branches, total, key = KNOWN[name]
            try:
                metadata = read_json(result_root / folder / 'controller.json')
                progress = read_json(result_root / folder / 'results.json', 64 * 1024 * 1024)
                if metadata['pid'] != pid:
                    raise ValueError('Controller PID differs from its progress record')
                if progress.get('status') in ('complete', 'stopped', 'failed', 'paused_pressure_or_error'):
                    # No more admissions; visible and reserved forks remain counted below.
                    continue
                remaining = max(0, total - int(progress[key]))
                remaining_slots = min(CPU_LIMIT, branches * remaining)
                completed = progress.get('rows', [])
                # A future legacy admission may select any remaining tier. Reserve
                # the largest possible memory unless all remaining maps are known.
                known_maps = {r.get('map') for r in completed}
                choices = progress.get('choices')
                tier = max((PER_BRANCH[m[0]] for m in choices if m not in known_maps), default=23 * GIB) if choices else 23 * GIB
                result.append({'pid': pid, 'slots': remaining_slots,
                               'memory': remaining_slots * tier, 'kind': name})
            except (OSError, ValueError, KeyError):
                result.append({'pid': pid, 'slots': CPU_LIMIT, 'memory': CPU_LIMIT * 23 * GIB,
                               'kind': name + ':unreadable_progress'})
        elif ('launch20_' in command and 'launch20_astra_dynamic_rsi_20261029' not in command
              and '--offline-check' not in row['args']):
            result.append({'pid': pid, 'slots': CPU_LIMIT, 'memory': CPU_LIMIT * 23 * GIB,
                           'kind': 'unleased_controller_unknown_capacity'})
    return result


def account(records, leases, legacy):
    groups = native_groups(records)
    covered, slots, growth = set(), 0, 0
    details = []
    for lease in leases:
        root = lease.get('native_pid')
        native = [pid for pid, row in records.items() if row['native'] and root and descendant(pid, root, records)]
        covered.update(native)
        actual = sum(records[p]['rss'] for p in native)
        n = max(lease['slots'], sum(records[p]['threads'] for p in native))
        slots += n
        expected = max(lease['memory_bytes'], max((records[p]['peak'] for p in native), default=0) * lease['slots'] * 1.25)
        growth += max(0, expected - actual)
        details.append({'kind': 'lease', 'slots': n, 'native': len(native)})
    for item in legacy:
        native = [p for p, row in records.items() if row['native'] and p not in covered and descendant(p, item['pid'], records)]
        covered.update(native)
        actual = sum(records[p]['rss'] for p in native)
        # Legacy remaining slots include roots and every pending descendant.
        n = max(item['slots'], sum(records[p]['threads'] for p in native))
        slots += n
        expected = max(item['memory'], max((records[p]['peak'] for p in native), default=0) * n * 1.25)
        growth += max(0, expected - actual)
        details.append({'kind': item['kind'], 'slots': n, 'native': len(native), 'pid': item['pid']})
    for root, members in groups.items():
        native = [p for p in members if p not in covered]
        if not native:
            continue
        n = max(branch_count(records[root]['command']), sum(records[p]['threads'] for p in native))
        slots += n
        actual = sum(records[p]['rss'] for p in native)
        growth += max(0, max(records[p]['peak'] for p in native) * n * 1.25 - actual)
        details.append({'kind': 'unleased_native_with_future_forks', 'slots': n, 'native': len(native), 'pid': root})
    return {'reserved_cpu_slots': slots, 'future_growth_bytes': growth,
            'actual_native_processes': sum(r['native'] for r in records.values()),
            'actual_native_threads': sum(r['threads'] for r in records.values() if r['native']), 'groups': details}


class HostLeases:
    def __init__(self, root, stage_id):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.root.lstat()
        if not self.root.is_dir() or self.root.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError('Shared lease root must be private and owned')
        self.stage_id = stage_id
        self.owner = os.getpid()
        self.owner_start = scan()[self.owner]['start']

    @contextmanager
    def locked(self):
        fd = os.open(self.root / 'lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def active(self, records):
        leases = []
        for path in self.root.glob('*.json'):
            item = read_json(path)
            if item['status'] != 'active':
                continue
            owner_live = records.get(item['owner_pid'], {}).get('start') == item['owner_start']
            native_live = records.get(item.get('native_pid'), {}).get('start') == item.get('native_start') if item.get('native_pid') else False
            if owner_live or native_live:
                leases.append(item)
            else:
                item.update(status='owner_and_native_gone', ended_ns=time.time_ns())
                atomic_json(path, item, replace=True)
        return leases

    def snapshot(self):
        with self.locked():
            records = scan()
            return account(records, self.active(records), legacy_reservations(records))

    def acquire(self, job, memory_bytes, total, available, *, cpu_capacity=CPU_LIMIT, memory_headroom=None):
        with self.locked():
            records = scan()
            load = account(records, self.active(records), legacy_reservations(records))
            headroom = available - total * .10 if memory_headroom is None else memory_headroom
            if load['reserved_cpu_slots'] + 2 > min(CPU_LIMIT, cpu_capacity) or headroom - load['future_growth_bytes'] - memory_bytes < 0:
                return None, load
            token = os.urandom(16).hex()
            item = {'token': token, 'owner_pid': self.owner, 'owner_start': self.owner_start,
                    'stage_id': self.stage_id, 'job': job, 'slots': 2, 'memory_bytes': memory_bytes,
                    'status': 'active', 'created_ns': time.time_ns(), 'native_pid': None}
            atomic_json(self.root / (token + '.json'), item)
            return token, load

    def bind(self, token, pid):
        with self.locked():
            path = self.root / (token + '.json')
            item = read_json(path)
            item.update(native_pid=pid, native_start=scan()[pid]['start'])
            atomic_json(path, item, replace=True)

    def release(self, token):
        with self.locked():
            path = self.root / (token + '.json')
            item = read_json(path)
            records = scan()
            root = item.get('native_pid')
            if root and any(r['native'] and descendant(p, root, records) for p, r in records.items()):
                raise RuntimeError('Cannot release lease with live native descendants')
            item.update(status='released', ended_ns=time.time_ns())
            atomic_json(path, item, replace=True)
