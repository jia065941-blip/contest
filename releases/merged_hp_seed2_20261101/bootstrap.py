"""Verify and unpack the exact native assets; compile copier; never start a game."""
from pathlib import Path, PurePosixPath
import argparse
import importlib.metadata
import os
import platform
import subprocess
import sys
import sysconfig
import tarfile
from support import ROOT, APP, read, sha, verify, assert_tmpfs, native_environment

def check_platform():
    if sys.flags.optimize:
        raise RuntimeError('Optimized Python disables required validation')
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Frozen native SDK requires Linux x86_64')
    reference = read(ROOT/'environment_reference.json')
    if sys.version != reference['python']:
        raise RuntimeError('Exact reference Python build required:\n'+reference['python']+'\nActual:\n'+sys.version)
    installed = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in importlib.metadata.distributions()}
    for line in (ROOT/'requirements.lock').read_text().splitlines():
        if not line or line.startswith(('#', '--')):
            continue
        name, version = line.split('==')
        if installed.get(name.lower().replace('_', '-')) != version:
            raise RuntimeError('Missing or different locked dependency: '+line)
    from packaging.requirements import Requirement
    for distribution in importlib.metadata.distributions():
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
                continue
            version = installed.get(requirement.name.lower().replace('_', '-'))
            if version is None or (requirement.specifier and version not in requirement.specifier):
                raise RuntimeError('Unsatisfied dependency of '+distribution.metadata['Name']+': '+raw)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    check_platform()
    for name, wanted in read(ROOT/'release_manifest.json')['files'].items():
        if sha(ROOT/name) != wanted:
            raise RuntimeError('Release checksum mismatch: '+name)
    manifest = read(ROOT/'assets_manifest.json')
    if sha(ROOT/'assets.tar.gz') != manifest['archive_sha256']:
        raise RuntimeError('Asset archive checksum mismatch')
    if not args.verify_only:
        with tarfile.open(ROOT/'assets.tar.gz') as tar:
            members = tar.getmembers()
            if len(members) != len(manifest['files']) or {m.name for m in members} != set(manifest['files']):
                raise RuntimeError('Archive inventory mismatch')
            for member in members:
                name = PurePosixPath(member.name)
                if not member.isfile() or name.is_absolute() or '..' in name.parts:
                    raise RuntimeError('Unsafe archive member')
                path = APP/member.name
                if any(p.is_symlink() for p in [path, *path.parents]):
                    raise RuntimeError('Symlink in asset destination')
                if path.exists():
                    if sha(path) != manifest['files'][member.name]:
                        raise RuntimeError('Existing asset changed: '+member.name)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(tar.extractfile(member).read())
    verify()
    sys.path.insert(0, str(APP/'performance_v2'))
    from copy_location import configure
    binary = configure()
    assert_tmpfs(binary.parent, executable=True)
    if not args.verify_only:
        binary.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = binary.with_name(binary.name+'.'+str(os.getpid())+'.tmp')
        try:
            subprocess.run(['cc', '-O3', '-shared', '-fPIC', '-Wall', '-Wextra', '-Wno-unused-parameter',
                            '-I'+sysconfig.get_path('include'), str(APP/'performance_v2/copy_core.c'),
                            '-o', str(temporary)], env={**os.environ, 'TMPDIR': str(binary.parent)}, check=True)
            temporary.replace(binary)
        finally:
            temporary.unlink(missing_ok=True)
    if not binary.is_file():
        raise RuntimeError('Compiled copier missing; run bootstrap without --verify-only')
    subprocess.run([sys.executable, '-B', str(ROOT/'verify_release.py'), '--native-imports'],
                   env=native_environment(Path('/tmp'), Path('/tmp')), check=True)

if __name__ == '__main__':
    main()
