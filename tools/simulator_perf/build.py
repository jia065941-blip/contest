"""Build the copy extension once into RAM; print only the library path."""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sysconfig
import tempfile

HERE = Path(__file__).resolve().parent


def build(source, cache):
    # Source + ABI + platform identify an artifact; no per-game compilation.
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if not suffix:
        raise RuntimeError('A CPython extension build is required')
    identity = source.read_bytes() + str((suffix, sysconfig.get_platform())).encode()
    digest = hashlib.sha256(identity).hexdigest()[:20]
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / (digest + suffix)
    # Multiple launchers share a single build. Lock lives in the RAM directory.
    import fcntl
    with (cache / (digest + '.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.is_file():
            with tempfile.TemporaryDirectory(prefix='build-', dir=cache) as folder:
                target = Path(folder) / output.name
                command = shlex.split(os.environ.get('CC', 'cc')) + [
                    '-O3', '-shared', '-fPIC', '-Wall', '-Wextra',
                    '-Wno-unused-parameter',
                    '-I' + sysconfig.get_paths()['include'],
                    str(source), '-o', str(target),
                ]
                env = dict(os.environ, TMPDIR=folder)
                subprocess.run(command, check=True, env=env)
                os.replace(target, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-launch20', action='store_true')
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('/dev/shm') / ('contest-copy-' + str(os.getuid())))
    args = parser.parse_args()
    source = HERE / ('official_launch20/copy_core.c' if args.official_launch20 else 'copy_core.c')
    print(build(source, args.cache_dir.resolve()))


if __name__ == '__main__':
    main()
