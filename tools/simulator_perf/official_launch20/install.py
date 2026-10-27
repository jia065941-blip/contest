"""Install the frozen launch20 SDK copy optimizations after checking all targets.

Pass the directory containing envengine, NOT contest/core. --check is read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def install(sdk, check=False):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    changes = []
    # Check every source and target before writing anything. Accept repeated use.
    for relative, hashes in manifest['files'].items():
        source = HERE / 'overlay' / relative
        destination = sdk / relative
        if digest(source) != hashes['after']:
            raise RuntimeError('Packaged source changed: ' + relative)
        actual = digest(destination)
        if actual == hashes['after']:
            continue
        if actual != hashes['before']:
            raise RuntimeError('Incompatible SDK or local modification: ' + str(destination))
        changes.append((source, destination))
    if not check:
        for source, destination in changes:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    return {'base_sdk_commit': manifest['base_sdk_commit'], 'check_only': check,
            'files_to_update': [str(destination.relative_to(sdk)) for _, destination in changes]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(install(args.sdk.resolve(), args.check), indent=2))
