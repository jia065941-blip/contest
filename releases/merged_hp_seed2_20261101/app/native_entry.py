"""Frozen native baseline, merged strategic tree, HP and same-memory audit."""
from pathlib import Path
import runpy
import sys

if sys.flags.optimize:
    raise RuntimeError('Native validation requires a non-optimized Python interpreter')

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE/'performance_v2'), str(HERE/'performance_v3'), str(HERE/'r3')]
from common import ENV, OLD, HARNESS, read, sha
sys.path[:0] = [str(ENV), str(OLD), str(HARNESS)]
for folder, name in [(HERE/'performance_v2', 'compiled_validation.json'),
                     (HERE/'performance_v3', 'compiled_validation.json'),
                     (HERE/'performance_v3', 'boundary_validation.json')]:
    proof = read(folder/name)
    assert proof['status'] == 'passed'
    for source, wanted in proof.get('sources', {}).items():
        assert sha(folder/source) == wanted, source
    if name == 'boundary_validation.json':
        assert sha(folder/'communication_copy.py') == proof['source_sha256']
import compiled_copy, ownership, communication_copy
from copy_location import configure
configure()
compiled_copy.install()
ownership.install()
communication_copy.install()
import matched
matched.install()
error = None
try:
    runpy.run_path(str(HERE/'entry.py'), run_name='__main__')
except SystemExit as exc:
    if exc.code not in (None, 0):
        error = str(exc.code)
    raise
except BaseException as exc:
    error = repr(exc)
    raise
finally:
    matched.finish(error)
