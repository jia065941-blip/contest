"""Process-local container copy acceleration, preserving existing copy boundaries.

Set CONTEST_COPY_LIBRARY to the artifact printed by tools/simulator_perf/build.py
before starting workers. Compilation never happens during simulator execution.
Without that setting, use the Python container path. Other classes retain their
standard copy protocol. CONTEST_COPY_BACKEND=stdlib restores the standard copier.
"""
import copy as _copy
import importlib.util
import os

_ATOMIC = frozenset((type(None), bool, int, float, complex, str, bytes))


def python_deepcopy(value, memo=None):
    if memo is None:
        memo = {}
    elif type(memo) is not dict:
        return _copy.deepcopy(value, memo)
    key = id(value)
    if key in memo:
        return memo[key]
    cls = type(value)
    if cls in _ATOMIC:
        return value
    if cls is dict:
        result = {}
        memo[key] = result
        for k, v in value.items():
            # Keep standard deepcopy's value-before-key callback order.
            result[python_deepcopy(k, memo)] = python_deepcopy(v, memo)
    elif cls is list:
        result = []
        memo[key] = result
        result.extend(python_deepcopy(v, memo) for v in value)
    else:
        return _copy.deepcopy(value, memo)
    _copy._keep_alive(value, memo)
    return result


BACKEND = os.environ.get('CONTEST_COPY_BACKEND', 'auto')
if BACKEND not in ('auto', 'python', 'c', 'stdlib'):
    raise ValueError('CONTEST_COPY_BACKEND must be auto, python, c or stdlib')
_library = os.environ.get('CONTEST_COPY_LIBRARY')
if BACKEND == 'stdlib':
    deepcopy = _copy.deepcopy
elif BACKEND in ('auto', 'c') and _library:
    _spec = importlib.util.spec_from_file_location('astra_copy_core', _library)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _module.configure(_copy.deepcopy)
    deepcopy = _module.deepcopy
    BACKEND = 'c'
elif BACKEND == 'c':
    raise RuntimeError('Build once and set CONTEST_COPY_LIBRARY before launching workers')
else:
    deepcopy = python_deepcopy
    BACKEND = 'python'
