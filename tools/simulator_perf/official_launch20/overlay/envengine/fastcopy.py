"""Specialized deep copy, preserving mutable isolation, cycles and aliases.

Only exact built-in dict/list and two audited plain SDK dataclasses have fast
paths. Subclasses and all other types retain Python's standard copy protocol.
No global monkey patch; no shared mutable communication/observation buffers.
"""
import copy as _copy
from envengine.sdk.base_struct.Basic import Vector3d, DetectInfo

_ATOMIC = frozenset((type(None), bool, int, float, complex, str, bytes))
_PLAIN = frozenset((Vector3d, DetectInfo))


def deepcopy(value, memo=None):
    cls = type(value)
    if cls in _ATOMIC:
        return value
    if memo is None:
        memo = {}
    key = id(value)
    if key in memo:
        return memo[key]
    if cls is dict:
        result = {}
        memo[key] = result
        for k, v in value.items():
            result[k if type(k) in _ATOMIC else deepcopy(k, memo)] = (
                v if type(v) in _ATOMIC else deepcopy(v, memo))
    elif cls is list:
        result = []
        memo[key] = result
        result.extend(v if type(v) in _ATOMIC else deepcopy(v, memo) for v in value)
    elif cls in _PLAIN:
        # These exact classes have no slots or custom copy/state/reduce hooks.
        result = object.__new__(cls)
        memo[key] = result
        result.__dict__.update(deepcopy(value.__dict__, memo))
    else:
        return _copy.deepcopy(value, memo)
    _copy._keep_alive(value, memo)
    return result
