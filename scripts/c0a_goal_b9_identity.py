"""Opt-in evidence for same-target b9 branch identity (simulation, not accuracy)."""
from __future__ import annotations
import hashlib
from collections import deque
import json
import random
import types
import numpy as np
import torch


def fingerprint(root):
    """Hash reachable Python state; enumerate opaque C++ types rather than hide them.

    Native objects are inherited by fork. This hash alone does not certify their
    internal bytes. Functions/types describe immutable code, not mutable state.
    """
    seen, opaque = {}, set()
    def walk(x):
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, (bytes, bytearray)):
            return [type(x).__name__, hashlib.sha256(x).hexdigest()]
        if isinstance(x, (types.FunctionType, types.BuiltinFunctionType, type)):
            return ["code", getattr(x, "__module__", ""), getattr(x, "__qualname__", str(x))]
        if isinstance(x, types.ModuleType):
            return ["module", x.__name__]
        identity = id(x)
        if identity in seen:
            return ["ref", seen[identity]]
        seen[identity] = len(seen)
        name = type(x).__module__ + "." + type(x).__qualname__
        if isinstance(x, torch.Tensor):
            a = x.detach().cpu().contiguous()
            return [name, str(a.dtype), list(a.shape), hashlib.sha256(a.numpy().tobytes()).hexdigest()]
        if isinstance(x, np.ndarray):
            return [name, str(x.dtype), list(x.shape), hashlib.sha256(x.tobytes()).hexdigest()]
        if isinstance(x, np.generic):
            return walk(x.item())
        if isinstance(x, random.Random):
            return [name, walk(x.getstate())]
        if isinstance(x, np.random.Generator):
            return [name, walk(x.bit_generator.state)]
        if isinstance(x, np.random.RandomState):
            return [name, walk(x.get_state())]
        if isinstance(x, torch.Generator):
            return [name, walk(x.get_state())]
        if isinstance(x, dict):
            return [name, [[walk(k), walk(v)] for k, v in x.items()]]
        if isinstance(x, (list, tuple, deque)):
            return [name, [walk(v) for v in x]]
        if isinstance(x, (set, frozenset)):
            return [name, [walk(v) for v in sorted(x, key=repr)]]
        if isinstance(x, types.MethodType):
            return ["method", walk(x.__func__), walk(x.__self__)]
        if hasattr(x, "__dict__"):
            return [name, walk(vars(x))]
        if hasattr(type(x), "__slots__"):
            return [name, [[k, walk(getattr(x,k))] for k in type(x).__slots__ if hasattr(x,k)]]
        opaque.add(name)
        return ["opaque", name, repr(x)]
    data = json.dumps(walk(root), ensure_ascii=False, allow_nan=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(data.encode()).hexdigest(), "opaque_types": sorted(opaque)}


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=float, separators=(",", ":")).encode()).hexdigest()


def rng_fingerprint():
    return fingerprint((random.getstate(), np.random.get_state(), torch.get_rng_state()))["sha256"]


class StepTrace:
    def __init__(self, path):
        self.path = path
        self.handle = path.open("w")
        self.digest = hashlib.sha256()
        self.count = 0

    def add(self, step, actions, observation, commander, health, reward, locked, suppressed):
        row = {
            "step": step, "actions": actions,
            "actions_sha256": json_sha(actions),
            "full_observation_sha256": fingerprint(observation)["sha256"],
            "teacher_state_sha256": fingerprint(commander)["sha256"],
            "public_rng_sha256": rng_fingerprint(),
            "objective_health": health, "official_return": reward,
            "locked": locked, "suppressed_total": suppressed,
        }
        line = json.dumps(row, sort_keys=True, default=float) + "\n"
        self.handle.write(line)
        self.handle.flush()
        self.digest.update(line.encode())
        self.count += 1

    def close(self):
        self.handle.close()
        return {"path": str(self.path), "sha256": self.digest.hexdigest(), "steps": self.count}
