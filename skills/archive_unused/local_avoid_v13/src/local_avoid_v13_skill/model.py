"""Public numerical Skill for the converted V13 Actor."""

from importlib.resources import files
from pathlib import Path

from skill_generation.registry import BackendRegistry
from skill_generation.skills import InferenceSkill

from .backend import BACKEND_ID, LocalAvoidV13Backend


def default_bundle_path() -> Path:
    return Path(str(files("local_avoid_v13_skill").joinpath("bundle")))


def create_v13_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(BACKEND_ID, LocalAvoidV13Backend)
    return registry


class LocalAvoidV13Skill(InferenceSkill):
    def __init__(self, bundle_dir: str | Path | None = None, **kwargs):
        if "registry" in kwargs:
            raise TypeError("LocalAvoidV13Skill owns its trusted backend registry")
        super().__init__(bundle_dir or default_bundle_path(),
                         registry=create_v13_registry(), **kwargs)
