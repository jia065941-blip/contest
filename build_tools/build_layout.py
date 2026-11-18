import hashlib
import platform
import shutil
import struct
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILD_ROOT = ROOT / "build"
COMPILED_ROOT = BUILD_ROOT / "envengine_compiled"
DIST_ROOT = BUILD_ROOT / "envengine_dist"
KEEP_SOURCE_DIRS = ()
NATIVE_DEPENDENCY_DIRS = ("envengine/simulator/models", "envengine/sdk/Util")
RUNTIME_PACKAGES = ("blue_defense", "example_evaluation", "scenarios", "user_agents")
MANIFEST_NAME = "build_manifest.json"


def require_runtime() -> None:
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11)
            or struct.calcsize("P") != 8 or sys.platform not in {"win32", "linux"}
            or platform.machine().lower() not in {"amd64", "x86_64"}):
        raise RuntimeError("构建及运行要求 Windows/Linux x86_64 的 CPython 3.11")


def runtime_identity() -> dict:
    return {"python": sys.implementation.cache_tag, "platform": sysconfig.get_platform(),
            "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX")}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_build_directory(path: Path) -> None:
    resolved = path.resolve()
    expected_build = ROOT.resolve() / "build"
    if BUILD_ROOT.resolve() != expected_build or resolved.parent != expected_build or path.is_symlink():
        raise ValueError(f"只能清理本项目 build 的直接子目录：{path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def collect_sources() -> list[tuple[str, str]]:
    base = ROOT / "envengine"
    if not base.is_dir():
        raise FileNotFoundError(base)
    result = []
    for path in sorted(base.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if "__pycache__" in path.parts:
            continue
        if any(relative.startswith(directory + "/") for directory in KEEP_SOURCE_DIRS):
            continue
        result.append((relative[:-3].replace("/", "."), relative))
    if not result:
        raise ValueError("没有找到需要编译的 envengine 模块")
    return result


def artifact_path(source: str) -> str:
    return str(Path(source).with_suffix(runtime_identity()["extension_suffix"])).replace("\\", "/")


def retained_engine_file(path: Path, relative: str) -> bool:
    if "__pycache__" in path.parts or path.suffix in {
        ".py", ".pyi", ".pyc", ".pyo", ".c", ".cpp", ".h", ".obj", ".lib", ".exp", ".pdb",
    }:
        return False
    vendor = any(relative.startswith(directory + "/") for directory in NATIVE_DEPENDENCY_DIRS)
    if path.suffix in {".pyd", ".so", ".dll"}:
        return vendor
    return True
