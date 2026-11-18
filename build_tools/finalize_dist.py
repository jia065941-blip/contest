from __future__ import annotations

import json
import shutil
from pathlib import Path

from build_layout import (
    COMPILED_ROOT, DIST_ROOT, MANIFEST_NAME, ROOT, RUNTIME_PACKAGES,
    artifact_path, clean_build_directory, collect_sources, require_runtime,
    retained_engine_file, runtime_identity, sha256,
)


def verified_build() -> dict:
    manifest = json.loads((COMPILED_ROOT / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["runtime"] != runtime_identity():
        raise ValueError("编译产物的 Python 或目标平台与当前解释器不符")
    sources = dict(collect_sources())
    if set(manifest["modules"]) != set(sources):
        raise ValueError("编译清单与当前 envengine 源码模块不一致，需要重新完整构建")
    for name, source in sources.items():
        entry = manifest["modules"][name]
        if entry["source"] != source or entry["artifact"] != artifact_path(source):
            raise ValueError(f"编译清单路径不符：{name}")
        if sha256(ROOT / source) != entry["source_sha256"]:
            raise ValueError(f"编译后源码发生变化：{source}")
        if sha256(COMPILED_ROOT / entry["artifact"]) != entry["artifact_sha256"]:
            raise ValueError(f"编译产物校验失败：{name}")
    return manifest


def copy_file(source: Path, stage: Path, relative: str) -> None:
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def require_native_dependencies(stage: Path) -> None:
    suffix = ".pyd" if runtime_identity()["platform"].startswith("win") else ".so"
    required = {
        "envengine/sdk/Util": "_UtilsPy",
        "envengine/simulator/models/ACMMissileModel": "_ACMMissilePy",
        "envengine/simulator/models/HXDMissileModel": "_HXDMissilePy",
        "envengine/simulator/models/OrbitModel": "_OrbitModelPy",
        "envengine/simulator/models/SM6_1BMissileModel": "_SM6_1BMissilePy",
    }
    for directory, name in required.items():
        if not (stage / directory / (name + suffix)).is_file():
            raise FileNotFoundError(f"缺少必要原生依赖：{directory}/{name}{suffix}")
        if directory != "envengine/sdk/Util":
            dependency = "Utils.dll" if suffix == ".pyd" else "libUtils.so"
            if not (stage / directory / dependency).is_file():
                raise FileNotFoundError(f"缺少必要动态库：{directory}/{dependency}")


def main() -> None:
    require_runtime()
    manifest = verified_build()
    stage = DIST_ROOT.with_name(DIST_ROOT.name + ".staging")
    clean_build_directory(stage)
    for entry in manifest["modules"].values():
        copy_file(COMPILED_ROOT / entry["artifact"], stage, entry["artifact"])
    for source in sorted((ROOT / "envengine").rglob("*")):
        if source.is_file():
            relative = source.relative_to(ROOT).as_posix()
            if retained_engine_file(source, relative):
                copy_file(source, stage, relative)
    for package in RUNTIME_PACKAGES:
        if not (ROOT / package / "__init__.py").is_file():
            raise FileNotFoundError(f"缺少运行包：{package}")
        for source in sorted((ROOT / package).rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix not in {".pyc", ".pyo"}:
                copy_file(source, stage, source.relative_to(ROOT).as_posix())
    for name in ("main.py", "simulator.py", "previewer.py", "config.yaml",
                 "requirements.txt", "README.md", "libUtils.so"):
        copy_file(ROOT / name, stage, name)
    require_native_dependencies(stage)
    manifest["files"] = {path.relative_to(stage).as_posix(): sha256(path)
                         for path in sorted(stage.rglob("*")) if path.is_file()}
    (stage / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # 只在新产物完整验证后替换旧目录，复制失败时保留上一份已完成产物。
    clean_build_directory(DIST_ROOT)
    DIST_ROOT.rmdir()
    stage.rename(DIST_ROOT)
    print(f"[info] 已生成完整独立产物：{DIST_ROOT}；文件 {len(manifest['files'])} 个")


if __name__ == "__main__":
    main()
