from __future__ import annotations

import importlib.metadata
import json
import os
import sys

from build_layout import (
    BUILD_ROOT, COMPILED_ROOT, MANIFEST_NAME, ROOT, artifact_path,
    clean_build_directory, collect_sources, require_runtime, runtime_identity, sha256,
)


def main() -> None:
    require_runtime()
    if sys.argv[1:] != ["build_ext"]:
        raise SystemExit("用法：python build_tools/cython_setup.py build_ext")

    from Cython.Build import cythonize
    from setuptools import Extension, setup

    sources = collect_sources()
    source_hashes = {source: sha256(ROOT / source) for _, source in sources}
    generated = BUILD_ROOT / "cython_gen"
    temporary = BUILD_ROOT / "cython_tmp"
    for directory in (generated, temporary, COMPILED_ROOT):
        clean_build_directory(directory)

    os.chdir(ROOT)
    print(f"[info] 串行编译 {len(sources)} 个模块", flush=True)
    distribution = setup(
        name="competition-platform-native", script_args=["build_ext"],
        ext_modules=cythonize(
            [Extension(name, [source]) for name, source in sources],
            compiler_directives={"language_level": "3", "annotation_typing": False, "infer_types": False},
            # 相对路径避免 setuptools 重复嵌入工程绝对路径，超过 MSVC 的文件路径上限。
            build_dir=generated.relative_to(ROOT).as_posix(), nthreads=0,
        ),
        options={"build_ext": {
            "build_lib": COMPILED_ROOT.relative_to(ROOT).as_posix(),
            "build_temp": temporary.relative_to(ROOT).as_posix(), "parallel": None,
        }},
    )
    modules = {}
    for name, source in sources:
        artifact = artifact_path(source)
        if sha256(ROOT / source) != source_hashes[source]:
            raise RuntimeError(f"编译期间源码发生变化：{source}")
        modules[name] = {"source": source, "source_sha256": source_hashes[source],
                         "artifact": artifact, "artifact_sha256": sha256(COMPILED_ROOT / artifact)}
    compiler = distribution.get_command_obj("build_ext").compiler
    manifest = {"schema_version": 1, "runtime": runtime_identity(), "modules": modules,
                "compiler": {"type": compiler.compiler_type, "executable": getattr(compiler, "cc", None)},
                "build_dependencies": {name: importlib.metadata.version(name)
                                       for name in ("Cython", "setuptools")}}
    (COMPILED_ROOT / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[info] {len(modules)} 个模块编译完成，来源清单位于 {COMPILED_ROOT / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
