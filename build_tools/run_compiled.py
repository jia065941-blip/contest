from __future__ import annotations

import importlib.machinery
import json
import runpy
import sys
from pathlib import Path

from build_layout import DIST_ROOT, MANIFEST_NAME, ROOT, require_runtime, runtime_identity, sha256


def validate_distribution(directory: Path) -> None:
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["runtime"] != runtime_identity():
        raise ValueError("独立产物与当前 Python 或平台不符")
    for relative, expected_hash in manifest["files"].items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            raise FileNotFoundError(f"独立产物缺少文件或含非法路径：{relative}")
        if sha256(path) != expected_hash:
            raise ValueError(f"独立产物文件校验失败：{relative}")
    for entry in manifest["modules"].values():
        path = directory / entry["artifact"]
        if (not path.resolve().is_relative_to(directory.resolve())
                or sha256(path) != entry["artifact_sha256"]):
            raise ValueError(f"编译模块完整性校验失败：{entry['artifact']}")
    spec = importlib.machinery.PathFinder.find_spec("envengine", [str(directory)])
    if spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        raise ImportError("独立产物必须包含编译后的 envengine 包入口")


def main() -> None:
    require_runtime()
    validate_distribution(DIST_ROOT)
    if any(name == "envengine" or name.startswith("envengine.") for name in sys.modules):
        raise RuntimeError("请在独立 Python 进程中启动编译产物")
    # 只保留产物及解释器依赖路径，防止从工程目录加载缺失模块。
    sys.path[:] = [str(DIST_ROOT), *(entry for entry in sys.path
                     if entry and not Path(entry).resolve().is_relative_to(ROOT.resolve()))]
    runpy.run_path(str(DIST_ROOT / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
