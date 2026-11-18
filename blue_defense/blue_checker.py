"""加载蓝方 Python 文件并检查固定的策略类入口。"""

from __future__ import annotations

import hashlib
from importlib import import_module, util
import inspect
from pathlib import Path
import sys

from .base import BaseBlueDefense


_BLUE_ROOT = Path(__file__).resolve().parent


def _load_module(path: Path):
    if path.is_relative_to(_BLUE_ROOT):
        parts = path.relative_to(_BLUE_ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(("blue_defense", *parts))
        module = import_module(name)
        if Path(module.__file__).resolve() != path:
            raise ValueError(f"模块 {name} 与指定路径不一致")
        return module

    name = "_competition_blue_" + hashlib.sha256(str(path).encode()).hexdigest()
    if name in sys.modules:
        return sys.modules[name]
    specification = util.spec_from_file_location(name, path)
    module = util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def check_blue_policy(path: str | Path) -> type[BaseBlueDefense]:
    """返回符合公共契约的 BlueDefense 类，实例化由运行入口负责。"""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".py" or not path.is_file():
        raise ValueError(f"蓝方策略路径必须是现有 Python 文件：{path}")
    try:
        module = _load_module(path)
        implementation = getattr(module, "BlueDefense", None)
        if not inspect.isclass(implementation) or not issubclass(implementation, BaseBlueDefense):
            raise ValueError("需要导出继承 BaseBlueDefense 的 BlueDefense 类")
        if implementation is BaseBlueDefense or inspect.isabstract(implementation):
            raise ValueError("BlueDefense 必须是可实例化的具体类")

        # 校验构造签名和同步实例方法的调用形式。
        inspect.signature(implementation).bind(seed=None, asset_values={})
        for method, arguments in (("decide", (None, None)), ("reset", (None,)),
                                  ("close", (None,)), ("asset_value", (None, 0, 0))):
            descriptor = inspect.getattr_static(implementation, method)
            function = getattr(implementation, method)
            if (isinstance(descriptor, (staticmethod, classmethod))
                    or not callable(function) or inspect.iscoroutinefunction(function)
                    or inspect.isasyncgenfunction(function) or inspect.isgeneratorfunction(function)):
                raise ValueError(f"{method} 必须是同步实例方法")
            inspect.signature(function).bind(*arguments)

        # 内置控制器通过 property 提供元数据，具体值在环境接入实例时检查。
        for name in ("name", "plans_launch_time", "preserves_pending_launches"):
            value = inspect.getattr_static(implementation, name)
            if isinstance(value, property):
                if value.fget is None or inspect.iscoroutinefunction(value.fget):
                    raise ValueError(f"{name} 必须是同步可读属性")
            elif name == "name":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("name 必须是非空字符串")
            elif type(value) is not bool:
                raise ValueError(f"{name} 必须是布尔值")
        return implementation
    except Exception as error:
        raise ValueError(f"蓝方策略检查失败：{path}：{error}") from error
