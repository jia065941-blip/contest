"""加载红方 Python 文件，分别检查单机策略类与部署类的调用契约。

同一个文件导出 RedAttack 和 Deployment。模块加载会执行文件顶层代码，
之后通过类继承关系与函数签名进行检查，再把类交给运行方创建实例。
策略内部的信息汇总与通信使用方式由实现代码及其审查约束。
"""

from __future__ import annotations

import hashlib
from importlib import import_module, util
import inspect
from pathlib import Path
import sys

from .base import BaseAgent, BaseDeployment


# 内置文件按包名加载，外置文件按完整路径生成独立模块名。
_USER_ROOT = Path(__file__).resolve().parent


def _external_module_name(path: Path) -> str:
    """把已规范化的绝对路径转成稳定模块名，区分不同目录下的同名文件。"""
    return "_competition_red_" + hashlib.sha256(str(path).encode()).hexdigest()


def _load_module(path: Path):
    """复用 Python 模块缓存，使部署检查与策略检查取得同一批类定义。"""
    if path.is_relative_to(_USER_ROOT):
        # 内置实现使用真实包名，保留组内和共享模块的相对导入。
        parts = path.relative_to(_USER_ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(("user_agents", *parts))
        module = import_module(name)
        if Path(module.__file__).resolve() != path:
            raise ValueError(f"模块 {name} 与指定路径不一致")
        return module

    # 外部同名文件按完整路径分别缓存，加载失败时清理本次模块。
    name = _external_module_name(path)
    if name in sys.modules:
        return sys.modules[name]
    specification = util.spec_from_file_location(name, path)
    module = util.module_from_spec(specification)
    # 先登记模块再执行顶层代码，使模块执行期间能够按自己的模块名找到定义。
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _check_implementation(path, entry_name, base_class, constructor_arguments, methods):
    """用给定构造参数和方法调用样例，核对一个具体入口类的接口。"""
    # 路径规范化及文件检查：后续模块缓存和错误信息均使用同一个绝对路径。
    path = Path(path).expanduser().resolve()
    label = "人族策略" if entry_name == "RedAttack" else "部署"
    if path.suffix.lower() != ".py" or not path.is_file():
        raise ValueError(f"{label}实现路径必须是现有Python文件：{path}")
    external_name = None if path.is_relative_to(_USER_ROOT) else _external_module_name(path)
    cached = external_name in sys.modules if external_name is not None else False
    try:
        # 类角色检查：导出对象须继承对应基类，并完整实现抽象方法。
        module = _load_module(path)
        implementation = getattr(module, entry_name, None)
        if not inspect.isclass(implementation) or not issubclass(implementation, base_class):
            raise ValueError(f"需要导出继承{base_class.__name__}的{entry_name}类")
        if implementation is base_class or inspect.isabstract(implementation):
            raise ValueError(f"{entry_name}必须是可实例化的具体类")

        # 签名绑定只验证参数能否按运行方的方式传入，实例及动作由调用方创建和执行。
        inspect.signature(implementation).bind(*constructor_arguments)
        for method, arguments, keywords in methods:
            # 同步实例方法使用普通顺序调用；同时读取静态描述符以识别 static/class 方法。
            descriptor = inspect.getattr_static(implementation, method)
            function = getattr(implementation, method)
            if (isinstance(descriptor, (staticmethod, classmethod))
                    or not callable(function) or inspect.iscoroutinefunction(function)
                    or inspect.isasyncgenfunction(function) or inspect.isgeneratorfunction(function)):
                raise ValueError(f"{method}必须是同步实例方法")
            inspect.signature(function).bind(*arguments, **keywords)
        return implementation
    except Exception as error:
        # 首次加载的外置实现若检查失败，移除缓存，允许修改同一文件后重新检查。
        if external_name is not None and not cached:
            sys.modules.pop(external_name, None)
        raise ValueError(f"{label}实现检查失败：{path}：{error}") from error


def check_red_agent(path: str | Path) -> type[BaseAgent]:
    """返回具体 RedAttack 类，供管理器按受控实体创建单机策略实例。"""
    # None 代表调用时绑定的 self；其余值模拟 get_obs/get_action 与各生命周期的实参。
    methods = (
        ("get_obs", (None, {}), {}),
        ("get_action", (None,), {}),
        ("reset", (None,), {}),
        ("on_reset", (None,), {}),
        ("record_step", (None,), {"observation": {}, "action": [], "reward": 0.0, "info": {}}),
        ("on_feedback", (None, {}, [], 0.0, None), {}),
        ("close", (None,), {}),
    )
    return _check_implementation(path, "RedAttack", BaseAgent, (1, {}), methods)


def check_deployment(path: str | Path) -> type[BaseDeployment]:
    """返回无参构造的 Deployment 类，供运行方在初始阶段生成位置方案。"""
    # 部署对象仅需生成方案并支持关闭，构造时所需参数为空元组。
    methods = (
        ("deploy", (None, {}), {}),
        ("close", (None,), {}),
    )
    return _check_implementation(path, "Deployment", BaseDeployment, (), methods)
