"""红方公共接口与实现文件检查入口。

选手实现从 base 继承策略类和部署类，运行入口通过 checker 加载选定的 Python 文件。
example 包提供可供阅读和修改的策略实现，其内部工具由示例自行组织。
"""

from .base import BaseAgent, BaseDeployment
from .red_checker import check_deployment, check_red_agent

# 对外导出基础契约和检查入口，保持使用者对公共接口的引用集中。
__all__ = ["BaseAgent", "BaseDeployment", "check_deployment", "check_red_agent"]
