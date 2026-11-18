"""红方公共契约：单实体策略与独立部署方案。

基础层定义调用方式、单机历史和生命周期；具体算法由选定的实现文件提供。
"""

from .base_agent import BaseAgent
from .deployment import BaseDeployment

__all__ = ["BaseAgent", "BaseDeployment"]
