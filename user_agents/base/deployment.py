"""红方初始位置方案接口，供运行入口在策略创建前调用一次。"""

from abc import ABC, abstractmethod


class BaseDeployment(ABC):
    """接收待部署实体与部署条件，返回实体编号到平面 x、y 坐标的映射。

    具体实现采用无参构造，所需清单与区域约束由 deploy 的观测参数提供。
    AgentManager 检查方案后修改仿真实体的位置，并统一启动本局红方单元。
    """

    @abstractmethod
    def deploy(self, observation: dict) -> dict[int, tuple[float, float]]:
        """生成本次全部受控实体的二维初始位置，坐标单位为米。

        observation 包含 step、entities、public_targets、coordinates、seed。
        entities 为 {实体 ID: {"type": 实体类型}}，不包含场景原位置。
        coordinates 是单个部署多边形的顶点列表，
        类型为 list[list[float]]，格式为 [[x, y], ...]。部署接口要求且仅接受一个区域。
        public_targets 提供公开母巢和哨卫巢的初态，格式与策略初始化的 entities
        目录一致；暗巢不在此目录中。seed 提供随机种子。
        返回 {实体 ID: (x, y)}，
        由管理器校验实体覆盖和区域范围，沿用每个实体的原始高度应用位置。
        """

    def close(self) -> None:
        """释放部署实现持有的资源；运行入口在方案处理结束后调用，默认无需清理。"""
