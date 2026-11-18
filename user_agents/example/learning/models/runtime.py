"""按示例配置创建一套学习机动策略。

随机策略直接使用 NumPy；选择 PPO/MAPPO 时再加载对应的 PyTorch 实现。
返回的模型对象由示例私有资源管理器持有，多个本机执行组件复用模型参数。
"""

from __future__ import annotations

from pathlib import Path

from ...common.config import LEARNING_MOTION_CHOICES
from .red_policy import RandomMaskedPolicy, SharedPolicy


def build_learning_motion_policy(
    name: str,
    *,
    seed: int | None = None,
    model_path: str | None = None,
    max_steps: int = 1000,
    training: bool = False,
    observation_dim: int = 85,
) -> SharedPolicy:
    """创建随机、PPO 或 MAPPO 策略，并设置训练或推理模式。

    observation_dim 与执行组件使用的编码器一致：普通输入为85维，R9为90维。
    推理时读取既有checkpoint；训练时可以从新网络开始，也可以恢复文件。
    max_steps 供 MAPPO 全局状态编码器归一化时间，seed 用于模型与采样初始化。
    """

    # 统一策略名称，随机掩码实现可直接用于检查动作和经验回传链路。
    policy_name = name.strip().lower().replace("-", "_")
    if policy_name == "random_masked":
        return RandomMaskedPolicy(seed=seed)
    if policy_name not in LEARNING_MOTION_CHOICES:
        valid_names = ", ".join(LEARNING_MOTION_CHOICES)
        raise ValueError(f"Unknown learning motion policy '{name}'. Valid policies: {valid_names}")
    # 神经网络推理需要模型文件；训练允许稍后将新模型保存到指定路径。
    if not training and not model_path:
        raise ValueError(f"model_path is required for learning motion policy '{policy_name}'")
    if model_path and not Path(model_path).is_file() and not training:
        raise FileNotFoundError(f"Learning checkpoint does not exist: {model_path}")

    # 按算法延迟导入，普通规则策略和随机机动无需加载深度学习依赖。
    try:
        if policy_name == "ppo":
            from .ppo_policy import PPOConfig, PPOSharedPolicy

            policy = PPOSharedPolicy(PPOConfig(
                seed=0 if seed is None else seed,
                observation_dim=observation_dim,
            ))
        else:
            from .mappo_policy import MAPPOConfig, MAPPOSharedPolicy

            policy = MAPPOSharedPolicy(
                MAPPOConfig(
                    seed=0 if seed is None else seed,
                    max_steps=max_steps,
                    observation_dim=observation_dim,
                )
            )
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise RuntimeError(
                "PPO/MAPPO requires PyTorch in the Python runtime. "
                "Use random_masked for wiring checks or install a compatible PyTorch build."
            ) from error
        raise

    # 文件同时恢复模型内部维护的优化器和训练计数，随后统一切换运行模式。
    if model_path and Path(model_path).is_file():
        policy.load(model_path)
    policy.set_training(training)
    return policy
