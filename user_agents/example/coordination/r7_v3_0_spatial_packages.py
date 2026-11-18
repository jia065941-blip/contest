# R7 3.0：空间波次协同，对应历史实验的 baseline。
# 按公开目标和部署区域安排主攻、掩护与搜索位置，协调预计到达步号。
# 包含主攻损失后的补充分配、通信群重组后的任务延续和短时避让。
# 与 R7 1.0/2.0 使用不同的部署及作战实现；R7 4.0 在此基础上增强搜索和任务恢复。
# 测试版本号用于区分方案，不表示分数排名；内部 variant 保持 baseline。
from user_agents.example.spatial.runtime import SpatialAttack, SpatialDeployment


class Deployment(SpatialDeployment):
    variant = 'baseline'


class RedAttack(SpatialAttack):
    variant = 'baseline'
