# R7 4.0：搜索覆盖与任务恢复，对应历史实验的 recovery。
# 共用 R7 3.0 的空间波次基础实现，将约半数清道夫用于分带搜索，其余保留掩护。
# 按距离联合分配已发现暗巢，设置任务认领期限，到期释放占用并允许冗余分配。
# 部署分工和任务恢复规则均有变化；原有任务延续和短时避让继续保留。
# 测试版本号用于区分方案，不表示分数排名；内部 variant 保持 recovery。
from user_agents.example.spatial.runtime import SpatialAttack, SpatialDeployment


class Deployment(SpatialDeployment):
    variant = 'recovery'


class RedAttack(SpatialAttack):
    variant = 'recovery'
