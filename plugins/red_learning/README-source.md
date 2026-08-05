# 红方 MAPPO 与毁伤链改动记录

## 1. 改动范围

本轮改动在已有红方 PPO 基线之上，主要完成以下工作：

1. 增加共享参数的 MAPPO 策略，实现集中训练、分散执行（CTDE）。
2. 扩充红方局部观测，并增加训练阶段使用的全局状态编码。
3. 重构训练环境的奖励、卫星使用和多智能体轨迹收集流程。
4. 完善模型评估指标和 WandB 训练记录。
5. 修复巡航导弹近炸、命中和毁伤命令链路。
6. 增加学习策略和毁伤链自动化测试。

训练产生的权重、评估 JSON、WandB 缓存及仿真结果仍由 `.gitignore` 排除，不进入 Git 历史。

## 2. MAPPO 网络

实现文件：`user_agents/learning/mappo_policy.py`

当前红方智能体共享同一套 Actor 和 Critic 参数：

| 网络 | 输入 | 结构 | 输出 |
| --- | --- | --- | --- |
| Actor | 85 维局部观测 | `85 -> 128 -> Tanh -> 128 -> Tanh -> 3` | 左转、停止机动、右转三个动作的 logits |
| Critic | 70 维全局状态与 85 维当前智能体局部观测 | `155 -> 256 -> Tanh -> 256 -> Tanh -> 1` | 当前智能体的状态价值 |

训练采用 PPO clipped objective、GAE、价值损失、熵奖励和梯度裁剪。轨迹按照 `agent_id` 分开计算 GAE，再统一更新共享网络。评估时 Actor 使用确定性的最大概率动作，Critic 不参与部署决策。

### 2.1 局部观测

局部观测编码位于 `user_agents/learning/red_policy.py`，共 85 维：

- 21 维自身状态：时间、位置、高度、血量、型号、发射状态、卫星状态、机动状态、速度、航向和角色等。
- 60 维目标状态：最多 5 个目标，每个目标 12 维，包括相对位置、距离、方位、探测状态、目标类型和当前分配标志。
- 4 维探测与通信统计。

### 2.2 全局状态

集中式 Critic 使用 70 维聚合全局状态：

- 当前时间。
- 三类红方导弹的存活率、平均血量、平均位置等。
- 五类蓝方单位的数量、存活率和平均血量。
- 最多五个主要蓝方目标的位置、血量和存活状态。
- 红方编队速度、空间离散程度和目标距离统计。

### 2.3 当前策略边界

Actor 当前只学习巡航导弹的横向机动。下列行为仍由规则控制：

- 目标按照智能体编号轮询分配。
- 导弹在首次学习决策时自动发射。
- 卫星由环境在指定仿真步统一触发。
- 初始部署由带随机种子的部署规则生成。

因此当前基线主要学习“如何机动接近已分配目标”，尚未学习目标选择、火力分配、发射时机和卫星调用时机。

## 3. 环境与奖励

主要修改文件：`envengine/environment/training_env.py`

- 在环境步开始和结束时向共享策略提供完整态势，用于构造当前和下一时刻的全局状态。
- 支持每个红方智能体分别提交 transition，并由共享 MAPPO 策略统一更新。
- 增加目标距离进展、距离里程碑、目标受损、目标击毁等稠密奖励。
- 增加红方受损、红方全灭、超时和步进成本等惩罚项。
- 增加奖励课程机制，使早期训练更强调接近目标，后期提高实际毁伤权重。
- 增加团队级卫星触发和使用统计。
- 结束条件支持蓝方主要目标全部被毁或红方全部失效。

需要注意：当前奖励仍包含红方受损及全灭惩罚。如果最终目标只关注蓝方目标是否被击毁，后续应继续降低这些惩罚的权重，避免训练目标偏离任务成功条件。

## 4. 毁伤链修复

新增公共实现：`envengine/simulator/simlulator_impl/cruise_missile_damage.py`

涉及三类巡航导弹模拟器：

- `CompCruiseMissileHSimulator.py`
- `CompCruiseMissileMSimulator.py`
- `CompCruiseMissileLSimulator.py`

主要改动：

- 将三类导弹重复的命中和毁伤逻辑抽取为公共 mixin。
- 增加约 1000 米范围的近炸/命中判断，避免导弹接近目标后因离散仿真步跨越而无法触发毁伤。
- 修复毁伤命令字段构造与序列化问题，使 `DamageC` 能够正确传入目标模拟器。
- 在模拟器工厂中接入新的公共毁伤实现。

## 5. 训练、评估与记录

### 5.1 训练

`main.py` 增加 `mappo` 策略选择、GAE 参数、共享策略注册、模型保存和 WandB 指标上传：

```powershell
python main.py --red-policy mappo --total-rounds 10 --max-steps 1000 `
  --render-mode none --policy-model results/red_mappo.pt `
  --wandb --wandb-mode online
```

### 5.2 评估

`evaluate_red_policy.py` 支持从 checkpoint 中恢复 MAPPO 配置，并记录：

- 回合奖励和奖励分项。
- 蓝方目标毁伤及击毁情况。
- 不同型号红方导弹的损失。
- 每枚导弹与目标的最小距离及距离区间命中情况。
- 动作分布、目标分配和卫星使用情况。

基本评估命令：

```powershell
python evaluate_red_policy.py --policy mappo --model results/red_mappo.pt `
  --episodes 10 --max-steps 1000 --output results/evaluation.json
```

### 5.3 可复现性和实验记录

- `user_agents/deploy_agent.py` 支持固定随机种子，减少训练和评估之间的部署差异。
- `docs/training_experiments.md` 用于记录每次训练的超参数、checkpoint 和评估结果。
- `docs/red_learning_baseline.md` 补充 MAPPO 结构、训练及评估说明。

## 6. 测试

- `tests/test_red_learning_policy.py`：覆盖局部观测、全局状态、动作映射、MAPPO transition 和网络更新。
- `tests/test_damage_chain.py`：覆盖近炸触发、毁伤命令格式及三种巡航导弹的公共毁伤链。

## 7. 主要文件清单

| 文件 | 类型 | 主要内容 |
| --- | --- | --- |
| `user_agents/learning/mappo_policy.py` | 新增 | MAPPO Actor、集中式 Critic、经验缓存、GAE 和 PPO 更新 |
| `user_agents/learning/red_policy.py` | 修改 | 85 维局部观测、70 维全局状态和三动作定义 |
| `user_agents/attack_missile_agent.py` | 修改 | 固定目标分配、自动发射、动作执行和 transition 收集 |
| `envengine/environment/training_env.py` | 修改 | MAPPO 环境钩子、奖励、卫星和结束条件 |
| `envengine/simulator/simlulator_impl/cruise_missile_damage.py` | 新增 | 统一近炸和毁伤命令逻辑 |
| `evaluate_red_policy.py` | 修改 | MAPPO checkpoint 加载和详细评估统计 |
| `main.py` | 修改 | MAPPO 训练入口、WandB 和模型保存 |
| `user_agents/deploy_agent.py` | 修改 | 可复现的随机部署 |
| `tests/test_red_learning_policy.py` | 修改 | 学习策略测试 |
| `tests/test_damage_chain.py` | 新增 | 毁伤链测试 |
| `docs/training_experiments.md` | 新增 | 训练参数和性能对比记录 |
