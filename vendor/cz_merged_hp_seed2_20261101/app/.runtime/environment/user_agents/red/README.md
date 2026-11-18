# 红方任务分配与执行

本目录包含红方部署、通信群内的任务分配和单个作战单位的机动策略。蓝方防御由 `blue_defense/` 管理。场景加载、公开搜索区域和部署约束位于 `scenarios/`。

## 运行过程

`main.py` 将示例配置和开局公开观测交给 `RedAgentFactory`，注册它创建的部署智能体及各单位的 `AttackMissileAgent`。任务分配、机动选择和通信群内协作均在 `user_agents` 内完成；主入口和环境只负责配置接线、系统状态、观测分发、动作校验与仿真推进。部署完成后，环境统一启动所有红方单位。单位在首个决策步即可选择目标或搜索航点，也可以同时执行横向机动。

每个决策步，引擎先更新通信连通群并融合合法侦察信息。AgentManager 先向所有存活单位分发各自的隔离观测，再依次调用 `get_action()`。红方示例利用观测中的 `commRangeInfo`，在用户实现内部收集本群本帧报告，为每个通信群维护独立的 `RedBaselineCommander`。群成员不变时沿用该群的策略状态；群合并或拆分时重新建立任务分配器，用新群成员各自已收到的侦察记录恢复可用目标。其他通信群的位置、目标发现和威胁信息不会进入本群的规划。

用户实现中的 `begin_step()` 只接收本群当前存活成员的观测。随后，各单位通过本群协作接口取得自身任务，再根据本机观测生成机动动作。通信群变化不会重置单位的出发时间、机动状态、天眼使用记录或学习权重。每局开始时重置本局历史和通信群状态。

本群压力只由合法收到的自爆虫群记录与当前成员数量计算。缺少某个成员的报告时，该成员退出本次规划；任务层不会据此认定它已被摧毁，也不会增加威胁压力。

## 单机接口与示例装配

`AttackMissileAgent(agent_id, entity_id, init_observation)` 保留单机接口，只接收单位身份和开局公开观测。Agent 自己接收 `set_observation()`、进行合法群内协作并通过 `get_action()` 返回动作。构造参数不再暴露 Commander、机动模式或共享学习策略；环境也不替 Agent 预计算动作。

`RedAgentConfig` 保存任务策略、机动策略、种子、回合时限、模型路径和训练开关。`RedAgentFactory(config).create_agents(public_observation, platforms, search_polygon=...)` 在用户包内完成示例装配并返回 Agent 列表，不接收环境对象。部署智能体同样由工厂创建。现有命令行参数由 `user_agents/red/config.py` 注册和校验。

## 目标与搜索航点

开局公开目标由 `TrainingEnv.get_public_targets_observation()` 提供，只包括母巢和哨卫巢。`initial_targets_from_observation()` 再次过滤目标类型。暗巢需要先出现在本机合法接收的 `detectInfo` 中。

`ReceivedTrackFusion` 保存本群已收到的目标快照，并按观测时间接收更新。旧快照和同一时刻的重复快照不会覆盖新记录。侦察记录没有目标血量，任务层不据此确认毁伤或摧毁。

当前任务策略为毁灭者、突击者分配母巢或哨卫巢，为清道夫分配已发现的暗巢。暂时缺少适合目标的单位会获得公开搜索区域中的航点。

动作数组保持 `N × 4`：

| 动作 | 格式 | 用途 |
|---|---|---|
| 横向机动 | `[0, 单位ID, 机动值, 0]` | 调整飞行机动 |
| 搜索航点 | `[2, 单位ID, x, y]` | 前往公开区域内的搜索位置 |
| 使用天眼 | `[3, 单位ID, 0, 0]` | 申请场景内的天眼能力 |
| 攻击目标 | `[4, 单位ID, 目标ID, 0]` | 攻击本机合法情报中的目标 |

攻击指令由引擎根据执行单位的合法目标快照解析位置，并在执行前校验。未经本机观测或通信获得的暗巢不能作为攻击目标。搜索航点与目标 ID 使用不同动作，任务层不再通过坐标动作传递目标位置。天眼的全局使用次数由环境控制。

## 当前任务策略

所有策略都使用同一套自动出发与通信约束。策略差异体现在目标分配、重规划和搜索方式。

| 策略 | 行为 |
|---|---|
| `r0_random` | 在适合本单位的已知目标中进行可复现的随机分配。 |
| `r1_priority` | 综合目标价值、距离、攻击能力和分配数量，定期更新任务。 |
| `r2_static_assignment` | 计算一次容量约束下的静态分配；小规模问题精确求解，较大问题采用确定性贪心。 |
| `r3_static_priority` | 缓存一次优先级贪心分配，后续保持该分配。 |
| `r4_rolling_rules` | 定期重规划，每次更新有限比例的单位。 |
| `r5_event_rolling` | 收到目标位置变化等事件后，在冷却间隔结束时重规划。 |
| `r6_pressure_assignment` | 按本群已观测威胁调整重规划数量；初期或高压力下优先处理清道夫任务。 |
| `r7_strike_packages` | 在压力感知的重规划中，为毁灭者和突击者安排同目标配对。 |
| `r7_static_search` | 用公开区域和哨卫巢位置规划清道夫三机搜索路线，发现暗巢后允许转换为攻击任务。其他单位继续执行 R7 分配。 |
| `r8_satellite_packages` | 在 R7 基础上，为同目标攻击组选择毁灭者申请天眼。 |
| `r9_hierarchical_learning` | 使用 R8 任务层，并向共享学习机动策略提供本通信群的任务上下文。 |

首轮分配会覆盖所有存活单位。后续的重规划数量只决定更新哪些单位的任务。搜索编组的路线规划以近距离队形为目标，实际通信关系始终由引擎根据当前位置判定。

## 学习机动层

`learning/` 提供随机掩码策略、PPO 和 MAPPO。Actor 的动作只有向左、向右和停止横向机动。目标分配仍由通信群内的任务层负责。

普通 Actor 保持 85 维观测；R9 使用 90 维观测，附加已分配目标类型、本通信群压力和本机出发状态。两个维度的模型需要分别训练和保存。MAPPO 的全局状态只在显式训练时供 Critic 使用，部署推理仅使用单机合法观测。

比赛运行的主循环不调用学习策略生命周期。`main.py --red-learning-train` 为兼容现有命令行，明确转入 `user_agents/red/train.py` 的 `run_training()`。该可信训练入口在每轮环境重置前重置训练会话；每步读取 `env.get_observation()` 的系统全局快照，设置训练上下文，再执行 `env.step()`，最后补齐全局下一状态并按原 Agent 顺序提交本地经验。Agent 接口只接收本机观测、调用本地推理并提交本地经验；全局快照由训练入口显式设置和消费。这是内置示例的调用约定，同进程 Python 对象并不构成不可信代码的安全沙箱。

这次职责调整保留原有随机调用顺序、PPO/MAPPO 更新时序、85/90 维编码和 checkpoint 格式，不改变训练算法或 R9 任务层行为。

运行示例：

```powershell
python main.py --scenario launch20/easy/E01 --red-policy r7_static_search --red-motion-policy reactive_evasion --blue-policy b9_min_cost_planned_multi_wave --render-mode none
```

训练示例（需先安装兼容的 PyTorch）：

```powershell
python main.py --scenario launch20/easy/E01 --red-policy r9_hierarchical_learning --red-motion-policy mappo --red-learning-train --red-learning-model results/models/r9_mappo.pt --render-mode none
```

PPO/MAPPO 推理需要通过 `--red-learning-model` 指定模型。新增策略应接入 `build_red_baseline()`，并补充目标合法性、通信域隔离和回合重置测试。
