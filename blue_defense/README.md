# 蓝方策略与探测信息

蓝方实现按决策和计划的执行机制归组：

```text
blue_defense/
  base/          公共接口、观测与决策类型、算法基类
  shared/        通用配置、控制器、信息库和策略注册表
  immediate/     B0—B3：即时目标分配
  coordination/  B4—B6：联合时序规划与协同
  scheduled/     B7—B9：策略内部延迟与波次计划
```

每个策略文件提供固定入口类 `BlueDefense`，继承 `blue_defense.base.BaseBlueDefense`。构造参数为随机种子 `seed` 和保护对象价值映射 `asset_values`：

```python
from blue_defense.immediate.b0_fixed_ratio_random import BlueDefense

defense = BlueDefense(seed=42, asset_values={101: 5.0})
decision = defense.decide(observation)
defense.reset()
defense.close()
```

`decide()` 接收 `BlueObservation`。控制节点按场景起点和步长创建并复用只读 `clock` 元数据，步号从0开始，表示观测时刻而非策略调用次数。

| 字段 | 含义 |
|---|---|
| `step` | 当前全局仿真步号 |
| `targets` | 自上次 `decide()` 后收到的原始探测记录，位置和速度属于记录的探测时刻；没有新探测时为空。记录可以包含同一目标的多个来源或时刻 |
| `interceptors` | 当前己方自爆虫状态与可用性 |
| `own_entities` | 己方实体的全量状态快照：编号、名称、类型、位置、速度、血量和可见状态；由策略自行选择保护对象、定义价值 |
| `launched_map` | 各目标已释放或被当前保留计划预留的自爆虫编号，只读映射 |
| `clock` | 场景时间起点和步长 |

蓝方的探测记录、记忆期限、决策间隔和计划调度统一使用步号或步数，允许计划保留小数步。位置统一使用 `pos`，速度统一使用 `vel`；坐标轴为 x 向东、y 向北、z 向上，位置单位为米，速度单位为米/秒。物理计算通过 `clock.steps_to_seconds()`、`clock.seconds_to_steps()` 换算，不把模型内部50ms积分步长作为全局步长。计划位置使用 `launch_pos` 和 `estimated_intercept_pos`。

正式仿真在注册和重置蓝方时调用 `bind_clock()`，从场景读取固定步长；选手无需另传时间。独立构造观测或信息库时应显式提供真实的 `BlueClock`，例如 `BlueClock(step_duration_ms=1000)`；默认1毫秒时钟仅供独立调用。当前 `Rtest` 每个全局步为1秒。

| 决策结果 | 环境处理 |
|---|---|
| `BlueDecision(assignments=...)` | 执行即时指派或登记计划释放，记录求解统计 |
| `BlueDecision()` | 本次评估完成，产生空指派 |
| `None` | 等待下一次决策时间，保留现有计划 |

`reset()` 清理单局状态，`close()` 释放策略资源。B0默认在多局之间延续随机流，`reset_rng_on_episode` 控制每局重新初始化随机流。

即时动作使用 `InterceptorAssignment(interceptor_id, target_id)`。计划动作使用 `ScheduledInterceptorAssignment`，通过 `decision_step`、`planned_launch_step`、`estimated_intercept_step` 表示全局步号，额外携带可选协同信息。内部延迟计划使用 `due_step`。执行时达到计划步才释放，不提前取整。公共动作不包含 `target_pos`、`target_vel` 或 `priority`；评分和排序由策略内部完成。

探测到达时，环境即时调用 `receive_tracks(tracks, sim_step)`，随后执行和制导通过 `get_track(target_id, sim_step)`、平台避让通过 `get_tracks(sim_step)` 查询蓝方维护的状态。这些参数均为全局步号。基类默认保存各目标最新的原始记录，不强制有效期或外推；自定义蓝方可覆盖接收与查询方法。覆写 `reset()` 时应调用 `super().reset()` 清理基类记忆。

环境仅允许查询曾由己方视野塔探测过的目标编号，并检查返回位置和速度为有限数值。它不从敌方实体实时状态补齐目标信息。记忆失效时查询返回 `None` 或不包含该目标：相应待释放计划取消，已释放自爆虫保留原生模型最后设置的引导状态，平台在缺少适用目标时返回初始位置。

自定义文件可以直接继承公共基类：

```python
from blue_defense.base import BaseBlueDefense, BlueDecision

class BlueDefense(BaseBlueDefense):
    name = "custom"

    def decide(self, observation):
        return BlueDecision()
```

YAML 的 `blue_mode` 指向策略 Python 文件，例如 `blue_defense/immediate/b0_fixed_ratio_random.py`。`blue_defense.blue_checker.check_blue_policy(path)` 加载并检查入口类；Simulator 使用共享seed和 `ScenarioConfig.asset_values` 的价值副本创建实例，再调用 `env.register_blue_policy(instance)`。每个运行环境对应一个蓝方实例和一个控制节点，由环境重置策略，由Simulator关闭策略。

`ScenarioConfig` 集中保存Profile、部署约束和评分规则，由 `scenarios.load_scenario()` 加载。蓝方算法的释放配额、决策间隔及波次参数由 `shared/config.py` 管理。

内置 B 的每个 `BlueDefenseController` 持有独立的 `BlueInformationStore`，重置时清空。控制器整理保护对象 `assets`，采用600步目标记忆和匀速外推，再将算法观测交给 B0—B9。首轮在第0步评估，后续间隔配置使用 `*_step`；`wave_gap_step=None` 时按原B8十秒、其他算法五秒的默认波次时长换算为步。显式指定的波次间隔直接按步使用。绑定时钟不会重建策略或重置B0随机流。

| 文件与接口 | 职责 |
|---|---|
| `shared/information.py`：`receive_tracks()` | 检查记录身份、时间与有限坐标，合并全部视野塔来源，保留各目标的最新记录 |
| `shared/information.py`：`get_track()`、`snapshot()` | 根据记录位置、速度和场景步长预测当前位置，清除超过600步的记录，返回观测快照 |
| `shared/information.py`：`update_own_state()` | 保存内置 B 整理的保护对象、自爆虫状态和已经释放或预留的清单 |
| `shared/config.py`：`BlueDefenseConfig` | 配置策略、随机种子、决策间隔及波次参数 |
| `shared/controller.py`：`BlueDefenseController.decide()` | 整理目标记忆与保护对象，控制决策间隔，将内部观测交给配置的算法 |
| `base/policy.py`：`ScheduledBluePolicy` | 推进B7—B9的延迟释放计划，输出到期的单位和目标编号 |

## 运行过程

1. `RadarModelSimulator.execute_detection()` 按现有20公里范围及飞行阶段条件生成探测记录，坐标与速度保存为独立快照。
2. `DefendCommanderModelSimulator.handel_detect_info()` 校验探测来源与数值，将原始记录即时交给蓝方，并收集到下一次决策输入中。
3. 控制节点推进时更新己方状态，将原始观测交给配置的蓝方策略，策略按步数控制决策间隔。即使策略返回 `None`，本批探测也已交付并消费。
4. 到期计划与即时动作都向蓝方查询目标状态，用同一份位置、速度进行距离和角度检查并生成底层发射命令。目标记忆失效、己方自爆虫已经释放或被摧毁时，取消对应计划。
5. `InterceptorSimulator.set_target_ecf()` 继续查询蓝方目标状态；`GeneralShipSimulator.detect_attacker()` 和 `InterceptFieldSimulator.detect_attacker()` 查询蓝方维护的目标，选择距离自身20公里以内的最近者进行避让。

内置 B 的目标记录自最后探测步起保留600步；新记录更新位置和速度，查询不会延长有效期。这项参数属于 B 的实现，公共蓝方接口不要求其他策略采用相同规则。

## 本地验证

保护价值的类型缺省值采用母巢5、哨卫巢2、暗巢1，按实体编号传入的 `asset_values` 仍优先使用。B4—B6的基础伤害估计采用毁灭者15、突击者5、清道夫对暗巢1；这些估计与当前引擎基础规则一致，天眼和夹击加成由引擎结算。

B0—B9的决策间隔、延迟、波次、单目标配额、记忆期限和合法信息来源保持各自原有规则。

原生模型使用CPython 3.11 x64。`test_blue_information.py` 加载E01场景并通过真实视野塔生成探测记录；`test_blue_defense_integration.py` 验证决策间隔、计划释放、随机流和释放检查：

```bash
python -m unittest discover -s tests -p test_blue_policies.py -v
python -m unittest discover -s tests -p test_blue_defense_integration.py -v
```
