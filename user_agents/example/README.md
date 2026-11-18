# 红方部署与单机策略示例

`user_agents/base/`保存基础契约，`red_checker.py`检查选定文件。example包含可直接选择的红方实现：

```text
example/
  deployment.py  独立部署方案生成
  assignment/    R0—R3：随机、优先级与静态分配
  rolling/       R4—R6：周期、事件及压力感知重规划
  coordination/  R7、R8、静态搜索及搜索几何
  learning/      R9入口、训练工具及models模型库
  common/        示例内部的执行、目标和通信工具
```

YAML示例：

```yaml
agent: user_agents/example/assignment/r0_random.py
```

同一文件导出`Deployment`和`RedAttack`。内置入口复用`example/deployment.py`的部署类，各自提供单机策略类。

## 独立部署阶段

`Deployment`继承`BaseDeployment`，采用无参构造。它接收一次部署观测，并返回完整位置映射：

```text
{实体ID: (x, y), ...}
```

部署观测固定包含`step`、`entities`、`public_targets`、`coordinates`和`seed`。`entities`是以实体ID为键的待部署人族清单，每项仅包含`type`，不提供场景原位置；`public_targets`是公开母巢和哨卫巢目录，包含名称、位置、血量、可见标记、类型与阵营，隐藏暗巢不进入目录。`coordinates`是单个共用部署多边形的顶点列表，类型为`list[list[float]]`，格式为`[[x, y], ...]`，单位为米。布局方案只返回水平坐标，高度沿用各实体原始状态。

部署接口要求且仅接受一个区域。场景/SDK的`redArea.coordinates`和环境全态的`deployment_area.coordinates`使用三层区域列表；部署观测在检查区域数为1后提取唯一多边形的顶点列表。

Simulator在环境reset后创建局部部署对象，传入`env.agent_manager.get_deployment_observation()`返回的资料。deploy生成的二维方案交给`manager.apply_deployment()`检查实体覆盖、编号、有限坐标和统一区域，验证原始高度后，通过Engine的set_position保留高度应用位置并统一启动。就绪初态从`env.get_observation()`读取。部署对象在finally中关闭，之后注册和重置策略实例，再开始日志与步进。

示例部署工具支持连续区域随机放置和网格方案。网格方案属于示例自身配置：

```python
from user_agents.example.deployment import Deployment

deployment = Deployment()
deployment.plan_path = "my_grid_plan.json"  # 省略时采用随机布局
positions = deployment.deploy(observation)
```

部署计算发生在`deploy()`调用期间。

## 单实体策略循环

`RedAttack`直接继承唯一策略基类`BaseAgent`，框架使用`(entity_id, init_observation)`创建实例。BaseAgent另提供关键字参数history_length，默认10。每个受控实体对应一个实例，实体ID同时用于部署方案、观测、动作和奖励路由。Manager先调用各自get_obs，再依次调用get_action。

`get_obs()`默认保存本机观测为`latest_observation`。最小策略实现只需返回动作；完整运行文件同时导出部署实现：

```python
import numpy as np
from user_agents.base import BaseAgent
from user_agents.example.deployment import Deployment

class RedAttack(BaseAgent):
    def get_action(self):
        return np.empty((0, 4), dtype=np.float64)
```

需要整理观测时，可覆写get_obs并调用super保存最新观测。基类提供历史记录、on_reset、on_feedback和close默认能力，各实现按需扩展。

策略初始化资料包含部署后的本机self、公开母巢和哨卫目录entities、地图与搜索边界，以及seed、max_steps、options和run_id等纯数据。`run_id`标识一次运行，训练跨局复用同一策略组时保持不变；训练回合状态由示例自身维护。示例可以自行汇总各实例实际收到的数据，并按当前通信连通群组织协作。训练全态由专用runner显式读取。

初始化本机、公开目标目录和逐帧本机状态统一使用`pos: {x, y, z}`与`vel: {x, y, z}`。位置单位为米，速度单位为米/秒；轴顺序为东、北、上。地图`map_area`使用`xMin/xMax/yMin/yMax`。搜索几何按平面距离计算，内部统计距离时可转换为千米。

红方初始化与逐帧观测的`self.detectInfo`均为`dict[int, dict]`，键为被探测实体ID。记录中的`detected_step: int`表示产生这条记录的零起始仿真步号，不是接收或转发时的当前步号；公开记录不含毫秒`time`。

步号由`max(0, floor((内部time - 场景simTime) / simStep))`换算，缓存和转发不改变它。例如当前`step=120`、记录`detected_step=100`，信息年龄为20步。同一步内的探测由引擎先按内部毫秒时间融合；示例允许后续相同步号的权威快照更新位置，完全重复的快照不触发重规划。内部SDK和环境全态仍保留毫秒时间，蓝方时间协议另行定义。

## 加载与生命周期

`check_deployment(path)`检查Deployment继承、无参构造及同步deploy/close；`check_red_agent(path)`检查RedAttack两参构造、get_obs/get_action及生命周期方法。同一文件的两个检查入口复用模块缓存，检查过程返回类。

部署对象只在初始化阶段使用。AgentManager接收Engine和场景配置，提供部署、校验应用与策略注册入口。程序化调用者通过`env.agent_manager.register_user_agent()`取得策略组并负责close。Env维护状态、step及通用观测，关闭时解除注册引用；运行入口持续管理环境、蓝方、红方实例集合和logger四个组件。

reset先清除公共历史和最新观测，再调用on_reset；record_step先记录本机历史，再调用on_feedback。

## 目标与动作

母巢和哨卫巢的初始位置与血量构成公开目录。暗巢由本机探测、通信或天眼信息获知，示例按记录时间更新目标快照。

| 动作 | 格式 | 用途 |
|---|---|---|
| 设置攻击目标 | `[1, 自身ID, x, y]` | 设置地图范围内的目标点 |
| 使用天眼 | `[2, 自身ID, 0, 0]` | 请求场景天眼能力 |
| 设置横向加速度 | `[3, 自身ID, 机动值, 0]` | 调整飞行机动 |

单位动作使用有限数值的`N × 4`数组，第二列是执行动作的己方实体ID。`x` 向东、`y` 向北，单位均为米，坐标引导的目标高度为0米。机动值为无量纲倍率：高空、低空单位按20G换算，中空单位按15G换算，G取9.8米/秒²；0停止机动。未使用的位置必须填0。

示例在内部保留目标ID，用于目标选择、类型适配、情报检查和分配记录；输出时从已有观测取目标平面坐标，攻击、搜索和即时重定向统一使用动作1。天眼和机动分别使用动作2、3。旧动作4、5不再属于公开协议；学习模型内部的离散分类0/1/2保持不变。

Manager检查控制实体归属，引擎检查目标点地图范围并控制天眼额度。引擎内部的目标ID解析能力继续保留，公开策略动作不再提交目标ID。

## 分类策略与内部工具

| 入口 | 行为 |
|---|---|
| `assignment/r0_random.py` | 对适合本单位的已知目标进行可复现随机分配 |
| `assignment/r1_priority.py` | 综合价值、距离、攻击能力和分配数量更新任务 |
| `assignment/r2_static_assignment.py` | 一次容量约束分配，小规模精确求解、大规模贪心 |
| `assignment/r3_static_priority.py` | 一次优先级分配后保持任务 |
| `rolling/r4_rolling_rules.py` | 定期重规划并限制更新数量 |
| `rolling/r5_event_rolling.py` | 合法状态变化事件触发冷却后的重规划 |
| `rolling/r6_pressure_assignment.py` | 根据本群威胁压力调整任务和重规划数量 |
| `coordination/r7_v1_0_strike_packages.py` | R7 1.0：基础协同编组，为毁灭者和突击者安排同目标攻击 |
| `coordination/r7_v2_0_static_search.py` | R7 2.0：在1.0上增加清道夫三机静态搜索 |
| `coordination/r7_v3_0_spatial_packages.py` | R7 3.0：空间波次部署、掩护搜索分工、任务延续和短时避让，对应 baseline |
| `coordination/r7_v4_0_search_recovery.py` | R7 4.0：在3.0上调整搜索部署，增加按距离分配、认领到期释放和冗余分配，对应 recovery |
| `coordination/r8_satellite_packages.py` | 为R7攻击组安排天眼 |
| `learning/r9_hierarchical_learning.py` | R8任务层提供上下文，学习机动层执行动作 |

common/execution.py复用单机执行，targets.py整理公开初值、价值与已接收轨迹，commander.py和coordination.py组织群内任务。R7搜索算法与几何位于coordination/search.py和coverage.py，配置及私有资源工具由各示例自行调用。

### R7版本与配对测试

1.0与2.0共用公共部署和单机执行代码，主要差别是三机静态搜索。3.0与4.0共用空间波次实现，4.0调整清道夫的搜索与掩护分工，并增加任务恢复规则。2.0到3.0同时涉及部署、任务组织和机动实现，比较时须记录这些变化。编号用于区分测试方案，不表示分数排名。

四个入口均保留精简规划快照优化。内部`policy_name`和`variant`保持原值，避免改变策略配置或历史结果解释；原始录像源码和哈希文件不变。

在工程根目录的`config.yaml`中设置对应入口，例如：

```yaml
agent: user_agents/example/coordination/r7_v1_0_strike_packages.py
```

使用`python main.py --config config.yaml`运行。对比四个版本时固定同一引擎提交、地图、蓝方和种子，分别设置输出目录；部署与作战都由所选文件导出。

## 学习与训练

`learning/models/`提供随机掩码、PPO和MAPPO机动策略，动作包括向左、向右和停止横向机动。普通Actor输入85维，R9输入90维并附加本通信群任务上下文。R9入口默认使用random_masked，PPO/MAPPO由具体配置选择模型与checkpoint。

编号特征使用示例资源内部登记的成员序号，实体ID仅表示身份。输入维数不变，既有checkpoint需要在新的编号特征下重新验证。

`learning/train.py`中的可信runner每局先reset，再经AgentManager取得部署观测和应用方案，从Env读取初态。首局通过Manager注册实例并reset一次，之后回合由Env.reset重置现有实例。训练循环显式读取全态、设置Critic上下文、提交经验并更新保存模型。策略推理使用单机观测，模型和队列属于示例内部工具。

训练采用程序化入口，配置直接使用本示例的 `RedAgentConfig`：

```python
from user_agents.example.common.config import RedAgentConfig
from user_agents.example.learning.train import run_training

run_training(
    "scenarios/cases/launch20/easy/E01/scenario.json",
    RedAgentConfig(
        policy_name="r9_hierarchical_learning",
        motion_policy="ppo",
        seed=7,
        model_path="results/checkpoints/r9_ppo.pt",
        training=True,
    ),
    rounds=5,
    output_dir="results/training",
    max_steps=300,
)
```

`max_steps`仅用于训练实验截断，省略时沿用场景时限。`blue_mode`可指定蓝方实现文件，`plan_path`可指定示例网格方案；两者由训练调用方提供。训练对象跨局复用，随机源延续；本次输出目录由函数返回，结束时保存checkpoint并关闭日志与过程写入器。

学习输入的位置和相对位移按地图范围归一化。
