# Competition Platform

红蓝对抗仿真平台。红方负责部署、目标选择和机动，AgentManager管理部署与策略接入；蓝方策略由配置中的Python文件提供，Env封装系统状态、观测和步进。

## `final` 分支：CZ + V12-R3 PPO 新版接口复现包

本分支以远端 `cz` 提交 `683e0d9570afebdcc22d6d508a43627b0cff3c7e` 为父提交，发布迁移到最新版公开接口后的完整可运行快照。算法结构保持为：A00--A11 初始方案、冻结选择、CZ Skill/决策树、V12-R3 PPO 底层机动；改动集中在笛卡尔坐标、逐实体观测与公开动作 1/2/3 的接口适配，以及只读赛后归因日志。

关键目录：

| 路径 | 内容 |
|---|---|
| `user_agents/cz_ppo/` | 新版接口适配器；不改变冻结上层规划、Skill和PPO决策语义 |
| `vendor/cz_merged_hp_seed2_20261101/` | 可移植最小依赖闭包，含规划器、Skill、旧Commander、V12-R3运行时与模型 |
| `evaluation/attribution.py` | 仿真外侧归因器，不向红方策略回灌引擎真值 |
| `results/cz_ppo_seed2_final_all20_attribution_20261118/` | seed2全20图终局结果、逐目标归因和控制台记录 |
| `CZ_PPO_20MAP_EVALUATION_REPORT_ZH.md` | 新版20图评估与后续优化建议 |

### 先验证下载完整性

克隆后无需安装仿真依赖即可验证源码、冻结模型和结果闭包：

```bash
git clone --branch final --single-branch https://github.com/jia065941-blip/contest.git
cd contest
python3 tools/verify_final_release.py
```

验证器会检查20/20地图覆盖、每图结果文件、得分重构，以及结果协议中记录的引擎、接口、规划器、Skill和PPO模型SHA-256。发布结果的20图算术均分为 `91.54934123847167`。由于按要求保留官方对象地址排序，同seed不是位级确定性协议；新重放应验证协议与统计结果，不应要求每图逐分相同。

### Docker一键重跑20图

宿主机需要Docker和足够的CPU时间。模型、CZ运行代码和场景均已在仓库内，Python依赖由镜像构建阶段安装；所有输出写到仓库下的 `reproduced_results/`：

```bash
WORKERS=20 sh tools/reproduce_seed2_all20.sh
```

完整20图在20并发CPU核上约需2小时。可先做单图验证：

```bash
docker build -t contest-final-repro .
docker run --rm -v "$PWD/reproduced_results:/results" contest-final-repro \
  python -B tools/run_cz_ppo_seed2_20.py \
  --out /results/e01_smoke --workers 1 --maps E01
```

本机已有CPython 3.11、GLIBC 2.38以上、PyTorch CPU和项目依赖时，也可直接执行：

```bash
CZ_RELEASE="$PWD/vendor/cz_merged_hp_seed2_20261101" \
  python3 -B tools/run_cz_ppo_seed2_20.py \
  --out "$PWD/reproduced_results/final20" --workers 20
```

### 在CZ基础上继续优化

- 上层方案入口：`vendor/cz_merged_hp_seed2_20261101/app/design/planner_v2.py`。
- 决策树与Skill：`vendor/cz_merged_hp_seed2_20261101/app/skills/`。
- PPO推理与模型：`vendor/cz_merged_hp_seed2_20261101/app/r3/`。
- 新版接口边界：`user_agents/cz_ppo/policy.py`和`compat.py`。
- 归因目标级证据：每张图的`attribution.json`；汇总见`attribution_summary.json`。

建议新实验复制一个新的vendor版本目录并通过`CZ_RELEASE`指向它，保留本发布目录和哈希不变，以便与91.549基线做受控比较。

## 环境准备

原生模型使用 CPython 3.11 x64。在对应环境中安装项目依赖：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux 原生模型依赖 Python 3.11 动态库、随项目提供的 Utils 动态库，以及 GLIBC 2.38、GLIBCXX 3.4.32、CXXABI 1.3.13 对应的运行库。

## 运行配置

`main.py` 接收 YAML 配置路径，并调用 `Simulator.run()`：

```shell
python main.py --config config.yaml
```

配置示例见 [config.yaml](config.yaml)。

| 配置项 | 用途 |
|---|---|
| `agent` | 红方Python文件路径，导出`Deployment`和`RedAttack`类 |
| `scenario` | 场景 JSON 文件路径 |
| `blue_mode` | 蓝方 Python 文件路径，固定导出 `BlueDefense` 类 |
| `seed` | 共享随机种子，null 时生成 |
| `render_mode` | human 打开窗口，none 关闭窗口 |
| `output_dir` | 运行输出目录 |
| `logging` | 过程记录开关、态势采样间隔和可选示例结果 |

相对路径以 YAML 文件所在目录为基准。日志采样间隔控制文件记录频率，仿真步长由场景定义。

场景初始化、红蓝策略注册和独立部署流程已接通。

## 输出通道

运行输出分为三路：

| 通道 | 用途 |
|---|---|
| 控制台 logger | 运行进度与诊断信息 |
| SDK writer | 配置、态势、事件和动作的过程文件 |
| `example_evaluation` | 可选的示例评分与结果统计 |

控制台复用`LogManager`，显示运行开始、结束和错误；内部模块默认使用`ERROR`级别。SDK writer复用环境已有的`write_*`调用，`logging.enabled`控制详细过程文件，`events`、`actions`和`state_interval_steps`分别控制事件、动作及态势采样，关闭writer自身的逐批提示。文件直接写入`output_dir`，不另建时间戳子目录。

`logging.example_evaluation`独立控制第三路，默认`false`；启用后在正常运行结束时，向`output_dir/example_evaluation.json`写入一次结果。该通道只输出示例结果，不参与仿真结束判定，也不向Agent提供输入。

## 离线回放视频

[独立回放工具](tools/replay_video/README.md)读取SDK过程日志生成1080p MP4，提供实时得分、兵力统计、首次发现与摧毁效果。工具使用单独的依赖文件，不调用在线可视化或仿真引擎。详细日志启用时，首次获知暗巢、视野塔首次获知毁灭者会沿现有事件通道记录21001事件，保留采样间隔内的发现时刻。

## 场景检查与预览

每张地图使用一个场景 JSON，可附带 `scenarios/images/` 中的同名预览图。地图范围、部署区域、实体位置、血量、数量和时限统一读取 `imagineProfile`；可选根级 `metadata` 保存编号、标题、难度和说明。场景字段与编辑步骤见 [JSON 格式说明](scenarios/FORMAT.md)。统一加载入口返回包含模型、部署约束和评分规则的 `ScenarioConfig`：

```python
from scenarios import load_scenario

scene = load_scenario("scenarios/E01.json")
```

自定义场景可复制现有 JSON 后直接编辑，在 YAML 的 `scenario` 中填写文件路径即可。实体使用唯一位置字段 `pos` 和速度字段 `vel`，均为三维 x/y/z；x 向东、y 向北、z 向上，位置单位为米，速度单位为米/秒。地图边界与部署区域使用同一平面坐标。

导出组织方预览：

```shell
python previewer.py scenarios/E01.json --output results/preview/E01
```

输出包括二维图片 `preview.png` 和初态数据 `state.json`。状态包含实体位置、血量、速度、父子关系、挂载数量及已知探测与通信范围；图片显示行动单位和感知节点，包括隐藏蓝方。控制节点保留在状态数据中。

`Previewer(path).preview()` 通过 `load_scenario(path, include_rules=False)` 取得初态配置并跳过计分规则装配，创建 `TrainingEnv(scene)`，将 `reset()` 返回的状态交给 `Renderer.render()`，最后关闭环境并返回 PNG 字节和状态字典。图片使用英文图例和实体编号，状态保留原始名称。

正式场景位于 `scenarios/`，共20张：简单6张、中等6张、困难8张；另提供 R01 和 Rtest。`manifest.json` 保留地图目录和发布文件哈希，清单格式版本为2，版本为20261103，坐标版本为 `local_xyz_m_20261103`。当前 YAML 示例使用 Rtest。全部场景的仿真步长为1000毫秒，时限为3000步。资源用途见 [场景说明](scenarios/cases/README.md)。运行 `python -m scenarios.generate --redraw` 可统一校验、重新导出场景并更新地图图片及校验值。

## 蓝方策略

蓝方代码按职责归组：

```text
blue_defense/
  base/          公共基类、观测和决策类型
  shared/        配置、控制器、信息库和策略索引
  immediate/     B0—B3：即时分配
  coordination/  B4—B6：联合规划与协同
  scheduled/     B7—B9：内部延迟与波次调度
```

每个策略文件提供 `BlueDefense` 类，继承 `BaseBlueDefense`。Simulator 通过 `blue_defense.blue_checker` 加载并检查类，传入共享种子和 `scene.asset_values` 的价值副本，然后交给环境注册和调度。场景加载器完成配置组装，Simulator在初始化阶段传入各组件，环境负责系统状态和推进。

蓝方统一接收 `BlueObservation`，包含本批原始探测 `targets`、己方实体状态 `own_entities`、自爆虫状态及已释放清单。哨卫巢的感知判定采用20公里三维空间距离，探测存活、可见且飞行阶段 `stage >= 3` 的红方实体。目标记忆、预测、保护对象选择与价值由蓝方管理。

即时动作仅提交 `InterceptorAssignment(interceptor_id, target_id)`，执行端向蓝方查询目标状态。计划动作额外携带执行时间等计划信息；动作不再要求目标位置、速度或优先级。即时动作按输出顺序执行，同一计划时刻的动作按登记顺序执行。

蓝方观测、探测记录、决策间隔和计划执行统一使用全局步号或步数。只读 `clock` 从场景读取每步时长，物理预测通过 `steps_to_seconds()`、`seconds_to_steps()` 换算；计划允许小数步，达到计划步才执行。详见 [蓝方接口说明](blue_defense/README.md)。

接口、算法分组和执行流程见 [蓝方说明](blue_defense/README.md)。

## 用户智能体

YAML的`agent`指向红方Python文件，例如`user_agents/example/assignment/r0_random.py`。同一文件提供两个入口：独立部署类`Deployment`继承`BaseDeployment`，单机策略类`RedAttack`继承`BaseAgent`。`user_agents.red_checker`分别检查并返回这两个类，文件加载缓存共用。

Simulator重置环境后，局部创建无参Deployment对象，传入`env.agent_manager.get_deployment_observation(seed=...)`提供的五字段观测：

| 字段 | 内容 |
|---|---|
| `step` | 部署阶段的步数，为0 |
| `entities` | 待部署人族单位，以实体ID为键，每项仅包含 `type` |
| `public_targets` | 开局公开的母巢、哨卫巢目录，包含名称、位置、血量、可见标记、类型和阵营 |
| `coordinates` | 单个部署多边形的顶点列表，类型为 `list[list[float]]`，格式为 `[[x, y], ...]`，单位为米 |
| `seed` | 随机种子 |

`public_targets`与后续策略初始化的`entities`目录格式一致，隐藏暗巢不进入目录。

部署接口要求且仅接受一个区域。场景/SDK中的`redArea.coordinates`和环境全态中的`deployment_area.coordinates`使用三层区域列表`list[list[list[float]]]`；部署观测在检查区域数为1后提取该多边形的顶点列表。

`deploy(observation)`返回`{实体ID: (x, y)}`完整方案。Manager全量校验后沿用各实体初始高度，通过Engine应用位置并统一启动。就绪初态由`env.get_observation()`读取，部署对象随后关闭。网格布局属于示例算法，方案路径通过示例部署对象的`plan_path`指定。

策略实例在部署后通过`env.agent_manager.register_user_agent()`按受控实体创建并重置一次。BaseAgent构造参数为`(entity_id, init_observation)`，可用关键字`history_length`设置历史长度，默认10。每个实体对应一个策略实例，实体ID同时用于部署方案、观测、动作和奖励路由。日常循环为`get_obs(observation)`接收本机观测，再由`get_action()`返回动作。基类保存最新观测和历史，最小策略实现只需提供`get_action()`。Manager先分发各自观测，再逐个收集动作。

策略初始化数据包含部署后的本机`self`、公开母巢和哨卫目录`entities`、地图与搜索边界，以及seed、max_steps、options和run_id等纯数据。`run_id`标识一次运行，训练跨局复用同一策略组时保持不变。每个实例根据这些资料自行组织helper、模型和合法协作；训练全态由示例训练入口单独读取。

红方单机初始化与逐帧观测中的`self.detectInfo`为`dict[int, dict]`。每条探测记录使用`detected_step: int`表示记录产生时的仿真步号，代替毫秒`time`；缓存和通信转发保留该步号，信息年龄按`当前step - detected_step`计算。内部SDK及环境全态保留毫秒`time`用于精确融合，蓝方时间协议另行定义。

`on_reset()`、`on_feedback(...)`和`close()`提供默认策略生命周期能力，供具体实现扩展本机状态和资源处理。部署计算在独立的`Deployment.deploy()`阶段执行。

单位动作采用有限数值的`N × 4`数组：

| 动作 | 数组格式 | 含义 |
|---|---|---|
| 设置攻击目标 | `[1, 自身编号, x, y]` | 设置目标点坐标 |
| 使用天眼 | `[2, 自身编号, 0, 0]` | 请求天眼支援 |
| 设置横向加速度 | `[3, 自身编号, 机动值, 0]` | 横向机动 |

第二列始终为执行动作的己方实体编号。`x`、`y` 为平面目标位置，单位为米，目标高度沿用现有坐标引导的0米。机动值为无量纲倍率：高空、低空单位按20G换算，中空单位按15G换算，G取9.8米/秒²；0表示停止机动。未使用的参数位置必须填0。

母巢和哨卫巢的初始位置、初始血量构成公开目录，暗巢通过探测获知。引擎按存活单位的位置和通信半径建立连通群，支持多跳中继；群内融合探测记录，失联单位保留历史观测。天眼激活期间更新全图共享信息。

公开动作统一为1、2、3，原动作4、5不再接受。策略自行依据已有观测选择目标并输出坐标；搜索规划也使用同一目标点动作。引擎检查实体控制归属、存活状态、数值及目标点的地图范围，并控制天眼额度。引擎内部按目标编号解析位置的指令能力继续保留。

基础接口与checker位于`user_agents/base/`和`user_agents/red_checker.py`。示例整体位于`user_agents/example/`，分类入口与内部部署、通信、执行和训练工具一起维护。接口扩展、示例策略和PPO/MAPPO流程见 [红方说明](user_agents/example/README.md)。

## 比赛计分

母巢 `9400`、哨卫巢 `9600`、暗巢 `9500` 的固定权重分别为5、2、1，计分目标和初始血量直接从场景实体读取。血量归零的目标计入摧毁权重：

```text
得分 = 100 × 已摧毁目标权重之和 ÷ 全部计分目标权重之和
```

得分范围为0至100。`raw_score` 和 `K` 记录摧毁权重占比，`T` 为0。计分实现位于 `scenarios/cases/reward.py`，可选示例统计位于 `example_evaluation/`。规则通过场景配置加载，示例结果通过上述第三路输出。

## 目录与验证

```text
main.py                  配置路径与程序入口
simulator.py             环境、双方策略和日志的运行编排
previewer.py             组织方初态预览
envengine/               观测、动作执行、物理仿真和渲染
envengine/agent_manager/  部署观测、方案校验应用与策略实例接入
user_agents/base/        独立部署与单机策略基类、历史接口
user_agents/red_checker.py 红方实现文件检查
user_agents/example/    分类基线、示例私有helper及训练工具
blue_defense/            蓝方公共接口、文件检查、分组策略与共用实现
scenarios/               场景加载校验、初始化配置、场景文件和规则资料
example_evaluation/      可选示例评分与结果输出
tests/                   接口、算法和原生环境验证
```

运行双方文件接入测试：

```shell
python -m unittest discover -s tests -p test_blue_plugin.py -v
python -m unittest discover -s tests -p test_red_plugin.py -v
```

## 构建

在目标平台的 CPython 3.11 x64 环境中安装构建依赖：

```shell
python -m pip install -r requirements.txt -r build_tools/requirements-build.txt
```

Windows 使用 `build_tools\build_x64.cmd`，Linux 使用 `bash build_tools/build_x64.sh`。Windows 构建需要 MSVC x64 C++ 工具和 VC++ 运行库。

编译模块保存到 `build/envengine_compiled/`，组装目录为 `build/envengine_dist/`。构建清单记录平台、解释器、源码、产物及文件哈希。根目录入口、配置和运行包一起复制并记入文件哈希清单。发布前检查见 [TODO](TODO/README.md)。

## 原生接口兼容

原生模型的注册名、动态库路径、SDK字段和消息编号沿用现有协议；它们与引擎及模型包配套。源码中的历史英文标识用于接口匹配，选手侧说明和显示名称统一使用当前游戏措辞。
