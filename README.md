# Competition Platform All

本项目用于运行红蓝对抗仿真。入口是 `run.py`，Windows 下建议通过 `run_local.cmd` 调用。红方可选择 R0--R3 基线，蓝方可选择四种拦截策略，场景内置九张竞赛地图。

## 快速开始

在项目根目录运行一局无渲染仿真：

```powershell
.\run_local.cmd run --scenario easy/E01 --red-policy r0_random --blue-policy b0_fixed_ratio_random --red-motion-policy reactive_evasion --run-id demo -- --total-rounds 1 --render-mode none
```

运行器会将结果写入：

```text
results/runs/<run-id>/<时间戳>/summary.json
```

控制台同时输出一行 `FINAL_SUMMARY`，其中包含红方发射/存活数、蓝方剩余总血量、各计分目标血量与得分。

如果本机没有可用 Python，可指定项目兼容的运行时：

```powershell
$env:COMPETITION_PYTHON = "C:\path\to\python.exe"
.\run_local.cmd run --help
```

## 常用命令

快速检查策略是否能启动：

```powershell
.\run_local.cmd run --scenario easy/E01 --red-policy r1_priority --blue-policy b2_threat_priority --run-id smoke -- --total-rounds 1 --max-steps 2 --render-mode none
```

对同一场景重复多局，并固定随机性：

```powershell
.\run_local.cmd run --scenario medium/M01 --red-policy r2_static_assignment --blue-policy b3_min_cost_assignment --red-motion-policy straight --seed 20260826 --run-id m01_r2_vs_mincost -- --total-rounds 10 --render-mode none
```

启动本地说明页：

```powershell
.\run_local.cmd ui
```

`--` 后面的参数会原样传递给 core；它不能省略，否则 core 参数会被外层运行器当作未知参数。

## 外层运行参数

这些参数写在 `run` 后、`--` 前。

| 参数 | 可选值/示例 | 默认值 | 作用 |
|---|---|---|---|
| `--scenario` | `easy/E01`、`easy/E02`、`easy/E03`、`medium/M01`--`M03`、`hard/H01`--`H03`、`default`、或场景 JSON 路径 | `default` | 选择场景。竞赛场景若未显式设置 `--max-steps`，会自动使用该地图的正式时限。 |
| `--red-policy` | `r0_random`、`r1_priority`、`r2_static_assignment`、`r3_rolling_rules` | `r0_random` | 选择红方目标分配/波次基线。 |
| `--blue-policy` | `b0_fixed_ratio_random`、`b1_nearest_interceptor`、`b2_threat_priority`、`b3_min_cost_assignment` | `b0_fixed_ratio_random` | 选择蓝方拦截分配策略。 |
| `--red-motion-policy` | `straight`、`reactive_evasion` | `reactive_evasion` | 红方平台发射后的底层飞行方式；前者直飞，后者探测到迎面拦截弹时机动并尝试使用卫星能力。 |
| `--seed` | 整数，例如 `20260826` | 未固定 | 同时设置红方、蓝方和仿真的随机种子，用于可复现实验。 |
| `--run-id` | 任意目录名，例如 `e01_r1` | `manual` | 本次结果在 `results/runs/` 下的归档名称。 |

## 红方 R0--R3

红方开局读取 core 的 `TrainingEnv._get_init_ship_observation()`。函数名含 `ship`，但实际提供完整的初始可用目标目录：`9400` 固定目标、`9500` 无人船、`9600` 拦截阵地，包含真实初始 ID、位置、血量和类型。

| 基线 | 行为 |
|---|---|
| `r0_random` | 在完整初始目录中可复现地随机分配，是随机对照。 |
| `r1_priority` | 基于目标价值、距离、已覆盖数量及平台—目标基础命中能力逐平台贪心分配。 |
| `r2_static_assignment` | 开局只求解一次分配；小规模使用精确分配，大规模使用确定性贪心。 |
| `r3_rolling_rules` | 每 20 步对未发射平台重新分配，每次最多释放 25%，用于波次对照。 |

H/M/L 的共同发射延后为 0/10/20 步。R1--R3 使用 core 当前基础毁伤能力进行匹配：H/M 优先 `9400/9600`，L 优先 `9500`；R0 则不引入该偏好。

R0--R3 的完整边界、当前 R3 的静态目录限制及未来 R4+ 规则见 [policies/red/README.md](policies/red/README.md)。

## 蓝方策略

| 策略 | 行为 |
|---|---|
| `b0_fixed_ratio_random` | 对每个已探测来袭平台发射固定数量的拦截弹，拦截弹随机选择。 |
| `b1_nearest_interceptor` | 对每个来袭平台优先分配距离最近的可用拦截弹。 |
| `b2_threat_priority` | 依据来袭平台类型、距受保资产距离和预计到达时间排序。 |
| `b3_min_cost_assignment` | 综合估算拦截时间、目标紧迫度、资产距离及重复覆盖惩罚。 |

蓝方策略只接收雷达已上报的来袭航迹。它目前没有显式的拦截成功概率模型；core 通过实际飞行轨迹和末端几何关系结算拦截。

## 传递给 core 的参数

这些参数必须写在 `--` 后。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--total-rounds N` | `100` | 连续运行 N 局；多局时分别产生 `summary_round_<N>.json`。 |
| `--max-steps N` | 场景正式时限或 `1000` | 每局最大步数。显式传入时覆盖竞赛场景的自动时限。 |
| `--render-mode none` | `human` | `none` 关闭渲染，适合批量试验；`human` 打开可视化。 |
| `--batch-size N` | `100` | 仿真写入器的批量写入大小。 |
| `--disable-log-color` | 关闭颜色 | 适合将日志重定向到文本文件。 |
| `--verbose` | 关闭 | 输出更详细的写入内容。 |
| `--enable-config` | 关闭 | 写出运行配置。 |
| `--enable-state` | 关闭 | 写出状态数据。 |
| `--enable-event` | 关闭 | 写出事件数据。 |
| `--enable-ai-action` | 关闭 | 写出 AI 动作数据。 |

例如，记录一局完整动作和事件：

```powershell
.\run_local.cmd run --scenario hard/H01 --red-policy r3_rolling_rules --blue-policy b2_threat_priority --seed 7 --run-id h01_trace -- --total-rounds 1 --render-mode none --enable-event --enable-ai-action
```

## 目录说明

```text
core/          仿真引擎、观测、通信、毁伤与拦截结算
policies/      红蓝运行时策略
scenarios/     九张竞赛地图、任务信息与计分规则
tools/         离线辅助工具，例如 evaluate.py
experiments/   尚未接入运行链路的学习与预测代码
tests/         策略契约测试
results/       本地运行结果
```
