# Competition Platform All

这是以 `env0727.rar` 为参考构建的插件化工作副本。原始压缩包与参考工程未被修改；本目录用于组合场景、红蓝方基线与评测工具。

## 启动前提

在本目录中通过 `run_local.cmd` 启动。该脚本依次查找 `COMPETITION_PYTHON` 指定的解释器、当前已激活的虚拟环境、仓库内 `.venv` 和系统 `python.exe`，并自动加入红、蓝方插件依赖路径。

```powershell
cd competition_platform_all
.\run_local.cmd list
```

查看本地控制页：

```powershell
.\run_local.cmd ui
```

## 运行一局对抗

以下命令运行 E01 场景：红方 B1 静态优先级规则，对阵蓝方威胁优先策略。`human` 会显示 pygame 窗口；关闭窗口或等待仿真结束即可返回终端。

```powershell
.\run_local.cmd run --scenario easy/E01 --red-policy b1_priority --blue-policy threat_priority --run-id e01_b1_threat -- --total-rounds 1 --max-steps 1200 --render-mode human
```

无界面快速运行时，将最后一项改为 `--render-mode none`：

```powershell
.\run_local.cmd run --scenario easy/E01 --red-policy b1_priority --blue-policy threat_priority --run-id e01_b1_threat_headless -- --total-rounds 1 --max-steps 1200 --render-mode none
```

E01 的环境回合终点为 1200 步。请显式保留 `--total-rounds 1`；当前核心默认也已改为单回合，显式写出可避免批处理时误解。

运行结束后，终端会输出 `FINAL_SUMMARY`，包括结束步数、红方存活/损失、红方实际发射数量、蓝方关键目标存活/毁伤及蓝方已发射拦截弹数。运行产物写入 `results/runs/<run-id>/`。

## 已接入的红方基线

红方基线由“平台局部上报 + 红方策略指挥器”运行。单个平台只提交自身状态、已接收探测和通信信息；指挥器保留跨步发射计划与已发射状态，不直接读取运行中的全局蓝方状态。

| 参数值 | 基线侧重点 | 当前火力规则 |
|---|---|---|
| `b0_random` | 固定种子的随机对照 | 全部可用平台一次性随机分配 |
| `b1_priority` | 按目标价值、距离、弹种成本排序的静态规则 | 全部可用平台按优先级分配主攻与补充任务 |
| `b2_static_assignment` | 一次性静态优化分配 | 全部可用平台按目标价值比例生成优化容量后分配 |
| `b3_rolling_rules` | 基于红方局部汇聚信息的周期性规则重规划 | 全部可用平台按约 25% 的滚动波次逐步释放 |

例如，改用 B2：

```powershell
.\run_local.cmd run --scenario easy/E01 --red-policy b2_static_assignment --blue-policy threat_priority --run-id e01_b2_threat -- --total-rounds 1 --max-steps 1200 --render-mode human
```

## 蓝方基线参数

`--blue-policy` 可选：

- `fixed_ratio_random`
- `nearest_interceptor`
- `threat_priority`
- `min_cost_assignment`

## 红方策略与运动参数

`--red-policy` 与 `--red-motion-policy` 分别控制不同层级，可自由组合：前者
决定目标分配和发射时机，后者决定每枚红方导弹发射后的飞行行为。

| 参数 | 控制层级 | 默认值 | 可选值 | 作用 |
|---|---|---|---|---|
| `--red-policy` | 上层任务分配 | `b0_random` | `b0_random`, `b1_priority`, `b2_static_assignment`, `b3_rolling_rules` | 为平台分配目标并确定发射帧。 |
| `--red-motion-policy` | 下层飞行运动 | `reactive_evasion` | `straight`, `reactive_evasion` | 选择导弹发射后的飞行与规避行为。 |
| `--blue-policy` | 蓝方拦截分配 | `fixed_ratio_random` | `fixed_ratio_random`, `nearest_interceptor`, `threat_priority`, `min_cost_assignment` | 选择蓝方拦截弹分配策略。 |
| `--scenario` | 环境 | `default` | `default`、`easy/E01`--`easy/E03`、`medium/M01`--`medium/M03`、`hard/H01`--`hard/H03` | 选择对抗场景。 |
| `--seed` | 可复现性 | 未设置 | 整数 | 设置红、蓝方随机策略的随机种子。 |

下层运动策略：

- `straight`：无规避运动基线。按分配任务发射后，不再产生额外规避动作。
- `reactive_evasion`：发射后的导弹探测到 `entity_type == 24000` 且其飞行方向
  指向本弹的拦截弹时，执行内置横向机动，并仅使用一次卫星动作。

例如，以下命令保持 B2 静态分配不变，并采用无规避运动基线：

```powershell
.\run_local.cmd run --scenario easy/E01 --red-policy b2_static_assignment --red-motion-policy straight --blue-policy threat_priority --run-id e01_b2_straight -- --total-rounds 1 --max-steps 1200 --render-mode none
```

仅切换下层运动策略，即可启用响应式规避：

```powershell
.\run_local.cmd run --scenario easy/E01 --red-policy b2_static_assignment --red-motion-policy reactive_evasion --blue-policy threat_priority --run-id e01_b2_evasion -- --total-rounds 1 --max-steps 1200 --render-mode none
```

## 单局汇总

每轮结束后，控制台会输出一行以 `FINAL_SUMMARY ` 开头的 JSON；同一内容会
同步写入 `results/runs/<run-id>/<timestamp>/summary.json`，不依赖可选的动作
或状态日志。

汇总包括：场景和策略参数、实际执行步数、终止原因、正式 K/T 得分、红方实际
发射/存活/损失数量、蓝方实体存活/毁伤数量与总剩余血量，以及每个计分目标的
初始血量、最终血量和首次毁伤帧。`突防数` 尚未写入正式汇总，因为引擎目前没
有统一的突防事件和边界定义。

场景可选 `easy/E01` 至 `easy/E03`、`medium/M01` 至 `medium/M03`、`hard/H01` 至 `hard/H03`。

## 已完成的 E01 冒烟对打

固定蓝方 `threat_priority`，四种红方策略均可运行至 1200 步环境终点。该组结果主要验证策略接入、局部信息指挥器与可解释火力规模的差异；不应将其视为策略强弱排名。
