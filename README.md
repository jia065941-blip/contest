# Competition Platform All

这是以 `env0727.rar` 为参考构建的插件化工作副本。原始压缩包与参考工程未被修改；本目录用于组合场景、红蓝方基线与评测工具。

## 启动前提

在本目录中通过 `run_local.cmd` 启动。该脚本会使用已配置的 Python 运行时，并自动加入红、蓝方插件依赖路径。

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

场景可选 `easy/E01` 至 `easy/E03`、`medium/M01` 至 `medium/M03`、`hard/H01` 至 `hard/H03`。

## 已完成的 E01 冒烟对打

固定蓝方 `threat_priority`，四种红方策略均可运行至 1200 步环境终点。该组结果主要验证策略接入、局部信息指挥器与可解释火力规模的差异；不应将其视为策略强弱排名。
