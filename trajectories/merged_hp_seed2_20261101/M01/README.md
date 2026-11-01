# M01 / seed2 / 合并收益树 + HP 轨迹

这是已完成的 `hp_composition_seed2_20261101` 同内存评估中 `merged_hp` 分支的原始保留记录，得分 **100**。本次上传未重跑仿真。

目录：`trajectories/merged_hp_seed2_20261101/M01/`

| 文件 | 内容 |
|---|---|
| `legal_terminal_trace.json` | 23,440 条 H/M 合法观察、原始动作、HP 决策及启用标记，记录覆盖 step 0–2999 |
| `execution.json` | 实际技能执行、绑定、指令与回执记录 |
| `combat_metrics.json` | 评估器统计的发射、伤害、拦截、目标结果等 |
| `summary.json` | 官方终局得分、目标摧毁时间及结束原因；3000 步，time_limit，100 分 |
| `branch_audit.json` | 同内存分支的 RNG、双地址及执行验证证据 |
| `manifest.json` | 原始归档来源、策略版本、逐文件 SHA-256 与记录范围 |

`legal_terminal_trace.json` 的 `records` 保存逐条 `step`、`unit`、`target`、`context`、`decision`、`enabled` 和 `original_actions`。它是 HP 相关 H/M 的合法状态/决策轨迹，并非所有单位的完整世界状态逐帧回放；不应直接当作 UI 原生回放文件。step 195 前为分叉前共享历史。评估器统计与地址审计不得作为在线策略输入。

原生全量逐帧 dump 未被历史归档保留；这里不伪造或重跑补齐。来源归档中的获胜分支文件原样提取，仅将 `native/episode/summary.json` 放到本目录的 `summary.json`。

策略代码在 `releases/merged_hp_seed2_20261101/`。官方调度含地址排序，新进程不保证逐分复现历史成绩。历史外层监控约 577.85 秒缺口和恢复根进程 OS 退出码缺失的限制见该发布目录说明；本分支保留的原生终局及 RNG/双地址验证已通过。
