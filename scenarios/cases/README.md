# 竞赛场景

九个场景按 `easy`、`medium`、`hard` 三个难度目录组织，每个案例包含：

- `scenario.json`：仿真器直接读取的场景；
- `case_info.json`：目标、时限与场景校验信息；
- `map_info.png`：不含推荐航迹、发射顺序或目标分配的信息图。

`manifest.json` 是全场景索引，`reward.py` 是统一 K/T 计分实现。运行单个案例：

```powershell
.\run_local.cmd run --scenario easy/E01 -- --total-rounds 1 --render-mode none
```
