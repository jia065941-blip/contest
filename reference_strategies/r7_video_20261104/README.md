# 两段演示录像使用的本地红方策略源码

本目录保留2026年11月4日补录使用的原始源码，文件内容逐一核对SHA256。两场实验共享基础代码，通过`variant`选择不同的部署和作战规则：

| 对局 | 方案选择参数 | 策略 | 录像完整摧毁得分 |
|---|---|---|---:|
| M01–B9，种子42 | `baseline` | 原R7+空间波次、任务延续与短时避让 | 89.85507246376811 |
| E06–B8，种子42 | `recovery` | 空间波次基础上的搜索覆盖与任务恢复 | 95.45454545454545 |

录像运行的仿真代码基于`e7365ad5b4e1c807ca63c63bd92e27d656d957cd`。上述分数属于历史录像，不能作为当前版本的复现结果。

## 原始文件

- `r7plus_port/adapter.py`：旧实验的部署与单机入口，部署构造接收`map_area`和`variant`。
- `r7plus_port/strategies.py`：搜索覆盖部署、`CoverageCommander`与`RecoveryCommander`。
- `r7plus_port/coordination.py`：通信群重组后的本机任务延续。
- `r7plus_port/legacy/`：当时冻结的波次、任务分配、搜索和避让实现。
- `SHA256.json`：本目录原始Python源码的逐文件哈希。

## 适配状态

`r7plus_port`是原始源码快照，不能直接填写到最新版`config.yaml`中运行。适配后的独立入口为 R7 3.0 `user_agents/example/coordination/r7_v3_0_spatial_packages.py` 和 R7 4.0 `user_agents/example/coordination/r7_v4_0_search_recovery.py`，具体接口、模块职责及验证方法见`user_agents/example/spatial/README.md`。原始Python源码和`SHA256.json`保持不变。

本目录另提供`M01_B9.yaml`与`E06_B8.yaml`，在工程根目录使用`python main.py --config 配置路径`运行当前版本。

本目录不包含实验成绩文件、对局日志、地图、视频或仿真引擎修改。
