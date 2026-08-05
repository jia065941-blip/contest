# 迁移边界

核心副本来源：`competition-platform-env0727.rar` 的已解压参考工程（当前保留于同级目录）。迁移过程不回写来源。

未迁入：`.git`、`.idea`、`.vscode`、`.venv`、`__pycache__`、`.pyc`、历史 `results/`、WandB 缓存，以及 `red_strategy_lab/.vendor`。

保留在 `core/` 的动态库为仿真环境运行依赖；插件不得复制这些动态库。新增插件须提供 `plugin.json`，并在 `plugins/registry.json` 中登记。
