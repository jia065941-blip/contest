# Red Strategy Lab

`red_strategy_lab` 是红方基线策略的独立实验包。

## 内容

- `B0RandomPolicy`：固定种子的随机策略；
- `B1PriorityPolicy`：固定优先级全局规则；
- `B2StaticAssignmentPolicy`：一次性全局整数分配；
- `B3RollingRulePolicy`：按固定周期重算的参数化全局规则。

## 运行测试

```powershell
python -m unittest discover -s red_strategy_lab/tests -v
```

测试只依赖 Python 标准库。环境适配代码位于 `src/red_strategy_lab/adapter.py`，策略模块可独立测试。

## 场景烟雾测试

```powershell
python red_strategy_lab/scripts/run_scenario.py --baseline b1 --max-steps 5
```

运行器通过 `adapter.py` 将基线决策转换为现有命令，并将每次结果写入 `results/red_strategy_lab/<run_id>/summary.json`。

## 场景基线比较

```powershell
python red_strategy_lab/scripts/compare_scenarios.py --max-steps 3
```

比较结果写入 `results/red_strategy_lab/comparison_<run_id>.json`。
