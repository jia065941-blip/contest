蓝方策略说明文档

本次已将蓝方拦截策略从原来的 `DefendCommanderModelSimulator` 内部硬编码逻辑中抽出，建立为独立策略接口。主程序和仿真引擎只负责收集态势、调用策略、发送拦截指令；具体蓝方策略放在 `user_agents/blue_strategies/` 下独立维护。

**一、修改的代码文件**

- [DefendCommanderModelSimulator.py](/C:/Users/Snow/Desktop/game/competition-platform-env/envengine/simulator/simlulator_impl/DefendCommanderModelSimulator.py:1)

主要改动：
- 原蓝方固定拦截逻辑改为策略接口调用。
- 新增环境变量选择策略：
  - `BLUE_POLICY`
  - `BLUE_INTERCEPTOR_RATIO`
  - `BLUE_POLICY_SEED`
- 新增 `_build_blue_observation()`，把仿真器内部状态整理成策略输入。
- 新增 `_assignment_to_command()`，把策略输出转换成仿真引擎的 `INTERCEPTOR_LAUNCH` 指令。
- 蓝方策略本身不直接依赖引擎指令类，便于后续独立建模和测试。

**二、新增的代码文件**

- [base.py](/C:/Users/Snow/Desktop/game/competition-platform-env/user_agents/blue_strategies/base.py:1)
- [__init__.py](/C:/Users/Snow/Desktop/game/competition-platform-env/user_agents/blue_strategies/__init__.py:1)
- [random_salvo.py](/C:/Users/Snow/Desktop/game/competition-platform-env/user_agents/blue_strategies/random_salvo.py:1)
- [nearest_interceptor.py](/C:/Users/Snow/Desktop/game/competition-platform-env/user_agents/blue_strategies/nearest_interceptor.py:1)
- [threat_priority.py](/C:/Users/Snow/Desktop/game/competition-platform-env/user_agents/blue_strategies/threat_priority.py:1)
- [min_cost_assignment.py](/C:/Users/Snow/Desktop/game/competition-platform-env/user_agents/blue_strategies/min_cost_assignment.py:1)

**三、蓝方策略统一接口**

所有蓝方策略统一继承 `BluePolicy`，核心接口是：

```python
decide(observation: BlueObservation) -> list[InterceptorAssignment]
```

输入 `BlueObservation` 包含：
- `sim_time`：当前仿真时间
- `targets`：雷达发现的红方威胁目标
- `interceptors`：蓝方拦截弹状态
- `assets`：蓝方被保护目标/资产
- `launched_map`：已经对每个目标发射过的拦截弹记录

输出 `InterceptorAssignment` 包含：
- `interceptor_id`：选择哪一枚拦截弹
- `target_id`：拦截哪个红方目标
- `target_pos_ecf`：目标当前位置
- `target_vel_ecf`：目标当前速度
- `priority`：策略内部优先级或评分，可用于后续分析

**四、已实现的蓝方基线策略**

`fixed_ratio_random`：固定比例随机分配。  
这是默认策略，每个目标最多分配 `BLUE_INTERCEPTOR_RATIO` 枚拦截弹，候选拦截弹随机选择。可用 `BLUE_POLICY_SEED` 固定随机性。

`nearest_interceptor`：最近拦截弹优先。  
对每个被探测目标，优先选择空间距离最近的可用拦截弹，属于简单几何规则基线。

`threat_priority`：威胁优先级策略。  
按目标类型、目标到被保护资产的距离、资产价值估计威胁程度。高性能目标优先级更高，距离高价值资产越近越优先。

`min_cost_assignment`：代价函数优化分配。  
对“目标-拦截弹”组合计算代价，考虑拦截距离、目标到资产距离、资产紧迫性、重复分配惩罚，然后用贪心方式选择当前最优分配。

**五、运行测试命令**

无界面测试建议：

```powershell
cd C:\Users\Snow\Desktop\game\competition-platform-env

$env:BLUE_POLICY='nearest_interceptor'
python main.py --render-mode none --total-rounds 1 --max-steps 200 --output-dir results\test_blue_nearest --disable-log-color

$env:BLUE_POLICY='threat_priority'
python main.py --render-mode none --total-rounds 1 --max-steps 200 --output-dir results\test_blue_threat --disable-log-color

$env:BLUE_POLICY='min_cost_assignment'
python main.py --render-mode none --total-rounds 1 --max-steps 200 --output-dir results\test_blue_mincost --disable-log-color
```

可视化界面目前会遇到仓库原有的 pygame 字体初始化问题。临时绕过命令如下：

```powershell
$env:BLUE_POLICY='min_cost_assignment'
python -c "import sys, pygame; pygame.font.SysFont = lambda name, size, bold=False, italic=False, constructor=None: pygame.font.Font(None, size); sys.argv = ['main.py', '--render-mode', 'human', '--total-rounds', '1', '--max-steps', '1000', '--output-dir', 'results/viz_blue_mincost', '--disable-log-color']; import main; main.main()"
```

**六、后续扩展方式**

后续如果合作者要新增蓝方策略，只需要：
- 在 `user_agents/blue_strategies/` 下新增一个策略文件。
- 新策略继承 `BluePolicy`。
- 实现 `decide(observation)`。
- 在 `user_agents/blue_strategies/__init__.py` 的 `_POLICIES` 中注册策略名。

主程序和仿真器侧不需要再改。当前这套接口已经支持独立建模、策略横向对比和后续优化算法接入。