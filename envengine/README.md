# Engine 输入输出说明

**状态：待人工确认草案。** 代码基准 `1e8bd24`，日期：2026-11-03。

本文只说明原生 `Engine`；公开位置为 `pos={x,y,z}`、速度为 `vel={x,y,z}`，x向东、y向北、z为高度，单位分别为米和米/秒。

## 入口方法

| 方法 | 输入 | 输出 |
|---|---|---|
| `Engine(profile, speed_multiplier=1)` | SDK `Profile`、运行倍速 | `Engine` 实例 |
| `reset()` | 无 | `None`，重置仿真 |
| `step(ai_commands=None)` | `list[Command]`；无指令可传 `[]` | `None`，推进一步 |
| `run(max_steps=None, step_callback=None)` | 运行步数、逐步回调 | `None`，循环推进 |
| `get_simulator_by_id(entity_id)` | 实体ID | `ISimulator`，不存在为 `None` |
| `get_simulators_by_type(entity_type)` | 实体类型 | `list[ISimulator]` |
| `get_simulators_by_side(side_id)` | 阵营ID | `list[ISimulator]` |
| `refresh_red_communications()` | 无 | 通信组 `tuple[tuple[int,...],...]`，同时刷新共享信息 |

来源：[Engine各方法](engine/engine.py)、[SimulatorFactory查询方法](simulator/simulator_factory.py)。

`Engine.step()`、`reset()`、`run()` 实际均返回 `None`；源码中step/run关于“返回态势”的注释与实现不一致。

## 最小输入

`Engine` 接收SDK对象。公开场景通过 `load_scenario(...).profile` 取得，坐标和SDK内部字段由适配器转换。

| Profile字段 | 用途 |
|---|---|
| `environmentProfile.dynamicLibraryConfigs` | 实体类型与模型映射 |
| `imagineProfile.entityList` | 实体初态和模型参数 |
| `imagineProfile.simStep` | 每步仿真毫秒数 |
| `imagineProfile.simTime` | 起始仿真毫秒时刻 |
| `imagineProfile.simEndLogicTime` | 总时长，用于计算默认运行步数 |
| `imagineProfile.mapArea / redArea` | xy米制地图范围与部署区域 |

`step()` 的每条输入是SDK `Command`，主要字段为 `executorId`、`commandTypeId`、`commandAttributes`。
`executorId` 为执行实体ID；当前原生红方指令如下：

| `commandTypeId` | `commandAttributes` |
|---|---|
| `3007`：机动 | `{"acc_z": value}` |
| `3012`：速度 | `{"vel_x": value}` |
| `3013`：天眼 | `{"requested": True}` |
| `3014`：目标点 | `{"intent": "waypoint", "target": {"x": x, "y": y, "z": z}}` |
| `3014`：目标编号 | `{"intent": "track", "target_id": entity_id}` |

来源：[Profile](sdk/base_struct/profile/profile.py)、[Imagine](sdk/base_struct/profile/Imagine.py)、[Command](sdk/base_struct/Message/Command.py)、[process_ai_commands / validate_red_command](simulator/simulator_factory.py)、[load_scenario](../scenarios/loader.py)、[profile_from_scene](environment/scene_adapter.py)。

## 回调与状态

`run()` 在每次推进前调用 `step_callback(run_index, situation)`，回调返回 `list[Command]`。
`run_index` 是本次运行的循环索引；摘要中累计步号字段如下：

| situation字段 | 含义 |
|---|---|
| `step` | `Engine.current_step`，累计步号 |
| `entity_count` | 已登记实体总数 |
| `timestamp` | `current_step * simStep / 1000`，经过秒数 |

`Engine.sim_time` 是内部仿真毫秒时刻，与对外累计步号区分。
实体状态通过上述查询方法读取；返回的是运行中的仿真器对象。
事件由工厂交给SDK `write_event()` 写入轮次文件，不作为 `step()` 返回值。
来源：[Engine.run / _get_current_situation](engine/engine.py)、[_send_events_callback](simulator/simulator_factory.py)、[write_event](sdk/writer/async_file_writer.py)。

## 调用结构示例

```python
from envengine import Engine
from scenarios import load_scenario

engine = Engine(load_scenario("E01").profile, speed_multiplier=0)
engine.reset()

def choose_commands(run_index, situation):
    return []

engine.step([])
engine.run(max_steps=10, step_callback=choose_commands)
```

## 与TrainingEnv的区别

`TrainingEnv.reset()` 返回观测字典。
`TrainingEnv.step()` 由管理层获取动作，不接收 `actions` 参数。
完整态势由上层 `TrainingEnv.step()` 的 `(observation, rewards, done, info)` 四元组返回，其中 `observation` 为态势；原生 `Engine.step()` 仍返回 `None`。
来源：[TrainingEnv.reset / step](environment/training_env.py)。
