# AttackMissileAgent 使用说明

`user_agents/attack_missile_agent.py` 中定义了红方**进攻弹智能体** `AttackMissileAgent`，继承自 `user_agents.base_agent.BaseAgent`。

该智能体以 `get_action(observation)` 作为与外部（训练环境 `TrainingEnv` / `AgentManager`）交互的核心接口：环境在每一步为其分发**隔离观测**，智能体返回一个 `numpy` 动作数组，环境再将其转换成引擎可执行的指令。

---

## 1. 类与构造

```python
class AttackMissileAgent(BaseAgent):
    def __init__(self, agent_id: int, entity_id: int, init_observation: dict)
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `agent_id` | `int` | 智能体 ID，由调用方分配（在 `main.py` 中由实体序号 `i + 1` 生成） |
| `entity_id` | `int` | 该智能体所控制的**实体 ID**（红方进攻弹） |
| `init_observation` | `dict` | 初始化态势观测，用于获取蓝方目标列表（见第 4 节） |

> 构造时会固定调用 `super().__init__(agent_id, entity_id, AgentType.AIRCRAFT, init_observation)`，即该智能体的 `agent_type` 恒为 `AgentType.AIRCRAFT`（飞行器）。

### 内部状态

| 属性 | 初值 | 含义 |
| --- | --- | --- |
| `set_acc_z_z` | `0` | 横向加速度状态机当前状态 |
| `acc_start_step` | `0` | 当前横向加速度状态的累计帧数 |
| `launch_step` | `-1` | 已发射时记录发射帧数；未发射为 `-1` |
| `sat_used` | `False` | 卫星是否已使用 |

---

## 2. 核心接口 `get_action`

```python
def get_action(self, observation: dict) -> np.ndarray
```

环境每步调用一次。流程为：

1. 读取 `observation["self"]["type"]`（实体类型）与 `observation["step"]`（当前帧）。
2. 依次调用：
   - `_launch(...)` —— 发射逻辑；
   - `_set_acc_z(...)` —— 横向加速度状态机（仅 `21001` 类型执行）；
   - `_use_satellite(...)` —— 卫星使用逻辑。
3. 返回一个 `dtype=np.float64` 的二维数组，每行代表一个动作（见第 3 节）。若某步无需动作则返回空数组 `np.array([], dtype=np.float64)`（或 `np.array([[]], ...)`，具体取决于 `actions` 是否为空）。

> **注意**：环境并不会直接把完整态势传给 `get_action`，而是通过 `AgentManager.extract_observation_for_agent` 提取出“仅包含本实体”的隔离观测（见第 4 节）。当实体 `health <= 0` 或 `isVisible == False` 时，隔离观测为空 `{}`，此时 `get_action` 不会被调用。

---

## 3. 动作数组格式（返回值）

`get_action` 返回 `np.array`，每行格式为：

```
[action_type, entity_id, arg1, arg2]
```

支持四类动作（顶层常量）：

| 常量 | 值 | 含义 | 行格式 |
| --- | --- | --- | --- |
| `ACTION_SET_ACC_Z` | `0` | 设置 Z 轴（横向）加速度 | `[0, 实体id, 加速度值, 0]` |
| `ACTION_LAUNCH` | `1` | 发射 | `[1, 实体id, 目标经度, 目标纬度]` |
| `ACTION_CHANGE_TARGET` | `2` | 修改目标点 | `[2, 实体id, 目标经度, 目标纬度]` |
| `ACTION_USE_SAT` | `3` | 使用卫星 | `[3, 实体id, 0, 0]` |

> 注：源文件顶部 docstring 中描述的部分 7 元素格式（如 `[ACTION_SET_ACC_Z, id, v, 0, 0, 0, 0]`）属于被注释掉的旧逻辑；**当前实际发出的动作均为 4 元素行**。请以下面“实际行为”为准。

### 实际发出的动作

- 发射：`[ACTION_LAUNCH, self.entity_id, target["position"]["lon"], target["position"]["lat"]]`
- 横向加速：`[ACTION_SET_ACC_Z, self.entity_id, 1, 0]` / `[ACTION_SET_ACC_Z, self.entity_id, 0, 0]`
- 卫星：`[ACTION_USE_SAT, self.entity_id, 0, 0]`

动作数组随后由 `CommandConverter.common_converter` / `CommandAdapter.common_adapter` 转换为引擎指令，调用方无需关心转换细节。

---

## 4. 观测格式

### 4.1 `get_action` 收到的隔离观测

由 `AgentManager` 构造，结构为：

```python
{
    "step": int,            # 当前帧号
    "entity_id": int,       # 本实体 ID
    "agent_id": int,        # 本智能体 ID
    "self": {               # 本实体的完整信息
        "nameChn": str,
        "position": {"lon": float, "lat": float, "alt": float},
        "health": int,          # survivePoints
        "isVisible": bool,
        "type": int,            # 实体类型（21000/21001/21002 等）
        "side": int,
        "detectInfo": ...,
        "commRangeInfo": ...
    }
}
```

智能体内部只用到 `observation["self"]["type"]` 与 `observation["step"]`。

### 4.2 构造时传入的 `init_observation`

由 `TrainingEnv._get_init_ship_observation()` 提供，仅含蓝方高价值目标（`type == 9400` 舰船 / `type == 9600` 拦截阵地）：

```python
{
    "step": int,
    "entities": {
        <entity_id>: {
            "nameChn": str,
            "position": {"lon": float, "lat": float, "alt": float},
            "health": int, "isVisible": bool,
            "type": int, "side": int,
            "detectInfo": ..., "commRangeInfo": ...
        },
        ...
    }
}
```

`get_random_target()` 在该 `entities` 中筛选 `type == 9400` 的目标并返回随机一个；若无可用目标则返回 `None`，对应的发射动作会被跳过。

---

## 5. 内部行为（默认策略）

### 5.1 发射 `_launch`

仅执行一次（由 `launch_step` 控制）。

| 实体类型 | 行为 |
| --- | --- |
| `21000`（高性能弹） | 等待 `step > 50` 后，随机选一个 `9400` 目标发射 |
| `21001` / `21002`（低性能弹） | 立即随机选一个 `9400` 目标发射 |

发射成功后记录 `self.launch_step = step`。

### 5.2 横向加速度状态机 `_set_acc_z`

**仅当 `entity_type == 21001` 且已发射后**才生效，否则直接返回。

状态转移（以发射时刻为 `acc_start_step` 起点逐帧 +1）：

| 当前状态 | 触发条件 | 发出动作 | 下一状态 |
| --- | --- | --- | --- |
| `0` | 发射后累计 `>= 100` 帧 | `SET_ACC_Z = 1` | `1` |
| `1` | 该状态累计 `>= 5` 帧 | `SET_ACC_Z = 0` | `2` |

> 后续“反向加速 / 重设目标点”等状态（`set_acc_z_z = 3/4` 及 `CHANGE_TARGET`）在当前源码中已被注释，默认不会执行。状态切换时通过 `logging.info` 输出 `[进攻弹智能体] 对应实体{id}, 横向加速度状态由{old}变为{new}`。

### 5.3 卫星 `_use_satellite`

已发射且 `step - launch_step >= 100`、且尚未使用过时，发出一次 `[ACTION_USE_SAT, id, 0, 0]`，并置 `sat_used = True`（每局仅一次）。

---

## 6. 其他方法

| 方法 | 说明 |
| --- | --- |
| `get_random_target() -> dict` | 从 `init_observation` 随机返回一个 `9400` 目标实体；无则返回 `None` |
| `reset()` | 重置内部状态（`set_acc_z_z=0`、`launch_step=-1`、`sat_used=False`）并清空历史记录，由环境在每轮重置时自动调用 |

（继承自 `BaseAgent` 的 `record_step` / `get_recent_observations` / `get_cumulative_reward` 等由环境自动维护，无需手动调用。）

---

## 7. 注册与运行（来自 `main.py`）

环境在初始化时为每个红方进攻弹实体（类型 `21000` / `21001` / `21002`）创建一个 `AttackMissileAgent` 并注册：

```python
from envengine import TrainingEnv, Profile
from user_agents import AttackMissileAgent

profile: Profile = Profile.from_dict(profile_data)   # 或 read_profile(url)
training_env = TrainingEnv(profile, render_mode="human")

simulators = training_env.engine.simulator_factory.get_all_simulators()
init_observation_ship = training_env._get_init_ship_observation()

for i, simulator in enumerate(simulators):
    entity_id = simulator.entity_ext.entity.id
    if simulator.entity_ext.entity.entityType in (21000, 21001, 21002):
        agent = AttackMissileAgent(agent_id=i + 1, entity_id=entity_id,
                                   init_observation=init_observation_ship)
        training_env.agent_manager.register_agent(agent)

observation = training_env.reset()
for step in range(1, max_steps + 1):
    obs, reward, done, info = training_env.step()   # 内部自动调用 get_action
    if done:
        break
training_env.reset()   # 触发各 agent.reset()
```

> 也可用 `python main.py` 直接运行（详见 `docs/main_usage.md`）。

---

## 8. 自定义扩展建议

该智能体当前为**随机/脚本示例**。若要实现自己的进攻策略，推荐做法：

1. **继承 `AttackMissileAgent` 或直接继承 `BaseAgent`**，重写 `get_action(observation) -> np.ndarray`：
   - 通过 `observation["self"]` 读取本方状态（位置、健康、可见性）；
   - 通过 `self.init_observation["entities"]` 读取蓝方目标；
   - 也可调用继承的 `get_recent_observations()` 等获取历史，做时序决策。
2. 返回值的每行必须形如 `[action_type, entity_id, arg1, arg2]`，`action_type` 取 `0~3` 之一（见第 3 节）。
3. 如需在每轮开始时重置内部状态，重写 `reset()`（记得 `super().reset()`）。
4. **实体 ID 务必使用 `self.entity_id`**（即构造时传入的 `entity_id`），不要硬编码。

### 最小自定义示例

```python
import numpy as np
from user_agents.attack_missile_agent import AttackMissileAgent, ACTION_LAUNCH

class MyMissileAgent(AttackMissileAgent):
    def get_action(self, observation):
        if self.launch_step >= 0:
            return np.array([], dtype=np.float64)
        target = self.get_random_target()
        if target:
            self.launch_step = observation["step"]
            return np.array([[ACTION_LAUNCH, self.entity_id,
                              target["position"]["lon"], target["position"]["lat"]]],
                            dtype=np.float64)
        return np.array([], dtype=np.float64)
```

---

## 9. 注意事项

1. **实体类型映射**：`21000`=高性能弹、`21001`=低性能弹（执行横向加速状态机）、`21002`=其它低性能弹；目标为 `9400` 舰船、`9600` 拦截阵地（具体含义以想定 `scenario` 配置为准）。
2. **隔离观测为空**：实体阵亡或不可见时 `get_action` 不会被调用，调用方无需在内部判空。
3. **动作维度**：当前实际发出的动作行为 4 元素行，请按第 3 节格式构造返回值；源文件 docstring 中的 7 元素格式为旧注释，已不生效。
4. **随机性**：默认实现含 `random.randint` 等随机逻辑，结果不可复现；如需可复现，请固定随机种子或替换为确定性策略。
5. **平台依赖**：运行环境依赖 `envengine` / `libUtils.so`（Linux 动态库），详见 `docs/main_usage.md` 的环境要求。
```