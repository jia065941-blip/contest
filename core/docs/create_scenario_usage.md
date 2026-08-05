# create_scenario.py 想定生成器配置说明

本文档面向**需要自定义仿真想定（scenario）的用户**，介绍如何使用 `create_scenario.py` 配置红/蓝双方实体、挂载武器与雷达，并生成可被 `main.py --scenario` 加载的 `scenario.json` 文件。

---

## 1. 工具作用

`create_scenario.py` 是一个**想定生成器**，基于 `envengine` 的 `Profile` 数据模型，自动构建：

- 红方实体（飞行器、巡航弹、无人机、卫星等）
- 蓝方实体（拦截阵地、目标、无人船等），并支持挂载**拦截弹**与**雷达**子节点
- 1 个固定的**拦截指控**实体（位于坐标原点）
- 想定范围（`mapArea`）、红方区域（`redArea`）等地图参数

最终输出一个标准 JSON 想定文件，可直接传给 `main.py --scenario ./scenario.json`。

---

## 2. 运行方式

在项目根目录、已激活 3.11 环境并执行过 `pip install -r requirements.txt` 的前提下：

```bash
python create_scenario.py      # 生成 scenario.json（默认输出文件名）
```

> 详见 `docs/main_usage.md` 第 2~3 节搭建环境与安装依赖。
> 生成后：`python main.py --scenario ./scenario.json`

---

## 3. 坐标系与阵营划分（重要概念）

| 项目 | 范围/位置 |
| --- | --- |
| 整体地图 `mapArea` | 经度 `[-4, 4]`，纬度 `[-2, 2]`（代码常量） |
| 红方区域 `redArea` | 左半边，经度约 `-4` 到 `-0.5` |
| 蓝方区域 | 右半边，经度 `0.5` 到 `4` |
| 拦截指控（35000） | 固定位于 `(0, 0)`（自动生成，无需配置） |

- **阵营 ID**：红方 `side_id = 0`，蓝方 `side_id = 1`（代码内部按此分配，无需用户设置）。
- **红方位置**：由配置的 `area` 范围**随机均匀采样**生成（见第 6 节）。
- **蓝方位置**：必须在配置中给出**具体坐标 `(x, y)`**，不再支持范围随机。

---

## 4. 实体类型速查表

配置 `type` 字段时请使用下表左侧的整型值（`MODEL_TYPES` 中的键）：

| type | 名称 | type_id | 是否可在红/蓝手动配置 |
| --- | --- | --- | --- |
| `21000` | 高性能飞行器 | 21000001 | 红方 |
| `21001` | 巡航弹 v2 | 21001001 | 红方 |
| `21002` | 低性能巡航弹（无人机） | 21002001 | 红方 |
| `9202` | 卫星 | 9202001 | 红方 |
| `9400` | 目标 | 9400001 | 蓝方 |
| `9500` | 无人船 | 9500001 | 蓝方 |
| `9600` | 拦截阵地 | 9600001 | 蓝方 |
| `24000` | 拦截弹（挂载子节点） | 24000001 | 由蓝方 `interceptors` 自动挂载 |
| `44000` | 普通雷达（挂载子节点） | 44000001 | 由蓝方 `radar_type` 自动挂载 |
| `35000` | 拦截指控 | 35000001 | 自动生成 1 个（无需配置） |

> 挂载的拦截弹（24000）、雷达（44000）作为蓝方父节点的**子节点**生成，坐标与父节点完全一致，`is_visible=False`（不可见）。

---

## 5. 用户配置区域总览

脚本底部（`# ==================== 用户配置区域 ====================`）是可编辑区，主要包含以下变量：

| 变量 | 作用 |
| --- | --- |
| `LAND_AREA` | 复用的红方区域字典（`x_min/x_max/y_min/y_max`） |
| `RED_CONFIG` | 红方实体配置列表 |
| `BLUE_CONFIG` | 蓝方实体配置列表（手动配置） |
| `generate_blue_config()` | 自动生成标准蓝方布防的函数 |
| `OUTPUT_FILE` | 输出文件名，默认 `scenario.json` |

---

## 6. 红方配置（RED_CONFIG）

每一项为一个实体种类，字段如下：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `type` | int（必填） | — | 实体类型，取值见第 4 节 |
| `count` | int | `1` | 生成数量，按 `area` 范围随机分布 |
| `area` | dict | `{}` | 位置范围 `{x_min, x_max, y_min, y_max}`，单位与地图一致 |
| `survive_points` | int | `1500` | 存活点数（血量） |
| `threat_level` | int | `0` | 威胁等级 |
| `custom_name` | str | `None` | 自定义中文名；不填则自动生成 `名称_typeid_id` |
| `custom_en_name` | str | `None` | 自定义英文名；缺省用 `custom_name` 去空格 |
| `is_visible` | bool | `True` | 是否可见 |

### 示例与特殊规则

```python
RED_CONFIG = [
    {
        "type": 21000,            # 高性能飞行器
        "count": 50,
        "survive_points": 1,
        "threat_level": 0,
        "area": LAND_AREA         # 复用公共区域
    },
    {
        "type": 21002,            # 无人机
        "count": 200,
        "survive_points": 1,
        "threat_level": 5,
        "area": {"x_min": -2.0, "x_max": -0.8, "y_min": -1.5, "y_max": 1.5}
    },
    {
        "type": 9202,             # 卫星（单点坐标也用 area 表示）
        "count": 1,
        "survive_points": 1500,
        "area": {"x_min": -0.6, "x_max": -0.6, "y_min": 0.8, "y_max": 0.8},
        "custom_name": "G_TEL_1_9202001001",
        "custom_en_name": "G_TEL_1_9202001001"
    },
]
```

**`redArea` 自动推导规则**（无需手动设置，由脚本依据 `type` 自动生成）：

- `type` 为 `21000` 或 `21001` → 写入 `redArea.coordinatesHM`
- `type` 为 `21002` → 写入 `redArea.coordinates`
- 若两类区域都未配置，脚本使用左半边默认值填充。

---

## 7. 蓝方配置（BLUE_CONFIG）

每一项为一个实体，**必须提供具体坐标 `x` / `y`**（非范围）。字段如下：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `type` | int（必填） | — | 蓝方实体类型：`9400` / `9500` / `9600` |
| `x`, `y` | float（必填） | — | 实体具体坐标，如 `{"x": 1.8, "y": 0.9}` |
| `survive_points` | int | `1500` | 存活点数 |
| `threat_level` | int | `0` | 威胁等级 |
| `interceptors` | int / None | `None` | 挂载拦截弹（24000）数量；`None` 表示不挂载 |
| `radar_type` | int / None | `None` | 挂载雷达类型，目前用 `44000`；`None` 表示不挂载 |
| `custom_name` | str | `None` | 自定义中文名 |
| `custom_en_name` | str | `None` | 自定义英文名 |
| `radar_name` | str | 自动生成 | 雷达子节点中文名（仅当挂载雷达时有效） |
| `radar_en_name` | str | 自动生成 | 雷达子节点英文名 |
| `is_visible` | bool | `True` | 主实体可见性；挂载的拦截弹/雷达恒为不可见 |

### 示例

```python
BLUE_CONFIG = [
    {
        "type": 9600,             # 拦截阵地
        "x": 1.8, "y": 0.9,       # 必须给出具体坐标
        "survive_points": 30,
        "interceptors": 50,       # 挂载 50 枚拦截弹
        "radar_type": 44000,      # 挂载 1 部雷达
        "custom_name": "拦截阵地_9400001004",
        "radar_name": "普通雷达_44000001001"
    },
    {
        "type": 9500,             # 无人船（不挂雷达）
        "x": 1.7, "y": 1.7,
        "survive_points": 10,
        "interceptors": 50,
        "radar_type": None
    },
    {
        "type": 9400,             # 目标（无挂载）
        "x": 1.9, "y": 0.7,
        "survive_points": 60,
        "interceptors": None,
        "radar_type": None
    },
]
```

**挂载行为说明：**

- 每枚拦截弹命名为 `拦截弹-<全局序号>-<父节点id>`，作为父节点的 `childrenId` 子节点，坐标与父节点相同。
- 雷达子节点的 `external` 默认写入 `{"raderExternal":{"range":{...},"distance":350000,"scanSpeed":5}}`（探测距离 350000，扫描速度 5）。

---

## 8. 蓝方自动生成开关

`main()` 中有一行开关，用于切换“手动 `BLUE_CONFIG`”与“自动标准布防”：

```python
BLUE_CONFIG = generate_blue_config()   # 取消注释 -> 自动生成；注释掉 -> 保留上方手动 BLUE_CONFIG
```

- **启用**（取消注释）：调用 `generate_blue_config()`，按固定部署计划自动生成蓝方布防（上/下区域各 20 个拦截阵地、左区域 50 个无人船、中间 10 个目标，坐标随机采样）。
- **禁用**（注释掉）：使用上方 `BLUE_CONFIG` 列表的手动配置。

> `generate_blue_config()` 部署计划（右半边 经度 0.5~4.0，纬度 -2.0~2.0）：
> - 上区域（纬度 0.2~1.8，经度 0.8~3.8）：20 个拦截阵地，各带 1 部雷达 + 50 枚拦截弹
> - 下区域（纬度 -1.8~-0.2，经度 0.8~3.8）：20 个拦截阵地，各带 1 部雷达 + 50 枚拦截弹
> - 左区域（经度 0.5~1.8，纬度 -1.8~1.8）：50 个无人船，各带 1 部雷达 + 50 枚拦截弹
> - 中区域（经度 1.8~3.0，纬度 -0.8~0.8）：10 个目标，无雷达/拦截弹

---

## 9. 进阶：自定义位置分布

红方实体的随机分布由 `get_position(area_config)` 函数决定，默认采用**均匀分布**：

```python
def get_position(self, area_config):
    x_min = area_config.get("x_min", -4.0)
    x_max = area_config.get("x_max", -0.5)
    y_min = area_config.get("y_min", -1.8)
    y_max = area_config.get("y_max", 1.8)
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    return x, y
```

如需改为高斯分布、网格布点等，可直接修改该函数体（注释 `# ===== 用户可在此修改分布逻辑 =====` 处）。

---

## 10. 输出与想定参数

运行后生成 `OUTPUT_FILE`（默认 `scenario.json`），并自动写入以下想定级参数（位于 `imagineProfile`）：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `simStep` | `1000` | 仿真步长 |
| `simTime` | `1783391450000` | 仿真起始时间戳 |
| `simEndLogicTime` | `50000000` | 逻辑结束时间 |
| `logicTime` | `0` | 初始逻辑时间 |
| `rules` / `eventList` / `taskList` | `[]` | 默认为空 |
| `mapArea` | `lon[-4,4], lat[-2,2]` | 地图范围 |

控制台会打印红/蓝实体数量、拦截弹总数等统计信息。

---

## 11. 注意事项

1. **蓝方必须给坐标**：蓝方配置项缺少 `x` / `y` 会触发 `config["x"]` 键错误，请务必填写。
2. **红方用范围、蓝方用点**：红方 `area` 是范围（随机采样），蓝方 `x/y` 是确定点，二者配置方式不同，不要混淆。
3. **拦截弹/雷达不可见**：挂载子节点强制 `is_visible=False`，即使父节点可见也不会在渲染中单独显示。
4. **实体 ID 自增**：脚本从 `entity_id_counter = 2` 开始自增分配实体 ID；如需指定固定 ID，需修改 `create_entity()`。
5. **环境参数为常量**：`mqttIp`、`domain`、`engineName` 等环境配置在 `_init_environment()` 中硬编码，一般无需改动；仅当接入真实 MQTT/跨域部署时才需修改。
6. **生成文件可直接用**：`scenario.json` 兼容 `main.py --scenario`，无需二次处理。
