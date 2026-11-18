# 场景 JSON 格式

每张地图保存在一个 JSON 文件中，使用 `environmentProfile` / `imagineProfile` 结构。完整例子见 [E01](E01.json)。位置使用同一米制坐标：**x向东、y向北、z为高度**；速度各分量为米/秒。

| 字段 | 类型、单位与约束 |
|---|---|
| `metadata` | 可选对象；`id`、`title`、`difficulty`、`description` 如提供，须为非空字符串。 |
| `environmentProfile.dynamicLibraryConfigs` | 非空列表；含唯一整数 `entityType` 和非空模型路径 `dynamicLibraryPath`。 |
| `imagineProfile.simStep` | 正整数，仿真步长，毫秒。 |
| `imagineProfile.simTime` | 有限非负绝对时间戳，毫秒。 |
| `imagineProfile.simEndLogicTime` | 有限逻辑总时长，毫秒，至少为一个步长。 |
| `imagineProfile.logicTime` | 可选，默认0；非负且小于总时长，毫秒。 |
| `imagineProfile.mapArea` | `xMin/xMax/yMin/yMax`，有限数值，米；各轴下界小于上界。 |
| `imagineProfile.redArea.coordinates` | 非空多边形列表；顶点为 `[x,y]`，米。至少三个不同顶点、面积非零、无自交；允许省略或重复首点闭合。 |
| `imagineProfile.entityList` | 非空列表；每项含 `entity` 和可选 `external`。 |
| `entity.id`、`entityType`、`sideId` | 整数；ID唯一，-1保留表示无父节点；实体类型已声明模型映射。 |
| `entity.pos` | 有限三维 `{x,y,z}`，米；挂载单位初始位置与其平台一致。 |
| `entity.vel` | 可选有限三维 `{x,y,z}`，米/秒；缺省为零。 |
| `entity.survivePoints` | 有限非负血量。 |
| `entity.parentId`、`childrenId` | 可选，默认-1和空列表；引用存在、双向匹配、无环。 |
| `entity.isVisible` | 可选布尔值。 |
| `entity.stage`、`step`、`time` | 可选；阶段为有限数，`step` 为非负整数毫秒，`time` 为非负绝对毫秒时间戳。 |
| `entity.nameChn`、`nameEn` | 可选字符串。 |
| `external` | 默认空字符串；非空时为 JSON 对象字符串，保存模型参数。 |
| `imagineProfile.eventList`、`taskList` | 可选，当前支持空列表。 |

数值必须有限，整数与布尔值严格区分，JSON对象键唯一。支援实体可以位于地图矩形外。保护目标类型9400、9600、9500的计分权重分别为5、2、1，初始血量读取各实体。时限为 `simEndLogicTime // simStep`。

正式20图的母巢、哨卫巢、暗巢初始血量分别为24、12、2。小／中／大图分别配置2／3／4个天眼实体（类型9202），当前引擎按实体数量计算使用额度；`satelliteMaxUseCount` 不再控制额度。`satelliteUseStep=80` 表示一次持续80步，`missilePincerAttackIntervalStep=20`、`missilePincerAttackMinAngle=30` 配置夹击窗口和角度。战斗单位数量、位置、高度和挂载关系保留既有设计。

部署方案提供 `(x,y)`；各单位沿用自身初始 `pos.z`。红方初始高度一致时，`deployment.launch_altitude` 为共同高度，否则为 `None`。部署区域为 `redArea.coordinates`。

编辑场景时，复制现有JSON，修改实体 `pos/vel`、边界和部署区；增减挂载单位时维护父子引用。调用 `load_scenario(path)` 后运行预览。自定义文件无需登记清单。

```python
from scenarios import load_scenario
scene = load_scenario("my_scene.json")
```

`load_scenario(path, include_rules=True)` 返回 `ScenarioConfig(path, profile, deployment, reward_policy)`。环境通过 `TrainingEnv(scene)` 使用配置；预览采用 `include_rules=False`。`check_scenario(path)` 返回规范化字典，补齐缺省 `parentId=-1`，错误通过带字段路径的 `SceneValidationError` 报告。

统一生成、校验与更新场景哈希：

```powershell
python -m scenarios.generate
python -m scenarios.generate my_scene.json
python -m scenarios.generate --redraw
```

生成入口保持实体ID、血量、父子拓扑及布局比例。正式20图由 `manifest.json` 索引，另有R01和Rtest练习场景。已使用当前格式的输入可重复生成，坐标不会再次缩放。图片显示x/y平面公里坐标，输出位于 `images/`；清单保存场景和图片哈希。
