# 离线战术回放视频

读取已有对局日志，生成带实时得分、兵力统计和关键事件效果的1080p MP4，或直接导出用于PPT的无损PNG。工具独立运行，不导入仿真引擎、选手策略或在线可视化模块。

## 安装与使用

Python 3.11。在本目录创建独立环境并安装工具依赖：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe replay_video.py render "D:\本局日志\1.json" "D:\本局视频.mp4" --fps 30 --duration 112
```

界面使用仿宋与Times New Roman，中英文分段排版并共享文字基线。Linux使用`python3.11 -m venv .venv`和`.venv/bin/python`，安装这两种字体，或通过`--font-file`与`--latin-font-file`分别指定文件。Windows已完成实际编码验证，Linux尚未完成整片编码验证。

导出指定仿真步的静态画面，用于展板或PPT：

```powershell
.\.venv\Scripts\python.exe replay_video.py frame "D:\本局日志\1.json" "D:\对局画面.png" --step 450 --map-only
```

`frame`直接绘制真实日志，不经过视频压缩。完整画面为1920×1080；`--map-only`保留地图与图例，省略右侧统计和时间轴，按实际画面比例输出。静态画面的主计数器、事件列表和底部进度统一显示仿真步；底部按真实记录的步数线性映射，不使用加速视频的播放进度，也不含片头或结算遮罩。需要与PPT字体一致时，可将同一个字体文件同时传给`--font-file`和`--latin-font-file`。图片旁保存`.validation.json`，记录来源、选取时刻、尺寸和校验值。

只检查记录：

```powershell
.\.venv\Scripts\python.exe replay_video.py audit "D:\本局日志\1.json"
```

将已有日志整理为gzip紧凑回放，再重复生成视频：

```powershell
.\.venv\Scripts\python.exe replay_video.py compact "D:\本局日志\1.json" "D:\紧凑回放"
.\.venv\Scripts\python.exe replay_video.py render "D:\紧凑回放" "D:\视频.mp4"
```

支持原生SDK逐行JSON日志（`config/state/event/aiAction`）、同格式的`.gz`文件，以及包含`replay.jsonl.gz`、`events.json`、`summary.json`的紧凑回放目录。目录另有`damage.json`时读取真实伤害来源，另有`task_changes.json`时读取实际目标分配。输入日志保留完整，已有输出文件拒绝覆盖。

## 记录要求

仿真配置启用`logging.enabled`、`logging.events`和`logging.actions`。`logging.state_interval_steps`控制态势采样间隔，不改变决策频率；工具保留日志实际保存的采样帧。

日志须从第0步开始并包含完整实体目录、位置、血量、类型、阵营和可见状态。缺少仿真时钟的历史日志须按实际步长指定`--step-seconds`，例如每步1秒时填写`--step-seconds 1`。

实体位置读取`pos: {x, y, z}`；日志速度字段为`vel: {x, y, z}`，回放运动方向按相邻记录位移计算。x向东、y向北、z向上，位置与速度单位分别为米和米/秒。地图使用`xMin/xMax/yMin/yMax`，部署区使用同一坐标系。画面使用等比例的米到像素映射，x向右、y向上；超出地图的逻辑控制节点不参与可视作战范围。紧凑数据标记`coordinate_system=local_xyz_m`，与原始日志保持相同的坐标值。

主标题显示“对局回放”。视频与静态帧底部均显示真实场景、虫族策略和种子，仅读取元数据的`map`、`blue`、`seed`。未记录的字段直接省略，不按文件夹名称猜测场次。

## 时间与显示规则

- 默认30帧/秒，可用`--fps`指定1至60的整数。日志开始时逐条读取，视频帧使用提取后的内存数据。
- `--duration`是时间预算：先分配片头、对局、尾段和结算时间，再将末尾快进之前的全部回放额外加速1.5倍。有三秒空闲尾段和五秒结算时，预算112秒得到约77.33秒视频。
- 发现、摧毁和天眼事件附近平滑减速；最后一个关键事件后保留12秒仿真时间，再将剩余空闲段压缩为最多三秒视频时间。结算保留五秒。
- 位置在相邻状态之间线性插值；人族箭头根据记录位移计算方向，再沿最短角度插值。血量、存活和兵力使用最近一条已到达的记录，不插值。
- 完整摧毁得分按母巢/哨卫巢/暗巢权重5/2/1重算。只累计血量归零的目标；摘要与重算不符会报错。得分在记录的摧毁时刻更新。
- 暗巢发现后使用蓝色`#5FA8FF`，图标、发现效果、事件文字和时间轴保持一致；未发现时使用灰蓝点阵。
- 仿宋与Times New Roman文字整体放大16%，人族图元为原尺寸1.95倍，母巢为1.5倍。计时器左移至标题右侧；地图上方的图例分为两行：上排为毁灭者、突击者、清道夫和目标分配，下排为母巢、哨卫巢、暗巢和自爆虫。计分卡上移至右上角，与下方统计区分开，地图、计分卡和统计区从不同高度开始。底部使用独立的118像素高时间轴。
- 关键事件最新在上、最多六条，表头以双横线和向下箭头图标表示顺序。同一秒的暗巢发现合并显示，摧毁事件单独显示。时间、图标和正文按实际字宽留出间距；独立卡片和列表画布裁剪，避免文字越过表头。
- 主时间轴显示母巢摧毁、哨卫巢摧毁、暗巢摧毁、暗巢发现四类关键事件；天眼保留地图与侧栏辅助提示。每条事件使用原有32像素标记，按真实发生时刻排成一行，不合并、不移动时间位置。三种摧毁标记使用彩色边框、透明内部和同色“×”，暗巢发现用带中心点的菱形，与地图及侧栏配色一致。
- 目标分配与航点引导使用琥珀色虚线箭头，实际航点另有空心端点；人族与虫族的实际命中分别用暖红和青蓝实线显示。`track`读取真实目标编号，`waypoint`读取真实航点，切换模式时清除旧线；不根据航点反推目标编号。紧凑日志在元数据`guidance`中保留航点XY、高度和变更时刻。原生AI指令没有独立时钟时按后续采样帧显示，并在核验报告注明精度；单位或目标失效后不显示。
- 采用锐利图形、短拖尾、得分胶囊和扫描效果。动画仅用于讲解，不表示额外攻击或探测。
- 天眼须有成功状态、紧凑事件或实际探测记录；单独的使用请求不算成功。原生9202成功探测记录可在状态采样之前确认启用，侧栏显示“天眼探测”；该时刻是首次成功探测证据，不冒充精确激活时刻。没有持续时间时只显示确认时刻的提示效果。

## 首次发现事件补充

`envengine/simulator/interfaces/isimulator.py`的`handel_detect_info()`在目标首次进入某接收单位的信息库时，通过现有事件通道增加`SENSOR_DETECT_INCREASE`（21001）记录。仅涉及人族获知暗巢、视野塔获知毁灭者两类事件，保存探测步号`step`、米制`pos/vel`、目标和来源；回放按日志实际步长换算显示时间。

此补充不调整态势采样、动作记录、单位状态、策略和计分。离线工具按同类目标选择全局最早的原生记录，避免通信转发产生重复提示。

缺少21001事件的旧日志按首次出现的采样帧显示发现，核验报告注明采样精度。历史首次发现时刻可能被后续探测覆盖，工具不会将最新探测时间冒充首次发现。没有正常结束确认的日志，片尾显示“日志结束”。

## 代码组织

| 文件 | 职责 |
| --- | --- |
| `replay_video.py` | `audit/render/compact/frame`命令入口，检查参数并调用读取和导出 |
| `replay_tool/log_input.py` | 逐行解析、原生/紧凑格式归一、事件提取、数据与得分核验 |
| `replay_tool/rules.py` | 单位名称与完整摧毁权重 |
| `replay_tool/icons.py` | 单位图形和配色，供地图和图例共用 |
| `replay_tool/typography.py` | 仿宋与Times New Roman分段测量、基线与绘制 |
| `replay_tool/video.py` | `Timeline`映射时间，`ReplayPainter`按米制坐标合成画面，`export()`编码，`export_frame()`保存静态图 |

`export()`把每帧RGB图片交给FFmpeg，输出H.264、yuv420p、faststart MP4，再严格解码全片并核对帧数，保存同名`.validation.json`。编码失败保留诊断和未完成文件；成功后清理空的编码日志。

日志单条限制32MiB、解压后限制4GiB、状态数值限制512MiB。全局回放包含组织方信息，不作为选手策略输入。

## 使用真实记录验证

```powershell
.\.venv\Scripts\python.exe verify_real_logs.py "D:\本局日志\1.json" "D:\检查.json"
.\.venv\Scripts\python.exe verify_presentation.py "D:\紧凑回放" "D:\界面检查.json"
.\.venv\Scripts\python.exe verify_projection.py "D:\紧凑回放" "D:\配色检查.json"
.\.venv\Scripts\python.exe verify_record_metadata.py "D:\新建检查目录" "D:\本局日志\1.json"
```

检查使用传入的真实数据，核验计分、无损转换、插值、事件列表和场次信息。工具不附带比赛数据或选手实验策略。
