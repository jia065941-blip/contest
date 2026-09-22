### 工程结构
```
competition-platform-env/
│
├── docs/                         # 描述文档
│   ├── create_scenario_usage.md    # 想定创建说明
│   └── main_usage.md               # 主入口使用说明
│
├── envengine/                    # 核心仿真引擎包
│   │
│   ├── sdk/                        # 基础数据结构与通信协议
│   │   ├── BaseStruct/             # 基础数据类（位置、实体、消息等）
│   │   ├── Struct/                 # 通信包结构体
│   │   ├── Transport/              # MQTT 通信实现
│   │   └── Util/                   # 工具函数
│   │
│   ├── common/                   # 通用命令与触发器
│   │   ├── BaseStruct/             # 基础结构体（攻击、探测、干扰等）
│   │   ├── Command/                # 命令类
│   │   ├── Trigger/                # 触发器类
│   │   ├── Enum/                   # 枚举类型
│   │   ├── SimmerCommandType.py    # 命令类型定义
│   │   └── SimmerTriggerType.py    # 触发器类型定义
│   │
│   ├── engine/                   # 核心仿真引擎
│   │   └── engine.py               # 驱动仿真步进、管理仿真器工厂、控制倍速
│   │
│   ├── simulator/                # 仿真器模块
│   │   ├── interfaces/             # ISimulator 接口定义
│   │   ├── simlulator_impl/        # 具体仿真器实现（飞行器、舰船、轨道模型等）
│   │   ├── models/                 # 底层物理模型（SWIG封装的C++高性能计算/Python计算）
│   │   ├── decorator/              # 仿真器注册装饰器
│   │   └── simulator_factory.py    # 仿真器工厂：创建、索引、查询仿真器
│   │
│   ├── environment/              # 训练环境层（RL适配）
│   │   ├── training_env.py         # 训练环境：封装Engine，提供reset/step接口
│   │   └── command_adapter.py      # 命令适配器
│   │
│   ├── agent_manager/            # 智能体管理器
│   │   └── agent_manager.py        # 智能体管理器：管理所有智能体，负责观测隔离分发和动作收集
│   │
│   ├── render/                   # 可视化渲染
│   │   ├── renderer.py             # Pygame独立线程渲染器，实时显示态势
│   │   ├── map.jpg                 # 地图素材
│   │   └── map.png                 # 地图素材
│   │
│   └── __init__.py               # 包初始化，导出核心类
│
├── user_agents/                  # 用户智能体实现
│   │── base_agent                  # 智能体基类目录
│   │   └── base_agent.py             # 智能体基类
│   └── xxx.py                      # 用户自定义智能体
│
│
├── scenarios/                    # 想定文件（JSON）
│   ├── platform.json               # 平台想定
│   └── platform_bak.json           # 备份想定
│
└── main.py                       # 主入口（想定解析工具）
```

### 导出requirements.txt
```
pigar generate
```

### 编译
```
python setup.py build_ext
```

### python 版本
```
python 3.11
```