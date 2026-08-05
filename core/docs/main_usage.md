# main.py 使用说明

本文档面向**首次下载本项目**的用户，介绍如何搭建运行环境、安装依赖，并通过命令行参数运行 `main.py` 启动仿真训练环境。

---

## 1. 环境要求

| 项目 | 要求 |
| --- | --- |
| Python | **3.11.x**（工程 `setup.py` 限定 `python_requires="==3.11.*"`） |
| 操作系统 | 推荐 **Linux**（官方 `Dockerfile` 基于 `python3.11-app` 镜像）。Windows 上因 `envengine` 内置 `.so`（Linux 动态库）及根目录 `libUtils.so`，需要自行重新编译为 `.pyd` 才能运行，见第 3 节 |
| 编译工具 | 仅当需重新编译 `envengine` 时需要：`Cython` + C/C++ 编译器（`gcc`/MSVC） |

> 建议：若本地为 Windows，可直接使用工程提供的 `Dockerfile` 构建镜像运行，避免平台兼容问题。

---

## 2. 建立虚拟环境（conda）

建议使用 `conda` 创建独立的 Python 3.11 环境，避免污染系统 Python。在项目根目录执行：

```bash
# 进入项目目录
cd competition-platform-env

# 创建名为 competition 的 conda 环境，指定 Python 3.11
conda create -n competition python=3.11 -y

# 激活环境
conda activate competition
```

激活后命令行前缀会变成 `(competition)`。

> 若使用 `mamba` / `micromamba`，命令相同，仅把 `conda` 换成对应命令即可，例如：
> `mamba create -n competition python=3.11 -y && mamba activate competition`

退出环境：`conda deactivate`

---

## 3. 安装第三方依赖

依赖列表见根目录 `requirements.txt`：

```
colorlog==6.10.1
dataclasses-json==0.6.7
fastdisjointset==1.0.3
paho-mqtt==2.1.0
pygame==2.6.1
pyproj==3.7.2
requests==2.34.2
websockets==16.0
numpy==2.4.6
msgpack==1.2.1
```

安装命令（已激活虚拟环境）：

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

如需使用清华镜像加速：

```bash
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 关于 envengine 包

`envengine/` 是工程内置的仿真引擎包：

- **Linux**：仓库已包含编译好的 `.so` 文件，安装 Python 依赖后通常可直接 `import envengine`，无需额外编译。
- **Windows / 需重新编译**：执行以下命令（依赖 `Cython`，已在 `setup.py` 的 `install_requires` 中声明）：

  ```bash
  pip install Cython==3.2.6
  python setup.py build_ext
  ```

  编译产物为 `.pyd`（Windows）/ `.so`（Linux），需保证生成的扩展文件位于 `envengine` 对应子目录中。

- 根目录 `libUtils.so` 为引擎底层公共库，Linux 下 Dockerfile 将其复制到 `/usr/lib/`。若在宿主机直接运行，需确保该库可被动态链接器找到（例如放到 `/usr/lib/` 或设置 `LD_LIBRARY_PATH`）。

---

## 4. 准备想定（scenario）文件

`main.py` 通过 `--scenario` 加载想定配置，支持**本地路径**或**远程 HTTP(S) URL**。

- 工程内置想定位于 `scenarios/`：`platform.json`、`platform_1.json`。
- 也可用 `create_scenario.py` 生成自定义想定：

  ```bash
  python create_scenario.py      # 生成 scenario.json
  ```

  生成的文件可直接传给 `--scenario ./scenario.json`。

---

## 5. 命令行参数说明

运行方式：

```bash
python main.py [参数...]
```

参数一览（括号内为默认值）：

| 参数 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- |
| `--scenario` | `./scenarios/platform.json` | 字符串（路径或 URL） | 想定配置文件路径，或 `http(s)://` 远程地址 |
| `--total-rounds` | `100` | 整数 | 总仿真轮数 |
| `--max-steps` | `1000` | 整数 | 每轮最大仿真步数 |
| `--render-mode` | `human` | `human` / `none` | 渲染模式。`human` 用 pygame 实时渲染（需图形界面）；`none` 为无界面（服务器/无头环境） |
| `--output-dir` | `results` | 字符串 | 输出目录基路径，运行时会自动追加时间戳子目录 |
| `--batch-size` | `100` | 整数 | 写入器批处理大小（多少条记录刷盘一次） |
| `--disable-log-color` | 默认开启彩色 | 开关（store_false） | 加上该参数可**禁用**日志彩色输出 |
| `--verbose` | 关闭 | 开关 | 启用详细输出（同时传给写入器 `verbose`） |
| `--enable-config` | 关闭 | 开关 | 启用配置写入 |
| `--enable-state` | 关闭 | 开关 | 启用状态写入 |
| `--enable-event` | 关闭 | 开关 | 启用事件写入 |
| `--enable-ai-action` | 关闭 | 开关 | 启用 AI 动作写入 |

> 说明：`--disable-log-color` 是“取反”型开关——不传时日志带颜色（`enable_log_color=True`），传了该参数则不带颜色。其余 `--enable-*`、`--verbose` 均为“出现即开启”的开关型参数。

---

## 6. 运行示例

最简运行（使用默认参数）：

```bash
python main.py
```

指定想定、关闭渲染、保存到指定目录：

```bash
python main.py \
  --scenario ./scenarios/platform_1.json \
  --render-mode none \
  --output-dir ./my_results \
  --total-rounds 10 \
  --max-steps 500
```

从远程 URL 加载想定：

```bash
python main.py --scenario https://example.com/platform.json
```

无头服务器上获取更完整日志并写入状态/事件：

```bash
python main.py \
  --render-mode none \
  --verbose \
  --disable-log-color \
  --enable-state \
  --enable-event
```

查看全部参数帮助：

```bash
python main.py --help
```

---

## 7. 输出说明

- 结果写入 `--output-dir` 指定的基目录下，自动创建带时间戳的子目录（如 `results/20260729120000/`）。
- 具体写入内容由 `--enable-config / --enable-state / --enable-event / --enable-ai-action` 控制；未指定这些开关时，对应内容不会被写出。
- 写入采用批处理机制，由 `--batch-size` 控制缓冲条数，每轮结束会强制刷盘（`write_immediately()`）。

---

## 8. 注意事项

1. **运行环境平台**：工程以 Linux（`Dockerfile`）为第一运行目标，`envengine` 与 `libUtils.so` 为 Linux 动态库。Windows 使用需重新编译为 `.pyd`。
2. **渲染依赖图形界面**：`--render-mode human` 依赖 `pygame` 与可用显示设备；无界面服务器请使用 `--render-mode none`。
3. **每轮等待**：`main.py` 在每轮开始时调用 `red_model_deploy()` 后会有约 `time.sleep(1000)` 的等待（源码内硬编码），单轮实际耗时通常远不止 `--max-steps` 仿真时间，属正常行为，请合理预估运行时间。
4. **Python 版本**：务必使用 3.11，否则 `envengine` 的编译扩展与依赖版本可能不兼容。
5. **想定 URL 鉴权**：远程想定需返回 JSON 且数据位于 `data` 字段（见 `read_profile()`），请确保接口格式符合要求。
