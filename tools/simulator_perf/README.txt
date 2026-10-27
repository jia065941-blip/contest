仿真复制加速（2026-10-27，直接集成到 cz）

一、cz 自带仿真器

基线：cz 的 57da02d。三个已有文件只替换 deepcopy 的调用来源：
TrainingEnv、ISimulator 和 SimulatorFactory。复制的时机、次数、隔离边界、
训练历史、奖励、通信、部署及事件调度均保留。其他对象和自定义复制协议
继续委托标准库处理；没有全局修改 copy.deepcopy。

默认使用 Python 容器复制路径；Linux CPython 3.11 可以先编译一次 C 扩展：

  export PYTHONDONTWRITEBYTECODE=1
  export CONTEST_COPY_LIBRARY="$(python tools/simulator_perf/build.py)"
  export CONTEST_COPY_BACKEND=c

然后使用原来的 run.py 命令。子进程继承环境变量。build.py 默认把产物和
编译临时文件放在 /dev/shm/contest-copy-<uid>；同一源码及 ABI 只编译一次，
各局不编译、不保存额外逐步日志。重启清空 tmpfs 后重跑编译命令即可。
有多个 Python 环境时，编译和运行必须使用同一个解释器。

  CONTEST_COPY_BACKEND=stdlib  使用标准库，便于诊断
  CONTEST_COPY_BACKEND=python 使用 Python 容器复制实现
  CONTEST_COPY_BACKEND=c      必须提供已编译的 CONTEST_COPY_LIBRARY

C 版本由原 launch20 复制器适配，额外遵守调用者对原子值预填的 memo，
不启用原版本忽略此类 memo 的标量字典快捷路径。它不是原始 C 文件的原样副本。

Linux 原生依赖若提示 libUtils.so 或 libpython3.11.so 找不到，请按现有运行器
配置库路径。当前服务器导入使用 core/envengine/simulator/models/HXDMissileModel
中的 libUtils.so；core/libUtils.so 与该模型的符号不匹配。这是基线自带库差异，
本提交未替换原生模型库。Python 标准库路径可由 sysconfig.get_config_var('LIBDIR')
查询。Windows 未提供 C 构建入口，使用 Python/stdlib 路径。

二、当前官方 launch20 环境的完整加速实现

official_launch20/ 收录此前实际运行的五个 SDK 优化文件、原始补丁、C 源码、
延迟复制通信表和可选 Astra 报告复用。来源：launch20_seed2_matrix_20261024/app。
适用 SDK 是 competition-platform-env 的 pku 提交：
74dc796390985bfb52341aabde9a7c47833d866e。

这个 SDK 与 cz 自带仿真器不是同一版本。cz 使用实体 ID 排列同刻事件，
官方 SDK 使用对象地址。两者的通信规则、观测结构也不同，所以不能把整个
官方 Engine/TrainingEnv 覆盖到 cz/core 后仍声称保持 cz 的游戏行为。
本包保留两套实现原来的调度；不把历史官方成绩当作 cz 的成绩。

安装到单独的、版本匹配的 SDK 工作目录（目录下应有 envengine）：

  python tools/simulator_perf/official_launch20/install.py --sdk /path/to/sdk --check
  python tools/simulator_perf/official_launch20/install.py --sdk /path/to/sdk
  python tools/simulator_perf/build.py --official-launch20

安装器先检查所有五个目标的原始/已优化哈希，再写入；重复安装无操作。
检测到不匹配的文件会报错，不适用于 cz/core。哈希检查覆盖被修改的 SDK 文件，
不能替代完整 SDK 版本管理；其他源文件应来自上述提交。

在现有评估入口构造智能体之前，把 official_launch20/ 和 SDK 加到 sys.path：

  import activate
  activate.install('/dev/shm/contest-copy-<uid>/<编译命令输出的文件名>')

该入口启用 C 复制器和通信子树按需复制。仅对已经隔离且后续只读报告的
原 Astra PlanGroup 流程，可额外传 astra_reports=True，以省掉成员报告的
重复复制。任务仍复制；不要对会修改观测的任意智能体启用这个选项。
训练智能体默认保留历史；只有明确设置 record_history=False 的评估智能体
跳过历史保存。这里不包括策略权重、地图或完整评估调度器。

三、检查范围

离线检查命令（不启动仿真）：

  python -B tests/test_simulator_fastcopy.py
  python -B tests/test_simulator_perf_install.py
  python -B tools/simulator_perf/official_launch20/test_communication_copy.py

先不设置 CONTEST_COPY_LIBRARY 执行复制测试，再按第一节编译并设置它，
执行相同复制测试覆盖 C 路径。检查包括循环引用、共享引用、独立修改、
自定义复制回调、回调顺序、容器遍历期间修改、错误恢复、引用释放、NumPy
数组及 memo。安装测试检查损坏源文件/目标文件时不写入、重复安装和打包哈希。

没有为此次上传新增仿真局数，也没有宣称测得 cz 上的加速倍数或成绩。
原官方环境曾做过有限轨迹检查，但其中 performance_v2/v3 的历史检查使用了
测试专用的稳定同刻排序，不能作为官方地址调度下的技能因果收益证据。
后续 skill 训练继续使用同一内存状态派生的分支并保留官方地址审计。

编译产物不上传 Git。评估仍需由调度器限制总进程和内存：父进程及分支总数
不超过 64、至少预留 10% 内存、瞬态输出放 tmpfs，仅持久化精简汇总。
本加速模块不负责关闭外部运行器已经配置的日志或限制其并发。
