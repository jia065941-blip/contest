# E01 Transformer 主模型：重构后的 C0–C5 课程计划

**版本时间**：2026-09-15 15:35 CST  
**地图**：`final20/easy/E01`  
**训练批次**：每批 $64$ 回合、$64$ 个互不重复的仿真 seed  
**正式验收**：独立 $128$ 回合，与教师使用完全相同的 seed 成对比较

## 决策与回滚基线

- Batch 1–35 的精简结论保留在 `EXPERIMENT_TRACKER.md`，逐回合 rollout、仿真输出和中间 checkpoint 已永久删除，不再作为新课程的训练历史。
- Batch 35 的 checkpoint `working_c0a_team_search_novelty_batch35.pt` 已删除，不能成为任何新训练的父 checkpoint。
- 新课程从 `checkpoints/initial.pt` 重新开始 $C_{0a}$。
- 仅保留 `best_c0_goal_so_far.pt` 和 `working_after_batch_00020_full_grid_then_centered.pt` 作为旧课程回归对照；不启动训练，直至课程实现和 smoke 测试完成。

## 主要主张

1. 将短攻击/部署决策与多 L 团队搜索拆开，能避免用单 L 稀疏发现奖励训练团队覆盖任务。
2. 在不向 Actor 暴露隐藏 9500 坐标的前提下，使用合法传感器扫掠覆盖、原生直接发现事件和受控规模扩张，可以学习可归因的团队搜索策略，同时保持 E01 与固定目标得分安全。

## 总体顺序

$$
C_{0a}\rightarrow C_{0b}\rightarrow C_{0s}\rightarrow C_1\rightarrow C_2\rightarrow C_{3a}\rightarrow C_{3b}\rightarrow C_4\rightarrow C_5
$$

任何阶段只有在独立 $128$ 回合同种子配对验收通过后才可晋级。训练批自身只决定“继续训练、回滚候选或进入正式验收”，不能直接跨阶段。

## $C_{0a}$：攻击与部署分量化单元学习

MAPPO 在高回报起始状态上只接管一个实体的一个短因果动作单元，依次执行：

1. `C0a_lifecycle`：学习 $\mathrm{WAIT}/\mathrm{LAUNCH}$。
2. `C0a_goal`：目标已合法发现时选择合法攻击目标。
3. `C0a_position`：学习初始发射位置。
4. `C0a_goal_position`：联合学习同一实体的目标和初始位置。
5. `C0a_joint`：联合验收同一实体当前合法开放的攻击、部署和辅助动作头。

受控实体按“类型 × 个体 × 边界时刻”分层轮换，不能连续固定某一种实体或固定某个 9600。每个训练批的 $64$ 个回合仍使用不同 seed；身份轮换是课程采样维度，不得用重复 seed 冒充。

## $C_{0b}$：支撑状态分布对齐

保持一个 `joint` 因果动作单元不变，仅将起始状态分布从高回报锚点扩展到全部 $24$–$32$ 条教师支撑轨迹：

$$
\rho_{\mathrm{high-return}}\rightarrow\rho_{\mathrm{support}},\qquad
\mathcal U_{C_{0b}}=\mathcal U_{C_{0a}}
$$

$24$–$32$ 条支撑轨迹是起始状态分层，不是 seed 数。每个训练批仍必须生成 $64$ 个互不重复的仿真 seed，并在所有支撑轨迹间平衡抽样。

## $C_{0s}$：多 L 团队搜索课程

### 搜索候选与实际接管集合

合法候选集合仍为：

$$
\mathcal E_t^L=\{i\mid i\text{ 为存活且已发射的 L，并满足 SEARCH 合法条件}\}
$$

但学生不能从第一批就接管整个 $\mathcal E_t^L$。在搜索选项边界只接管一个分层轮换的子集：

$$
\mathcal L_{t,K}^{\mathrm{search}}=\operatorname{Select}_K(\mathcal E_t^L),
\qquad K\in\{8,16,32,64,\mathrm{all}\}
$$

每个 $K$ 都要先完成 $64$ 回合训练观察，再通过独立 $128$ 回合配对安全验收，才能扩大到下一个 $K$。未被选入的 L、H、M 和未开放动作头均由教师在当前事实状态上闭环控制。

`Select` 必须随 seed 和批次分层轮换实体，不能根据隐藏目标坐标、未来毁伤结果或“历史上命中过 9500 的实体编号”挑选学生或教师。

### 动态攻击资源约束

不再固定预留某 $9$ 个历史命中 L。用最近已验收数据中 L 外部毁伤率的保守上界 $\bar p_{\mathrm{int}}$ 计算所需可用资源：

$$
R=\min\left\{r:\Pr\left[\operatorname{Binomial}
\left(r,1-\bar p_{\mathrm{int}}\right)\ge 9\right]\ge0.95\right\}
$$

这是团队资源约束而非固定身份约束；发现 9500 后可从仍存活且合法的 L 中动态分配攻击者。扩大 $K$ 后若不再满足 $R$，该候选直接拒绝。

### `C0s_team_route`

- 学生只控制 $\mathcal L_{t,K}^{\mathrm{search}}$ 的 SEARCH 航路点。
- 教师控制所有 L 的发射、初始位置、机动、卫星和攻击。
- 搜索选项在抵达航路点邻域、合法发现目标、实体终止或合法超时条件满足时终止。
- 不使用“持续存活时间”作为无条件正奖励，避免把无效飞行超时训练成好行为。

真实传感器覆盖使用比 $16\times12$ 动作格点更细的 $64\times48$ 诊断网格。对 L 的合法探测半径 $R_L=30\,\mathrm{km}$，给定高度 $h_{i,t}$，水平覆盖半径为：

$$
r_{i,t}^{xy}=\sqrt{\max(0,R_L^2-h_{i,t}^2)}
$$

只在仿真实际传感器扫描时刻累计覆盖集合，覆盖奖励不读取隐藏 9500 坐标。团队航段奖励改为：

$$
G_k=
w_{\mathrm{cov}}\frac{\Delta|\mathcal C_k|}{|\mathcal M|}
+w_{\mathrm{det}}\frac{\Delta N_{9500,k}^{\mathrm{first}}}{9}
-w_{\mathrm{kill}}N_k^{\mathrm{external\ kill}}
-w_{\mathrm{drift}}\mathrm{Drift}_k
$$

其中 $\mathcal C_k$ 是本航段新增合法传感器覆盖单元，$\mathcal M$ 是诊断网格全集。重复覆盖自然不会增加 $\Delta|\mathcal C_k|$，因此 $\mathrm{Dup}_k$ 先保留为诊断指标，不再叠加一个可能重复惩罚同一行为的大权重项。

每个合法边界使用可计算的边际新增覆盖作为稠密信用。精确差分反事实为：

$$
A_{i,k}=G_k(a_{i,k},\mathbf a_{-i,k})-
G_k(a_{i,k}^{\mathrm{null}},\mathbf a_{-i,k})
$$

但完整仿真反事实只对每回合分层抽样的 $2$–$4$ 个边界执行，用来校准信用和审计偏差；其余边界使用新增覆盖和直接发现事件的在线边际信用。不得为数万边界全部复制完整回合。

### `C0s_team_attack`

`C0s_team_route` 通过后开放：

- 9500 尚未合法发现时，学生只选择 SEARCH 航路。
- 9500 进入当前合法目标 mask 后，学生可从 SEARCH 切换到攻击选项。
- 未合法发现的 9500 不得进入 Actor 输入、终止条件或奖励归因。
- 学生启动攻击选项后遵守 $\mathrm{KEEP}/\mathrm{RETARGET}$ 状态机，教师不得在选项终止前改写目标。

本阶段学习：

$$
\mathrm{SEARCH}\rightarrow\mathrm{LEGAL\ DETECTION}\rightarrow\mathrm{ATTACK}
$$

### `C0s_team_evasion`

固定已通过的搜索与目标切换策略，只在合法局部观测出现来袭拦截弹时开放 L 的机动头：

$$
m_{i,t}^{(\mathrm{maneuver})}=1
\iff \mathcal D_{i,t}^{\mathrm{int}}\ne\varnothing
$$

在进入正式训练前，必须先完成机动语义校准 smoke：验证当前 `SetDesiredAccZ` 对 $-1/0/+1$ 的真实飞行响应、持续步数、过载和高度安全范围。此前 $\pm20g$ 脉冲曾造成异常下降，不能直接把它解释为世界坐标中的“左/直/右”。语义未校准则不得开放本子阶段。

奖励比较存活、继续产生新覆盖、继续接近或发现合法 9500，以及任务偏离。未进入 $\mathcal D_{i,t}^{\mathrm{int}}$ 的拦截弹不得参与 Actor 动作、选项终止或奖励归因。

### `C0s_team_joint`

按下列顺序逐头解冻，而不是同一步骤同时放开所有变量：

$$
\mathrm{route+attack}\rightarrow
\mathrm{route+attack+evasion}\rightarrow
\mathrm{route+attack+evasion+satellite}
$$

联合动作的策略概率仍为：

$$
\log\pi_\theta(a_{i,t}\mid o_{i,t})=
\sum_{q\in\mathcal Q}m_{i,t}^{(q)}
\log\pi_\theta^{(q)}(a_{i,t}^{(q)}\mid o_{i,t})
$$

教师控制或确定性 mask 固定的动作头不进入该 step 的 PPO 对数概率和熵。

## $C_1$–$C_5$

- $C_1$：保持 $C_{0b}$ 的攻击支撑状态分布，接管首个因果时间组和相邻的下一个必要时间组。已通过的搜索策略冻结；只有遇到 SEARCH 合法边界时才调用，不能自动把全部搜索实体重新纳入接管。
- $C_2$：接管一个分配目标从首次决策到毁伤结果的完整必要攻击链，其他目标攻击链由教师闭环控制。
- $C_{3a}$：学生在合法边界启动攻击选项后持续控制该实体直到 $\beta_{\omega_i}(o_{i,t})=1$，期间教师不得改写目标。
- $C_{3b}$：从较晚课程起始状态接管全部红方实体的连续后缀，后缀内全部学生动作进入 on-policy rollout。
- $C_4$：按 $\mathrm{satellite}\rightarrow\mathrm{regional\ search}\rightarrow\mathrm{initial\ launch}\rightarrow\mathrm{deployment}\rightarrow t=0$ 逐个向前移动接管边界，每次只改变一个边界。
- $C_5$：从 $t=0$ 独立执行完整回合；教师不参与事实轨迹，只作相同 seed 的固定配对评价基线。

## 修订后的 $C_{0s}$ 指标

原“本回合至少发现一个 9500”的指标只保留为 smoke 健康指标：

$$
P_{\mathrm{any}}=
\frac1N\sum_{e=1}^N
\mathbf1\left(|\mathcal D_e^{\mathrm{student\ L}}|>0\right)
$$

它不能晋级，因为发现 $1/9$ 和发现 $9/9$ 都会记为成功。正式指标改为：

$$
P_{\mathrm{cov}}=
\frac1{9N}\sum_{e=1}^N
|\mathcal D_e^{\mathrm{student\ L}}|
$$

$$
P_{\mathrm{full}}=
\frac1N\sum_{e=1}^N
\mathbf1\left(|\mathcal D_e^{\mathrm{student\ L}}|=9\right)
$$

首次发现时间按目标计算，未发现目标记为回合终点 $T$：

$$
\mathrm{AUC}_{\mathrm{det}}=
\frac1{9NT}\sum_{e=1}^N\sum_{g=1}^9
\left(T-\min(t_{e,g}^{\mathrm{first}},T)\right)
$$

所有 $\mathcal D_e^{\mathrm{student\ L}}$ 和 $t_{e,g}^{\mathrm{first}}$ 必须来自每枚 L 仿真器的原生 $30\,\mathrm{km}$ 直接探测事件，不能使用可能被通信簇同时间戳覆盖的 `detect_from` 字段。

## 晋级规则

每个子阶段先看 $64$ 回合训练批，再运行独立 $128$ 回合同种子配对验证。所有门槛同时满足才通过：

1. $P_{\mathrm{cov}}$ 不低于同种子教师减去预先固定的非劣界 $\epsilon_{\mathrm{det}}$。
2. $P_{\mathrm{full}}$ 达到预定门槛；若目标门槛为 $80\%$，应使用二项置信下界判断，不能仅用样本均值大于 $80\%$。
3. $\mathrm{AUC}_{\mathrm{det}}$ 不劣于教师；同时报告逐目标首次发现时间。
4. 新增覆盖率提高、重复覆盖率下降，且改善在未参与训练的 seed 上复现。
5. L 外部毁伤率相对教师的单侧配对上界不超过容忍量 $\delta_{\mathrm{int}}$。
6. 9400/9600 固定目标分数的单侧配对置信下界不低于 $-\delta_{\mathrm{fixed}}$。
7. E01 配对分数的单侧置信下界不低于 $-\delta_{E01}$。
8. 无非法目标采样、无隐藏 9500 信息进入 Actor、seed 全部唯一、KL 不超限、PPO 指标有限。

$\epsilon_{\mathrm{det}}$、$\delta_{\mathrm{int}}$、$\delta_{\mathrm{fixed}}$ 和 $\delta_{E01}$ 必须在首次正式验证前写入配置并冻结，不能看完结果后调整。

## 实现位置

| 文件 | 修改内容 |
|---|---|
| `tools/train_start_state_option_curriculum.py` | 将阶段表重构为 $C_{0a}/C_{0b}/C_{0s}/C_1/C_2/C_{3a}/C_{3b}/C_4/C_5$；增加子阶段状态、$K$ 级别、实体分层轮换、动态资源门槛和逐级验收；移除按历史 `damage_anchors` 选择固定攻击 L 的逻辑。 |
| `tools/train_start_state_option_curriculum.py` | 将 `team_search_boundary_credits()` 的发现输入改为逐 L 原生直接探测事件；加入 $P_{\mathrm{cov}}$、$P_{\mathrm{full}}$、$\mathrm{AUC}_{\mathrm{det}}$ 和配对非劣验收，`P_{\mathrm{any}}` 只作 smoke。 |
| `core/main.py` | 汇总每枚 L 的直接发现时刻和传感器扫掠覆盖；将通信簇 `detect_from` 仅保留为诊断；按子阶段只开放被选 L 的对应动作头。 |
| `core/envengine/simulator/simlulator_impl/CompCruiseMissileLSimulator.py` | 在不改变探测物理的前提下导出直接探测事件与扫描位置/高度，用于环境侧覆盖和评价；数据不得进入 Actor 隐藏特征。 |
| `policies/red/learning/runtime.py` 与 `policies/red/learning/mappo_policy.py` | 增加课程 action-head mask 与冻结表，保证教师头和确定性头不进入 PPO log-prob/entropy。 |
| `tools/build_native_guidance_manifest.py` | 区分“支撑轨迹分层”和“仿真 seed”；每批生成 $64$ 个唯一 seed，并在 $24$–$32$ 条支撑轨迹间平衡抽样。 |
| `tests/test_temporal_attack_options.py` | 增加学生选项目标不被教师提前改写、SEARCH→ATTACK、逐头 mask 和 $K$ 级接管测试。 |
| `tests/test_run_summary_detection.py` | 增加直接探测与通信簇来源隔离、$1/9$ 不得等价于 $9/9$、未发现时间截尾和覆盖去重测试。 |
| 新增 `tests/test_restructured_curriculum.py` | 验证阶段不能跳级、Batch 35 checkpoint 被拒绝、$K$ 只能逐级扩大、128 回合配对门槛以及唯一 seed。 |

## 实施与运行顺序

1. 先实现直接探测归因、指标和阶段状态机单元测试。
2. 再实现 $K=8$ 的 `C0s_team_route`，只做 $1$–$4$ 回合 smoke，验证接管数量、teacher fallback、覆盖计算和无隐藏信息。
3. 从 `checkpoints/initial.pt` 重跑 `C0a_lifecycle`；未通过不得进入下一组件。
4. 每个正式批次输出 MAPPO/教师 E01、配对差、固定目标分、直接目标覆盖、全覆盖、发现 AUC、外部毁伤率、动作头 KL 与课程状态。
5. 第一个完整 $64$ 回合新批完成后再决定是否发起 $128$ 回合正式验收。

## 停止条件

- 任一阶段出现隐藏真值进入 Actor、教师改写活动学生选项、seed 重复、动作 mask 错误或 checkpoint 父链错误，立即停止该批并标记无效。
- E01 或固定目标分触发预设安全下界，回滚本候选，不继续扩大 $K$ 或开放新动作头。
- 最终目标保持：

$$
\mathbb E[S_{\mathrm{MAPPO}}^{E01}]>
\mathbb E[S_{\mathrm{teacher}}^{E01}]
$$
