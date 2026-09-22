# cz 分支：当前实现与 b14 检查点

此分支基于远程 main 的 `09cb91ec2c3962f19c97f2d340756c3f1291bc34`，保存 competition-platform-env 当前相关实现与原始 b14 模型。main 不随此提交更新。

## 模型

- 文件：[`../../models/b14_c0a_goal.pt`](../../models/b14_c0a_goal.pt)，原始完整 checkpoint（含配置、模型张量及目标头优化器状态）。
- SHA256：`4609add14e70a5e25df87ddb3377756c2e45b0d3fa136d4de66406f26ca2be6e`。
- 架构：`target_conditioned_ctde_mappo_v11`，Transformer 目标头。
- b14 从 b13 完成 48 次目标头更新；本分支包含后续位置课程 b15 的实现，但没有上传 b15 权重。

在配置了项目依赖的 Python 环境中，从仓库根目录运行：

```bash
python tools/inspect_b14_checkpoint.py
sha256sum -c models/b14_c0a_goal.pt.sha256
```

检查命令验证原始文件哈希、严格加载模型和有限值前向推理，不代表完整回合评估。

## 实现入口

- `experiments/unified_mappo/model.py`：主模型、Transformer 目标头与 MAPPO 实现。
- `policies/red/learning/unified_mappo_policy.py`：观测、合法动作掩码与运行时接管。
- `scripts/train_c0a_goal_b11_diagnostic.py`：b11–b14 使用的 regret + KL 训练及梯度/接受更新诊断。
- `scripts/c0a_goal_b10_objective.py`：完整合法候选的期望 regret 损失。
- `scripts/c0a_goal_b9_runtime.py`：同状态 fork、随机流恢复、冻结教师闭环续跑。
- `scripts/evaluate_c0a_goal_b14_e01.py`、`verify_c0a_goal_b14_e01.py`：历史 32 种子配对评估和原始产物核验。
- `scripts/train_c0a_position_b15.py`、`scripts/c0a_position_runtime/`：下一课程位置头实现。

脚本中的仿真根路径及默认 R9 权重路径已改为仓库相对路径。历史训练/评估入口仍要求原实验清单、教师轨迹、候选回报数据和前序模型；这些大型实验数据没有随本次代码与模型上传。历史凭据中的原绝对路径和哈希保留用于溯源，不能把它们当作当前克隆上的可用路径。原始实验代码哈希也不等同于做过路径适配的分支脚本哈希。

## b14 的实际证据范围

- E01：32 个不同种子，32 条重新仿真的教师参考；均分 **82.006414**，教师 **80.430170**，配对差 **+1.576244**。
- 总分配对差单侧 95% 下界 **+0.679832**；固定目标分数下界 **+0.567954**；后缀回报比 **1.019598**；非法目标 0。本批数值条件 PASS。
- 每回合只接管一个目标边界，其余动作和续跑由冻结教师控制。这不是完整学生独立部署评估。
- 训练集为 H 50 个、M 14 个状态，没有 L 或 SEARCH 候选。评估的 9 个 SEARCH 状态只有一个合法选项，不能作为 L 学会探索 9500 的证据。
- 评估种子与本轮训练以及 b10–b13 已用种子隔离，但来自经过教师表现筛选的历史验证池。
- 只完成本模型的 32 对评估；用户随后明确授权直接晋级到 C0a_position，豁免该次 128 回合要求。`evaluation_summary.json` 保留当时 formal_promotion=false，授权见 `promotion.json`；没有把 32 改记为 128。
- 独立复核数值完整性通过，审计 WARN 仅涉及范围/保证边界，见 `EXPERIMENT_AUDIT.md`。原生状态以 fork 内存继承，b14 未开启全对象图逐步身份审计。

## R9 与学生的信息条件

本分支忠实保留 competition-platform-env 的当前行为：环境初始化目录包含 9400、9500、9600 的真实坐标；R9 任务层可使用全部初始目录。学生的动态生命周期模式则将 9500 排除出初始合法目标，并施加可见性限制。二者的信息条件不对称；不能称为同样需要先搜索的基线。该行为与原远程 main 当前仅提供 9400/9600 的初始化过滤不同，并已明确记录。

评分汇总见 `evaluation_summary.json`，逐种子值见 `episodes.csv`，条件通过分数见 `course_acceptance_scores.json`。原始大体积仿真产物没有包含在此次提交中。
