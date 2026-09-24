# PPO Evasion Skill

这是一个与上层目标分配解耦的单弹规避技能。R9 上层仅负责部署、目标分配和发射；技能只输出横向 `-20G / 0 / +20G` 三个动作。

## 约束

- Actor 只读取本弹状态和 `detect_from == 本弹 ID` 的拦截弹航迹。
- 来袭几何统一投影到本弹局部坐标，并显式编码预计最近交会距离（CPA）。
- V3 只把 `CPA <= 15 km`、`TGO <= 120 s` 且闭合速度不低于 `50 m/s` 的航迹视为入射威胁，避免被其他导弹附近的过航拦截弹污染奖励。
- V4 对每条本机经验生成左右镜像经验并交换动作标签，抑制策略退化成“始终向同一侧转”；镜像样本使用独立轨迹 ID，不会串接不同导弹的 GAE。
- 不读取其他红弹状态、通信群信息、团队得分或全局 Critic。
- 技能不会请求卫星，因此执行阶段不存在导弹间航迹共享。
- 所有导弹可以共享同一套网络参数，但每枚弹独立推理、独立轨迹、独立奖励。
- 当前平台只有高性能弹能自行探测拦截弹。因此训练基准默认移除中低性能弹，避免把“无传感器输入”误当作 PPO 能解决的问题。

## 指标

一次来袭事件从首次出现有效本机拦截弹航迹开始。连续 3 步脱离威胁判为成功；在拦截弹最近距离 2 km 内消失判为失败。最终门槛要求滚动 2000 次事件中：

1. 至少完成 1000 次事件；
2. 躲避率不低于 99%；
3. 95% Wilson 置信下界不低于 99%。

训练和评估会输出 `EVASION_METRICS` JSON。只有独立评估满足门槛，才能宣称达到 99%。

## 入口

```bash
python tools/train_evasion_ppo.py --rounds 100
python tools/prepare_evasion_v3_checkpoint.py \
  models/ppo_evasion_v2_final20_e01_best.pt \
  models/ppo_evasion_v3_final20_e01.pt
python tools/prepare_evasion_v4_checkpoint.py \
  models/ppo_evasion_v3_final20_e01.pt \
  models/ppo_evasion_v4_final20_e01.pt
python tools/evaluate_evasion_skill.py \
  --model models/ppo_evasion_final20_e01_best.pt \
  --seed-count 20 --rounds-per-seed 2
```

## 局部四阶段训练

`LocalAvoidEnv` 是独立于 R9 上层和完整仿真 SDK 的中层规避训练环境，对应《人族远征_分层决策与中层PPO规避训练方案》的第一阶段：

- Stage 0：无自爆虫，先学会稳定直飞；
- Stage 1：单个正面来袭自爆虫；
- Stage 2：单个任意方向来袭自爆虫；
- Stage 3：双弹同时来袭、左右夹击或同向延迟 `5--15` 步来袭；
- Actor 使用本机局部观测，Critic 额外使用训练期特权状态；
- 动作为 `LEFT / STRAIGHT / RIGHT`，只输出相对上层航向的残差机动；
- 速度、转弯率、碰撞半径、课程长度和全部奖励权重均可在配置中调整。

```bash
python tools/train_local_avoid_ppo.py --smoke --device cpu
python tools/train_local_avoid_ppo.py --device cuda
python tools/evaluate_local_avoid_ppo.py \
  --model models/local_avoid_ppo_final.pt --episodes 2000 --stages 2
```

V2 从 Stage 1 checkpoint 继续，把任意方向威胁拆成三个渐进阶段，并启用危险交会筛选、累积转向、CPA 风险势函数奖励和 Stage 0/1 混合回放：

```bash
python tools/train_local_avoid_ppo_v2.py --device cuda
```

Stage 3 从 Stage 2 最优点继续，默认混合回放 `30%` Stage 2 和 `70%`
Stage 3。训练入口支持只运行指定课程，并可选左右镜像一致性损失：

```bash
python tools/train_local_avoid_ppo_v4_s3.py \
  --resume models/local_avoid_ppo_v3_stage2_best.pt \
  --model models/local_avoid_ppo_v4_stage3.pt \
  --device cuda

python tools/evaluate_local_avoid_ppo.py \
  --model models/local_avoid_ppo_v4_stage3_best.pt \
  --episodes 2000 --stages 2,3 --seed 20269901 --device cuda
```

当前 Stage 3 选择模型是 `models/local_avoid_ppo_stage3_selected.pt`，内容与 V9
恢复性微调 best checkpoint 相同。训练先混合正常 S3 与同侧双弹 hard case，再冻结
V4 的观测归一化统计，并以 `90%` Stage 2 回放做单次恢复 update。固定 seed
`20269901` 的 2000 局结果如下：

| 模型 | Stage 2 躲避率 | Stage 3 躲避率 | Stage 3 命中率 |
|---|---:|---:|---:|
| V4 baseline | `98.68%` | `79.35%` | `19.35%` |
| V9 selected | `98.14%` | `83.30%` | `16.00%` |

V9 在维持 Stage 2 超过 `98%` 的同时，将 Stage 3 提升约 `3.95` 个百分点。结果见
`results/local_avoid_ppo/evaluation_v9_stage2_recovery_best_2000.json`。当前仍不能宣称
Stage 3 达到 `99%`。

固定 seed `20271001` 的 3000 局分模式诊断显示：同时来袭从 `74.88%` 提升到
`78.60%`，夹击从 `91.20%` 提升到 `93.65%`，延迟接续从 `76.01%` 提升到
`79.63%`，同侧双弹从 `74.21%` 提升到 `78.71%`。代价是 Stage 3 平均额外航程
从约 `14.1 km` 增至 `17.7 km`。诊断结果见
`results/local_avoid_ppo/stage3_diagnosis_v9_selected_3000.json`。

该环境用于先验证规避技能的可学习性，不替代完整仿真验收。达到局部环境指标后，还需接回真实观测、动力学和拦截弹模型做迁移训练与固定种子评估。

## 当前验收结果

当前选择模型为 `models/ppo_evasion_selected.pt`，内容与 V3 final checkpoint 相同。V4 的左右镜像增强在独立评估中没有超过 V3，因此未被选中。

V3 final 在 B0、全新固定 seed 401--418 上完成 1080 次入射事件：成功 1025 次、失败 55 次，躲避率 `94.91%`，95% Wilson 下界 `93.43%`。结果见 `results/evasion_ppo/eval_v3_final_heldout_1000.json`。

这还没有达到 99% 目标。训练中出现的单局 100% 不能替代大样本验收；在指标达到门槛前，模型只应称为当前候选，不应称为 99% 躲避模型。
