# Skill 功能与输入输出接口

本文档描述 `/home/amax/ry/competition/skill_generation/skills` 下当前任务栈的职责、纯 Skill API、运行时适配关系和合法信息边界。

## 总体层级

```text
PlanSpec
  初始部署、初始目标、初始波次、搜索 cell、资源框架
        |
        v
MissionReplanSkill
  launch/hold/supplement/cancel/retarget 全局仲裁
        |
        +------------------+-------------------+
        |                  |                   |
        v                  v                   v
HStrikePackageSkill  MStrikePackageSkill  L Search/Response Skills
        |                  |                   |
        +------------------+-------------------+
                           |
                           v
                 V12-R3 / V13 Local Policy
                    LEFT / STOP / RIGHT
```

支撑层：`TargetDamageBeliefSkill` 为 Mission 和 H/M/L package 提供统一目标毁伤 belief。

## 权限边界

| 决策 | 唯一所有者 |
|---|---|
| 初始部署、初始目标和波次 | PlanSpec |
| 执行期 launch/hold/cancel/语义换靶审批 | MissionReplanSkill |
| 同 target ID 坐标刷新 | Execution Layer |
| H 包少量联合机动 | HStrikePackageSkill |
| M/L 单弹机动 | V12-R3 / V13 |
| 每枚弹最终 ACC_Z 命令 | UnifiedAttackMissileAgent |
| 目标毁伤 belief | TargetDamageBeliefSkill |
| 真实命中诊断 | Sidecar，仅离线分析 |

纯 skill 不直接读取 `competition_envs`、`envengine`、simulator 或 sidecar 真值。环境 observation、PlanSpec binding 和合法 InfoCenter 信息只允许由 adapter 转换成冻结 DTO。

---

# 1. TargetDamageBeliefSkill

路径：`target_damage_belief/src/target_damage_belief_skill/belief.py`

## 功能

- 维护目标 HP 下界、期望、上界和 projected HP。
- 维护 delivery、receipt 和在飞 commitment。
- 支持 commitment 原子换靶转移。
- 使用 event ID 保证 delivery/hit 幂等。
- 不读取环境真实剩余 HP。

## 输入

```python
TargetPriorEvidence(
    target_id: int,
    target_type: int,
    initial_health: float,
    discovered_step: int = 0,
    source: str = "initial_catalogue",
)
```

```python
DeliveryEvidence(
    missile_id: int,
    missile_type: int,
    target_id: int,
    step: int,
    last_stage: int,
    minimum_target_distance_metres: float | None,
    disappeared: bool = True,
    satellite_active: bool = False,
    event_id: str | None = None,
)
```

未来可选合法回执：

```python
HitReceipt(
    missile_id: int,
    missile_type: int,
    target_id: int,
    step: int,
    applied_damage: float,
    target_destroyed: bool = False,
    event_id: str | None = None,
)
```

## 方法

```python
register_target(evidence)
commit_missile(missile_id, missile_type, target_id, satellite_expected=False)
transfer_missile(missile_id, target_id)
release_missile(missile_id)
report_delivery(evidence)
report_hit_receipt(receipt)
confirm_destroyed(target_id)
estimate(target_id)
snapshot()
reset()
```

## 输出

```python
TargetHealthBelief(
    target_id,
    target_type,
    initial_health,
    lower_hp,
    expected_hp,
    upper_hp,
    projected_expected_hp,
    confidence,
    confirmed_destroyed,
    committed_missiles,
    delivery_missiles,
    receipt_missiles,
    delivered_expected_damage,
    confidence_source,
)
```

`projected_expected_hp` 不能单独作为确认摧毁证据。

---

# 2. MissionReplanSkill

路径：`mission_replan/src/mission_replan_skill/controller.py`

## 功能

- 作为 PlanSpec 的事件驱动执行期 overlay。
- 分阶段释放 L response package。
- 根据统一 belief 补包或取消未发射包。
- 对高置信满足目标阻止 M followup/backup 发射。
- 对完成目标的存活未投送 H/M/L package 做一次性整包换靶。
- 不输出 LEFT/STOP/RIGHT，不接入 R9。

## 输入

```python
ResponsePackagePlan(
    package_id: str,
    response_group_index: int,
    sequence_index: int,
    member_ids: tuple[int, ...],
)
```

```python
MissionTargetBelief(
    response_group_index,
    target_id,
    expected_hp,
    projected_expected_hp,
    confidence,
    confirmed_destroyed,
    initial_health=0.0,
    delivered_expected_damage=0.0,
    confidence_source="heuristic_delivery",
)
```

## 配置

```python
MissionReplanConfig(
    required_discoveries,
    release_delay_steps=20,
    delivery_timeout_steps=700,
    subsequent_release_interval_steps=250,
    hard_deadline_steps=900,
    initial_release_fraction=0.50,
    minimum_initial_members=3,
    destroyed_confidence_threshold=0.75,
)
```

当前 adapter 使用 Easy/Hard 首轮 timeout 600/800，后续 250/300，初始 L response 至少释放 50%。

## 输出

```python
MissionReplanDecision(
    launch_package_ids,
    launch_member_ids,
    held_package_ids,
    held_member_ids,
    cancelled_package_ids,
    cancelled_member_ids,
    target_ids_by_response_index,
)
```

Adapter 还负责 package 换靶、60-step 目标冷却、100-step 25% 重分配上限、PlanSpec goal slot 转换和 belief commitment 原子迁移。

Trace：`mission_replan.jsonl`

---

# 3. HStrikePackageSkill

路径：`three_missile_joint/src/three_missile_joint_skill/controller.py`

`ThreeMissileJointSkill` 是固定三成员兼容包装；核心类为 `HStrikePackageSkill`。

## 适用成员

```text
2～6 枚 H / 21000
同 target_id / target_type
计划发射时间差在配置窗口内
```

单 H 直接使用 local network。

## 输入

```python
JointMemberSnapshot(
    entity_id,
    missile_type,
    target_id,
    position: Vector2,
    velocity: Vector2,
    goal_position: Vector2,
    threats: tuple[JointThreat, ...],
    baseline_action,
    launch_step=0,
    alive=True,
)
```

可选任务上下文：

```python
JointMissionContext(
    target_expected_hp,
    target_confidence,
    target_confirmed_destroyed,
    authorized_fallback_target_ids,
)
```

## 功能

- 每枚 H 先独立调用 local network。
- 判断 24000 threat owner。
- 计算包内最小间距和 ETA spread。
- mission-aware 模式保护 primary、禁止 LEFT/RIGHT 直接反转。
- 极少量情况下覆盖一枚非 owner 的动作。
- 输出角色、包状态和 RetargetRequest。
- 成员损失后降级，不跨包补员。

## 输出

```python
JointDecision(
    actions,
    baseline_actions,
    overridden_entity_id,
    reason,
    threat_owners,
    baseline_min_separation_metres,
    selected_min_separation_metres,
    eta_spread_seconds,
    status,
    retarget_request,
)
```

动作 ABI：`0=LEFT, 1=STOP, 2=RIGHT`。

Trace：`joint_decisions.jsonl`

---

# 4. MStrikePackageSkill

路径：`m_strike_package/src/m_strike_package_skill/controller.py`

## 适用成员

```text
2～6 枚 M / 21001
同 target_id
目标类型 9400 或 9600
计划发射跨度不超过 10 steps
```

单 M 直接使用 local network。

## 输入

```python
MMemberSnapshot(
    entity_id,
    missile_type,
    target_id,
    target_type,
    planned_release_step,
    alive,
    launched,
    delivered,
    baseline_action,
)
```

```python
MStrikeMissionContext(
    belief: TargetHealthBelief,
    current_step,
    last_accepted_release_step,
    delivery_timeout_steps,
    authorized_fallback_target_ids=(),
)
```

## 功能

- 维护 M 包角色和毁伤承诺。
- 保护 PRIMARY/SUPPORT 初始必要火力。
- 为未发射 followup/backup/reserve 生成 StopRequest。
- 目标缺伤且 timeout 后生成 SupplementRequest。
- 目标完成后生成 RetargetRequest。
- 不修改 local network maneuver。

## 输出

```python
MStrikePackageDecision(
    status,
    motion_overrides=(),
    stop_request,
    supplement_request,
    retarget_request,
)
```

Trace：`m_strike_status.jsonl`

---

# 5. LSearchCoordinationSkill

路径：`l_search_coordination/src/l_search_coordination_skill/controller.py`

## 输入

```python
SearchMemberSnapshot(
    entity_id,
    search_cell,
    alive,
    launched,
    report_step,
    discovered_target_ids=(),
    baseline_action=1,
)
```

## 功能

- 验证 4×8 search cell ownership。
- 聚合合法暗巢发现 ID。
- 判断 required discovery quorum。
- 成员死亡/报告过期时生成 SearchReassignmentRequest。
- 超过 deadline 且无新发现时生成 SatelliteRequest。
- 不修改 waypoint 或 local network maneuver。

## 输出

```python
SearchCoordinationDecision(
    actions,
    status,
    reassignment_requests,
    satellite_request,
)
```

Trace：`l_search_status.jsonl`

---

# 6. LResponsePackageSkill

路径：`l_response_package/src/l_response_package_skill/controller.py`

## 适用成员

```text
2～4 枚 L / 21002
同一个已合法发现的 9500
```

## 输入

```python
ResponseMemberSnapshot(
    entity_id,
    target_id,
    alive,
    launched,
    delivered,
    launch_step,
    estimated_eta_seconds,
    baseline_action=1,
)
```

并传入：

```python
belief: TargetHealthBelief
authorized_fallback_target_ids: tuple[int, ...]
```

## 功能

- 维护暗巢攻击包状态。
- 输出 StopCommitmentRequest、SupplementRequest、RetargetRequest。
- 计算 ETA spread。
- 不修改 local network maneuver。

## 输出

```python
ResponsePackageDecision(
    actions,
    status,
    stop_commitment_request,
    supplement_request,
    retarget_request,
)
```

Trace：`l_response_status.jsonl`

---

# 7. Local Network Runtime

原 V13：

```text
Runtime: /home/amax/ry/competition/jmy_ppo/contest
Checkpoint: local_avoid_ppo_v13_gate_aligned_s3_best.pt
Gate: mission_v12
```

新版 V12-R3：

```text
Runtime: /home/amax/ry/competition/jmy_ppo/contest-jmy
Checkpoint: local_avoid_set_attention_v12_r3_timely_arrival.pt
Gate: none
```

统一接口：

```python
encode_motion_observation(observation, **context)
prepare_action_mask(encoded, action_mask)
select_action(encoded, action_mask) -> 0 | 1 | 2
reset_episode()
```

---

# 8. Runtime 模式

| `--lower-policy` | 功能 |
|---|---|
| `local-all` | H/M/L 全部直接 local network |
| `hybrid` | 旧 H 单弹规则，M/L local network |
| `joint-h-v13` | H package + local network |
| `joint-m-local` | M package status/request + local network |
| `joint-l-local` | L search/response status/request + local network |
| `joint-hl-local` | H + L package/coordination |
| `joint-hml-local` | H + M + L 全部 package/coordination |

| `--mission-replan` | 功能 |
|---|---|
| `off` | 不过滤 PlanSpec 任务决策 |
| `l-response` | 仅 L response staged release |
| `all-packages` | L staged release、M stop、H/M/L 整包换靶 |

推荐完整模式：

```bash
--lower-policy joint-hml-local \
--joint-rule-profile mission-aware \
--mission-replan all-packages
```

---

# 9. 合法信息边界

在线 Skill 可使用：

- PlanSpec 初始公开目标目录。
- InfoCenter 接受后的合法 target track。
- discovery events。
- 隔离后的本弹 observation。
- 通信/卫星合法融合轨迹。
- 自身发射、stage、位置和速度历史。
- TargetDamageBeliefSkill 输出。
- Package status/request。

在线 Skill 禁止使用：

- `TrainingEnv._get_observation()` 原始全局实体表。
- simulator 对象或蓝方私有状态。
- 未发现的 9500。
- `target_hit_relation`。
- `summary.json`、`hit_events.jsonl`、`analysis_summary.json`。

Sidecar 仅用于离线调参和归因分析。

---

# 10. 赛后诊断 Sidecar

官方 `summary.json` 保持原结构。

新增：

```text
hit_events.jsonl
analysis_summary.json
```

两者明确标记为赛后分析，在线 policy 不可访问。

---

# 11. 当前未完成能力

- L search cell 重分配请求尚未实际修改 waypoint。
- L search satellite request 尚未进入完整预算仲裁。
- M SupplementRequest 尚未从全局未发射库存分配新成员。
- 整包换靶已支持目标完成后的 cleanup retarget，但尚未做复杂多包全局优化。
- 速度控制存在 native 命令，但当前红方公共 Action ABI 未开放。
- H package 主动机动仍处于实验阶段，跨地图不保证非劣。

R9 保持官方 baseline，不使用上述 Mission/Package 扩展。
