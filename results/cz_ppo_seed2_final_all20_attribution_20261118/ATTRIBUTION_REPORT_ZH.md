# CZ + PPO seed2 新版 20 图失分归因

本报告由仿真外侧只读评估器生成。引擎真值仅用于赛后归因，未回灌红方策略。

- 20 图算术均分：91.549341
- 平均失分：8.450659
- 未摧毁目标数：78

## 与历史 91.117 分运行的关系

- 历史运行均分：91.116996
- 本次归因重放均分：91.549341
- 分数完全一致地图：9/20
- 分数不同地图：11/20
- 官方对象地址排序保持不变，因此同 seed 不是位级复现。下面的因果标签只属于本次新鲜重放；历史运行只能做终局目标类型统计，不能倒推出逐单位终止原因。

## 按主因归因

| 主因 | 目标数 | 20 图失分点数之和 |
|---|---:|---:|
| `arrived_but_no_effective_hit` | 6 | 25.797101 |
| `boundary_constrained_no_hit` | 2 | 9.246377 |
| `discovered_but_not_dispatched` | 20 | 32.129117 |
| `dispatched_but_did_not_reach_damage_radius` | 15 | 42.342556 |
| `hit_but_insufficient_damage` | 24 | 44.230567 |
| `undiscovered_or_uncovered` | 11 | 15.267457 |

## 按目标类型归因

| 类型 | 目标数 | 20 图失分点数之和 |
|---|---:|---:|
| dark | 59 | 83.707510 |
| mother | 12 | 68.610013 |
| sentinel | 7 | 16.695652 |

## 分图

| 图 | 得分 | 未摧毁目标 | 缺失权重 | 主因 |
|---|---:|---:|---:|---|
| E01 | 88.636364 | 1 | 5.0 | hit_but_insufficient_damage:1 |
| E02 | 93.181818 | 3 | 3.0 | discovered_but_not_dispatched:1, dispatched_but_did_not_reach_damage_radius:2 |
| E03 | 97.727273 | 1 | 1.0 | undiscovered_or_uncovered:1 |
| E04 | 97.727273 | 1 | 1.0 | discovered_but_not_dispatched:1 |
| E05 | 90.909091 | 4 | 4.0 | discovered_but_not_dispatched:1, hit_but_insufficient_damage:1, undiscovered_or_uncovered:2 |
| E06 | 93.181818 | 3 | 3.0 | discovered_but_not_dispatched:3 |
| M01 | 92.753623 | 4 | 5.0 | arrived_but_no_effective_hit:1, hit_but_insufficient_damage:3 |
| M02 | 94.202899 | 4 | 4.0 | discovered_but_not_dispatched:3, undiscovered_or_uncovered:1 |
| M03 | 94.202899 | 4 | 4.0 | discovered_but_not_dispatched:3, hit_but_insufficient_damage:1 |
| M04 | 91.304348 | 2 | 6.0 | boundary_constrained_no_hit:1, discovered_but_not_dispatched:1 |
| M05 | 85.507246 | 9 | 10.0 | arrived_but_no_effective_hit:1, discovered_but_not_dispatched:2, dispatched_but_did_not_reach_damage_radius:2, hit_but_insufficient_damage:4 |
| M06 | 95.652174 | 2 | 3.0 | discovered_but_not_dispatched:1, dispatched_but_did_not_reach_damage_radius:1 |
| H01 | 92.000000 | 3 | 8.0 | discovered_but_not_dispatched:1, dispatched_but_did_not_reach_damage_radius:1, hit_but_insufficient_damage:1 |
| H02 | 94.000000 | 2 | 6.0 | dispatched_but_did_not_reach_damage_radius:1, undiscovered_or_uncovered:1 |
| H03 | 91.000000 | 4 | 9.0 | boundary_constrained_no_hit:1, dispatched_but_did_not_reach_damage_radius:2, undiscovered_or_uncovered:1 |
| H04 | 89.000000 | 6 | 11.0 | discovered_but_not_dispatched:1, dispatched_but_did_not_reach_damage_radius:2, hit_but_insufficient_damage:3 |
| H05 | 87.000000 | 9 | 13.0 | arrived_but_no_effective_hit:1, discovered_but_not_dispatched:1, hit_but_insufficient_damage:5, undiscovered_or_uncovered:2 |
| H06 | 92.000000 | 4 | 8.0 | arrived_but_no_effective_hit:1, discovered_but_not_dispatched:1, hit_but_insufficient_damage:2 |
| H07 | 83.000000 | 8 | 17.0 | arrived_but_no_effective_hit:1, dispatched_but_did_not_reach_damage_radius:4, hit_but_insufficient_damage:2, undiscovered_or_uncovered:1 |
| H08 | 88.000000 | 4 | 12.0 | arrived_but_no_effective_hit:1, hit_but_insufficient_damage:1, undiscovered_or_uncovered:2 |

## 未摧毁目标明细

| 图 | ID | 类型 | 扣分 | 末血 | 派发角色 | 被拦截/派发 | 命中 | 主因 |
|---|---:|---|---:|---:|---|---:|---:|---|
| E01 | 52 | mother | 11.363636 | 6.000 | H:2,M:4 | 1/6 | 1 | `hit_but_insufficient_damage` |
| E02 | 2884 | dark | 2.272727 | 2.000 | L:4 | 0/4 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| E02 | 2892 | dark | 2.272727 | 2.000 | L:4 | 0/4 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| E02 | 2893 | dark | 2.272727 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| E03 | 2573 | dark | 2.272727 | 2.000 | 无 | 0/0 | 0 | `undiscovered_or_uncovered` |
| E04 | 2886 | dark | 2.272727 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| E05 | 2573 | dark | 2.272727 | 1.000 | L:9 | 0/9 | 1 | `hit_but_insufficient_damage` |
| E05 | 2884 | dark | 2.272727 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| E05 | 2892 | dark | 2.272727 | 2.000 | 无 | 0/0 | 0 | `undiscovered_or_uncovered` |
| E05 | 2893 | dark | 2.272727 | 2.000 | 无 | 0/0 | 0 | `undiscovered_or_uncovered` |
| E06 | 169 | dark | 2.272727 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| E06 | 2892 | dark | 2.272727 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| E06 | 2893 | dark | 2.272727 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| M01 | 2741 | sentinel | 2.898551 | 12.000 | H:1 | 0/1 | 0 | `arrived_but_no_effective_hit` |
| M01 | 2763 | dark | 1.449275 | 1.000 | L:3 | 0/3 | 1 | `hit_but_insufficient_damage` |
| M01 | 3293 | dark | 1.449275 | 1.000 | L:6 | 0/6 | 1 | `hit_but_insufficient_damage` |
| M01 | 3294 | dark | 1.449275 | 1.000 | L:3 | 0/3 | 1 | `hit_but_insufficient_damage` |
| M02 | 2759 | dark | 1.449275 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| M02 | 2764 | dark | 1.449275 | 2.000 | 无 | 0/0 | 0 | `undiscovered_or_uncovered` |
| M02 | 3289 | dark | 1.449275 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| M02 | 3295 | dark | 1.449275 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| M03 | 2762 | dark | 1.449275 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| M03 | 2763 | dark | 1.449275 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| M03 | 3295 | dark | 1.449275 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| M03 | 3296 | dark | 1.449275 | 1.000 | L:4 | 0/4 | 1 | `hit_but_insufficient_damage` |
| M04 | 2764 | dark | 1.449275 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| M04 | 3302 | mother | 7.246377 | 24.000 | H:2,M:3 | 0/5 | 0 | `boundary_constrained_no_hit` |
| M05 | 169 | dark | 1.449275 | 1.000 | L:4 | 0/4 | 1 | `hit_but_insufficient_damage` |
| M05 | 2747 | sentinel | 2.898551 | 12.000 | H:1 | 0/1 | 0 | `arrived_but_no_effective_hit` |
| M05 | 2755 | dark | 1.449275 | 1.000 | L:4 | 0/4 | 1 | `hit_but_insufficient_damage` |
| M05 | 2757 | dark | 1.449275 | 2.000 | L:4 | 0/4 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| M05 | 2762 | dark | 1.449275 | 2.000 | L:4 | 0/4 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| M05 | 2763 | dark | 1.449275 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| M05 | 3289 | dark | 1.449275 | 1.000 | L:4 | 0/4 | 1 | `hit_but_insufficient_damage` |
| M05 | 3294 | dark | 1.449275 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| M05 | 3296 | dark | 1.449275 | 1.000 | L:4 | 0/4 | 1 | `hit_but_insufficient_damage` |
| M06 | 2741 | sentinel | 2.898551 | 12.000 | H:1 | 0/1 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| M06 | 2763 | dark | 1.449275 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| H01 | 2955 | mother | 5.000000 | 24.000 | H:2,M:3 | 0/5 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| H01 | 2959 | sentinel | 2.000000 | 6.000 | H:1,M:5 | 0/6 | 1 | `hit_but_insufficient_damage` |
| H01 | 2985 | dark | 1.000000 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| H02 | 2955 | mother | 5.000000 | 24.000 | H:2,M:4 | 0/6 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| H02 | 3867 | dark | 1.000000 | 2.000 | 无 | 0/0 | 0 | `undiscovered_or_uncovered` |
| H03 | 2955 | mother | 5.000000 | 24.000 | H:2,M:3 | 0/5 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| H03 | 2969 | sentinel | 2.000000 | 12.000 | H:1,M:5 | 0/6 | 0 | `boundary_constrained_no_hit` |
| H03 | 3863 | dark | 1.000000 | 2.000 | 无 | 0/0 | 0 | `undiscovered_or_uncovered` |
| H03 | 3866 | dark | 1.000000 | 2.000 | L:3 | 0/3 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| H04 | 54 | sentinel | 2.000000 | 12.000 | H:1,M:5 | 0/6 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| H04 | 2955 | mother | 5.000000 | 24.000 | H:2,M:3 | 0/5 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| H04 | 2994 | dark | 1.000000 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| H04 | 3857 | dark | 1.000000 | 0.800 | L:4 | 0/4 | 1 | `hit_but_insufficient_damage` |
| H04 | 3862 | dark | 1.000000 | 1.000 | L:8 | 0/8 | 1 | `hit_but_insufficient_damage` |
| H04 | 3867 | dark | 1.000000 | 1.000 | L:4 | 0/4 | 1 | `hit_but_insufficient_damage` |
| H05 | 53 | mother | 5.000000 | 24.000 | H:2,M:3 | 0/5 | 0 | `arrived_but_no_effective_hit` |
| H05 | 2993 | dark | 1.000000 | 1.000 | L:5 | 0/5 | 1 | `hit_but_insufficient_damage` |
| H05 | 2994 | dark | 1.000000 | 1.000 | L:6 | 0/6 | 1 | `hit_but_insufficient_damage` |
| H05 | 2997 | dark | 1.000000 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| H05 | 2998 | dark | 1.000000 | 2.000 | 无 | 0/0 | 0 | `undiscovered_or_uncovered` |
| H05 | 2999 | dark | 1.000000 | 1.000 | L:2 | 0/2 | 1 | `hit_but_insufficient_damage` |
| H05 | 3859 | dark | 1.000000 | 1.000 | L:3 | 0/3 | 1 | `hit_but_insufficient_damage` |
| H05 | 3860 | dark | 1.000000 | 2.000 | 无 | 0/0 | 0 | `undiscovered_or_uncovered` |
| H05 | 3866 | dark | 1.000000 | 1.000 | L:3 | 0/3 | 1 | `hit_but_insufficient_damage` |
| H06 | 2955 | mother | 5.000000 | 24.000 | H:2,M:3 | 0/5 | 0 | `arrived_but_no_effective_hit` |
| H06 | 2989 | dark | 1.000000 | 1.000 | L:5 | 0/5 | 1 | `hit_but_insufficient_damage` |
| H06 | 2999 | dark | 1.000000 | 1.000 | L:7 | 0/7 | 1 | `hit_but_insufficient_damage` |
| H06 | 3863 | dark | 1.000000 | 2.000 | 无 | 0/0 | 0 | `discovered_but_not_dispatched` |
| H07 | 54 | sentinel | 2.000000 | 12.000 | H:1 | 0/1 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| H07 | 169 | dark | 1.000000 | 1.000 | L:8 | 0/8 | 1 | `hit_but_insufficient_damage` |
| H07 | 301 | mother | 5.000000 | 24.000 | H:3,M:4 | 0/7 | 0 | `arrived_but_no_effective_hit` |
| H07 | 2955 | mother | 5.000000 | 24.000 | H:3,M:4 | 0/7 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| H07 | 2987 | dark | 1.000000 | 1.000 | L:4 | 0/4 | 1 | `hit_but_insufficient_damage` |
| H07 | 2991 | dark | 1.000000 | 2.000 | L:4 | 0/4 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| H07 | 2992 | dark | 1.000000 | 2.000 | L:2 | 0/2 | 0 | `dispatched_but_did_not_reach_damage_radius` |
| H07 | 2993 | dark | 1.000000 | 2.000 | 无 | 0/0 | 0 | `undiscovered_or_uncovered` |
| H08 | 52 | mother | 5.000000 | 9.000 | H:2,M:3 | 0/5 | 1 | `hit_but_insufficient_damage` |
| H08 | 168 | dark | 1.000000 | 2.000 | 无 | 0/0 | 0 | `undiscovered_or_uncovered` |
| H08 | 2955 | mother | 5.000000 | 24.000 | H:2,M:3 | 0/5 | 0 | `arrived_but_no_effective_hit` |
| H08 | 2993 | dark | 1.000000 | 2.000 | 无 | 0/0 | 0 | `undiscovered_or_uncovered` |
