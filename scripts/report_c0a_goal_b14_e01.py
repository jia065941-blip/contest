#!/usr/bin/env python3
"""Report b14 training and new32 E01 gates, including conditional mean-score cutoffs."""
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
from evaluate_c0a_goal_b14_e01 import BASE,file_sha256,summarize,write_json

def main():
    if sys.flags.optimize:
        raise RuntimeError("Integrity checks require Python assertions; rerun without -O/-OO")
    directory=BASE/'validations/b14_e01_new32_20260921'
    protocol=json.loads((directory/'protocol.json').read_text())
    rows=json.loads((directory/'episodes.json').read_text())
    summary=json.loads((directory/'evaluation_summary.json').read_text())
    assert summarize(rows,protocol)==summary
    assert len(rows)==len({r['seed'] for r in rows})==32
    assert not set(protocol['seeds'])&set(protocol['training_seeds'])
    assert not set(protocol['seeds'])&set(protocol['excluded_previous_evaluation_seeds'])
    assert all('reused_from' not in r for r in rows)
    checks=json.loads((directory/'deterministic_checks.json').read_text())
    assert checks['status']=='PASS' and checks['exact_logits_checks']==32
    train=BASE/'runs/c0a_goal_b14_continuation_20260921/training'
    training=json.loads((train/'training_summary.json').read_text())
    train_protocol=json.loads((train/'protocol.json').read_text())
    assert training['output_sha256']==protocol['checkpoint_sha256']
    assert training['attempted_updates']==training['accepted_updates']==48
    assert not training['changed_frozen_tensors']
    gates=summary['gates'];cutoffs={}
    for name in ['e01_score_noninferiority','fixed_target_score_noninferiority']:
        gate=gates[name]
        cutoff=gate['teacher_mean']+gate['required_lower']+gate['critical_value']*gate['standard_error']
        margin=gate['student_mean']-cutoff
        assert abs(margin-(gate['one_sided_95_lower']-gate['required_lower']))<1e-10
        assert (margin>=0)==gate['pass']
        cutoffs[name]={'actual_student_mean':gate['student_mean'],'teacher_mean':gate['teacher_mean'],
                       'paired_standard_error':gate['standard_error'],'required_student_mean_given_observed_standard_error':cutoff,
                       'score_margin':margin,'pass':gate['pass']}
    acceptance={'status':'complete','episodes':32,'course_numeric_gates_decision':summary['course_numeric_gates_decision'],
                'cutoff_formula':'teacher_mean - 0.5 + t(0.95,31) * paired_standard_error',
                'interpretation':'conditional on this sample paired standard error; not a universal score cutoff',
                'mean_score_cutoffs':cutoffs,'formal_promotion':False,'formal_promotion_min_episodes':128,
                'source_summary_sha256':file_sha256(directory/'evaluation_summary.json'),
                'report_code_sha256':file_sha256(Path(__file__))}
    write_json(directory/'course_acceptance_scores.json',acceptance)
    score=gates['e01_score_noninferiority'];fixed=gates['fixed_target_score_noninferiority'];suffix=gates['suffix_return_ratio'];outcome=summary['score_outcomes']
    lines=['# b14：续训及不同 32 种子 E01 课程评估','',f'生成时间：{datetime.now(timezone.utc).isoformat()}。','',
           f'**b14 平均得分 {summary["score_mean"]:.6f}，标准差 {summary["score_std"]:.6f}；本次课程数值判定 {summary["course_numeric_gates_decision"]}。**','',
           '## 续训结果','',
           '从已评估的 b13 检查点继续 48 次全批更新，沿用 64 个训练状态、960 个完整合法候选回报。固定 b9 锚点、λ=0.01、新 Adam 和固定学习率 2e-5，与 b13 续训规则相同。',
           f'尝试/保留更新 48/48，三种回滚均为 0（硬验收门未启用）；目标头 {training["changed_target_tensors"]} 个张量改变，非目标张量改变为 0。',
           f'训练集预期 regret 从 {training["before"]["expected_regret_score_points"]:.6f} 降到 {training["after"]["expected_regret_score_points"]:.6f} 分；最优目标平均概率从 {100*training["before"]["optimal_probability_mean"]:.6f}% 到 {100*training["after"]["optimal_probability_mean"]:.6f}%。',
           f'末步加权 KL/regret 梯度范数比 {training["gradient_ratio_last"]:.6f}，方向余弦 {training["gradient_cosine_last"]:.6f}。这些均为训练集诊断。','',
           '## 32 回合评估结果','',
           '32 个学生回合及 32 个教师参考分支全部重新仿真，将历史 128 种子清单按种子编号排序，排除此前 b10/b11/b12/b13 使用过的 96 个种子后取剩余 32 个，保留原先分配的目标；与此前评估种子及实际训练种子均无交集。本轮取剩余种子的规则不依赖 b14 的评估结果；上游历史清单保留 teacher_score>0 的轨迹，并按教师得分排序分配目标，因此这是经过教师表现筛选的历史验证池。',
           f'学生均分 {summary["score_mean"]:.6f}，教师均分 {score["teacher_mean"]:.6f}；配对均差 {score["mean_delta"]:+.6f} 分，胜/平/负 {outcome["wins"]}/{outcome["ties"]}/{outcome["losses"]}。',
           f'学生均分双侧 95% t 区间 [{summary["score_mean_95_ci"][0]:.6f}, {summary["score_mean_95_ci"][1]:.6f}]，最低/最高分 {summary["score_min"]:.6f}/{summary["score_max"]:.6f}。','',
           '| 课程条件 | 本次值 | 要求 | 判定 |','|---|---:|---:|---|',
           f'| 总分配对差单侧 95% 下界 | {score["one_sided_95_lower"]:.6f} | ≥ −0.5 | {"PASS" if score["pass"] else "FAIL"} |',
           f'| 固定目标分数配对差单侧 95% 下界 | {fixed["one_sided_95_lower"]:.6f} | ≥ −0.5 | {"PASS" if fixed["pass"] else "FAIL"} |',
           f'| 后缀回报均值比 | {suffix["ratio"]:.9f} | ≥ 0.8 | {"PASS" if suffix["pass"] else "FAIL"} |',
           f'| 非法目标选择 | {gates["target_legality"]["illegal_samples"]}/32 | 0 | {"PASS" if gates["target_legality"]["pass"] else "FAIL"} |',
           '| 唯一种子、训练池无交集、同状态/随机流配对 | 32/32 | 全部满足 | PASS |','',
           '## 本批数据对应的通过分数','',
           '课程不是固定的绝对分数线。给定本批教师均分和配对标准误，总分或固定目标均分须达到：教师均分 − 0.5 + t(0.95,31) × 配对标准误。','',
           '| 分数项目 | 实际学生均分 | 本批条件通过均分 | 距离门槛 |','|---|---:|---:|---:|']
    for name,label in [('e01_score_noninferiority','E01 总分'),('fixed_target_score_noninferiority','9400/9600 固定目标')]:
        c=cutoffs[name];lines.append(f'| {label} | {c["actual_student_mean"]:.6f} | {c["required_student_mean_given_observed_standard_error"]:.6f} | {c["score_margin"]:+.6f} |')
    lines+=['','以上分数线仅在本批配对标准误保持不变时成立；更换策略或样本会改变标准误，不能当作通用及格分。各项条件需同时满足。正式晋级另要求原方案的 128 个配对回合，本次严格保留 32 回合，不合并不同模型的评估样本。','',
            '## 动作集合分组','', '| 分组 | 回合数 | 学生均分 | 教师均分 | 均差 |','|---|---:|---:|---:|---:|']
    for name,g in summary['decision_groups'].items():
        lines.append(f'| {name} | {g["episodes"]} | {g["student_mean"]:.6f} | {g["teacher_mean"]:.6f} | {g["mean_delta"]:+.6f} |')
    lines+=['','SEARCH-only 回合没有目标选择空间，不能单独作为目标选优能力的证据。','',
            '## 逐种子原始得分','', '| seed | b14 目标 | 教师目标 | b14 得分 | 教师得分 | 配对分差 |','|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["seed"]} | {r["student_target_id"]} | {r["teacher_target_id"]} | {r["student_score"]:.6f} | {r["teacher_score"]:.6f} | {r["student_score"]-r["teacher_score"]:+.6f} |')
    lines+=['','目标 −100 表示 SEARCH。','', '## 范围与复核','',
            '- 本轮只改变学生检查点和 32 个种子；仍是每个回合只接管一个目标边界，其余动作和续跑由冻结教师执行。',
            '- 种子与训练及此前三组评估分离，但来自保留正教师得分、按教师得分安排目标的历史验证池，不是从未用于任何历史模型的新随机测试集。',
            '- 本轮种子与上一轮不同，不将两个模型的均分差解释为配对模型提升。',
            '- 32 次目标 logits 与合法 argmax 精确重建，192 项原始证据哈希、64 个分支契约及汇总重算通过，见 deterministic_checks.json。',
            '- 学生和教师继承同一 live fork 状态，并核对物理状态及 Python/NumPy/Torch 公共随机流指纹；本轮未启用完整对象图的 identity_audit，不能将这些指纹检查解释为已独立验证全部私有/native 随机流。',
            '- 机器可读课程分数线：course_acceptance_scores.json；全部逐回合数据：episodes.json / episodes.csv；原始协议：protocol.json。',
            f'- b14 检查点：`{protocol["checkpoint"]}`。',f'- 检查点 SHA256：`{protocol["checkpoint_sha256"]}`。','']
    recovery_path=directory/'INTERRUPTION_RECOVERY.json'
    if recovery_path.exists():
        recovery=json.loads(recovery_path.read_text())
        lines+=['## 中断恢复记录','',
                f'运行中原评估进程及其仿真进程消失，具体外部原因未确认。保留并核验了 {recovery["kept_complete_episodes"]} 个已完成回合，归档未完成的部分轨迹后按原协议恢复其余回合。',
                '中断前的未完成分支不计入得分；最终仍为 32 个唯一、完整的 b14 学生回合及 32 个教师参考。未按中途成绩筛选、替换种子或改变检查点。详见 INTERRUPTION_RECOVERY.json 与 interrupted_attempts/。','']
    (directory/'RESULTS.md').write_text('\n'.join(lines))
    write_json(BASE/'runs/c0a_goal_b14_continuation_20260921/completion.json',{'status':'complete','training_summary':str(train/'training_summary.json'),'evaluation_summary':str(directory/'evaluation_summary.json'),'report':str(directory/'RESULTS.md'),'course_numeric_gates_decision':summary['course_numeric_gates_decision'],'formal_promotion':False})
    print(json.dumps({'score_mean':summary['score_mean'],'score_std':summary['score_std'],'gates':gates,'acceptance_scores':acceptance},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
