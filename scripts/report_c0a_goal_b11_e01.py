#!/usr/bin/env python3
"""Report completed b11 E01 evaluation and the paired, fixed-pool b10 comparison."""
from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import t
import torch

from evaluate_c0a_goal_b11_e01 import BASE, file_sha256, outcomes, summarize, write_json


def verify_training_provenance(directory):
    evaluation = json.loads((directory / 'protocol.json').read_text())
    b11_path = BASE / 'runs/c0a_goal_b11_diagnostics_20260921/b11/protocol.json'
    b10_path = BASE / 'runs/c0a_goal_b10_regret_64_20260920/training/protocol.json'
    b11 = json.loads(b11_path.read_text())
    b10 = json.loads(b10_path.read_text())
    dataset = Path(b11['candidate_batch'])
    digest = file_sha256(dataset)
    if not (digest == b11['data_sha256'] == b10['data_sha256']):
        raise ValueError('b11/b10 actual training data differs')
    if Path(b10['candidate_batch']) != dataset:
        raise ValueError('training dataset paths differ')
    if file_sha256(Path(b11['source_checkpoint'])) != b11['source_sha256']:
        raise ValueError('b11 source checkpoint differs')
    baseline = json.loads((BASE / 'validations/b10_e01_32_20260921_v2/protocol.json').read_text())
    if b11['source_sha256'] != baseline['checkpoint_sha256']:
        raise ValueError('b11 source is not evaluated b10')
    batch = torch.load(dataset, map_location='cpu', weights_only=True)
    seeds = sorted({int(item['seed']) for item in batch['items']})
    if seeds != b11['validation']['training_seeds'] or seeds != evaluation['training_seeds']:
        raise ValueError('actual b11 training seeds differ from evaluation contract')
    overlap = sorted(set(seeds) & set(evaluation['seeds']))
    if overlap:
        raise ValueError('b11 training/evaluation seed overlap')
    summary_path = b11_path.parent / 'training_summary.json'
    training_summary = json.loads(summary_path.read_text())
    checkpoint_digest = file_sha256(Path(evaluation['checkpoint']))
    if checkpoint_digest != evaluation['checkpoint_sha256'] or checkpoint_digest != training_summary['output_sha256']:
        raise ValueError('b11 evaluated checkpoint changed')
    receipt = {
        'status': 'PASS', 'training_states': len(batch['items']),
        'training_unique_seeds': len(seeds), 'training_seeds': seeds,
        'evaluation_seeds': evaluation['seeds'], 'overlap': overlap,
        'b11_and_b10_same_dataset_sha256': digest,
        'b11_source_is_evaluated_b10': True,
        'checkpoint_sha256': checkpoint_digest,
        'b11_training_protocol_sha256': file_sha256(b11_path),
        'b10_training_protocol_sha256': file_sha256(b10_path),
        'b11_training_summary_sha256': file_sha256(summary_path),
        'b11_training_protocol': str(b11_path),
        'b11_training_summary': str(summary_path),
        'candidate_batch': str(dataset),
        'scope': 'actual b11 continuation dataset seed disjointness; historical validation pool, not a never-used test set',
    }
    write_json(directory / 'training_provenance.json', receipt)
    return receipt


def main():
    directory = BASE / 'validations/b11_e01_32_20260921'
    baseline = BASE / 'validations/b10_e01_32_20260921_v2'
    verify_training_provenance(directory)
    protocol = json.loads((directory / 'protocol.json').read_text())
    rows = json.loads((directory / 'episodes.json').read_text())
    old_rows = json.loads((baseline / 'episodes.json').read_text())
    summary = json.loads((directory / 'evaluation_summary.json').read_text())
    if summarize(rows, protocol) != summary:
        raise ValueError('evaluation summary does not reproduce')
    if len(rows) != 32 or len(old_rows) != 32:
        raise ValueError('incomplete comparison')
    if [r['seed'] for r in rows] != [r['seed'] for r in old_rows]:
        raise ValueError('seed order differs')
    comparison = []
    for row, old in zip(rows, old_rows):
        # These fields must agree across the two separately simulated runs.
        for key in ['order', 'seed', 'start_state_id', 'decision_step', 'executor_id',
                    'teacher_target_id', 'physical_state_sha256', 'public_rng_sha256',
                    'legal_target_count', 'teacher_target_legal_in_student_mask',
                    'teacher_end_step']:
            if row[key] != old[key]:
                raise ValueError(f'paired protocol/state mismatch: {row["seed"]} {key}')
        for key in ['teacher_score', 'teacher_fixed_target_score', 'teacher_suffix_return']:
            if abs(row[key] - old[key]) > 1e-9:
                raise ValueError(f'teacher replay mismatch: {row["seed"]} {key}')
        if 'reused_from' in row:
            raise ValueError('b11 report expects all episodes freshly simulated')
        for path, sha in row['artifact_sha256'].items():
            if file_sha256(Path(path)) != sha:
                raise ValueError(f'episode artifact mismatch: {path}')
        comparison.append({
            'order': row['order'], 'seed': row['seed'],
            'b10_target_id': old['student_target_id'], 'b11_target_id': row['student_target_id'],
            'teacher_target_id': row['teacher_target_id'],
            'b10_score': old['student_score'], 'b11_score': row['student_score'],
            'teacher_score': row['teacher_score'],
            'b11_minus_b10': row['student_score'] - old['student_score'],
            'b11_minus_teacher': row['student_score'] - row['teacher_score'],
        })
    deltas = [r['b11_minus_b10'] for r in comparison]
    delta_mean = statistics.fmean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(32)
    half = float(t.ppf(.975, 31)) * se
    paired = {
        'status': 'complete', 'pairs': 32,
        'b10_score_mean': statistics.fmean(r['b10_score'] for r in comparison),
        'b11_score_mean': summary['score_mean'],
        'mean_delta': delta_mean,
        'relative_change_percent': 100 * delta_mean / statistics.fmean(r['b10_score'] for r in comparison),
        'paired_delta_two_sided_95_ci': [delta_mean - half, delta_mean + half],
        'outcomes': outcomes(deltas),
        'changed_targets': sum(r['b10_target_id'] != r['b11_target_id'] for r in comparison),
        'fresh_b11_student_episodes': 32, 'fresh_teacher_references': 32,
        'all_teacher_scores_reproduce_b10': True,
        'all_initial_states_and_public_rng_match_b10': True,
        'scope': 'same fixed historical validation pool, disjoint from training; no new-test/generalization claim',
        'baseline_episodes_sha256': file_sha256(baseline / 'episodes.json'),
        'candidate_episodes_sha256': file_sha256(directory / 'episodes.json'),
        'report_code_sha256': file_sha256(Path(__file__)),
    }
    write_json(directory / 'b10_comparison.json', paired)
    with (directory / 'b10_comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    gates = summary['gates']
    score = gates['e01_score_noninferiority']
    fixed = gates['fixed_target_score_noninferiority']
    suffix = gates['suffix_return_ratio']
    verdict = summary['course_numeric_gates_decision']
    outcome = summary['score_outcomes']
    report = [
        '# b11：E01 独立 32 种子课程评估', '',
        f'生成时间：{datetime.now(timezone.utc).isoformat()}。', '',
        f'**b11 平均分 {summary["score_mean"]:.6f}，标准差 {summary["score_std"]:.6f}；本次 32 回合课程数值门槛 {verdict}，不正式晋级。**', '',
        '完成 32 个唯一学生回合及 32 个同状态、同随机流教师参考分支，全部重新仿真。',
        f'教师均分 {score["teacher_mean"]:.6f}；b11 对教师配对均差 {score["mean_delta"]:+.6f} 分，胜/平/负为 {outcome["wins"]}/{outcome["ties"]}/{outcome["losses"]}。',
        f'学生均分双侧 95% t 区间为 [{summary["score_mean_95_ci"][0]:.6f}, {summary["score_mean_95_ci"][1]:.6f}]；最低/最高分 {summary["score_min"]:.6f}/{summary["score_max"]:.6f}。', '',
        '## 课程通过条件', '',
        '| 判据 | 本次值 | 要求 | 结果 |', '|---|---:|---:|---|',
        f'| E01 总分配对差单侧 95% 下界 | {score["one_sided_95_lower"]:.6f} | ≥ −0.5 | {"PASS" if score["pass"] else "FAIL"} |',
        f'| 9400/9600 固定目标分数配对差单侧 95% 下界 | {fixed["one_sided_95_lower"]:.6f} | ≥ −0.5 | {"PASS" if fixed["pass"] else "FAIL"} |',
        f'| 正式后缀回报均值比（学生/教师） | {suffix["ratio"]:.9f} | ≥ 0.8 | {"PASS" if suffix["pass"] else "FAIL"} |',
        f'| 非法目标选择 | {gates["target_legality"]["illegal_samples"]}/32 | 0 | {"PASS" if gates["target_legality"]["pass"] else "FAIL"} |',
        '| 种子唯一、训练池无交集、同状态和随机流配对 | 32/32 | 全部满足 | PASS |', '',
        '配对非劣门槛使用单侧 95% Student t 下界、df=31。未通过表示尚未证明达到预设非劣界，不等于已经证明显著劣于教师。正式晋级仍要求原方案的 128 个配对回合；本次按用户要求固定为 32 回合。', '',
        '## 与 b10 的同种子配对比较', '',
        '| 模型 | 平均得分 |', '|---|---:|',
        f'| b10 | {paired["b10_score_mean"]:.6f} |',
        f'| b11 | {paired["b11_score_mean"]:.6f} |',
        f'| b11 − b10 | {delta_mean:+.6f} |', '',
        f'配对分差双侧 95% t 区间 [{delta_mean-half:.6f}, {delta_mean+half:.6f}]；相对均分变化 {paired["relative_change_percent"]:+.4f}%。',
        f'b11 对 b10 胜/平/负：{paired["outcomes"]["wins"]}/{paired["outcomes"]["ties"]}/{paired["outcomes"]["losses"]}；{paired["changed_targets"]}/32 个状态改变贪心目标。所有教师参考得分和起始物理状态/随机流均复现 b10。', '',
        '## 动作集合分组', '',
        '| 状态类型 | 回合数 | b11 均分 | 教师均分 | 均差 |', '|---|---:|---:|---:|---:|',
    ]
    for name, group in summary['decision_groups'].items():
        report.append(f'| {name} | {group["episodes"]} | {group["student_mean"]:.6f} | {group["teacher_mean"]:.6f} | {group["mean_delta"]:+.6f} |')
    report += ['', 'SEARCH-only 状态没有目标选择空间，不单独作为目标选优能力的证据。', '',
               '## 逐种子原始结果', '',
               '| seed | b10 目标 | b11 目标 | 教师目标 | b10 得分 | b11 得分 | 教师得分 | b11−b10 |',
               '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparison:
        report.append(f'| {row["seed"]} | {row["b10_target_id"]} | {row["b11_target_id"]} | {row["teacher_target_id"]} | {row["b10_score"]:.6f} | {row["b11_score"]:.6f} | {row["teacher_score"]:.6f} | {row["b11_minus_b10"]:+.6f} |')
    report += ['', '目标 ID −100 表示 SEARCH。', '', '## 范围与证据', '',
               '- 每个种子仅由学生接管一个 C0a_goal 目标边界，其余边界动作、教师内部状态及续跑规则与 b10 相同；不是学生整队独立控制。',
               '- 同一组预先固定的 32 个历史验证种子，与训练池无交集；另从 b11 实际训练数据复核了种子集合和 b10/b11 相同数据哈希，见 training_provenance.json。它不是全新的测试集，不能据此声称泛化改善。',
               '- 教师使用原生目标，SEARCH-only 状态下教师目标可能不在学生观察掩码中；没有扩充学生合法集合。',
               '- 只更换学生检查点，未修改 KL、模型参数或课程门槛。',
               f'- 检查点：`{protocol["checkpoint"]}`。',
               f'- 检查点 SHA256：`{protocol["checkpoint_sha256"]}`。',
               '- 原始结果：episodes.json / episodes.csv；汇总：evaluation_summary.json；配对比较：b10_comparison.json / b10_comparison.csv；协议：protocol.json。',
               '- 每回合 capture、branch_input、branch_output、decision 及两个分支 JSON 的路径和哈希保存在 episodes/*/result.json。', '']
    (directory / 'RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'summary': summary, 'comparison': paired}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
