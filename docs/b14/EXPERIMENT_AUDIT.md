# b14 E01 Completed Experiment Integrity Audit

**Date:** 2026-09-21  
**Auditor:** GPT-5.6-Sol ultra (fresh same-family reviewer)  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Overall verdict:** **WARN — numeric integrity passes; scope and assurance limitations remain**  
**Evaluation classification:** `simulation_only`, with a frozen-teacher comparator rather than real ground truth

The completed 32-pair result has no numeric-integrity blocker. Independent read-only reconstruction passed for every raw artifact, decision, branch, score, suffix return, CSV row, aggregate, cutoff, and report row. The WARN verdict is driven by the curated upstream pool, the narrow one-boundary simulation scope, disabled full object-graph/private-native RNG fingerprinting, and incomplete transitive hash coverage in the immutable protocol. These limits are disclosed in `RESULTS.md:16,37,46,89-93` and `SCOPE_QUALIFICATIONS.json:5-8`.

## Completed result

- 32 fresh b14 student branches and 32 fresh frozen-teacher branches were verified. No row contains reuse metadata.
- Student mean: **82.0064139660**; teacher mean: **80.4301697531**; paired mean difference: **+1.5762442130**; one-sided 95% lower bound: **+0.6798323592** versus the required **−0.5** (`evaluation_summary.json:17-27`).
- Fixed-target paired difference: **+1.4605034722**; one-sided 95% lower bound: **+0.5679542667** versus **−0.5** (`evaluation_summary.json:28-38`).
- Suffix-return mean ratio: **1.0195976736** versus the required **0.8** (`evaluation_summary.json:39-45`).
- Wins/ties/losses: **16/11/5** (`evaluation_summary.json:59-64`). All course numeric gates pass; formal promotion remains false because this run has 32 pairs rather than the planned 128 (`evaluation_summary.json:15,84-87`; `course_acceptance_scores.json:25-26`).

## Independent evidence reconstruction

The audit script in `audits/completed/trace/independent_verify.py` did not invoke the simulator. It verified:

- 32 ordered unique rows and exact seed agreement with the immutable protocol; zero overlap with the actual b14 training seeds and prior 96 evaluation seeds.
- The fixed b14 checkpoint SHA-256 `4609add14e70a5e25df87ddb3377756c2e45b0d3fa136d4de66406f26ca2be6e` and exact inference for all 32 captures: logits matched bit-for-bit, maximum absolute difference 0, and all 32 actions matched the legal argmax.
- 192/192 declared raw artifact hashes; 32 capture-to-branch-input bindings; 32 branch-input-to-output bindings; 32 request bindings; 32 result-row bindings; and all 32 artifact chronology chains.
- 64/64 branch JSON bindings and 64/64 branch-control contracts, including physical-target locking, immediate SEARCH release, zero non-target boundary changes, and common physical/public-RNG digests.
- 64/64 official E01 scores reconstructed directly from `case_info.json` using fixed target/site/ship weights 5/2/1 and total weight 54; 64/64 fixed-target contributions; and 64/64 official suffix returns from boundary and final objective health.
- 32/32 exact CSV rows, all 32 `RESULTS.md` per-seed rows, the complete aggregate summary, and both conditional cutoff calculations.

The project verifier independently reports 32 exact-logit checks, 32 legal-argmax checks, 192 artifact hashes, 64 branch contracts, six immutable code hashes, 38 source/trace hashes, scalar rebinding, and summary reproduction (`deterministic_checks.json:2-15`). My independently derived aggregate differs from the serialized summary only in four SciPy t-quantile-dependent floats by at most `2.220446049250313e-16` (one ULP). This is numerical roundoff, changes no displayed digit or decision, and means the cross-process replay is numerically exact within `1e-12` rather than bit-exact for those four fields. The full diff is preserved in `audits/completed/trace/INDEPENDENT_VERIFICATION.json`.

All entries in the mid-run transitive dependency snapshot were rehashed after completion and are unchanged. The protocol, checkpoint, six protocol-covered source files, 32 teacher traces, six named source artifacts, verifier, reporter, official scoring sources, target-slot helper, manifest builder, and policy/model modules match their recorded hashes. See `audits/completed/trace/FINAL_DEPENDENCY_HASHES.json`.

## A. Reference provenance: WARN

The teacher checkpoint, scenario, manifest, seed-selection artifact, training protocol, previous protocol, and all 32 teacher traces are hash-bound. All 32 jobs match the manifest target, damage anchor, timestep, executor, teacher trace command, and coordinate. The comparator is consistently a frozen model-generated teacher reference, not ground truth.

The current b14 remaining-seed rule is deterministic and does not inspect b14 outcomes (`scripts/evaluate_c0a_goal_b14_e01.py:203-213`). Its eligible historical manifest retained only `teacher_score > 0` trajectories and assigned targets while iterating in teacher-score order (`/home/ubuntu/yuanlei/cz/competition-platform-env/tools/build_native_guidance_manifest.py:206-222`). The result therefore supports a conditional claim on this curated historical pool, not an outcome-independent random E01 population. `RESULTS.md:16,90` states this correctly.

## B. Official score and normalization: PASS

The official score uses fixed scenario objective identities, initial health, and weights 5/2/1 (`/home/ubuntu/yuanlei/cz/competition-platform-env/scenarios/cases/reward.py:12-18,68-141,144-224`). No denominator is derived from b14 outputs. The fixed-target metric uses the 9400/9600 contribution divided by the same fixed total (`/home/ubuntu/yuanlei/cz/competition-platform-env/tools/train_start_state_option_curriculum.py:806-821`). The suffix reward uses those same weights and telescoping objective-health changes (`scripts/b10_evaluation_runtime/c0a_goal_b9_runtime.py:238-243,342-364`). All 64 branches reproduced independently.

## C. Artifact existence and numeric provenance: PASS

All expected final artifacts exist. The raw evidence, JSON rows, CSV, summary, deterministic checks, conditional cutoffs, and report are mutually bound. Published cutoffs are explicitly conditional on the observed paired standard error and are not presented as universal score thresholds (`course_acceptance_scores.json:5-23`; `RESULTS.md:28-37`).

## D. Dead code and execution reachability: PASS

Active score, suffix, branch-validation, aggregation, verification, and reporting paths all produced bound artifacts. `reuse_episode` remained dormant because `reuse_run` is null, and no result contains `reused_from`. The optional `identity_audit` path remained dormant and is treated as an assurance limitation, not as evidence that a check ran. The verifier and reporter reject optimized Python before assertion-backed checks (`scripts/verify_c0a_goal_b14_e01.py:10-12`; `scripts/report_c0a_goal_b14_e01.py:9-11`).

## E. Scope and seed independence: WARN

The 32 seeds are unique and disjoint from the hashed training batch and prior 96 evaluation seeds. There is no b14-outcome-dependent inclusion, replacement, or stopping. Evidence remains limited to one E01 simulator setting, one b14 goal-boundary decision per episode, and frozen-teacher control of all other actions. Nine episodes were SEARCH-only singleton decisions and supply no target-ranking choice evidence (`evaluation_summary.json:65-82`; `RESULTS.md:43-46`). Cross-model means on different seed sets are not paired improvement evidence. Formal promotion and general/full-team claims are unsupported.

## F. Evaluation classification: PASS — `simulation_only`

This is a paired counterfactual simulator study against a frozen-teacher reference. Supported wording is that b14 passed the stated 32-pair course gates for one goal-boundary intervention on this curated E01 pool. It is not `real_gt`, full-team deployment evidence, a general E01 result, or formal promotion.

## Special contracts

- **Immutable checkpoint and inference: PASS.** The hard-pinned checkpoint matches training provenance; all 32 logits and legal argmax decisions reproduced exactly (`scripts/evaluate_c0a_goal_b14_e01.py:127-140,187-213`; `scripts/verify_c0a_goal_b14_e01.py:50-65`).
- **Fork state and RNG assurance: WARN.** Both candidates inherit one live process through `os.fork`; physical state and public Python/NumPy/Torch RNG digests agree for all 32 pairs (`scripts/b10_evaluation_runtime/c0a_goal_b9_runtime.py:265-287`). Full object-graph `identity_audit` was not enabled (`:246-255,288-292`), so no claim should extend to independently verified private/native RNG state. No mismatch was found.
- **Branch control and official suffix reward: PASS.** All 64 artifacts satisfy target-lock/SEARCH-release semantics, teacher closed-loop step counts, common fork digests, fixed weights, and endpoint reward reconstruction (`scripts/evaluate_c0a_goal_b14_e01.py:93-109`).
- **Fresh teacher results: PASS.** All 32 teacher branches were produced in the completed run, bind to their job targets, and contain no reuse provenance.
- **Transitive immutability: WARN in protocol, verified by audit receipt.** The immutable protocol hashes six direct files (`protocol.json:116-123`) but omits important transitive inference and scoring sources. The independent before/after snapshot found no change; future protocols should include them directly.

## Claim disposition

- **Supported, with WARN qualifiers:** the stated 32-pair course-gate result and its exact raw score evidence for this one-boundary, curated-pool, simulation-only protocol.
- **Unsupported:** formal promotion, a general E01 conclusion, full-team student performance, comparison of b14 versus earlier model means as a paired improvement, unconditional random-test performance, or proof of every private/native RNG component.

The machine-readable verdict is `EXPERIMENT_AUDIT.json`; raw independent checks and final dependency hashes are under `audits/completed/trace/`.
