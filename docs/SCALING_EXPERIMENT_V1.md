# Nomos targeted scaling experiment v1

Date: 2026-08-24
Decision: **do not promote a new checkpoint; do not generate a second cohort**

## Plain-language result

The experiment produced the requested 25,000-row targeted cohort and trained
all six controlled branches. The new data is useful: it increases ranking
margins, improves several old and sealed retrieval measurements, and exposes a
clear curriculum effect. It does not, however, produce a better agentic
coprocessor. Every shortlisted new checkpoint makes the local Qwen3-0.6B agent
complete fewer raw sessions than the current model.

The current contrast-replay encoder therefore remains promoted. Its existing
FP32 ONNX package remains the production CPU artifact. No ONNX export was made
from a rejected checkpoint.

## Generation accounting

| Item | Count |
|---|---:|
| Requested accepted rows | 25,000 |
| Primary requests | 1,647 |
| Total API requests | 1,695 |
| Rows returned by DeepSeek | 26,935 |
| Accepted rows | 25,000 |
| Rejected/quarantined rows | 1,306 |
| Retry requests | 34 |
| Batch splits | 7 |
| Malformed responses | 41 |
| Timeouts | 0 |
| Replacement rounds | 3 |
| Teacher fallbacks | 0 |
| Prompt / completion tokens | 5,234,484 / 2,728,672 |
| Total tokens | 7,963,156 |
| Generation time | 277.2 seconds |

Assignment matrix SHA-256:
`797c50a576c80d4444f541863a5a9653ae3cb1bb6587732d7ec3d85df5565c54`.
Accepted cohort SHA-256:
`ccabef958b880adc2e921af27666957faedbc8311d04066dd389b242cb86ce8a`.

The 25,000 accepted rows are unique by decision-state ID, normalized question,
semantic signature, assignment, matrix cell, and source lineage. Independent
contract validation accepted 25,000/25,000 rows. A stratified 230-row quality
audit covered all 23 scenario families and passed 230/230 rows.

Rejected rows were retained with machine-readable reasons. The main causes
were duplicate questions (1,115), incorrect completed-step counts (153),
duplicate semantic signatures (25), weak recovery wording (8), padding (3),
and text length (2).

## Matrix and holdout

The frozen matrix allocates 25,000 exact slots across 23 difficult scenario
families. It independently balances:

- pools of 10, 17, 34, 50, and 100 tools;
- text, structured-data, document, code, image, audio, and mixed modalities;
- five side-effect policies, four session positions, and three history lengths;
- initial and recovery routing, prior failures, completed neighbors, and stale
  or conflicting history;
- opaque versus semantic capabilities, five registry-description styles, and
  five argument-schema styles;
- four wording styles, four initial target positions, and five history
  transitions;
- all requested close distinctions, including exact versus broad search,
  listing versus metadata, search versus read, planning versus execution,
  evidence inspection/comparison/assessment, requirement update/finalization,
  provenance, schema/record, document structure/pages, and code symbol/source;
- no-suitable-tool, missing-prerequisite, illegal-attractive, modality, and
  side-effect cases.

The post-scaling holdout was frozen before training. It contains 756 states:
720 answer-present and 36 no-suitable-tool cases, evenly distributed across
34-, 50-, and 100-tool registries. It uses four unseen registry styles, three
unseen schema styles, unseen opaque names, and 21 held-out workflows. Question,
template, registry, scenario, and source overlap with training are all zero.

## Training accounting

All branches start from `artifacts/nomos_bge_contrast_replay_ablation`. The
encoder loss requires a positive tool, so the 1,000 no-suitable rows are
retained for confidence work and evaluation but do not create false triplets.
Thus 24,000 of the 25,000 rows enter gradient training in the full-data
branches. This is explicit in every checkpoint manifest.

| Checkpoint | New input | Gradient rows | Replay | Duration | Training loss |
|---|---:|---:|---:|---:|---:|
| scaling5k | 5,000 | 4,783 | 0 | 82 s | 0.002645 |
| scaling10k | 10,000 | 9,565 | 0 | 161 s | 0.001963 |
| scaling25k | 25,000 | 24,000 | 0 | 384 s | 0.001196 |
| scaling25k-replay | 25,000 | 24,000 | 10,896 | 429 s | 0.004851 |
| staged broad | 12,500 | 11,500 | 0 | 183 s | 0.001376 |
| staged hard | 12,500 | 12,500 | broad stage | 205 s | 0.000851 |

The full branches have approximately 60,081 unique examples in their total
lineage: the current model's approximately 35,081 unique records plus 25,000
new unique decision states. Exactly 24,000 new rows affect encoder gradients;
the other 1,000 encode abstention outcomes.

## Complete retrieval comparison

`Frozen` is the 903 answer-present test states from the 10k frozen synthetic
corpus. `Generic` is the old 1,000-row holdout. `Sealed` uses the production
multiview ranker over 720 answer-present states. All values are percentages
except margin.

| Variant | Frozen R@3 | Generic R@3 | ToolRet R@3 | Final R@3 | Promotion R@1 / R@3 | Sealed R@1 / R@2 / R@3 | Sealed margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current baseline | 99.78 | 94.20 | 80.00 | 98.65 | 97.37 / 100 | 100 / 100 / 100 | 0.2193 |
| 5k | 99.78 | 94.10 | 81.67 | 98.65 | 97.37 / 100 | 100 / 100 / 100 | 0.2391 |
| 10k | 99.78 | 94.20 | 78.33 | 98.65 | 97.37 / 100 | 100 / 100 / 100 | 0.2498 |
| 25k | 99.56 | 94.30 | 81.67 | 99.32 | 97.37 / 100 | 100 / 100 / 100 | 0.2590 |
| 25k + replay | 100.00 | 95.00 | 76.67 | 100.00 | 97.37 / 100 | 100 / 100 / 100 | 0.2648 |
| Staged | 99.78 | 94.10 | 81.67 | 100.00 | 97.37 / 100 | 100 / 100 / 100 | 0.2616 |

The single-view diagnostic is intentionally harsher than production
multiview ranking. On that view, sealed R@3 is 97.08% for baseline and replay,
97.36% for plain 25k, and 97.50% for 5k, 10k, and staged. The common weakness
is the fully opaque bounded-change-preflight family. Candidate multiviews
recover every miss, which is why runtime sealed R@3 is 100% for every branch.

## Confidence and abstention

Calibration is fitted on the pre-existing frozen validation split, never on
the post-scaling sealed answers. `Old absent` contains 221 no-suitable-tool
test states. `Sealed absent` contains 36.

| Variant | Old absent recall | Old false abstain | Sealed absent recall | Sealed false abstain |
|---|---:|---:|---:|---:|
| Current baseline | 97.29 | 0.66 | 100.00 | 5.42 |
| 5k | 95.93 | 0.22 | 100.00 | 3.06 |
| 10k | 95.48 | 0.11 | 100.00 | 2.08 |
| 25k | 93.67 | 0.00 | 100.00 | 1.67 |
| 25k + replay | 96.38 | 0.11 | 100.00 | 1.25 |
| Staged | 95.02 | 0.00 | 100.00 | 1.53 |

The new branches are more willing to answer and therefore falsely abstain less,
but most lose recall on old unsupported requests. Replay has the best new
tradeoff, yet still does not preserve the baseline's old absent recall.

## Raw Qwen3-0.6B contribution

After retrieval screening, 5k, replay, and staged were the non-dominated
shortlist. Plain 10k and 25k were not subjected to costly end-to-end execution
because another branch matched or beat each on the selection metrics.

| Encoder, raw top-3 | Completed | Stage completion | Selection | Invalid calls | Wrong executions | Oracle visible |
|---|---:|---:|---:|---:|---:|---:|
| Current baseline | **23/32** | 77.71 | 95.38 | 6 | 3 | 100 |
| 5k | 19/32 | 72.69 | 93.60 | 8 | 5 | 100 |
| 25k + replay | 17/32 | 66.17 | 92.11 | 10 | 5 | 100 |
| Staged | 17/32 | 70.96 | 92.80 | 9 | 6 | 100 |

Every new model keeps the correct tool visible in every attempt. The regression
comes from candidate ordering and the weak agent's downstream choice, not a
top-three retrieval miss. Training increased embedding margins but changed the
context/order in ways that made the small agent choose worse tools more often.

## Complete coprocessor contribution

| Encoder, complete top-3 | Completed | Stage completion | Selection | Invalid calls | Wrong executions |
|---|---:|---:|---:|---:|---:|
| Current baseline | **26/32** | 86.88 | 92.11 | 11 | 6 |
| 5k | 26/32 | 88.51 | 92.36 | 14 | 7 |
| 25k + replay | 26/32 | 89.14 | 90.80 | 16 | 10 |
| Staged | 25/32 | 86.43 | 91.19 | 16 | 9 |

Validation, schema-derived repair, and candidate recovery rescue seven sessions
for 5k and nine for replay, but neither becomes better than the current system.
The validator itself is unchanged by encoder training: legal filtering,
schema/prerequisite/modality/policy checks, and no-repeat recovery remain
deterministic and covered by the repository test suite.

## Top-k ablation

The retained baseline was executed raw with one, two, and three visible tools.

| Visible tools | Completed | Oracle visible | Prompt tokens/attempt | Description reduction |
|---|---:|---:|---:|---:|
| Top 1 | 18/32 | 96.55 | 189 | 97.06 |
| Top 2 | 18/32 | 98.36 | 259 | 94.12 |
| Top 3 | **23/32** | 100.00 | 328 | 91.18 |

Top three remains the correct default. Smaller pages save more tokens but hide
needed tools and reduce completed sessions.

## Promotion and scaling decision

No new checkpoint qualifies. The desired raw target was at least 28/32 and the
current baseline is 23/32; the best new branch reaches only 19/32. Complete
coprocessor performance does not exceed 26/32, and new branches require more
validation/recovery work. ToolRet improves by at most one query out of 60 and
is not monotonic with scale.

A second 25,000-row cohort is therefore not justified. The 5k-to-25k learning
curve is flat or negative on the decisive agent metric. Future work should not
add more paraphrases. It should target the interface between ranking order and
weak-agent choice: order-aware distillation, listwise objectives, explicit
top-three set/order supervision, and harder malformed-argument examples.

## Retained deployment

- PyTorch: `artifacts/nomos_bge_contrast_replay_ablation`
- FP32 ONNX: `artifacts/nomos_bge_contrast_replay_onnx_fp32_ablation`
- Size: 127.6 MiB
- Existing measured warm ranking p95: 244 / 422 / 222 ms for pools 10 / 30 / 100
- Existing measured cold indexing: 0.59 / 1.66 / 4.45 s
- Existing validator p50 / p95: 1.16 / 1.68 ms

No CPU benchmark was rerun for rejected checkpoints because all use the same
architecture and none passed the quality gate required for export.

## Reproduction

The source workflows are:

- `tools/materialize_scaling_matrix_v1.py`
- `tools/freeze_post_scaling_holdout_v1.py`
- `tools/generate_scaling_deepseek_v1.py`
- `tools/validate_scaling_cohort_v1.py`
- `tools/audit_scaling_cohort_v1.py`
- `tools/select_scaling_curriculum_v1.py`
- `tools/train_dense_triplet_router.py`
- `tools/evaluate_dense_router.py`
- `tools/evaluate_dense_confidence.py`
- `tools/audit_real_session_retrieval.py`
- `tools/evaluate_real_agent_sessions.py`

Generated data, run reports, traces, model weights, and API responses remain
ignored. The committed matrix and holdout manifests contain the hashes needed
to reproduce the boundaries without publishing secrets or generated text.
