# Nomos balanced salvage and full-replay experiment

Date: 2026-08-24  
Decision: **retain the current promoted checkpoint**

## Plain-language result

The earlier scaling experiment did not replay all of the data that created the
current model. This follow-up fixes that mistake. It reconstructs all 35,081
unique old training states, measures which of the 25,000 new rows are actually
hard for the current model, and trains on the full old lineage plus a balanced
5,100-row hard subset.

The correction prevents the worst forgetting, but it still does not make the
small Qwen agent better. The conservative finalist completes 20 of 32 raw
agent sessions; the current model completes 23. No new checkpoint is promoted.

## What entered training

The old unique lineage is reconstructed as:

| Source | Trainable rows |
|---|---:|
| Generic portability | 20,000 |
| Agentic v2 | 4,185 |
| ToolRet breadth | 4,096 |
| State transitions | 3,400 |
| Opaque contrasts | 3,400 |
| **Old total** | **35,081** |

The promoted model scored all 24,000 answer-present rows in the new cohort.
It already ranked the answer first on 23,670 rows (98.625%). There were only
330 non-top-one rows: 281 at rank two, 25 at rank three, and 24 below rank
three. The remaining 1,000 cohort rows are no-suitable-tool states and cannot
train a positive-only encoder loss.

The salvage rule keeps mistakes first and then the smallest
positive-minus-best-negative margins, capped at 300 rows per canonical target.
It retains all 330 top-one mistakes across the 17 positive target families and
produces exactly 5,100 rows. This prevents large easy families from
dominating merely because they have more generated examples.

## Controlled training

The corrected full-replay branch starts from the unchanged promoted checkpoint
and sees 40,181 unique trainable states for one epoch at learning rate `2e-6`.
It uses in-batch multiple-negative ranking loss (MNRL).

Preliminary full-specialist and 25% interpolation sweeps drifted on the sealed
unseen-style set. Linear interpolation at 10% specialist / 90% promoted
baseline preserves most of the new signal. The 10% MNRL blend is the corrected
finalist taken through downstream evaluation.

## Retrieval comparison

All values are recall percentages. `Sealed` is the single-view 720-state
post-scaling holdout; the production multiview audits are reported separately.

| Encoder | Frozen R@3 | Generic R@3 | Sealed R@1 / R@3 | ToolRet R@1 / R@3 | Final R@1 / R@3 | Promotion R@1 / R@3 |
|---|---:|---:|---:|---:|---:|---:|
| Current promoted | 99.78 | 94.20 | 95.97 / 97.08 | 61.67 / 80.00 | 91.22 / 98.65 | 97.37 / 100 |
| Corrected 10% MNRL blend | 99.67 | 94.40 | 96.25 / 97.50 | 63.33 / 80.00 | 92.57 / 98.65 | 97.37 / 100 |

The MNRL blend is the retrieval winner: it gains one ToolRet top-one query,
improves sealed recall, and preserves promotion top-three. This is not enough
to call it a better coprocessor.

## Confidence and abstention

The candidate reuses the production calibration fitted before the sealed
holdout. On 221 old no-suitable states, abstention recall rises from 97.29% to
97.74%, while false abstention on old answer-present states rises from 0.66%
to 0.89%. On the sealed set it exactly preserves 100% absent recall and 5.42%
false abstention, with a slightly lower Brier score (0.0676 versus 0.0685).
This is a small mixed change, not a promotion-level gain.

## Decisive raw-agent comparison

The raw test exposes exactly three ranked tools to local Qwen3-0.6B. It has no
repair feedback and no candidate recovery. The correct tool remains visible in
100% of attempts for every encoder.

| Encoder | Completed | Stage completion | Selection | Invalid calls | Wrong executions |
|---|---:|---:|---:|---:|---:|
| Current promoted | **23/32** | **77.71%** | **95.38%** | **6** | **3** |
| Corrected 10% MNRL blend | 20/32 | 70.89% | 92.62% | 7 | 5 |

The new model does not hide the right tool. It reorders the same useful set in
ways that make the weak agent choose a wrong tool more often. New failures
appear in SDK-symbol resolution, catalog provenance, implementation
discrepancy, and release-readiness sessions.

## Conclusion and next data refinement

This closes the replay ambiguity: insufficient replay was a real flaw in the
old experiment, but it was not the whole cause of the regression. The present
25,000-row cohort mostly teaches semantic answer-versus-negative separation.
That objective raises margins and benchmark recall while failing to teach
which top-three ordering helps a weak decoder act correctly.

The next cohort should not simply add more paraphrases. Its matrix needs a new
agent-transfer layer containing:

- the same legal candidate set under controlled top-three permutations;
- weak-agent choice and argument validity for each permutation;
- cases where the semantic top-one is unchanged but ordering changes success;
- malformed-argument and no-call outcomes;
- full workflow stage coverage, especially finalization, document pages, and
  structured-record search, which were absent as positive targets here;
- training-only workflows and registries distinct from every frozen holdout.

An order-aware/listwise objective can then optimize demonstrated agent utility
while retaining semantic routing and full old replay. Until that data exists,
the production artifacts remain:

- PyTorch: `artifacts/nomos_bge_contrast_replay_ablation`
- ONNX FP32: `artifacts/nomos_bge_contrast_replay_onnx_fp32_ablation`

## Reproduction

The committed source workflow is:

```text
python -m tools.audit_scaling_salvage --base-model artifacts/nomos_bge_contrast_replay_ablation --input data/generated/nomos_scaling_targeted_v1_25000.jsonl --scores-output runs/nomos_scaling_targeted_v1_hardness.jsonl --salvage-output data/generated/nomos_scaling_targeted_v1_balanced_hard.jsonl --manifest-output runs/nomos_scaling_targeted_v1_balanced_hard.manifest.json --max-per-capability 300 --device cuda
```

The corrected training and conservative interpolation are:

```text
python -m tools.train_dense_router --base-model artifacts/nomos_bge_contrast_replay_ablation --input data/generated/nomos_generic_portability_100000.jsonl --input data/generated/nomos_agentic_v2_frozen_10000.jsonl --input data/generated/nomos_toolret_agentic_v1_4096.jsonl --input data/generated/nomos_agentic_transitions_v3_3400.jsonl --input data/generated/nomos_agentic_contrasts_v4_3400.jsonl --input data/generated/nomos_scaling_targeted_v1_balanced_hard.jsonl --output artifacts/nomos_bge_scaling_balanced_fullreplay_mnrl_v2 --limit-per-input 20000 --epochs 1 --batch-size 64 --learning-rate 2e-6 --seed 20260824
python -m tools.interpolate_dense_models --base artifacts/nomos_bge_contrast_replay_ablation --specialist artifacts/nomos_bge_scaling_balanced_fullreplay_mnrl_v2 --specialist-weight 0.10 --output artifacts/nomos_bge_scaling_balanced_mnrl_soup10_v2
```

Generated rows, score reports, traces, and model artifacts remain ignored.
