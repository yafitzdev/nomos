# Nomos production promotion report

Date: 2026-08-24  
Decision: **promote the contrast-replay encoder with the FP32 ONNX coprocessor runtime**

## Plain-language result

Nomos is now a CPU-runnable tool-calling coprocessor. An agent gives it the
current objective, observable state, and a registry of legal tools. Nomos:

1. ranks the three most useful tools;
2. can return a fresh page that excludes rejected tools;
3. abstains when confidence is unsafe;
4. verifies a proposed call deterministically;
5. returns schema-derived repair guidance without accepting or executing an
   invalid call.

The original assisted evaluation found that a local Qwen3-0.6B agent completed
2/8 tasks with the full registry and 8/8 with the complete Nomos runtime. That
comparison measured the whole system, not the encoder alone. A subsequent
one-shot ablation removes repair feedback, retries, and replacement candidates:
the weak agent completes 0/8 with all 34 tools, 4/8 with the old encoder's raw
top three, and 6/8 with the promoted encoder's raw top three. This isolates a
real encoder contribution without attributing the complete runtime's safety
net to the learned weights.

## Deploy this

- Encoder checkpoint: `artifacts/nomos_bge_contrast_replay_ablation`
- CPU package: `artifacts/nomos_bge_contrast_replay_onnx_fp32_ablation`
- Runtime entry point: `fitz_tool.dense_selector.DenseToolRanker`
- Policy shell: `fitz_tool.coprocessor.coprocessor_response`
- External contract runner: `tools/run_router_contract.py --mode dense`

The ONNX package is numerically equivalent to the training checkpoint at the
embedding level and preserves every measured frozen ranking metric. The int8
package is rejected: it introduced three top-three misses on the promotion
suite.

## Frozen promotion evidence

The final suite was authored after model selection and never entered training.
It contains four unseen registry families, opaque tool and capability labels,
new schemas and descriptions, eight unseen multi-step workflows, and 34 tools
per registry: 17 relevant tools plus 17 unrelated distractors.

| Metric | Result |
|---|---:|
| Promotion states | 152 |
| Recall@1 | 97.37% |
| Recall@3 | 100.00% |
| Top-three misses | 0 |
| Strong-agent completion, full registry | 8/8 |
| Strong-agent completion, Nomos | 8/8 |
| Weak-agent completion, full registry | 2/8 |
| Weak-agent completion, Nomos | 8/8 |
| Strong prompt tokens, full / Nomos | 92,216 / 12,875 |
| Weak prompt tokens, full / Nomos | 104,085 / 13,949 |
| Tool-description reduction | 91.2% |

Actual local tools read Markdown and Python files, inspect AST structure, parse
CSV records, compare evidence, update state, and finalize only after the
required sequence succeeds. A rank hit alone is not counted as task success.

## Safety, recovery, and confidence gates

The frozen 1,500-state coprocessor test reports:

| Gate | Result |
|---|---:|
| Raw synthetic Recall@3 | 99.78% |
| Lowest known capability Recall@3 | 97.06% |
| Illegal recommendation rate | 0.00% |
| Deterministic validation accuracy | 100.00% (376 calls) |
| False accepts | 0 / 341 invalid calls |
| False rejects | 0 / 35 valid calls |
| Recovery Recall@3 | 99.30% |
| No-repeat recovery | 100.00% (359 states) |
| No-suitable-tool abstention recall | 97.29% |
| False abstention on suitable states | 0.66% |
| Test selective risk | 0.78% |

The invalid-call suite covers unknown and illegal IDs, repeated candidates,
missing, extra, and wrong-typed arguments, modality mismatches, unmet state
prerequisites, and disallowed side effects. In executed promotion traces,
Nomos emitted no illegal calls. The weak-agent Nomos condition had three schema
rejections and one wrong-tool execution; all were recovered and all eight tasks
completed.

## Old-model comparison

The old mixed encoder also completes 8/8 weak-agent promotion tasks when placed
inside the new coprocessor runtime. This matters: the large improvement from
2/8 to 8/8 comes primarily from candidate filtering, deterministic validation,
repair guidance, and recovery—not from claiming magical new weights.

The refined encoder still contributes measurable value:

| Weak-agent Nomos condition | Old mixed encoder | Promoted encoder |
|---|---:|---:|
| Completed tasks | 8/8 | 8/8 |
| Tool-call events | 45 | 42 |
| Invalid calls | 4 | 3 |
| Wrong-tool executions | 3 | 1 |
| Prompt tokens | 15,067 | 13,949 |

On ToolRet's 60-query customized holdout, top-three recall rises from 68.3% for
the old mixed encoder and 73.3% for the previous curriculum checkpoint to
80.0% for the promoted encoder. Promotion Recall@1 rises from 95.39% to 97.37%.

The old generic holdout moves from 95.6% for the old mixed checkpoint to 94.2%
for the promoted model, a 1.4-point regression and inside the two-point gate.

## Raw end-to-end ablation

"Raw" here means one attempt per workflow stage. The agent sees either the
entire legal registry or the encoder's first three candidates. Deterministic
validation is used only as the scoring and execution gate. It does not return
repair instructions, retry a malformed call, exclude a wrong tool, or request
another candidate page.

| Agent and condition | Tasks | Mean stage completion | Tool selection | Invalid calls | Wrong executions |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B, all 34 tools, raw | 0/8 | 4.3% | 40.0% | 3 | 5 |
| Qwen3-0.6B, old encoder top-3, raw | 4/8 | 63.2% | 93.3% | 2 | 2 |
| Qwen3-0.6B, promoted encoder top-3, raw | 6/8 | 77.5% | 97.0% | 1 | 1 |
| Qwen3-0.6B, promoted complete coprocessor | 7/8 | 87.5% | 97.4% | 3 | 1 |
| DeepSeek-v4-flash, all 34 tools, raw | 7/8 | 87.5% | 96.9% | 0 | 1 |
| DeepSeek-v4-flash, old encoder top-3, raw | 8/8 | 100.0% | 100.0% | 0 | 0 |
| DeepSeek-v4-flash, promoted encoder top-3, raw | 8/8 | 100.0% | 100.0% | 0 | 0 |
| DeepSeek-v4-flash, promoted complete coprocessor | 8/8 | 100.0% | 100.0% | 0 | 0 |

The correct tool was present somewhere in every raw top-three set for both
encoders. The weak agent nevertheless made fewer wrong selections with the
promoted ordering. The strong agent saturates this eight-task suite with either
encoder, so it provides evidence that raw top-three filtering works but cannot
resolve the old/new difference. The weak-agent result is directional rather
than definitive because eight workflows are a small sample.

For the strong agent, full-registry raw prompts averaged 2,377 tokens per tool
call attempt; promoted raw top-three prompts averaged 339, an 85.7% reduction.
The complete coprocessor added no benefit in this strong-agent run because all
raw top-three calls were already valid and correct. In the fresh weak-agent
run, it recovered one additional task over raw top-three, not all six gains.

Local run outputs (intentionally ignored by git with the other generated runs):

- `runs/nomos_contrast_replay_raw_ablation_weak_8_v1.json`
- `runs/nomos_mixed_raw_ablation_weak_8_v1.json`
- `runs/nomos_contrast_replay_raw_ablation_strong_8_v1.json`
- `runs/nomos_mixed_raw_ablation_strong_8_v1.json`

## CPU deployment

Measured on Windows 11 with a 12-logical-core AMD CPU:

| Metric | FP32 ONNX package |
|---|---:|
| Artifact size | 127.6 MiB |
| Model load | 0.60 s |
| Load-time RSS increase | 181.9 MiB |
| Final RSS after registry caches | 486 MiB |
| Warm ranking p95, pools 10 / 30 / 100 | 244 / 422 / 222 ms |
| Cold registry+query, pools 10 / 30 / 100 | 0.59 / 1.66 / 4.45 s |
| Deterministic verifier p50 / p95 | 1.16 / 1.68 ms |
| Deterministic verifier throughput | ~815 calls/s |

Registry embeddings are cached, so cold indexing is paid once when a registry
is registered. Warm online routing stays below half a second in this test.

## Data decision

No valid corpus was deleted or overwritten. Fitz-Sage-specific data remains
quarantined. The broad 30k DeepSeek slice is preserved as an experiment but is
not in the winning training path because its model regressed.

The winning path is a curriculum:

1. retained compact ToolRet breadth training;
2. 3,400 train-only state-transition rows;
3. replay of those transitions plus 3,400 new opaque-tool contrast rows.

The v4 contrast matrix balances every target capability and ten confusable
operation families, with opaque capability labels, ambiguous verbs, stale or
completed history, recovery pages, and pools of 10, 14, and 17. This focused
3.4k addition improved the learning curve; scaling broad paraphrases to 60k or
100k was therefore not justified.

## Remaining weaknesses

- ToolRet Recall@3 is 80%, not universal retrieval perfection. Arbitrary public
  tool descriptions remain harder than registries that provide useful metadata.
- Confidence misses about 2.7% of no-suitable-tool cases and falsely abstains on
  about 0.7% of suitable frozen cases.
- A 100-tool registry takes about 4.5 seconds to index on the measured CPU,
  although this is cached and online query latency is much lower.
- Dynamic int8 quantization is not quality-safe for this checkpoint. FP32 ONNX
  is the promoted package.
- The executed suite is deliberately local and read-only. External services,
  authentication failures, timeouts, and irreversible writes require adapter-
  specific integration tests before enabling those effects.

These limitations do not invalidate promotion: every stated gate passes, the
sealed executable sessions show a large real-agent benefit, and the remaining
risks are explicit rather than hidden behind synthetic recall.
