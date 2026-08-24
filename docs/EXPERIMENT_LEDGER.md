# Nomos experiment ledger

This ledger records retained and rejected branches. Generated datasets, model
weights, and run JSON remain ignored locally; source workflows are committed.

## Encoder and retrieval experiments

| Experiment | Main evidence | Decision |
|---|---|---|
| Old mixed BGE checkpoint | 95.6% old holdout; 68.3% ToolRet R@3 with dual views | Retain as baseline |
| Transition-only continuation | 93.5% old holdout; 41.7% ToolRet | Reject: over-specialized |
| 4k ToolRet breadth | 95.4% old holdout; 60.0% single-view ToolRet | Retain as curriculum base |
| ToolRet + 3.4k transitions | 94.7% old holdout; 73.3% dual-view ToolRet; 100% development R@3 | Previous best |
| 30k broad DeepSeek ToolRet | 94.4% old holdout; 50.0% single-view ToolRet | Reject: more data regressed |
| Tool-REX dense 1k / 10k | 68.3% ToolRet R@3, flat from 1k to 10k | Reject scaling to 50k |
| Generic and Tool-REX cross-encoders | 66.7–73.3% ToolRet R@3 | Reject: no top-three gain |
| Lexical RRF at 0.25 / 0.50 | Better broad recall, 68.3% / 61.7% top-three | Reject as default |
| Tool-Embed-0.6B reference | 75.0% ToolRet R@3; ~1.5 GB weights | Reject for compact CPU target |
| Candidate multiview late interaction | Development R@3 rose to 100%; ToolRet to 73.3% | Retain runtime architecture |
| 3.4k opaque contrast + transition replay | 99.78% synthetic R@3; 80.0% ToolRet; 100% promotion R@3 | Promote |
| Targeted scaling 5k | 99.78% frozen R@3; 81.7% ToolRet; Qwen raw 19/32 | Reject: raw agent regressed from 23/32 |
| Targeted scaling 10k | 99.78% frozen R@3; 78.3% ToolRet | Reject: dominated before agent execution |
| Targeted scaling 25k | 99.56% frozen R@3; 81.7% ToolRet | Reject: non-monotonic scaling and dominated shortlist |
| Targeted 25k + 10,896 replay | 100% frozen/final/promotion R@3; 76.7% ToolRet; Qwen raw 17/32 | Reject: retrieval gains did not transfer to agent |
| Targeted staged 12.5k + 12.5k | 100% final/promotion R@3; 81.7% ToolRet; Qwen raw 17/32 | Reject: ordering/agent-choice regression |
| Balanced 5,100 hard rows + exact 35,081 full replay | 10% MNRL blend: 94.4% generic R@3, 80.0% ToolRet R@3, Qwen raw 20/32 | Reject: corrected replay still regressed raw agent choice |

## Deployment experiments

| Package | Size | Promotion R@3 | Decision |
|---|---:|---:|---|
| PyTorch FP32 | 128.0 MiB | 100.0% | Reference checkpoint |
| ONNX dynamic int8 | 67.0 MiB | 98.0% | Reject: three promotion misses |
| ONNX FP32 | 127.6 MiB | 100.0% | Promote: exact quality, much lower load overhead |

## Learning-curve conclusion

The observed curve does not support a second 25k cohort, 60k, or 100k
generation. The 2026-08-24 targeted experiment accepted and validated exactly
25,000 new rows, but raw Qwen completion fell from 23/32 to 19/32 at 5k and
17/32 for both replay and staged finalists. Synthetic margins improved while
real agent choices worsened. See [`SCALING_EXPERIMENT_V1.md`](SCALING_EXPERIMENT_V1.md).
Future work should target order-aware weak-agent transfer rather than row count.
The corrected full-replay follow-up reached the same conclusion without the
original replay confound; see
[`SCALING_SALVAGE_EXPERIMENT_V1.md`](SCALING_SALVAGE_EXPERIMENT_V1.md).

## Source references

Method selection used primary sources for [ToolRet](https://aclanthology.org/2025.findings-acl.1258/),
[Re-Invoke](https://aclanthology.org/2024.findings-emnlp.270/),
[ToolRerank](https://aclanthology.org/2024.lrec-main.1413/),
[Meta-Tool](https://aclanthology.org/2025.acl-long.1481/), and
[Tool-REX](https://arxiv.org/abs/2510.22670). The production ONNX path follows
the official [Sentence Transformers inference documentation](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html).
