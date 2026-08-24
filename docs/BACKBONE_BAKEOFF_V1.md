# Nomos backbone bakeoff v1

## Outcome

The BGE-small family remains the best foundation for Nomos. A fresh BGE-small
control trained on the common 40,181-pair corpus has the highest macro-average
raw top-three recall and the highest independent ToolRet top-three recall. The
accepted BGE soup remains substantially stronger on the frozen agentic suite,
so `nomos-v1-nano-g1` remains the stable published release rather than being
silently replaced by this experiment.

LFM2.5-Embedding-350M and LFM2.5-ColBERT-350M are viable CPU-runnable
alternatives, and both perform very well on several internal suites. Neither
beats BGE on the independent ToolRet benchmark or on the sealed unseen-style
suite. GTE-ModernBERT is not suitable under the common full-fine-tuning recipe:
it catastrophically forgets its general retrieval geometry.

## Locked training comparison

Every fresh model used the same accepted answer-present training records:

- 20,000 generic portability pairs;
- 4,185 agentic-v2 pairs;
- 4,096 ToolRet-derived agentic pairs;
- 3,400 transition pairs;
- 3,400 contrast pairs;
- 5,100 balanced hard-slice pairs.

That is 40,181 pairs total. All runs used one epoch, seed `20260824`, cached
multiple-negatives ranking loss, effective batch 64, mini-batch 16, learning
rate `2e-5`, and a 512-token maximum. ColBERT retained native token-level
MaxSim scoring and used minimum query expansion with a 512-token query cap.

| Fresh model | Train time | Mean loss |
|---|---:|---:|
| BGE-small control | 3m 24s | 0.986 |
| GTE-ModernBERT | 7m 20s | 1.631 |
| LFM2.5-Embedding-350M | 9m 33s | 0.587 |
| LFM2.5-Encoder-230M | 7m 38s | 0.888 |
| LFM2.5-ColBERT-350M | 10m 10s | 0.601 |

Training losses are not directly comparable across architectures. Held-out raw
ranking is the promotion criterion.

## Raw top-three tool recall

| Model | Frozen | Generic | Sealed | Final | Promotion | ToolRet | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Accepted BGE soup | 99.67% | 94.40% | 97.50% | 98.65% | 100.00% | 80.00% | 95.04% |
| **Fresh BGE control** | 96.12% | 95.20% | **100.00%** | **100.00%** | **100.00%** | **81.67%** | **95.50%** |
| GTE-ModernBERT | 100.00% | 93.90% | 86.25% | 60.81% | 35.53% | 3.33% | 63.30% |
| LFM2.5-Embedding-350M | **100.00%** | **98.20%** | 92.22% | **100.00%** | 98.68% | 70.00% | 93.18% |
| LFM2.5-Encoder-230M | **100.00%** | 96.00% | 97.78% | 97.30% | 96.71% | 36.67% | 87.41% |
| LFM2.5-ColBERT-350M | **100.00%** | 97.00% | 93.06% | **100.00%** | **100.00%** | 75.00% | 94.18% |

ToolRet contains 60 independently sampled queries in this comparison, so a
single example changes recall by 1.67 percentage points. The fresh BGE lead
over the accepted BGE model is one query and must not be overstated. The much
larger gaps against the other backbones are still decision-relevant.

## CPU deployment measurements

These are unoptimized PyTorch CPU measurements on the same workstation. Tool
descriptions are cached for warm measurements; each request still encodes the
current objective and agent-state views. The three latency values correspond
to the representative 10-, 30-, and 100-tool states in the frozen suite. The
state texts differ, so latency is not expected to rise monotonically with pool
size after candidates are cached.

| Model | Artifact | Load RSS delta | Warm p50 (10 / 30 / 100 tools) |
|---|---:|---:|---:|
| Accepted BGE soup | 128.0 MiB | 734.1 MiB | 187 / 292 / 190 ms |
| Fresh BGE control | 129.0 MiB | 734.0 MiB | 187 / 295 / 191 ms |
| GTE-ModernBERT | 288.7 MiB | 770.0 MiB | 4,376 / 7,032 / 4,088 ms |
| LFM2.5-Embedding-350M | 681.7 MiB | 810.0 MiB | 475 / 779 / 439 ms |
| LFM2.5-Encoder-230M | 881.8 MiB | 809.5 MiB | 906 / 1,416 / 862 ms |
| LFM2.5-ColBERT-350M | 679.7 MiB | 809.2 MiB | 476 / 790 / 518 ms |

The user's latency intuition is directionally right for the Liquid retrieval
models: an extra roughly 0.25-0.50 seconds can be acceptable beside an LLM call
that takes seconds. It is not true for this unoptimized ModernBERT path, which
adds several seconds per warm decision and takes 85 seconds to encode and rank
a cold representative 100-tool registry.

## ModernBERT diagnosis

The untouched GTE-ModernBERT checkpoint scored 87.49% frozen, 88.10% generic,
98.33% sealed, 99.32% final, 95.39% promotion, and 68.33% ToolRet top-three.
After full fine-tuning it scored 100.00%, 93.90%, 86.25%, 60.81%, 35.53%, and
3.33%, respectively. The training recipe therefore over-specialized the model;
the backbone itself was not initially incapable.

Any future ModernBERT experiment should use a much smaller learning rate,
parameter-efficient tuning, or base/specialist weight interpolation. It must
gate every checkpoint on sealed, final, promotion, and ToolRet transfer during
training. Even a successful accuracy rescue would still need ONNX or another
CPU optimization path to address the measured latency.

## Recommendation

1. Keep the published accepted BGE soup as `nomos-v1-nano-g1`.
2. Treat the fresh BGE control as the leading G2 candidate, not an automatic
   promotion. Expand independent external evaluation before choosing it over
   the accepted model.
3. Do not promote any LFM or ModernBERT candidate from this bakeoff.
4. Retain LFM2.5-Embedding and ColBERT support in the repository for future
   experiments; their latency is practical and their internal retrieval is
   strong, but their external transfer is currently insufficient.

The locked run is reproduced with:

```powershell
python -m tools.run_backbone_bakeoff --device cuda
python -m tools.summarize_backbone_bakeoff `
  --run-dir runs/backbone_bakeoff_v1 `
  --output runs/backbone_bakeoff_v1/summary.json
```
