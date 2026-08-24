# Nomos agentic data v1

This document describes the generic agentic extension to the registry-aware
router. It is separate from the quarantined Fitz-Sage-shaped material and from
the frozen generic baseline.

## What the extension teaches

Each row is a decision state with a deterministic registry and labels. The
language teacher supplies only a question and a difficult paraphrase. The
local generator supplies the registry, legal candidate set, call proposal,
verification outcome, recovery context, provenance and uniqueness signatures.

Nomos is trained to rank legal candidates, including a top-3 proposal, and to
recognize the recovery action `request_more_tool_candidates`. The deterministic
call validator checks:

- candidate membership and tool identity;
- required fields, types and enum values in the tool schema;
- available input modality;
- state prerequisites and previously rejected candidates; and
- permitted side-effect classes.

This validator proves contract legality. It does not prove that an external
tool returned factually correct evidence; that remains an execution and
governance responsibility of the host agent.

## Matrix

The executable matrix is [`configs/matrix.agentic.v1.json`](../configs/matrix.agentic.v1.json).
Its controlled dimensions include:

| Dimension | Values exercised |
|---|---|
| task | route, recover, verify |
| candidate pool | 10, 30, 100 |
| output target | top-1, top-3 |
| validation case | valid, unknown, illegal, schema missing/wrong type, modality, stale state, prerequisite, side effect |
| recovery trigger | none, empty, partial, contradictory, tool error, stale assessment, budget pressure |
| novelty | familiar, unseen tool ID, unseen tool family |
| history | none, failed first page, previously rejected candidate |
| terminal state | selected, expanded, accepted, rejected, abstain |

The generator creates opaque generic registries from capabilities and metadata;
it does not import Fitz-Sage and does not use Fitz-Sage tool names as generic
training vocabulary. Every row records a matrix-cell ID, teacher/model,
prompt version, dataset version, registry fingerprint, provenance and both
type- and instance-level uniqueness signatures.

## DeepSeek generation measurements

The capacity probe used `deepseek-v4-flash` with 256 concurrent workers. The
largest successful tested request contained 32 rows. It used 6,104 total
tokens for 32 rows, or approximately 191 tokens per row. The 10,000-row
extension used batch size 32 and 256 workers.

The measurement artifacts are:

- `runs/deepseek_agentic_batch_probe.json` for batches 1, 4, 8, 10, 12 and 16;
- `runs/deepseek_agentic_batch_probe_extended.json` for batches 20, 24 and 32.

The API key is read only from the process environment and is never written to
the repository, prompts, manifests or generated rows.

## Generated cohorts

Generated data is ignored by git and must be verified through its manifest and
validation report before training:

| Cohort | Rows | Teacher | Purpose |
|---|---:|---|---|
| `nomos_agentic_pilot_1000_v2` | 1,000 | DeepSeek | balanced pilot and 25-row sample gate |
| `nomos_agentic_scale_10000_v1` | 10,000 | DeepSeek | disjoint targeted extension |

The 10,000-row manifest reports 10,000 unique matrix/type/instance/question
signatures, zero final generation errors and balanced coverage across the
controlled dimensions. Its full validation report has zero invalid rows and
zero sample failures. The cross-cohort validator reports zero overlap in
matrix-cell, type or instance signatures between the pilot and scale cohort.
It does report 42 repeated short question surfaces; these are distinct states
with distinct type and instance signatures, and are retained as a wording
quality warning rather than treated as duplicate semantic data. The pilot
remains a separate diagnostic cohort; it is not overwritten by the scale run.

Validation commands:

```text
python -m tools.validate_agentic_v1 --input data/generated/nomos_agentic_pilot_1000_v2.jsonl --sample-size 25 --seed 20260825 --report runs/nomos_agentic_pilot_1000_v2_validation.json --expected-count 1000
python -m tools.validate_agentic_v1 --input data/generated/nomos_agentic_scale_10000_v1.jsonl --sample-size 25 --seed 20260826 --report runs/nomos_agentic_scale_10000_v1_validation.json --expected-count 10000
```

## Training comparison policy

The old generic corpus remains the baseline. Fitz-Sage-specific legacy data is
not eligible for either experiment. The comparison uses three models:

1. the existing 100k generic baseline;
2. a fresh targeted-only model trained on the 1k pilot plus 10k extension; and
3. a mixed model trained on a deterministic 10k sample of the clean generic
   baseline plus the 1k pilot and 10k extension.

The targeted-only model answers whether fresh data can stand on its own. The
mixed model answers whether retaining the old clean corpus is useful. Both are
evaluated on the frozen old-generic holdout and on the targeted agentic rows.
The comparison report is written to `runs/` and must be read together with
the model manifests; an aggregate improvement cannot hide a regression on the
old holdout or on unseen families.

## Reproducible generation and training commands

```text
python -m tools.probe_deepseek_agentic --batch-sizes 1,4,8,10,12,16,20,24,32 --requests 1 --concurrency 256 --model deepseek-v4-flash --output runs/deepseek_agentic_batch_probe_full.json
python -m tools.generate_agentic_ninfer_v1 --count 10000 --seed 20260826 --output data/generated/nomos_agentic_scale_10000_v1.jsonl --manifest runs/nomos_agentic_scale_10000_v1_manifest.json --teacher deepseek --model deepseek-v4-flash --batch-size 32 --concurrency 256
python -m tools.train_encoder_v2 --input data/generated/nomos_agentic_pilot_1000_v2.jsonl --input data/generated/nomos_agentic_scale_10000_v1.jsonl --output artifacts/nomos_agentic_fresh_11000.pt --epochs 12 --feature-dim 512 --hidden-dim 128 --learning-rate 0.002 --seed 20260826 --batch-size 4096
python -m tools.train_encoder_v2 --input data/generated/nomos_generic_baseline_train_10000.jsonl --input data/generated/nomos_agentic_pilot_1000_v2.jsonl --input data/generated/nomos_agentic_scale_10000_v1.jsonl --output artifacts/nomos_agentic_mixed_21000_v1.pt --epochs 12 --feature-dim 512 --hidden-dim 128 --learning-rate 0.002 --seed 20260826 --batch-size 4096
```

The commands above assume the DeepSeek key is supplied through an ignored
environment variable. Never put the literal key in a shell history committed
to the project, source file or artifact.

## Scaled comparison result

On the 10,000-row targeted extension, the existing baseline remains strongest
on top-1 routing overall, while the fresh targeted model is strongest on
targeted top-3 routing:

| Artifact | Targeted R@1 | Targeted R@3 | Old-generic holdout R@1 | Old-generic holdout R@3 |
|---|---:|---:|---:|---:|
| existing generic baseline | 0.502 | 0.757 | 0.864 | 0.966 |
| fresh targeted-only | 0.492 | 0.774 | 0.527 | 0.800 |
| clean-generic + targeted mix | 0.456 | 0.728 | 0.547 | 0.851 |

All three artifacts report a zero invalid-candidate rate. The targeted-only
model improves verification-row recall over the baseline (R@1 0.108 vs 0.019;
R@3 0.251 vs 0.108) and the mixed model improves recovery no-repeat rate
(0.414 vs 0.373), but neither preserves the old generic capability. The
deterministic call validator agrees with every one of the 3,333 verification
labels in the scale cohort; this is contract-validation accuracy, not factual
tool-result accuracy.

The fresh targeted model is therefore a useful research checkpoint for the new
agentic behavior, not a replacement for the existing generic baseline. The
safe deployment decision is to retain the old baseline for the established
generic route and evaluate the targeted model as a separate agentic branch
until a larger or better-balanced training run closes the old-holdout gap.

For the scale matrix, presenting three candidates instead of the full legal
pool would reduce candidate-description entries from 466,630 to 30,000 across
the 10,000 rows: a structural estimate of 93.6% fewer candidate descriptions.
The actual token saving depends on the host agent's schema serialization.

The comparison report is
`runs/nomos_agentic_model_comparison_scale_v1.json`.
