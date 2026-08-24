# Nomos targeted scaling cohort v1

This cohort is a fixed-slot, project-agnostic tool-routing experiment. The
matrix contains exactly 25,000 accepted target slots. DeepSeek supplies natural
wording only; registries, legal candidates, positives, hard negatives,
prerequisites, modalities, side-effect policy, and recovery state are created
and checked deterministically.

The allocation intentionally gives extra weight to measured weak families:
exact matching, source inventory, code-symbol inspection, evidence sufficiency,
provenance, planning, and close finalization alternatives. Independent balanced
axes cover 10/17/34/50/100-tool pools, seven modalities, five side-effect
policies, initial through terminal session positions, empty through long
history, semantic and opaque metadata, five schema styles, five registry
description styles, candidate position, recovery, stale state, and abstention.

Generation uses `deepseek-v4-flash`, disabled thinking, batch size 16,
concurrency 512, an 8,000-token response limit, strict JSON, and a 180-second
timeout. A failed batch is retried once, then split from 16 to 8 to 4. Accepted
rows are appended idempotently; rejects and request events are retained in
ignored local JSONL files. Replacement assignments preserve the fixed slot's
dimensions but receive new deterministic assignment and matrix-cell IDs.

Before generation, a 756-state post-scaling holdout was frozen. It contains 720
answer-present states and 36 abstention states across unseen 34/50/100-tool
registries, four unseen description styles, three unseen schema families, and
three-stage workflows with recovery. Its row, source, registry, template, and
scenario hashes are recorded in
`configs/post_scaling_holdout.v1.manifest.json`. This holdout must not be used
for training or model selection feedback before the controlled variants are
trained.

The final prompt-v2 canary accepted 115 matrix-unique rows and covered all 23
scenario families. The stratified quality audit passed 115/115 rows. It found
no fallback text, identity leakage, contract failure, state inconsistency,
modality mismatch, duplicate accepted question, or holdout overlap.

Reproducible commands:

```powershell
python -m tools.materialize_scaling_matrix_v1 --output data/generated/nomos_scaling_matrix_v1_25000.jsonl --manifest runs/nomos_scaling_matrix_v1_25000_manifest.json
python -m tools.freeze_post_scaling_holdout_v1 --output data/generated/nomos_post_scaling_holdout_v1.jsonl --manifest runs/nomos_post_scaling_holdout_v1_manifest.json
python -m tools.generate_scaling_deepseek_v1 --assignments data/generated/nomos_scaling_matrix_v1_25000.jsonl --holdout data/generated/nomos_post_scaling_holdout_v1.jsonl --output data/generated/nomos_scaling_targeted_v1_25000.jsonl --rejects data/generated/nomos_scaling_targeted_v1_25000_rejected.jsonl --request-log runs/nomos_scaling_targeted_v1_25000_requests.jsonl --manifest runs/nomos_scaling_targeted_v1_25000_manifest.json --batch-size 16 --concurrency 512 --max-tokens 8000 --timeout 180
python -m tools.audit_scaling_cohort_v1 --input data/generated/nomos_scaling_targeted_v1_25000.jsonl --sample-size 115 --output runs/nomos_scaling_targeted_v1_25000_audit.json
```
