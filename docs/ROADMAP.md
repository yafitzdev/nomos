# Fitz-Tool execution roadmap

> **Current status:** The historical Fitz-Sage-shaped bootstrap described below
> is retained only as design history. It is not active training data. The
> project-agnostic 50k corpus has now passed validation, training and held-out
> evaluation; see [`GENERIC_NINFER_50K.md`](GENERIC_NINFER_50K.md).

## Objective

Train a first state-aware encoder that predicts the next legal operation from a
question, agent state and an external tool registry.

## Working agreement

The execution order is:

1. approve the dimensions and conditional matrix rules;
2. generate a 1,000-row NInfer slice;
3. validate all rows structurally and audit a reproducible sample of 25;
4. generate a disjoint 1,000-row DeepSeek slice and audit its 25-row sample;
5. scale only after both pilot slices pass, preserving the matrix ledger so a
   semantic type is never generated twice;
6. train and evaluate the first encoder on deterministically accepted states.

The current recommendation is to interpret “10k–20k” as accepted decision
states, because those are the encoder’s training examples. We will report the
number of source-grounded scenarios separately. If we later choose a
10k–20k-question target instead, the same gates apply and the required number
of executed trajectories will be larger.

The dataset unit is a decision state. A generated scenario can produce several
decision states after execution through the external Fitz-Sage V2 runner.
Scenario, trajectory and decision-state counts must therefore be reported
separately. The 10k–20k training target refers to accepted decision-state rows,
not merely unexecuted scenario definitions; at roughly 7–10 states per useful
trajectory, this will require substantially fewer but fully grounded questions.

## Gates

1. **Matrix v1**: controlled dimensions, conditional validity rules, quotas and
   uniqueness signatures are approved.
2. **Pilot contracts**: source-card, scenario, trajectory and decision-state
   records validate deterministically.
3. **Teacher slices**: create 1,000 NInfer scenarios and 1,000 DeepSeek
   scenarios from disjoint reserved matrix cells while preserving comparable
   dimension coverage. Every later slice must pass earlier accepted slices via
   `--exclude-slice`; duplicate type or instance signatures are rejected.
4. **Audit**: run cheap structural, provenance, leakage and duplicate checks on
   every row. Run the expensive external-runner and manual audit on a
   reproducible stratified sample of at least 25 rows per teacher. The current
   deterministic grounding audit is a prerequisite only; it does not create
   router labels. The sample must still pass the state-aware `runner.v1`
   trajectory audit before scaling.
5. **Scale**: generate enough unique scenarios in batches to yield 10k–20k
   accepted decision states only after both pilot gates pass. Repeat
   deterministic checks every batch and periodically refresh the semantic audit
   sample. Learning curves at 2k, 5k, 10k, 15k and 20k accepted states decide
   whether further generation is useful.
6. **Train**: extract accepted decision states with deterministic positive and
   hard-negative labels. Train the first encoder only from accepted states;
   teacher proposals alone are not labels.
7. **Evaluate**: use frozen held-out documents, question templates and matrix
   cells. Report Recall@1/3, invalid-call rate, premature finalization,
   downstream selection quality, abstention quality, retrieval steps and
   latency against the no-encoder baseline.

The first encoder gate is accepted-tool Recall@3 of at least 98% overall,
without any important tool family below 95%, plus an illegal-tool proposal rate
below 0.5% and no regression on the frozen end-to-end retrieval benchmark.

## Uniqueness policy

Every row has a canonical `type_signature` and `instance_signature`. A type
signature covers the matrix cell, source cards, expected facts and terminal
condition; an instance signature also covers the question and difficult
paraphrase. Duplicate type or instance signatures are rejected within a
generation ledger. Planned paraphrase and counterfactual pairs are the only
future exception and must be explicitly marked as such.

The matrix currently uses an explicit terminal profile with approximately 74%
ongoing decision states and 26% terminal decisions. This prevents
`finalize_document_selection` from dominating the first router slice while
still exercising selection, abstention, clarification, contradiction and
step-limit outcomes.

The first text-only pilot excludes `absent` evidence topology because one
positive source card cannot prove absence. Absence cases enter the dataset only
after a multi-source/negative-source manifest exists.

## Current status

The active generic line is complete through the first training/evaluation gate:

- `data/generated/nomos_generic_ninfer_50000.jsonl` contains 50,000 rows from
  the project-agnostic v3 matrix. Fitz-Sage-shaped data is quarantined and is
  not used by the generic model.
- `runs/nomos_generic_ninfer_50000_validation.json` reports zero invalid rows,
  zero duplicate questions, 50,000 unique matrix/type/instance signatures,
  and no held-out registry, family, source or question-template overlap.
- `runs/nomos_generic_ninfer_50000_sample_audit.json` is a reproducible
  100-row audit with zero errors.
- `artifacts/nomos_generic_ninfer_full.pt` trains on only the 34,000-row
  `train` cohort. Its overall Recall@1/Recall@3/MRR are 87.4%/96.9%/0.923;
  the illegal-candidate rate is 0%.
- Held-out Recall@1 is 84.7% for an unseen tool family, 86.3% for unseen tool
  IDs, 81.0% for unseen question templates and 88.3% for unseen sources.
  Candidate-order, renamed-ID and sampling-context invariance all pass.

The next engineering gate is integration testing against external runner
contracts and a real unseen registry supplied by another agent, not more
Fitz-Sage-specific training data.

## Current commands

```text
python -m tools.materialize_matrix --count 1000 --seed 20260823 --output data/generated/matrix_cells.jsonl
python -m tools.validate_matrix_slice data/generated/matrix_cells.jsonl --min-per-value 1
python -m tools.generate_teacher_slice --count 1000 --teacher ninfer --base-url http://127.0.0.1:19003/v1 --model Qwen/Qwen3.8-27B --no-api-key --source-card tests/fixtures/payments_migration_source_card.json --output data/generated/ninfer_matrixbound_completion_1000_v2.jsonl
python -m tools.validate_slice data/generated/ninfer_matrixbound_1000_v5.jsonl --source-card tests/fixtures/payments_migration_source_card.json --audit-size 25 --audit-output runs/ninfer_matrixbound_1000_v5_audit.json

# Cheap grounding check; this does not produce training labels.
python -m tools.run_grounding_audit --scenarios data/accepted/ninfer_matrixbound_scenario_slice_1000_v5.jsonl --audit-manifest runs/ninfer_matrixbound_1000_v5_audit.json --source-card tests/fixtures/payments_migration_source_card.json --source-root tests/fixtures/pilot_corpus --v2-root ../fitz-sage-v2 --output runs/ninfer_matrixbound_1000_v5_grounding.json

# Bounded runner-contract smoke; governance-off output is not training truth.
python -m tools.run_runner_audit --scenarios data/accepted/ninfer_matrixbound_scenario_slice_1000_v5.jsonl --audit-manifest runs/ninfer_matrixbound_1000_v5_audit.json --output data/trajectories/ninfer_matrixbound_1000_v5_v2_runner_audit_25_clean.jsonl --runner-command python -m tools.run_v2_runner --v2-root ../fitz-sage-v2 --source-root tests/fixtures/pilot_corpus --source-card tests/fixtures/payments_migration_source_card.json --base-url http://127.0.0.1:19003/v1 --model qwen3.8-27b --backend llama-cpp --max-steps 8 --governance off --scenario-timeout 120 --no-prewarm
python -m tools.extract_decision_states --trajectories data/trajectories/ninfer_matrixbound_1000_v5_v2_runner_audit_25_clean.jsonl --output data/accepted/ninfer_matrixbound_1000_v5_v2_runner_states_25_clean.jsonl --accepted-only

# Disjoint 20k development scale and deterministic encoder bootstrap.
python -m tools.generate_matrix_template_slice --count 20000 --seed 20260824 --source-card-manifest runs/v2_source_cards.jsonl --source-card-manifest runs/v2_code_source_cards.jsonl --exclude-slice data/generated/matrix_template_scale_20000.jsonl --exclude-slice data/generated/matrix_template_pilot_1000.jsonl --exclude-slice data/accepted/ninfer_matrixbound_scenario_slice_1000_v5.jsonl --output data/generated/matrix_template_completion_scale_20000.jsonl
python -m tools.validate_slice data/generated/matrix_template_completion_scale_20000.jsonl --source-card-manifest runs/v2_source_cards.jsonl --source-card-manifest runs/v2_code_source_cards.jsonl --audit-size 25 --audit-output runs/matrix_template_completion_scale_20000_audit.json
python -m tools.train_encoder --input data/accepted/matrix_template_completion_scale_20000_states.jsonl --output artifacts/router_v1_completion_20k.pt --epochs 40 --feature-dim 4096 --hidden-dim 128 --learning-rate 0.002 --seed 20260824

# A second teacher slice must reserve the first slice's cells and signatures.
# Configure FITZ_TOOL_DEEPSEEK_API_KEY in an ignored local environment first.
python -m tools.generate_teacher_slice --count 1000 --teacher deepseek --source-card tests/fixtures/payments_migration_source_card.json --exclude-slice data/accepted/ninfer_matrixbound_scenario_slice_1000_v5.jsonl --output data/generated/deepseek_matrixbound_1000_v1.jsonl
python -m tools.validate_slice data/generated/deepseek_matrixbound_1000_v1.jsonl --source-card tests/fixtures/payments_migration_source_card.json --audit-size 25 --audit-output runs/deepseek_matrixbound_1000_v1_audit.json
```
