# Fitz-Tool execution roadmap

> **Current status:** The historical Fitz-Sage-shaped bootstrap described below
> is retained only as design history. It is not active training data. The
> retained project-agnostic baseline has passed validation, training and
> held-out evaluation. The additive generic agentic extension is documented in
> [`AGENTIC_DATA_V1.md`](AGENTIC_DATA_V1.md).

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

### Generic agentic extension

The agentic v1 matrix and deterministic validator produced a 1,000-row pilot and
a 10,000-row DeepSeek extension. Both are preserved, but v1 is not eligible for
the production training line. Its generator accidentally coupled task type to
pool size (route=10, recover=30, verify=100), its language validator checked
format rather than intent preservation, and verification was mixed into a
retrieval objective despite being a deterministic contract problem.

The existing generic baseline remains the comparison point. The corrected
test-only report is `runs/nomos_agentic_model_comparison_test_only_v1.json`.
On accepted agentic states, baseline/targeted/mixed Recall@3 is
0.743/0.731/0.705. On the old frozen holdout it is 0.966/0.800/0.851. Earlier
agentic headline numbers evaluated the entire 10k file and therefore included
training rows; they must not be used. None of the new models replaces the
baseline.

`matrix.agentic.v2` is now the active design. It independently samples task
kind and candidate-pool size, adds no-suitable-tool abstention cases, separates
call verification from retrieval, varies hard-negative type and description
quality, and freezes registry and question-template namespaces by split. No v2
language rows should be generated until its deterministic skeleton and frozen
evaluation suite pass.

The production-facing coprocessor shell is also implemented. It guarantees that
ranked outputs are legal, excludes prior candidates during recovery, validates
calls deterministically, emits machine-readable reasons and refuses to describe
raw scores as calibrated confidence. A first calibration audit of the retained
100k baseline could satisfy a 1% selective-risk target on only 0.7% of test
requests. This is evidence that the ranking model must improve; thresholding or
scaling the existing recipe is not enough.

The first v2 architecture ablations now favor a compact dense bi-encoder. A
one-epoch BGE-small rehearsal mix used 20,000 retained generic train pairs plus
4,185 answer-present v2 train pairs. On untouched tests it reaches 99.22%
Recall@3 on the deterministic v2 suite and 95.6% on the old generic holdout,
which is a one-point regression from the retained baseline. On DeepSeek v1
wording it reaches 93.6% for route states but only 61.2% for underspecified
recovery states. The 99.22% v2 number is not a promotion result because v2's
deterministic train/test wording still shares capability-focus phrases.

Adding absolute cosine features to the confidence layer raises v2 test coverage
at a 1% selective-risk target from less than 1% to 70.5%. Observed test risk is
0.63%, no-suitable-tool recall is 97.7%, and false abstention is 12.8%. The
validation/test coverage gap and false-abstention rate remain too large. The
next evidence gate is independently worded and executed multi-step sessions,
not a larger synthetic generation run.

The active generic line is complete through the 100k portability gate:

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

The first external portability gate reached a mean 57.2% Recall@1 and 83.5%
Recall@3, versus 14.2% and 41.5% for the candidate-order baseline, with zero
illegal-candidate outputs. A concentrated 70k retrain regressed to 42.9%/76.5%,
so it was rejected. A controlled 55k mix recovered and improved the mean to
59.5%/84.0%.

The accepted 100k line combines the frozen 50k base with 20k targeted and 30k
balanced portability rows. It validates with 100,000 unique matrix/type/
instance/question signatures and zero invalid rows. Training uses a balanced
54k-row subset and reaches 85.5%/96.0% internal Recall@1/Recall@3. On the same
four unseen registries it reaches 68.5%/88.0% mean Recall@1/Recall@3, with zero
illegal-candidate outputs.

A development runner-request.v2 execution harness now crosses a separate
router process and simulates state updates across 800 multi-step tasks. It
confirms the router is useful: task completion is 0.88% for candidate order,
12.38% for the 50k model and 12.25% for the 100k model, with zero illegal
selections. The 100k model improves first decisions but does not yet improve
complete multi-step tasks, so it is not promoted over the 50k checkpoint. The
next gate is actual external-agent execution; more synthetic rows are not yet
justified.

The first external Fitz-Sage V2 smoke produced schema-valid traces but zero
accepted trajectories because the local backend repeated incomplete or
non-visible tool calls. The V2-specific `tools.nomos_openai_proxy` bridge now
places the 100k Nomos checkpoint in the live candidate loop, enforces the
external runner's legal candidates, repairs one-step output failures and keeps
source discovery available when a named source is unresolved. The aligned
payments fixture now completes a 10-decision trajectory and passes deterministic
acceptance. A first 25-case governance-enforced fresh-start slice now produces
17 fully accepted trajectories (68%), 103 accepted decision states, zero illegal
tool selections and 23 retained hard-negative rows. Four cases timed out and
three selected traces missed expected facts; the sample is useful for finding
weaknesses, but it is not yet a training-promotion gate. The immediate next
step is to add a multi-location fixture for comparison questions, repair the
multi-fact evidence misses, and rerun this gate before collecting more data.

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
