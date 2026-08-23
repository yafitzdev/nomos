# Fitz-Tool workflows

Executable generation and evaluation workflows belong here.

Planned slices:

1. Validate and materialize matrix cells.
2. Build grounded source cards from local documents.
3. Generate synthetic testcase candidates through the local Qwen3.8 NInfer teacher.
4. Execute trajectories through an external Fitz-Sage V2 runner contract.
5. Validate provenance, governance and terminal outcomes deterministically.
6. Export accepted decision states and hard negatives.
7. Train and evaluate the tool-routing encoder.

The runner boundary is documented in `docs/EXTERNAL_RUNNER_CONTRACT.md`.
The first local commands are:

```text
python -m tools.materialize_matrix --count 1000 --seed 20260823 --output data/generated/matrix_cells.jsonl
python -m tools.validate_matrix_slice data/generated/matrix_cells.jsonl --min-per-value 1
python -m tools.validate_slice data/accepted/ninfer_matrixbound_scenario_slice_1000_v5.jsonl --source-card tests/fixtures/payments_migration_source_card.json --audit-size 25 --audit-output runs/ninfer_matrixbound_1000_v5_audit.json
python -m tools.run_grounding_audit --scenarios data/accepted/ninfer_matrixbound_scenario_slice_1000_v5.jsonl --audit-manifest runs/ninfer_matrixbound_1000_v5_audit.json --source-card tests/fixtures/payments_migration_source_card.json --source-root tests/fixtures/pilot_corpus --v2-root ../fitz-sage-v2 --output runs/ninfer_matrixbound_1000_v5_grounding.json
python -m tools.run_runner_audit --scenarios data/accepted/ninfer_matrixbound_scenario_slice_1000_v5.jsonl --audit-manifest runs/ninfer_matrixbound_1000_v5_audit.json --output data/trajectories/ninfer_matrixbound_1000_v5_runner_audit.jsonl --runner-command <external-runner> <runner-args>
python -m tools.extract_decision_states --trajectories data/trajectories/ninfer_matrixbound_1000_v5_runner_audit.jsonl --output data/accepted/ninfer_matrixbound_1000_v5_states.jsonl --accepted-only
python -m tools.train_encoder --input data/accepted/matrix_template_completion_scale_20000_states.jsonl --output artifacts/router_v1_completion_20k.pt --epochs 40 --feature-dim 4096 --hidden-dim 128 --learning-rate 0.002 --seed 20260824
```

For a second teacher, pass the first accepted slice as `--exclude-slice`.
This reserves its matrix cell IDs and rejects repeated type or instance
signatures before the slice can be used. DeepSeek uses
`FITZ_TOOL_DEEPSEEK_BASE_URL`, `FITZ_TOOL_DEEPSEEK_MODEL` and
`FITZ_TOOL_DEEPSEEK_API_KEY`; keep those in an ignored local environment.

The current NInfer slice has passed structural validation and 25-row grounding.
A bounded no-governance V2 adapter sample is also available for runner-contract
testing, but it is not production acceptance: governance, freshness and clean
state extraction must pass before its rows can train the encoder. The
template/oracle workflow below is only a development bootstrap and must remain
tagged separately.

The local adapter smoke command is:

```text
python -m tools.run_runner_audit --scenarios data/accepted/ninfer_matrixbound_scenario_slice_1000_v5.jsonl --audit-manifest runs/ninfer_matrixbound_1000_v5_audit.json --output data/trajectories/ninfer_matrixbound_1000_v5_v2_runner_audit_25_clean.jsonl --runner-command python -m tools.run_v2_runner --v2-root ../fitz-sage-v2 --source-root tests/fixtures/pilot_corpus --source-card tests/fixtures/payments_migration_source_card.json --base-url http://127.0.0.1:19003/v1 --model qwen3.8-27b --backend llama-cpp --max-steps 8 --governance off --scenario-timeout 120 --no-prewarm
python -m tools.extract_decision_states --trajectories data/trajectories/ninfer_matrixbound_1000_v5_v2_runner_audit_25_clean.jsonl --output data/accepted/ninfer_matrixbound_1000_v5_v2_runner_states_25_clean.jsonl --accepted-only
```

Use `--governance shadow` or the production governance mode only for the
state-aware acceptance gate. The bounded no-governance output is evidence that
the external process contract works, not a training-label source.

When the external teacher is unavailable, the development-only matrix-template
bootstrap keeps scale and encoder plumbing testable without masquerading as
teacher data. The current validated scale command is:

```text
python -m tools.generate_matrix_template_slice --count 20000 --seed 20260824 --source-card-manifest runs/v2_source_cards.jsonl --source-card-manifest runs/v2_code_source_cards.jsonl --exclude-slice data/generated/matrix_template_scale_20000.jsonl --exclude-slice data/generated/matrix_template_pilot_1000.jsonl --exclude-slice data/accepted/ninfer_matrixbound_scenario_slice_1000_v5.jsonl --output data/generated/matrix_template_completion_scale_20000.jsonl
python -m tools.validate_slice data/generated/matrix_template_completion_scale_20000.jsonl --source-card-manifest runs/v2_source_cards.jsonl --source-card-manifest runs/v2_code_source_cards.jsonl --audit-size 25 --audit-output runs/matrix_template_completion_scale_20000_audit.json
Get-Content data/generated/matrix_template_completion_scale_20000.jsonl -Encoding utf8 | python -m tools.run_matrix_oracle --source-card-manifest runs/v2_source_cards.jsonl --source-card-manifest runs/v2_code_source_cards.jsonl | Set-Content data/trajectories/matrix_template_completion_scale_20000.jsonl -Encoding utf8
python -m tools.extract_decision_states --trajectories data/trajectories/matrix_template_completion_scale_20000.jsonl --output data/accepted/matrix_template_completion_scale_20000_states.jsonl --accepted-only
python -m tools.train_encoder --input data/accepted/matrix_template_completion_scale_20000_states.jsonl --output artifacts/router_v1_completion_20k.pt --epochs 40 --feature-dim 4096 --hidden-dim 128 --learning-rate 0.002 --seed 20260824
```

Template/oracle rows must be tagged and reported separately from NInfer,
DeepSeek, and real Fitz-Sage V2 runner rows.

## Registry-aware router.v2 foundation

```text
python -m tools.validate_registry configs/tool_registry.fitz_sage_v2.json
python -m tools.materialize_matrix_v2 --count 5000 --seed 20260823 --output data/generated/matrix_v2_pilot_5000.jsonl
python -m tools.adapt_v1_states_to_v2 --input data/accepted/matrix_template_completion_scale_20000_states.jsonl --output data/accepted/router_v2_adapted_bootstrap.jsonl
python -m tools.train_encoder_v2 --input data/accepted/router_v2_pilot_states.jsonl --output artifacts/router_v2_pilot.pt --epochs 20 --feature-dim 4096 --hidden-dim 128
python -m tools.evaluate_router_v2 --artifact artifacts/router_v2_pilot.pt --input data/accepted/router_v2_pilot_states.jsonl
python -m tools.predict_tools_v2 --artifact artifacts/router_v2_pilot.pt --request request.v2.json --top-k 3
```

The completed registry-aware pilot uses the following reproducible workflow.
Generated rows, reports and model artifacts stay ignored by git:

```text
python -m tools.generate_router_v2_pilot --count 5000 --seed 20260823 --output data/generated/router_v2_pilot_5000.jsonl --manifest runs/router_v2_pilot_5000_manifest.json
python -m tools.validate_router_v2_pilot --input data/generated/router_v2_pilot_5000.jsonl --expected-count 5000 --min-per-target 200 --report runs/router_v2_pilot_5000_validation.json
python -m tools.audit_router_v2_holdouts --input data/generated/router_v2_pilot_5000.jsonl --report runs/router_v2_pilot_5000_holdout_audit.json
python -m tools.train_encoder_v2 --input data/generated/router_v2_pilot_5000.jsonl --output artifacts/router_v2_pilot_1k.pt --train-count 1000 --epochs 15 --feature-dim 2048 --hidden-dim 128 --learning-rate 0.002 --seed 20260823
python -m tools.train_encoder_v2 --input data/generated/router_v2_pilot_5000.jsonl --output artifacts/router_v2_pilot_2_5k.pt --train-count 2500 --epochs 15 --feature-dim 2048 --hidden-dim 128 --learning-rate 0.002 --seed 20260823
python -m tools.train_encoder_v2 --input data/generated/router_v2_pilot_5000.jsonl --output artifacts/router_v2_pilot_full.pt --train-count 3400 --epochs 15 --feature-dim 2048 --hidden-dim 128 --learning-rate 0.002 --seed 20260823
python -m tools.evaluate_router_v2 --artifact artifacts/router_v2_pilot_full.pt --input data/generated/router_v2_pilot_5000.jsonl --output runs/router_v2_pilot_full_evaluation.json
```

If the local NInfer endpoint is available, proposals remain separate from
deterministic labels and require an explicit 25-row validation sample:

```text
python -m tools.generate_ninfer_router_v2_slice --count 100 --seed 20260824 --model Qwen/Qwen3.8-27B --no-api-key --batch-size 4 --concurrency 2 --output data/generated/ninfer_router_v2_proposals_100.jsonl
python -m tools.validate_ninfer_router_v2_slice --input data/generated/ninfer_router_v2_proposals_100.jsonl --expected-count 100 --sample-size 25 --seed 20260824 --report runs/ninfer_router_v2_proposals_100_validation.json
```

Those proposals are marked `not_executed` and `unverified_teacher_proposal`;
they cannot become training labels until an external runner verifies evidence,
provenance, governance freshness and terminal correctness.

The V2 architecture, observable-state boundary and holdout requirements are
documented in `docs/GENERIC_ROUTER_V2.md`.
