# Nomos generic router foundation (`router.v2`)

`router.v2` keeps the candidate-conditioned core from V1 while removing the
literal tool ID from learned features. An external agent supplies a validated
tool registry and the legal candidate IDs for each decision. The router ranks
that set and cannot invent, authorize, or execute a tool.

This is an implementation foundation, not a trained general-purpose release.
The existing `router.v1` code and artifacts remain unchanged.

## Boundaries

```text
external agent
  -> runner-request.v2
      -> observable state
      -> tool-registry.v2
      -> legal_candidate_ids
          -> router.v2 scores each candidate
              -> ranked legal candidates
```

The generic core has no Fitz-Sage imports. Fitz-Sage V2 names and metadata live
in `configs/tool_registry.fitz_sage_v2.json`; translation of V1 decision states
lives in `fitz_tool/adapters/fitz_sage_v2.py`.

## Registry contract

Each concrete tool records:

- stable `tool_id` for lookup and output only;
- tool family and description;
- composable capabilities;
- input and output modalities;
- evidence roles;
- side-effect class;
- JSON-compatible argument schema;
- constraints and prerequisites.

The registry is canonicalized and SHA-256 fingerprinted. Training provenance
records the exact registry fingerprint. Renaming `tool_id` while preserving the
semantic metadata produces the same candidate features.

## Runtime contract

`runner-request.v2` contains the question, observable state, history, evidence,
governance, resources, source/query state, an embedded registry and the current
legal candidate IDs. Contract validation rejects empty candidate sets, unknown
IDs, duplicates and candidates whose side-effect class is disallowed by the
request governance snapshot.

`decision-state.v2` adds deterministic labels and provenance for training. It
supports more than one acceptable tool.

## Feature boundary

The scorer consumes only:

- question and current agent phase/state;
- prior actions and current plan;
- inspected evidence and current governance facts;
- actual remaining steps, requirement/evidence counts and distractor counts;
- source inventory, available modalities and inspection depth;
- query operation, specificity and match strategy;
- identity-free candidate metadata and candidate/state interactions.

The scorer does not consume `matrix_context`, `sampling_context`, labels, target
capability, future governance paths or terminal outcomes. These fields may be
retained for generation, stratification and audit but are not runtime features.
The current feature encoder is `registry-features.v2.2`; it adds lexical
question-intent/candidate-intent interactions derived from the observable
question, without reading the matrix target label. The intent lexicon includes
paraphrased evidence outcomes such as page ranges, field layouts, provenance,
surrounding context and sufficiency; it does not expose a target capability
field at runtime.

## Question-generalization gate

The original `router_v2_pilot_5000.jsonl` is a frozen benchmark. Its existing
`heldout_questions` cohort is intentionally retained, including its domain-only
wording; that cohort is useful as an underspecified negative control. The
derived `question-generalization.v1` view changes only the questions for
training and the indirect heldout cohort. It uses eight training surfaces and
four disjoint holdout surfaces, with a token-level audit rejecting canonical
target capability phrases.

Training interleaves the original 3,400 training rows with their 3,400
generalized counterparts. The full checkpoint therefore sees 6,800 training
states while all frozen validation/test rows remain unchanged. The formal gate
requires indirect heldout Recall@1 >= 0.70, <= 0.05 Recall@1 regression on
unseen IDs, renames, heldout families and alternate registries, stable metadata
ablation, and invariance on both views.

## Matrix v2 refinements

`configs/matrix.v2.json` replaces literal next-tool targets with target
capabilities and introduces the distinctions exposed by the V1 error analysis:

| Area | V2 control |
|---|---|
| retrieval intent | exact vs semantic lookup, source/record enumeration, comparison and absence verification |
| source readiness | source inventory unknown/partial/known |
| query readiness | broad, entity-bound, exact identifier or schema-bound |
| retrieval method | lexical, identifier, metadata, structural, semantic or hybrid |
| evidence readiness | none, snippet, partial context, full context or multi-source inspected |
| governance | requirement progress and current assessment freshness |
| resources | explicit remaining steps, unresolved requirements, evidence, distractor and search counts |
| candidate difficulty | same-family neighbors, capability overlap, schema-only and constraint differences |
| generalization | held-out tool ID, family, source, question template and external-agent profile |

`target_capability`, terminal outcome and other generator labels are explicitly
sampling-only.

## Evaluation requirements

Do not use a random trajectory split as the only test. Reserve independent
cohorts for:

1. familiar tools on held-out states;
2. unseen tool IDs from trained families;
3. tool-ID renames with identical metadata;
4. schema and modality variants;
5. entirely held-out tool families;
6. held-out source documents and question templates;
7. a non-Fitz-Sage registry/agent contract.

Report Recall@1, Recall@3, MRR, fixed-order baseline, invalid-candidate rate,
candidate-order invariance, ID-renaming invariance and question/matrix
ablations. Include an indirect question-template counterfactual; capability
phrases in training questions can otherwise make the question signal look
stronger than it is. Metrics from deterministic matrix-oracle data must remain
separate from NInfer, DeepSeek and governance-validated external runner data.

## Commands

```text
python -m tools.validate_registry configs/tool_registry.fitz_sage_v2.json
python -m tools.validate_registry configs/tool_registry.alternate_v2.json
python -m tools.validate_registry configs/tool_registry.heldout_v2.json
python -m tools.generate_router_v2_pilot --count 5000 --seed 20260823 --output data/generated/router_v2_pilot_5000.jsonl --manifest runs/router_v2_pilot_5000_manifest.json
python -m tools.validate_router_v2_pilot --input data/generated/router_v2_pilot_5000.jsonl --expected-count 5000 --min-per-target 200 --report runs/router_v2_pilot_5000_validation.json
python -m tools.audit_router_v2_holdouts --input data/generated/router_v2_pilot_5000.jsonl --report runs/router_v2_pilot_5000_holdout_audit.json
python -m tools.generate_router_v2_question_generalization --frozen-input data/generated/router_v2_pilot_5000.jsonl --output data/generated/router_v2_question_generalization_5000.jsonl --manifest runs/router_v2_question_generalization_5000_manifest.json --training-output data/generated/router_v2_question_generalization_training_8400.jsonl --training-manifest runs/router_v2_question_generalization_training_8400_manifest.json
python -m tools.validate_router_v2_question_generalization --frozen-input data/generated/router_v2_pilot_5000.jsonl --derived-input data/generated/router_v2_question_generalization_5000.jsonl --training-input data/generated/router_v2_question_generalization_training_8400.jsonl --report runs/router_v2_question_generalization_5000_validation.json
python -m tools.train_encoder_v2 --input data/generated/router_v2_question_generalization_training_8400.jsonl --output artifacts/router_v2_question_generalization_full.pt --train-count 6800 --epochs 30 --feature-dim 2048 --hidden-dim 128 --learning-rate 0.002 --seed 20260823
python -m tools.evaluate_router_v2 --artifact artifacts/router_v2_question_generalization_full.pt --input data/generated/router_v2_pilot_5000.jsonl --output runs/router_v2_question_generalization_full_frozen_evaluation.json
python -m tools.evaluate_router_v2 --artifact artifacts/router_v2_question_generalization_full.pt --input data/generated/router_v2_question_generalization_5000.jsonl --output runs/router_v2_question_generalization_full_derived_evaluation.json
python -m tools.check_router_v2_question_gate --baseline-frozen runs/router_v2_pilot_full_evaluation.json --candidate-frozen runs/router_v2_question_generalization_full_frozen_evaluation.json --candidate-generalized runs/router_v2_question_generalization_full_derived_evaluation.json --report runs/router_v2_question_generalization_full_gate.json
python -m tools.generate_ninfer_router_v2_slice --count 100 --seed 20260824 --model Qwen/Qwen3.8-27B --no-api-key --batch-size 4 --concurrency 2 --output data/generated/ninfer_router_v2_proposals_100.jsonl
python -m tools.validate_ninfer_router_v2_slice --input data/generated/ninfer_router_v2_proposals_100.jsonl --expected-count 100 --sample-size 25 --seed 20260824 --report runs/ninfer_router_v2_proposals_100_validation.json
```

After the question gate passes, the lower-bound scale pilot is:

```text
python -m tools.generate_router_v2_scale --count 30000 --seed 20260923 --output data/generated/router_v2_scale_30000.jsonl --manifest runs/router_v2_scale_30000_manifest.json
python -m tools.validate_router_v2_pilot --input data/generated/router_v2_scale_30000.jsonl --expected-count 30000 --min-per-target 1000 --report runs/router_v2_scale_30000_validation.json
python -m tools.audit_router_v2_holdouts --input data/generated/router_v2_scale_30000.jsonl --report runs/router_v2_scale_30000_holdout_audit.json
python -m tools.generate_router_v2_question_generalization --frozen-input data/generated/router_v2_scale_30000.jsonl --output data/generated/router_v2_scale_question_generalization_30000.jsonl --manifest runs/router_v2_scale_question_generalization_30000_manifest.json --training-output data/generated/router_v2_scale_question_generalization_training_50400.jsonl --training-manifest runs/router_v2_scale_question_generalization_training_50400_manifest.json --expected-count 30000
python -m tools.validate_router_v2_question_generalization --frozen-input data/generated/router_v2_scale_30000.jsonl --derived-input data/generated/router_v2_scale_question_generalization_30000.jsonl --training-input data/generated/router_v2_scale_question_generalization_training_50400.jsonl --expected-count 30000 --report runs/router_v2_scale_question_generalization_30000_validation.json
```

The adapted V1 rows are useful for compatibility and smoke testing. Because
their labels and source matrix remain V1-specific, they are not evidence of
tool-family generalization.
