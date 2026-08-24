# Nomos production coprocessor contract

Nomos has two separate responsibilities:

1. a learned, candidate-based ranker proposes at most three legal tools from an
   external registry;
2. deterministic policy validates proposed calls and enforces legality,
   schema, modality, prerequisites, state, side effects and recovery history.

The separation is intentional. A classifier score cannot prove that an
argument object matches JSON Schema or that an external write is allowed.

## Operations and actions

Requests use one of three operations:

- `recommend_tools` ranks an initial legal candidate set;
- `request_more_tool_candidates` ranks a later page after excluding all prior
  or rejected IDs;
- `verify_tool_call` checks a concrete proposed call without model scoring.

Responses choose `recommend_tools`, `request_more_tool_candidates`,
`accept_tool_call`, `reject_tool_call`, or `abstain`. Recommendations include
reason codes and matched semantic terms. The backward-compatible `ranked_tools`
field remains available to development evaluators; production clients should
send only the top-three `recommendations` descriptions to their LLM.

For a repairable schema rejection, `verify_tool_call` also returns a `repair`
object containing the same tool ID, required and allowed argument names, an
exact call shape, and a warning that placeholders must be replaced. This is
guidance only: the malformed call remains rejected and is never executed.

## Confidence

Raw logits and softmax diagnostics are not confidence. An artifact may claim
`calibrated: true` only when it embeds a calibration record fitted on the frozen
validation partition. The abstention threshold is selected for a declared
maximum selective risk and is then measured once on the untouched test split.

The promoted dense artifact embeds a multiview logistic calibration fitted on
validation only. On the untouched test split it reaches 0.78% selective risk,
97.29% recall for no-suitable-tool abstention, and 0.66% false abstention on
suitable states. Calibration does not replace deterministic validation.

## Data v2 boundary

`matrix.agentic.v2` replaces the coupled v1 generator. The v2 line must keep
task type independent from pool size, represent abstention explicitly, balance
valid and invalid verification cases, freeze registries/templates by split,
and treat recovery as the same routing problem with a deterministic exclusion
set. V1 rows remain immutable audit inputs and are not silently appended to v2.

## Promotion evidence

The promoted model and FP32 ONNX runtime pass 100% top-three recall over 152
states from unseen 34-tool registries. Frozen deterministic evaluation reports
zero illegal recommendations, 100% validation accuracy, zero false accepts in
341 invalid calls, 100% no-repeat expansion, and 99.30% recovery Recall@3.
Executed weak-agent sessions improve from 25% to 100% completion. Full details
and remaining weaknesses are in [`PROMOTION_REPORT.md`](PROMOTION_REPORT.md).
