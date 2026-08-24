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

## Confidence

Raw logits and softmax diagnostics are not confidence. An artifact may claim
`calibrated: true` only when it embeds a calibration record fitted on the frozen
validation partition. The abstention threshold is selected for a declared
maximum selective risk and is then measured once on the untouched test split.

The retained 100k baseline is not production-calibrated. On agentic v1, a
logistic confidence layer had to reduce coverage to 0.7% to keep observed test
risk below 1%. This shell therefore provides the mechanism, not evidence that
the existing ranker is ready.

## Data v2 boundary

`matrix.agentic.v2` replaces the coupled v1 generator. The v2 line must keep
task type independent from pool size, represent abstention explicitly, balance
valid and invalid verification cases, freeze registries/templates by split,
and treat recovery as the same routing problem with a deterministic exclusion
set. V1 rows remain immutable audit inputs and are not silently appended to v2.
