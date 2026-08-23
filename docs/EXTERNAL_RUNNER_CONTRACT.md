# External runner contract v1

Fitz-Tool does not import Fitz-Sage V2 Python modules. A runner is an external
process that receives UTF-8 scenario JSON objects, one per line, on standard
input and emits exactly one UTF-8 `trajectory.v1` JSON object per input scenario
on standard output.

For bootstrap development, `tools/run_matrix_oracle.py` implements the same
contract as `matrix-oracle.v1`. It is deterministic synthetic labeling, not a
substitute for Fitz-Sage V2 execution; its rows must remain marked with the
oracle version and should be replaced or compared with V2-validated rows before
production integration.

## Input

Each input line is one `scenario.v1` object. The runner may use a local source
root and its own database, but those paths and implementation details stay
outside Fitz-Tool.

## Output

Each output line is one `trajectory.v1` object with:

- the matching `scenario_id`;
- runner identity and `runner.v1` contract version;
- ordered events, including one `decision` event before each model tool call;
- the exact legal tool set at that decision;
- the compact agent state, evidence summary and governance snapshot;
- `acceptable_tools` and `hard_negative_tools` supplied by deterministic V2
  validation, not by the teacher;
- terminal outcome, provenance and `validation.trajectory_accepted`.

The runner must not write to Fitz-Tool's source or generated-data directories.
Fitz-Tool validates the returned trace, extracts decision-state rows, and
retains failed-but-safe prefixes as hard negatives.

## Decision event minimum

```json
{
  "step": 0,
  "kind": "decision",
  "agent_state": {"state_name": "initial"},
  "legal_tools": ["set_retrieval_plan", "search_bm25"],
  "observed_evidence": [],
  "governance": {"assessment_fresh": false, "requirements": []},
  "proposed_tool": "search_bm25",
  "executed_tool": "search_bm25",
  "acceptable_tools": ["search_bm25"],
  "hard_negative_tools": ["set_retrieval_plan"]
}
```

`acceptable_tools` may contain more than one tool. A trajectory is positive
only when deterministic validation marks it accepted. Teacher proposals and
the model's selected tool are audit fields, not labels by themselves.

## Development router execution harness

The generic router also exposes a separate `runner-request.v2` process for
portability testing:

```text
python -m tools.run_router_contract --mode model --artifact artifacts/nomos_generic_portability_100000.pt
```

It accepts one validated request per JSONL line and returns one
`router-response.v2` object containing the ranked legal candidates. The
development evaluator sends these requests through the process boundary and
simulates a tool result/state update after each correct choice:

```text
python -m tools.evaluate_runner_contract --input data/generated/nomos_generic_portability_100000.jsonl --limit 200 --output runs/nomos_runner_contract_100k.json --trace-output runs/nomos_runner_contract_100k_traces.jsonl
```

This evaluator deliberately uses a deterministic capability simulator. It
checks request validation, legal-candidate handling and multi-step state
transitions; it is not a substitute for a real API/tool runner or for
Fitz-Sage V2 validation.

## Fitz-Sage V2 smoke result

The external `runner.v1` adapter was also exercised against two generated
fixture scenarios using a local Fitz-Sage V2 subprocess. The adapter returned
two schema-valid `trajectory.v1` traces in the expected order, confirming that
the process boundary and trace validation work. Neither trajectory was
accepted: V2 repeated an incomplete `<tool_call>` argument payload, produced
no legal executed tool according to its own diagnostics, and reached its step
limit with no selected evidence.

This is an agent-output/parsing blocker, not a router-quality result. The
smoke did not yet place Nomos in V2's live decision loop, so those traces must
not be used as positive training data. The next integration gate is to make
the external agent emit complete, parseable decision requests and then route
the supplied legal candidates through `runner-request.v2` before measuring
real tool execution.
