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

The first direct smoke exposed an agent-output problem: the local Qwen backend
sometimes emitted incomplete or non-visible tool calls and reached the step
limit without selecting evidence. Fitz-Tool now provides
`tools.nomos_openai_proxy`, a V2-specific adapter-layer process that:

- translates each observable OpenAI request into `runner-request.v2`;
- ranks only the candidate functions supplied by the external runner;
- preserves modality constraints while allowing unresolved source discovery;
- raises the tool-call output budget and retries once when the response is
  incomplete; and
- rejects or repairs malformed/non-legal model output into the selected legal
  function shape without importing Fitz-Sage code.

On the aligned payments fixture, the repaired path produced one valid
`trajectory.v1` trace with 10 decisions. The external V2 run ended in
`selected`, retrieved and inspected the expected AUTH-409 evidence, and passed
the deterministic acceptance check. The proxy routed three multi-candidate
decisions and repaired one backend response. This proves that Nomos is in the
live candidate-selection loop; it is not yet a production-quality benchmark.

Run the local bridge with the 100k checkpoint like this:

```text
python -m tools.nomos_openai_proxy --artifact artifacts/nomos_generic_portability_100000.pt --target-url http://127.0.0.1:19003/v1 --listen-port 19004 --source-modality text --min-max-tokens 512 --retry-max-tokens 1024 --trace-output runs/nomos_proxy_trace.jsonl
python -m tools.run_runner_audit --scenarios runs/v2_runner_aligned_scenario.jsonl --audit-manifest runs/v2_runner_aligned_audit.json --output runs/v2_runner_nomos_aligned_trajectories.jsonl --runner-command python -m tools.run_v2_runner --v2-root ../fitz-sage-v2 --source-root tests/fixtures/pilot_corpus --source-card tests/fixtures/payments_migration_source_card.json --base-url http://127.0.0.1:19004/v1 --model qwen3.8-27b --backend llama-cpp --max-steps 14 --governance off --scenario-timeout 480 --no-prewarm
```

The proxy is an integration adapter, not part of the generic router core. Its
repair path is a legal-candidate safety boundary, not a substitute for tool
execution, evidence validation or deterministic terminal acceptance. The next
gate is a stratified external-runner sample large enough to measure accepted
trajectories and extract only verified decision states for training.
