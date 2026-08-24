# Nomos token-savings A/B harness

## Purpose

The harness measures the product claim that Nomos can reduce the tool context
shown to an agent model without hiding the tool needed for the next action.
It replays the same OpenAI-compatible tool-decision request in two conditions:

| Condition | Visible tools | What it isolates |
|---|---:|---|
| `full` | every legal tool in the captured request | Existing agent behavior and token cost. |
| `nomos` | Nomos top-k, normally three | Candidate reduction from the raw encoder. |

The harness does not execute tools, repair calls, or force Nomos's top-one
choice. The downstream model still chooses among the visible candidates. This
keeps the test focused on candidate retrieval and token cost rather than the
deterministic coprocessor policy.

## What already exists

The frozen 34-tool evaluation reported an 86.0% prompt-token reduction for the
local Qwen agent and 85.7% for DeepSeek when all tools were replaced by raw
Nomos top-three. Those results establish feasibility, but they use eight local
workflow templates across four registry styles. The replay harness exists to
measure the same effect on captured agent decisions from unrelated projects.

## Input

Input is JSONL with one captured decision per line:

```json
{
  "case_id": "session-17-step-4",
  "request": {
    "model": "agent-model",
    "messages": [{"role": "user", "content": "Find the implementation of this symbol."}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "inspect_code",
          "description": "Inspect symbols and implementation structure in a source repository.",
          "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
          }
        }
      }
    ]
  },
  "acceptable_tools": ["inspect_code"]
}
```

`acceptable_tools` is optional but strongly recommended. It must come from the
successful trajectory, deterministic task oracle, or independent annotation;
it must not be inferred from the Nomos ranking being evaluated.

Standard OpenAI descriptions and parameter schemas are sufficient. Registries
can provide richer identity-free metadata under `function["x-nomos"]`:

```json
{
  "tool_family": "code_inspection",
  "capabilities": ["inspect_code_structure"],
  "input_modalities": ["code", "text"],
  "output_modalities": ["symbols"],
  "evidence_roles": ["observation"],
  "side_effect_class": "read",
  "constraints": ["read_only"],
  "prerequisites": ["repository_available"]
}
```

The adapter never turns concrete tool names into output classes. When richer
metadata is absent, it ranks from the function description and argument schema
with neutral fallback fields.

## Offline replay

Offline mode is deterministic and makes no agent-model requests. Supply the
tokenizer of the downstream agent model to estimate request and schema tokens:

```powershell
python -m tools.benchmark_nomos_openai_ab `
  --input data/accepted/real_agent_decisions.jsonl `
  --nomos-model artifacts/nomos_v1_nano_g1_release `
  --tokenizer <agent-model-or-tokenizer-id> `
  --top-k 3 `
  --output runs/nomos_openai_ab_offline.json `
  --trace-output runs/nomos_openai_ab_offline_traces.jsonl
```

Tokenizer counts cover the serialized request and are estimates because hosted
providers may add hidden formatting. UTF-8 request and schema byte reductions
are always reported as a tokenizer-independent check.

## Live paired replay

Live mode sends both conditions to an OpenAI-compatible endpoint and records
provider-reported prompt/completion tokens, selected tools, and latency. The
condition order is randomized per case to reduce ordering and cache bias.

```powershell
$env:AGENT_API_KEY = "<temporary runtime secret>"
python -m tools.benchmark_nomos_openai_ab `
  --input data/accepted/real_agent_decisions.jsonl `
  --nomos-model artifacts/nomos_v1_nano_g1_release `
  --endpoint https://example.invalid/v1/chat/completions `
  --api-key-env AGENT_API_KEY `
  --top-k 3 `
  --input-cost-per-million 1.00 `
  --output-cost-per-million 2.00 `
  --output runs/nomos_openai_ab_live.json `
  --trace-output runs/nomos_openai_ab_live_traces.jsonl
```

Keys stay in environment variables and are never written to reports. A live
replay performs model calls and therefore incurs provider cost. It does not
execute returned tool calls.

## Primary metrics

| Metric | Why it matters |
|---|---|
| provider prompt-token reduction | Direct live measurement of input savings. |
| estimated request-token reduction | Reproducible offline comparison. |
| tool-schema byte reduction | Provider- and tokenizer-independent context reduction. |
| Nomos top-k recall | Whether an independently acceptable tool remains visible. |
| selected-tool accuracy | Whether the agent still chooses an acceptable visible tool. |
| estimated dollar-cost reduction | Prompt and completion cost at supplied rates. |
| Nomos ranking latency | CPU overhead added before the agent request. |

The first practical gate should be at least 95% top-three recall on labeled
real decisions, a large positive provider prompt-token reduction, no material
selected-tool accuracy regression, and warm routing latency small relative to
the downstream generation call. Results should be stratified by registry size,
tool family, project, agent model, and decision depth before making a broad
deployment claim.

## Next data collection

Capture at least 500 real decision points across multiple projects, with a
useful spread of 10-, 30-, 60-, and 100-plus-tool registries. Keep complete
messages, legal tools, provider usage, subsequent selected tool, whether the
call succeeded, and the final task outcome. Redact secrets and user content
before placing traces in the repository, and keep the raw captures ignored.
