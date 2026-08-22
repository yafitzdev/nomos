# Fitz-Tool session handoff

This document captures the durable decisions from the initial design session
so work can continue in a new chat without reconstructing the conversation.

## Current objective

Build a portfolio-grade, state-aware tool-routing encoder for technical
integration research agents.

The first learned artifact is:

```text
user question + current agent state + legal tool set
    -> ranked acceptable next-tool proposals
```

This is not an answer generator and not a replacement for the LLM, Pi, tool
execution or deterministic governance.

## Repository boundaries

| Repository | Owns | Boundary |
|---|---|---|
| `C:\Users\yanfi\PycharmProjects\fitz-sage` | V1 traditional RAG | Read-only corpus/benchmark reference; never modify or import |
| `C:\Users\yanfi\PycharmProjects\fitz-sage-v2` | Agentic RAG product | External system under test; V2 retains Pi, tools, governance and the runtime loop |
| `C:\Users\yanfi\PycharmProjects\fitz-tool` | Tool router/data factory | Owns matrices, generated data, trajectories, labels, training and artifacts |

Fitz-Tool must not import private V1 or V2 Python modules. The integration
with V2 will use a stable external runner/trace contract. V2 must not become a
runtime dependency of this repository.

## Runtime architecture

```text
User question
    -> small encoder ranks promising tools
    -> local LLM reasons, chooses arguments and calls a tool through Pi
    -> tool result updates the agent state
    -> encoder runs again
    -> governance/controller allows search, comparison, abstention or finalization
```

The encoder proposes; it does not issue an unconditional “must call” command.
Deterministic code masks illegal tools, enforces step limits, validates
evidence identity/provenance, tracks governance freshness and validates final
selection.

The first encoder target is next-tool ranking. It should not initially attempt
to generate arbitrary tool-call JSON or an entire trajectory.

## Tool families in the V2 system under test

- Planning: `set_retrieval_plan`.
- Broad retrieval: `search_bm25`.
- Exact/metadata retrieval: `grep_search`, `search_metadata`.
- Tables: `list_tabular_sources`, `inspect_table_schema`, `search_table_rows`.
- PDFs: `list_pdf_sources`, `inspect_pdf_structure`, `search_pdf_pages`.
- Code/files: `read_file`, `inspect_code`.
- Evidence: `inspect_evidence`, `expand_context`.
- Comparison: `compare_evidence`.
- Progress/governance: `update_requirement_progress`, `assess_evidence`.
- Terminal action: `finalize_document_selection`.

Multiple tools can be valid for the same state. Labels should support an
acceptable set or ranked alternatives, not force one arbitrary sampled route.

## Data-generation pipeline

```text
matrix cells + grounded source cards
    -> synthetic testcase candidates
    -> deterministic grounding, deduplication and leakage checks
    -> real Fitz-Sage V2 runner trajectories
    -> deterministic evidence/provenance/governance validation
    -> accepted decision states + hard negatives
    -> encoder training and ablation evaluation
```

A testcase is not the same as a training row. One question can produce many
decision states: planning, first retrieval, recovery, inspection, governance,
comparison and finalization.

Teacher output is never automatically ground truth. A positive state requires
valid tool use, verified evidence, correct provenance, fresh governance and a
correct terminal outcome. Failed but informative trajectories become hard
negatives where safe.

## Model roles

- Qwen3.5-2B remains the deployment-oriented V2 baseline.
- Qwen3.8-27B through local NInfer is the primary trajectory teacher and can
  produce Qwen-aligned on-policy examples.
- DeepSeek V4 Flash is optional for offline breadth generation, alternative
  proposals or critique. It must never be a hidden runtime dependency.
- No API keys belong in this repository or generated artifacts.

The useful hybrid is: DeepSeek can cheaply create broad grounded testcase
variants; Qwen/NInfer can execute selected cases through the real local Pi/V2
loop; deterministic validation produces the labels. We should also retain
Qwen-only, DeepSeek-only and mixed dataset tags for ablations.

## NInfer benchmark facts

The local test instance was run without an API key at:

```text
http://127.0.0.1:19003/v1
model: qwen3.8-27b-nvfp4
```

Best tested startup profile:

```text
text-only
thinking disabled
MTP3 + optimized draft head
INT8 KV
32K context
max-concurrency 8
max-pending-requests 32
CUDA graphs and prefix reuse enabled
```

Measured with valid JSON testcase outputs:

| Engine concurrency | Requests | Aggregate decode | Effective testcase rate |
|---:|---:|---:|---:|
| 4 | 8 | 408.4 tok/s | ~6,390/hour |
| 6 | 24 | 535.9 tok/s | ~7,967/hour |
| 8 | 32 | 636.2 tok/s | ~9,670/hour |
| 10 | startup rejected | unsupported (`[1,8]`) | — |

The eight-slot profile estimates roughly one hour for 10,000 compact testcase
definitions, excluding validation and trajectory execution. Use `127.0.0.1`
rather than Windows `localhost`; the latter added approximately two seconds of
client-side delay in testing. The local server was restored at concurrency 8
after the benchmark.

The reusable benchmark is
`tools/benchmark_teacher_throughput.py`. It supports keyless local mode with
`--no-api-key` and has a streaming timing probe.

## DeepSeek planning facts

These figures were researched from the official DeepSeek API documentation on
2026-08-22 and must be rechecked before spending money:

- V4 Flash input cache miss: `$0.14 / 1M tokens`.
- V4 Flash input cache hit: `$0.0028 / 1M tokens`.
- V4 Flash output: `$0.28 / 1M tokens`.
- Official account concurrency limit: `2,500`.
- Supports JSON output and tool calls.

Approximate cost for 10,000 testcase definitions:

| Assumption per case | Estimated no-cache cost |
|---|---:|
| 1,000 input + 300 output tokens | $2.24 |
| 2,000 input + 500 output tokens | $4.20 |
| 4,000 input + 1,000 output tokens | $8.40 |

These figures do not represent the cost of executing 10,000 full multi-step
agent trajectories. Official references:

- https://api-docs.deepseek.com/quick_start/pricing/
- https://api-docs.deepseek.com/quick_start/rate_limit/
- https://api-docs.deepseek.com/guides/thinking_mode

## Matrix design

The draft taxonomy is in `docs/DATA_GENERATION_MATRIX.md`. Primary axes are:

- integration domain;
- information operation;
- source modality;
- evidence topology;
- retrieval obstacle;
- agent state;
- next-tool target;
- governance path;
- terminal condition;
- resource pressure.

Important counterfactual families include:

- same evidence, different wording: route should remain stable;
- same question, different modality: tool family should change;
- remove one source: sufficient becomes insufficient;
- add a contradiction: sufficient becomes disputed;
- perturb an identifier: hit becomes no-match/reformulation;
- change “current” to “compare versions”: temporal lookup becomes comparison.

Do not generate the full Cartesian product. Use valid conditional combinations:
LSP requires code, PDF tools require PDF, table tools require structured data,
contradiction requires multiple claims, and finalization requires a fresh
assessment of the exact canonical evidence set.

## Evaluation plan

Compare at least:

1. Qwen/Pi without encoder.
2. Qwen/Pi with encoder top-k tool proposals.
3. Deterministic retrieval-only baseline.
4. Qwen-only training data versus mixed-teacher data.

Measure next-tool Recall@1/Recall@3, invalid-call rate, premature-finalization
rate, downstream document-selection quality, abstention quality, evidence
precision, retrieval steps, latency, CPU time and memory.

Include held-out corpora, held-out question templates and model-transfer tests.
The frozen V2 benchmark must never be used as synthetic training input.

## Current repository state

The repository currently contains the charter, scaffold, draft matrix,
environment example, ignored data/artifact directories and the NInfer
benchmark helper. It contains no generated dataset, no trained encoder and no
formal versioned trace schema yet. The initial scaffold is not committed.

## Next implementation order

1. Review and approve the matrix dimensions and valid combinations.
2. Define versioned scenario, trajectory and decision-state schemas.
3. Add matrix materialization and quota validation.
4. Add source-card ingestion for local corpora; keep raw documents hashed and
   immutable.
5. Define the external Fitz-Sage V2 runner/trace contract.
6. Generate a small audited pilot before scaling.
7. Validate and export accepted next-tool labels plus hard negatives.
8. Train the first encoder and run the ablations.
9. Scale only after the pilot passes manual and deterministic audits.

The next chat should start in
`C:\Users\yanfi\PycharmProjects\fitz-tool`, read `AGENTS.md` and this file,
inspect Git status, and agree on the matrix/schema before generating a large
dataset.
