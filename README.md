# Nomos / Fitz-Tool

Nomos is a project-agnostic learned tool-routing coprocessor. Fitz-Tool owns
the matrix, registries, synthetic data, validation, training and evaluation
workflows. Fitz-Sage V2 is only one external integration adapter and is not the
training universe or a runtime dependency.

The core research question is:

> Can a small state-aware encoder reduce invalid, irrelevant and premature
> tool calls for different LLMs while preserving adaptive retrieval and
> abstention behavior?

The first model is a tool router, not an answer generator:

```text
question + agent state + legal tool schemas
    -> ranked next-tool proposals
```

Pi, the language model and the deterministic controller remain responsible
for tool execution, arguments, provenance, governance and final selection.

## Production status

Nomos now passes the production promotion gates. On a sealed suite with 34
tools per registry, the correct tool is top-three in 152/152 unseen states. A
local Qwen3-0.6B agent was tested across all 32 workflow-by-registry
combinations: it improves from 0/32 completed tasks with one-shot access to the
raw full registry to 23/32 with one-shot raw Nomos top-three proposals. The
complete coprocessor reaches 26/32. Raw top-three uses 86.0% fewer prompt tokens
per call attempt.

The promoted deployment is a 127.6 MiB FP32 ONNX encoder plus deterministic
validation, confidence, repair, and no-repeat recovery code. See
[`docs/PROMOTION_REPORT.md`](docs/PROMOTION_REPORT.md) for the evidence and
[`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md) for retained and
rejected ablations.

Install the lightweight CPU runtime with:

```powershell
python -m pip install -e ".[runtime]"
python tools/run_router_contract.py `
  --mode dense `
  --artifact artifacts/nomos_bge_contrast_replay_onnx_fp32_ablation
```

The contract runner reads one `runner-request.v2` JSON object per input line
and writes one `router-response.v2` object per line.

## Generic router v2 and v3 data line

The generic router accepts a versioned tool registry and an external agent's
legal candidate set. It represents candidates through capability, modality,
evidence-role, side-effect, prerequisite and schema metadata rather than
literal tool names. See `docs/GENERIC_ROUTER_V2.md` for the core contract.

The retained generic baseline is the `matrix.generic.v3` line. The matrix
creates unique state cells, opaque registries and deterministic legality/oracle
labels. Local NInfer/Qwen is the primary language teacher; DeepSeek can provide
a separate breadth slice or continue an interrupted run. Each row records its
actual teacher and model. The separation is deliberate: teacher wording is
real LLM output, while acceptance and labels remain reproducible and cannot be
silently changed by a teacher hallucination.

`nomos-agentic.v1`, documented in [`docs/AGENTIC_DATA_V1.md`](docs/AGENTIC_DATA_V1.md),
is retained as an audited development cohort, not as the production data line.
An honest test-only rerun found coupled dimensions and semantically underspecified
teacher rows. `matrix.agentic.v2` replaces it with independently sampled pool sizes,
task kinds, abstention cases, hard-negative classes, balanced call verification,
and split-specific registry/template namespaces. Existing clean generic data and
artifacts remain preserved; Fitz-Sage-specific legacy data remains quarantined.

The runtime contract now separates learned ranking from deterministic safety.
Nomos returns at most three described recommendations, exposes calibrated versus
uncalibrated confidence, can abstain, supports `request_more_tool_candidates`
with guaranteed no-repeat behavior, and validates proposed calls against tool
identity, legality, schema, modality, prerequisites, state and side-effect policy.
Rejected repairable calls include an exact schema-derived retry shape but are
never silently accepted or executed.

The complete workflow and cohort plan are in
`docs/GENERIC_NINFER_50K.md`. Older Fitz-Sage-shaped generated data is legacy
material and must not be used as generic training data.

## Repository layout

```text
fitz-tool/
├── fitz_tool/          # reusable library code
├── tools/              # executable generation, validation and training workflows
├── schemas/            # versioned dataset and trace contracts
├── configs/            # matrix, teacher and runner configurations
├── docs/               # design notes and dataset cards
├── data/
│   ├── raw/            # immutable local inputs; ignored by git
│   ├── generated/      # synthetic testcase candidates; ignored by git
│   ├── trajectories/   # executed teacher traces; ignored by git
│   └── accepted/       # validated training examples; ignored by git
├── runs/               # manifests, metrics and SQLite ledgers; ignored
└── artifacts/          # trained models and exports; ignored
```

## Current generation pipeline

```text
generic matrix cells + opaque registry factory
        -> NInfer-generated question surfaces
        -> deterministic legal-candidate and provenance validation
        -> accepted labels + hard negatives
        -> encoder training and held-out evaluation
```

Fitz-Sage V2 is consumed as an external system under test through a stable
runner contract. Its mappings live under `fitz_tool/adapters/` and are not
imported by the generic generator or router core.

## Local teacher configuration

Use environment variables or an ignored local `.env` file. Never commit keys.

```text
FITZ_TOOL_TEACHER_BASE_URL=http://127.0.0.1:19003/v1
FITZ_TOOL_TEACHER_MODEL=Qwen/Qwen3.8-27B
FITZ_TOOL_TEACHER_API_KEY=

# Optional external breadth teacher; keep the key in an ignored local .env.
FITZ_TOOL_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
FITZ_TOOL_DEEPSEEK_MODEL=deepseek-v4-flash
FITZ_TOOL_DEEPSEEK_API_KEY=
```

The local NInfer teacher is a data-generation dependency only. It is not the
deployment model for the Fitz-Sage agent.
