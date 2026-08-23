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

## Generic router v2 and v3 data line

The generic router accepts a versioned tool registry and an external agent's
legal candidate set. It represents candidates through capability, modality,
evidence-role, side-effect, prerequisite and schema metadata rather than
literal tool names. See `docs/GENERIC_ROUTER_V2.md` for the core contract.

The current 50k data line is `matrix.generic.v3`. The matrix creates unique
state cells, opaque registries and deterministic legality/oracle labels. Local
NInfer/Qwen is the primary language teacher; DeepSeek can provide a separate
breadth slice or continue an interrupted run. Each row records its actual
teacher and model. The separation is deliberate: teacher wording is real LLM
output, while acceptance and labels remain reproducible and cannot be silently
changed by a teacher hallucination.

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
