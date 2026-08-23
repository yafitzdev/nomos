# Nomos / Fitz-Tool

Nomos is the learned tool-routing model line built by Fitz-Tool. Fitz-Tool is
the data-generation, validation and training companion for Fitz-Sage: a
portfolio artifact for making technical research agents better at choosing
their next tool.

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

## Generic router v2

The repository now includes a side-by-side `router.v2` foundation for Nomos.
External agents supply a versioned tool registry plus their current legal
candidate set. V2 scores capability, description, modality, evidence-role,
side-effect and schema metadata without learning the literal tool ID.

See `docs/GENERIC_ROUTER_V2.md` for the architecture and matrix refinements.
The paste-ready pilot/training request is in `docs/GENERIC_ROUTER_GOAL.md`.
V1 code and artifacts remain supported and unchanged.

The first completed V2 pilot is a 5,000-state deterministic matrix-oracle
slice with frozen unseen-ID, renamed-ID, schema, modality, family, source,
question-template and alternate-registry cohorts. The current encoder feature
version is `registry-features.v2.1`; NInfer proposals are generated and
validated separately from labels.

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
source cards + matrix cells
        -> teacher-generated testcase candidates
        -> deterministic grounding/deduplication checks
        -> real Fitz-Sage V2 runner trajectories
        -> accepted next-tool labels + hard negatives
        -> encoder training and ablation evaluation
```

Fitz-Sage V2 is consumed as an external system under test through a stable
runner contract. This repository remains independently runnable and does not
require the V1 or V2 checkout at import time.

## Local teacher configuration

Use environment variables or an ignored local `.env` file. Never commit keys.

```text
FITZ_TOOL_TEACHER_BASE_URL=http://127.0.0.1:19003/v1
FITZ_TOOL_TEACHER_MODEL=qwen3.8-27b-nvfp4
FITZ_TOOL_TEACHER_API_KEY=

# Optional external breadth teacher; keep the key in an ignored local .env.
FITZ_TOOL_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
FITZ_TOOL_DEEPSEEK_MODEL=deepseek-chat
FITZ_TOOL_DEEPSEEK_API_KEY=
```

The local NInfer teacher is a data-generation dependency only. It is not the
deployment model for the Fitz-Sage agent.
