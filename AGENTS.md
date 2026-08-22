# Fitz-Tool Project Charter

## Product identity

Fitz-Tool is an independent portfolio project for learning and evaluating a
state-aware tool-routing encoder for technical research agents.

The first learned artifact maps:

```text
user objective + current agent state + legal tools
    -> ranked next-tool proposals
```

It is not a replacement for an LLM, Pi orchestration, tool execution,
evidence governance, or deterministic provenance validation.

## Repository ownership

This repository owns:

- the multidimensional testcase-generation matrix;
- corpus/source-card manifests;
- synthetic questions and mutations;
- teacher trajectories and decision-state records;
- deterministic acceptance and hard-negative labels;
- encoder training and evaluation code;
- trained router artifacts and dataset cards.

Fitz-Sage V2 is an external system under test. Fitz-Tool must not import its
private Python modules or become a runtime dependency of V2. Integration with
V2 must use an explicit runner/trace contract. The V1 Fitz-Sage repository is
read-only external input and must never be modified or imported.

## Teacher and runtime boundaries

- Qwen3.8-27B through the local NInfer endpoint is the primary trajectory
  teacher.
- DeepSeek is optional for offline breadth generation or critique only, never
  a hidden runtime dependency.
- No API keys may be committed, embedded in prompts, or written to generated
  artifacts.
- Production/runtime agent behavior remains local and is evaluated separately
  from synthetic-data generation.

## Data rules

- Raw source documents are immutable inputs and receive content hashes.
- Every generated item records the corpus, source, prompt, model, artifact,
  seed and validator versions.
- Teacher proposals are not automatically ground truth.
- A decision state is accepted only when deterministic execution verifies legal
  tools, evidence identity, provenance, governance freshness and terminal
  correctness.
- Failed trajectories are retained as labeled hard negatives when safe.
- Frozen evaluation questions and holdout documents must never enter training.

## Scope

Keep the domain focused on technical integration/API documentation research.
Do not turn Fitz-Tool into a general autonomous researcher or an autonomous
external-write system.

Tools and executable workflows belong in the repository-root `tools/` folder.
