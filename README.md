<div align="center">

# Nomos

### A CPU-runnable co-processor for tool-using agents

**Find the next useful tool. Check the proposed call before it runs.**

[Why it exists](#why-it-exists) · [How it works](#how-it-works) · [What it enables](#what-it-enables) · [Boundaries](#boundaries)

</div>

---

## Why it exists

An agent with many tools must choose what to do next from the tools that are legal *right now*. The right choice depends on its objective, what it has already learned, and what each tool can do. Putting an entire registry of descriptions and schemas into every prompt costs tokens and leaves the agent to sort through irrelevant options.

Nomos gives the agent a smaller, state-aware set of choices. As a co-processor, it ranks useful next tools and checks a proposed call against the current state and tool contract before execution.

> A tool can be relevant to the task and still be premature: the agent may need to inspect a file before it can safely edit it.

---

## How it works

Nomos runs locally on CPU. A small encoder ranks candidates; a deterministic policy layer handles validation and recovery.

| Stage | What Nomos reads | What it returns |
| --- | --- | --- |
| **Rank** | The objective, observable state, and legal tool registry | Up to three next-tool recommendations |
| **Check** | A proposed tool call and its arguments | Acceptance or a specific reason and repair guidance |
| **Recover** | Rejected candidates and the updated state | Fresh candidates without repeats, or abstention |

```text
objective + state + legal tools
             ↓
Nomos ranks the next-tool candidates
             ↓
agent chooses a tool and supplies arguments
             ↓
Nomos checks the proposed call
             ↓
runtime executes an accepted call, repairs it, or asks for more candidates
```

The deployed encoder uses an FP32 ONNX package for CPU inference. Nomos itself requires neither a GPU nor a hosted language model at inference time. The surrounding agent remains responsible for choosing arguments and running tools.

---

## What it enables

**Smaller tool prompts.** Show the agent a short, relevant menu instead of the whole registry at each step.

**State-aware routing.** Recommend a different tool as the objective, available evidence, and legal actions change.

**Call validation.** Reject unknown or illegal tools, invalid arguments, unmet prerequisites, and disallowed side effects before execution.

**Targeted recovery.** Give the agent a repair shape for a rejected call, or offer new candidates without repeating ones it has already declined.

**Safe abstention.** Say when there is no suitable tool or the routing confidence is too low.

---

## Boundaries

Nomos proposes and validates tool calls; it does not execute them, generate final answers, or guarantee that a recommended tool will complete a task. The runtime defines which tools are legal, applies policy, and decides whether to execute, retry, or stop. Its behavior should be evaluated with the tool registry and workflows where it will be used.

This repository is a public overview of the project.

## License

The repository text is licensed under [CC BY-NC 4.0](LICENSE).
