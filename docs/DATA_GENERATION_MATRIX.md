# Data-generation matrix (draft v0)

This is intentionally a draft. We should approve the taxonomy and sampling
quotas before producing the large dataset.

The dataset unit is an agent decision state, not just a user question. One
question can produce several states as the agent searches, inspects evidence,
reassesses governance and finalizes.

## Primary axes

| Axis | Examples |
|---|---|
| Integration domain | auth, payments, refunds, webhooks, schemas, errors, migrations, security, reconciliation |
| Information operation | lookup, enumerate, compare, join, latest-value selection, compatibility, contradiction, absence |
| Source modality | text, PDF, CSV, Excel, SQLite, code/LSP, mixed |
| Evidence topology | one passage, multiple passages, complementary sources, cross-format, contradictory, absent |
| Retrieval obstacle | paraphrase, acronym, identifier variation, near-duplicate, long-document needle, split context, version noise |
| Agent state | initial, no hits, noisy hits, partial evidence, expansion needed, contradiction, insufficient, disputed, fresh sufficient |
| Next-tool target | BM25, grep, metadata, table, PDF, file/LSP, inspect, expand, compare, assess, finalize |
| Governance path | sufficient, insufficient→sufficient, repeated insufficient, disputed→sufficient, sufficient→disputed, stale assessment |
| Terminal condition | selection, abstention, clarification, unresolved contradiction, step-limit termination |
| Resource pressure | requirement count, evidence count, distractor count, retrieval-pass budget, remaining steps |

## Controlled counterfactuals

Generate related cases where one variable changes:

- same evidence with different wording: route should remain stable;
- same question in a different modality: tool family should change;
- remove one required source: sufficient becomes insufficient;
- add a contradictory source: sufficient becomes disputed;
- perturb an identifier: hit becomes no-match or reformulation;
- change “current” to “compare versions”: temporal lookup becomes comparison.

## Validity constraints

- LSP targets require source-code evidence.
- PDF tools require PDF inputs.
- Table tools require a structured source.
- Contradiction requires at least two claims or sources.
- Cross-format reconciliation requires at least two modalities.
- Governance assessment requires inspected evidence and tracked requirements.
- Finalization requires a fresh assessment of the exact canonical evidence set.
- Abstention requires deterministic exhaustion of materially different searches,
  unless the corpus manifest proves that the requested source is absent.

## Balance rule

Do not balance only by question count. Report balance at the decision-state
level, especially for:

- first-tool selection;
- recovery after weak retrieval;
- tool switching;
- progressive governance;
- contradiction handling;
- abstention and clarification;
- finalization.
