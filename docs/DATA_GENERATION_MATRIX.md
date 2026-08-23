# Data-generation matrix (draft v0; executable candidate matrix.v1)

This is intentionally a draft. We should approve the taxonomy and sampling
quotas before producing the large dataset.

The executable candidate is `configs/matrix.v1.json`. It adds an `ongoing`
terminal-condition value so nonterminal decision states do not get forced into
`finalize_document_selection`. The candidate sampling profile targets roughly
74% ongoing states and 26% terminal states, while still requiring coverage of
every reachable controlled value in a pilot slice. The materializer seeds one
legal example for each reachable next-tool target before filling the remainder
randomly; this prevents rare tool families from disappearing by chance.

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
| Next-tool target | retrieval planning, BM25, grep, metadata/source listing, table, PDF, file/LSP, inspect, expand, compare, assess, finalize |
| Governance path | sufficient, insufficient→sufficient, repeated insufficient, disputed→sufficient, sufficient→disputed, stale assessment |
| Terminal condition | ongoing, selection, abstention, clarification, unresolved contradiction, step-limit termination |
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
- Initial source listing and structure inspection are legal discovery actions;
  they are validated as retrieval, not treated as premature finalization.

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

## Uniqueness rule

Each generated row claims a canonical matrix/source `type_signature` and a
wording-sensitive `instance_signature`. Exact and semantic-type duplicates are
rejected by the generation ledger. Planned paraphrase or counterfactual pairs
must be explicitly marked; they are not accidental duplicates.
