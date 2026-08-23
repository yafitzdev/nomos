# Next goal prompt: registry-aware pilot and generalization gate

Copy the prompt below into a new goal request.

```text
Build, validate and evaluate the first registry-aware Nomos router.v2 pilot in
the current repository. This is an execution goal, not a design-only report.

Read AGENTS.md and docs/GENERIC_ROUTER_V2.md first. Preserve router.v1 and all
existing V1 artifacts. Do not upload or publish an artifact during this goal.

Current foundation:
- tool-registry.v2 validation and fingerprints exist;
- runner-request.v2 and decision-state.v2 contracts exist;
- router.v2 scores identity-free tool metadata against observable state;
- matrix.v2 uses target capabilities and explicit holdout axes;
- the Fitz-Sage V2 vocabulary is isolated in an adapter and registry config;
- invariance and contract smoke tests exist.

Objective:
Produce a 5,000-decision-state pilot that can determine whether the architecture
generalizes to unseen tool IDs and whether it is ready to scale to 30k-50k
states.

Required work:

1. Audit the V2 foundation and fix any contract, matrix, adapter or feature bug
   exposed by end-to-end use. Do not reintroduce literal tool IDs or
   sampling-only oracle fields into model features.

2. Implement the matrix.v2 scenario/state generator. Every row must record:
   - matrix cell and canonical uniqueness signature;
   - registry ID and fingerprint;
   - source cards and immutable content hashes;
   - prompt/model/teacher identity when applicable;
   - seed, feature version and validator version;
   - explicit evaluation cohort and split-group IDs.

3. Generate exactly 5,000 unique decision states with balanced capability
   coverage. Target at least 200 positive states per important target
   capability. Prefer same-family and capability-overlap hard negatives over
   obviously unrelated negatives. Support multiple acceptable tools where the
   deterministic validator permits them.

4. Use multiple registries totaling at least 30 concrete tool IDs and at least
   8 tool families. The additional registries must use realistic descriptions,
   capabilities, modalities and schemas; do not create meaningless aliases
   whose only difference is the ID.

5. Freeze disjoint evaluation cohorts before training:
   - familiar tools with held-out states;
   - unseen tool IDs from trained families;
   - exact semantic tool-ID renames;
   - held-out schema/modality variants;
   - at least one entirely held-out tool family;
   - held-out question templates and source documents;
   - at least one alternate external-agent registry.
   No held-out tool descriptor, source document or question template may enter
   training.

6. Train learning-curve checkpoints at approximately 1k, 2.5k and the full
   training partition. Use deterministic seeds and keep artifacts separately
   named. Report candidate-pair counts and class/capability balance.

7. Evaluate every cohort with:
   - Recall@1 and Recall@3;
   - MRR;
   - fixed candidate-order and frequency baselines;
   - invalid-candidate rate;
   - candidate-order invariance;
   - tool-ID-renaming score invariance;
   - question-removal ablation;
   - tool-metadata removal ablation;
   - sampling-context leakage test.

8. Treat these as hard correctness gates:
   - the router never returns an ID outside legal_candidate_ids;
   - candidate order does not change candidate scores;
   - renaming an ID with identical metadata does not change its score beyond
     floating-point tolerance;
   - sampling-only fields do not change any score;
   - the generic router imports no Fitz-Sage implementation;
   - frozen holdouts have zero train overlap.

9. Use NInfer for up to 1,000 matrix-bound teacher scenarios only if the local
   endpoint is available. Validate a reproducible 25-row sample and retain
   teacher proposals as proposals until deterministic execution accepts them.
   If NInfer or governance-valid execution is unavailable, finish the
   deterministic pilot and report the external blocker precisely. Do not use
   a missing external service as a reason to stop safe local work.

10. Decide whether to scale. Recommend the 30k-50k run only if unseen-tool
    performance beats both baselines, all correctness gates pass, question
    ablation shows that the objective contributes useful signal, and the
    learning curve has not already plateaued. Do not claim unseen-family
    generalization merely because unseen IDs from known families perform well.

Run the full test suite, Ruff and all new smoke/evaluation commands. Finish with
a concise report covering changed files, dataset composition, validation,
metrics by cohort, failed gates, and the exact next scaling recommendation.
```
