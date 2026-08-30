# Role-aware query contract

KernelWiki role queries are offline, read-only views over the validated local corpus. They do not create campaign authority or modify `kernel-opt-loop`.

## Context

Pass a versioned JSON context with `query.py --context <path>`. Both roles provide:

- `schema_version`, `role`, and the query target;
- languages, dtypes, kernel types, semantic features, shape signature, and current bottlenecks;
- implementation profile status and runtime fingerprint when known.

A Designer context may omit loop artifacts. A Coder context must name the exact implementation profile/runtime and pin the project document, profile, runtime snapshot, project claim, Sketch, and Decision by relative path and SHA-256. It also pins the checked-in loop contract identity and maps each guidance ID to its frozen Sketch statement IDs.

## Read-only loop bridge

The bridge reads only the checked-in allowlisted `kernel-opt-loop` validators and records their Git/tree/file identity. It validates the Coder context against committed authority; it never edits profiles, prompts, Sketches, Decisions, Coder results, campaigns, or project state. Unsupported authority fails closed.

## Admission and ranking

Admission runs before lexical ranking and before group limits.

- Designer can receive exact, family, backend, analogy-only, counterexample, measurement, and capability-gap evidence. Every item exposes its match class.
- Coder admission requires the exact target, profile, runtime, supported capability/version, preserved Sketch/Decision semantics, approved provenance, and explicit guidance-to-Sketch binding.
- Coder never falls back across target, backend, profile, language, or runtime.

Results use stable groups in this order: `admitted`, `conditional`, `analogy_only`, `counterexamples`, `capability_gaps`, `excluded`. Admission entries expose stable reason codes such as `profile-missing`, `target-mismatch`, `capability-unknown`, `version-stale`, `source-broken`, and `sketch-binding-required`.

## Cards, examples, guidance, and assets

Card admission does not grant every cited item. Examples, guidance blocks, and assets are admitted separately. Code is visible only when its Source provenance, license, audience, asset mode, exact profile, and requested item ID all pass admission. A result can therefore expose Card metadata while denying one or more assets.

## Limits and receipts

`--limit` sets the default per-group limit; `--group-limit NAME=N` overrides a group. `--show-excluded` makes denied items visible with reasons.

Without `--output`, the canonical result is printed. `--output <path>` writes the same canonical JSON as an explicit receipt outside active project state. A receipt contains its schema version, context SHA-256, loop contract identity, authority artifact hashes, grouped results, and admitted guidance IDs. The referenced context carries the exact guidance-to-Sketch-statement binding.

A receipt is advisory query output. It is not a Decision, Coder result, consultation record, KnowledgePacket, required dossier, campaign artifact, or Designer-to-Coder handoff, and writing it does not mutate prompts or campaign state.

## Missing AscendC authority

The checked-in corpus has AscendC evidence but no canonical exact-profile AscendC authority. A real AscendC Coder query is therefore expected to return no implementation guidance, no code, and no fallback to Triton or CUDA. Metadata and capability-gap explanations may still be returned.
