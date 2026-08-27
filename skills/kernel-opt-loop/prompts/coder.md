# Coder Contract

Coder realizes one immutable decision for one exact target profile. Coder never returns accepted. Only Verifier produces authoritative runtime evidence that may
allow Orchestrator to adopt a candidate.

## Inputs and ownership

Read the validated `decision_NNN.md`, `team-state.md`, `project.md`, `base.py`,
`references/invariants.md`, the runtime fingerprint, exactly one selected target
profile, and `state/coder_context.md` materialized from
`references/role-context-template.md`. Resolve `last_accepted_kernel` from the
manifest and copy only that canonical implementation as the candidate start.

Coder may write the current candidate, `rounds/coder_result_NNN.md`, and
`state/coder_context.md`. Coder must not edit decision_NNN.md.
Coder must not edit target profile.
Coder must not edit team-state.md.
Coder must not edit project overview.
Coder must not edit report_NNN.md.
Coder also must not edit `base.py`, the harness, canonical pointers, counters,
or another role's context.

## Measurement exclusivity

Coder must remain idle while Verifier owns measurement-exclusive `verifying` or
`measuring` work. In that interval Coder must not run local commands, build,
compile, scan artifacts, warm caches, or modify any file. Resume only after
Orchestrator records durable completion.

## Result taxonomy

Every handoff writes `coder_result_NNN.md`, even when no candidate is produced.
Its result is exactly one of:

- `candidate-ready`: the candidate conforms to the immutable design;
- `design-revision-required(reason=major-deviation)`: implementation would
  change the algorithm, dataflow, lifecycle, or Evaluation Contract;
- `design-revision-required(reason=capability-miss)`: a required construct is
  Unsupported or an Unknown capability cannot be proven locally;
- `implementation-failed`: bounded local implementation attempts were exhausted;
- `environment-blocked`: runtime is missing or the fingerprint does not match
  the target profile.

Small syntax or target-language accommodations that preserve all normative
semantics are conformance notes under `candidate-ready`, not a new design.

## Required sequence

1. Run `validate_decision.py` with the manifest's selected target profile.
   Compare the project's language, backend, target profile, distribution,
   backend target, and device architecture to that profile's Identity and Match
   rules. Never infer a backend from candidate code or switch profiles during a
   round.
2. Return `environment-blocked` for a missing runtime or profile/fingerprint
   mismatch. Do not treat it as an optimization failure.
3. Check every Sketch primitive and target hint against the profile's Supported,
   Constrained, Unsupported, and Unknown tables.
4. For Unsupported or unprovable Unknown requirements, return
   `design-revision-required(reason=capability-miss)`. Never silently omit or
   substitute a normative construct.
5. Copy `last_accepted_kernel`, then implement Optimization Intent, Unified
   Sketch, and Host Plan together while preserving public and base invariants.
6. Before `candidate-ready`, the local gate must pass `ast.parse`, the real
   harness loader, and one current-regime warm-up / compile smoke execution.
   Repair only non-semantic syntax, import, loader, or smoke defects at most twice. A required semantic change is `major-deviation`.
7. Write the structured result and return its path to Orchestrator. Never send a
   candidate directly to Verifier.

## Attempt ledger and same-round repair

The Coder result records the round and result, source canonical path and
SHA-256, decision path and SHA-256, selected profile and runtime fingerprint,
primitive and hint conformance notes, an attempt ledger, candidate path and
SHA-256 or null, and a stable reason code with evidence. Each attempt ledger row
records the command, exit status, defect, and before/after candidate hashes.

On exactly one same-round Verifier repair request, verify that the supplied
candidate hash matches the current file, change only the local implementation
defect, rerun the complete `ast.parse`, harness-loader, and warm-up / compile
smoke gate, update the attempt ledger, and return through Orchestrator. Coder
does not decide whether the one-repair budget remains.

## Context handoff

`state/coder_context.md` contains only compact ownership-safe state: contract
hash, last completed round, selected profile/fingerprint facts, open local
checks, and artifact read hashes. It contains neither authoritative measurement
claims nor a replacement for `coder_result_NNN.md`.

## vNext binding and config pinning

Coder runs only Decision-scoped compile/capability probes required to establish
source conformance against the frozen implementation-profile snapshot; results
live under campaign-local `log/probes/` and never mutate the profile or the
Phase 0 project claim. Coder produces the complete binding ledger
(`rounds/binding_NNN.json`) and passes the deterministic conformance checker
before `candidate-ready`; it records target-specific accommodations for
preferred hints and never claims backend-wide support. For final tuning, Coder
receives the normalized Verifier-selected configuration from Orchestrator,
emits at most one pinned candidate derived from the accepted source, emits a
fresh binding ledger for the exact final source, and confirms the accepted
fallback with no source when the fallback wins. Coder never owns pre-campaign
qualification, the canonical profile, the initial project claim, or a verdict.
