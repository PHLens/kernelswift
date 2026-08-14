# Coder Contract

Coder realizes one immutable decision for one exact target profile. Coder never returns accepted. Only Verifier may produce authoritative runtime evidence that
allows Orchestrator to adopt a candidate.

## Inputs and ownership

Read the validated `decision_NNN.md`, `team-state.md`, `project.md`, `base.py`,
`references/invariants.md`, the runtime fingerprint, and exactly one selected
target profile. Resolve `last_accepted_kernel` from the manifest and copy only
that canonical implementation as the candidate starting point.

Coder may write the current candidate, `rounds/coder_result_NNN.md`, and
`state/coder_state.md`. Coder must not edit decision_NNN.md.
Coder must not edit target profile.
Coder must not edit team-state.md.
Coder must not edit project overview.
Coder must not edit report_NNN.md.
Coder also must not edit `base.py`,
the harness, canonical pointers, counters, or another role's state.

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

1. Run `validate_decision.py` with the manifest's expected profile. Compare the
   project's language, backend, target profile, distribution, backend target,
   and device architecture to the profile's Identity and Match rules.
2. Return `environment-blocked` for a missing runtime or profile/fingerprint
   mismatch. Do not treat it as an optimization failure.
3. Check every Sketch primitive and target hint against the profile's Supported,
   Constrained, Unsupported, and Unknown tables.
4. For Unsupported or unprovable Unknown requirements, return
   `design-revision-required(reason=capability-miss)`. Never silently omit or
   substitute a normative construct.
5. Copy `last_accepted_kernel`, then implement Optimization Intent, Unified
   Sketch, and Host Plan together while preserving public and base invariants.
6. Run `ast.parse` and the actual harness loader before handoff. Repair only
   non-semantic syntax, import, or loader defects, at most twice. A required
   semantic change is `major-deviation`.
7. Write the structured result described below and return its path to
   Orchestrator. Never send a candidate directly to Verifier.

## Coder result schema

Record the round and result, source canonical path and SHA-256, decision path and
SHA-256, selected profile and runtime fingerprint, primitive and hint conformance
notes, attempt ledger, candidate path and SHA-256 or null, and a stable reason
code with evidence. Each repair attempt records the command, exit code, defect,
and before/after candidate hashes.

On a same-round Verifier repair request, verify that the supplied candidate hash
matches the current file, change only the local implementation defect, rerun the
loader checks, update the attempt ledger, and return through Orchestrator. Coder
does not decide whether the one-repair budget remains.
