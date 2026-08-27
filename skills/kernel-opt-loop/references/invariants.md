# Workflow Invariants

These rules apply to every kernel-opt-loop project. Project-specific semantic and
environment invariants belong in `project.md`; a decision may narrow its allowed
change boundary but may not weaken this file.

## Immutable Reference and Harness

- `base.py` is user-owned and immutable. No role edits, formats, replaces, or
  commits a generated version over it.
- Phase 0 generates `baseline_adapter.py` from `base.py`. The adapter is the
  initial executable canonical implementation; its generation must leave the
  bytes of `base.py` unchanged.
- The public constructor, forward signature, output structure, documented
  numerical semantics, dtype, shape, and tolerance/tie rules remain compatible
  with the reference unless the user explicitly changes the project contract.
- The harness, shape, dtype, device, warmup/repeat counts, profiler settings, and
  reference bytes are part of `measurement_fingerprint`. A change requires a new
  comparable baseline and must not be presented as an in-regime speedup.

## AST Loader Behavior

- Phase 0 inspects the actual harness loader and records its behavior. A harness
  that filters a module with an AST may preserve only selected top-level nodes;
  Coder must write imports, definitions, and initialization in forms that this
  discovered loader retains.
- The adapter generator identifies exactly one top-level `Model` class and
  renames it to `ModelNew`. Zero or multiple top-level `Model` definitions are an
  error, not a guess.
- Generated and candidate modules must expose the entry points required by the
  discovered harness, including `ModelNew`, `get_inputs`, and `get_init_inputs`
  when the harness requires them.
- Historical observations such as a particular loader stripping non-literal
  module assignments are project evidence, not universal Triton or MLU behavior.

## Canonical Pointers

- `last_accepted_kernel` and `last_accepted_report` are the only canonical
  comparison pointers. Every candidate starts from `last_accepted_kernel` and
  every benchmark compares against it.
- Only an `accepted` terminal result advances the canonical pointers.
  `no-improvement`, `design-rejected`, `candidate-failed`, and `aborted` leave
  them unchanged.
- A rejected or failed candidate remains auditable but never becomes a future
  starting point.
- A validated `decision_NNN.md` is immutable after coding dispatch. A required
  Sketch, Host Plan, algorithm, dataflow, lifecycle, or evaluation change closes
  the round as `design-rejected`; the replacement decision belongs to a new
  round after the terminal commit.
- An environment incident blocks the current safe step. It does not complete a
  round, advance a canonical pointer, or modify either progress streak.
- `total_rounds` increments exactly once for each `accepted`, `no-improvement`,
  `design-rejected`, `candidate-failed`, or `aborted` terminal result.
  `performance_miss_streak` increments only for `no-improvement`;
  `failed_attempt_streak` increments only for `design-rejected`,
  `candidate-failed`, or `aborted`. Either failure class resets the other
  consecutive streak, and `accepted` resets both.

## Buffer, Device, and Stream Lifecycle

- Buffer allocation, reuse, cache keys, invalidation, ownership, and lifetime
  must follow the validated Host Plan. Coder does not introduce implicit global
  caches or longer-lived state outside that boundary.
- A cached buffer is reused only when every declared cache-key component matches;
  shape, dtype, and device are mandatory whenever they affect compatibility.
- The Host Plan declares concurrency assumptions. State owned by one model
  instance is not silently shared across model instances or concurrent forwards.
- Candidate code preserves the caller-selected device and current stream unless
  the Host Plan explicitly specifies and justifies another behavior.
- Removing a device context or caching an output can be considered only after
  the local runtime and harness prove that the lifecycle remains correct. Prior
  project wins are not universal target guarantees.

## Role Ownership

- **Orchestrator** alone writes `team-state.md`, project overview rows, canonical
  pointers, counters, round transitions, workflow commits, and verdict
  artifacts. Before campaign state exists, Orchestrator owns pre-campaign profile
  onboarding and may write only the isolated probe root, normalized qualification
  input, results, evidence, promotion candidate, and note; it never edits the
  canonical implementation profile.
- **Designer** alone writes the current uncommitted decision, the typed Sketch,
  causal graph references, and `state/designer_context.md`. Designer does not
  write candidate code, runtime measurements, a runtime fact pack, or a verdict.
  In read-only capability preflight Designer identifies explicit primary/fallback
  pairs without writing campaign files; it never equates an Unknown capability
  with unavailable.
- **Coder** alone writes the current candidate, `rounds/coder_result_NNN.md`, the
  Decision-local binding ledger, and `state/coder_context.md`. Coder never
  returns `accepted`, never changes canonical state, and never owns the canonical
  profile, the initial project capability claim, or a verdict.
- **Verifier** alone performs authoritative runtime execution and writes the
  current report/status, profiler outputs, the structured fact pack, observed
  lowering, and `state/verifier_context.md`. Verifier does not edit the
  candidate, the decision, the binding, or canonical pointers, and assigns no
  design/code blame.
- Roles may exchange advisory context, but every state-changing response goes to
  Orchestrator and every actionable result is recorded in a durable artifact.

## Probe, Profile, and Attribution Boundaries

- Profile probing may run versioned probe definitions, emit hashed run-local
  evidence and a proposed promotion candidate, and stop without creating a
  campaign. It never mutates the canonical implementation profile and never
  allocates `team-state.md`, rounds, Decisions, candidates, reports, verdicts,
  or benchmark rankings.
- `promotion-pending` is an onboarding disposition, not a campaign terminal
  result. Raw pre-campaign, Coder, or Verifier probe evidence never satisfies a
  campaign claim and never updates the canonical profile during a campaign.
- The immutable `state/project_capability_claim.json` is the only fallback
  authority. A fallback Decision references the embedded disposition id/hash and
  never a raw probe-result reference; deleting the pre-campaign run must not
  invalidate campaign history.
- Source binding proves source-level conformance to the Decision. Observed
  lowering is a separate Verifier-owned claim; a source statement is never
  required to map one-to-one to a final device kernel.
- `lowering-unknown` terminates as `design-rejected` but does not increment
  `failed_attempt_streak`; explicit `design-error` does increment it. The run
  policy consumes the attribution counter effect rather than treating every
  `design-rejected` round as equivalent.
- Submission finalization is one offline bounded configuration-only gate per
  eligible Triton submission snapshot. It uses the existing Decision, report,
  binding, and verdict families with `artifact_kind: submission-finalization`
  and an artifact index; it never updates campaign-round pointers, terminal
  fields, attribution, or counters, and adds no manifest pointer. The final
  candidate contains one fixed selected configuration and no runtime/online
  autotune, first-use search, or cache-dependent selection.

## Measurement Attribution

- Level 0 correctness and interleaved paired wall timing are required for every
  candidate. Benchmark wall time, using unrounded medians, controls adoption.
- After correctness passes, Level 1 records device time per call, kernel count
  per call, and top-k kernel breakdowns in separate reference and candidate scopes.
- Multi-iteration profiler totals are divided by the number of forward calls
  before comparison. Reference and candidate events are never combined.
- Profiler time is diagnostic evidence, not a substitute for benchmark wall
  time. Adoption requires correctness, every guardrail, and at least 5% unrounded
  median wall improvement against `last_accepted_kernel`.
- Each Evaluation Contract observable is mirrored by exact name with expectation,
  observation, verdict, and evidence. Missing required evidence produces
  `measurement-incomplete`, not an inferred success or failure.
- Mixed kernel/host changes are allowed only when inseparable and separately
  observable. Otherwise each round changes one intervention so its result remains
  attributable.
