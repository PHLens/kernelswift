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
  pointers, counters, round transitions, and workflow commits.
- **Designer** alone writes the current uncommitted decision and
  `state/designer_state.md`. Designer does not write candidate code or runtime
  measurements.
- **Coder** alone writes the current candidate, `rounds/coder_result_NNN.md`, and
  `state/coder_state.md`. Coder never returns `accepted` and never changes
  canonical state.
- **Verifier** alone performs authoritative runtime execution and writes the
  current report/status, profiler outputs, and `state/verifier_state.md`.
  Verifier does not edit the candidate or the decision.
- Roles may exchange advisory context, but every state-changing response goes to
  Orchestrator and every actionable result is recorded in a durable artifact.

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
