# kernel-opt-loop v2 Continuous-Run Design

**Created**: 2026-08-14

**Status**: Approved

**Scope**: A delta specification for continuous, bounded optimization runs on top
of the approved v1 `kernel-opt-loop` contracts. It does not replace the v1
artifact, correctness, target-profile, or canonical-pointer semantics.

## 1. Motivation and scope

v1 makes individual rounds reproducible, but a practical optimization campaign
still has two costly failure modes: a terminal round can be mistaken for the end
of the workflow, and roles can repeatedly reload the same durable history. v2
turns a live invocation into one bounded optimization run with explicit global
stopping, inexpensive rejection of clearly poor candidates, and durable compact
role context.

v2 has these goals:

- continue automatically after every non-stopping terminal round;
- stop after a user-defined success target, a fixed optimization budget, or a
  verified performance plateau;
- report global progress every three completed terminal rounds without pausing;
- preserve authoritative measurement while avoiding expensive profiling of
  clearly poor candidates;
- reuse role identities and compact context when the runtime supports it; and
- retain a Git-auditable project evidence ledger without committing raw runtime
  noise.

v2 remains a single-machine, single-candidate workflow. It preserves the v1
invariants that `base.py` and harness semantics are user-owned and immutable,
and that only the accepted canonical implementation may seed a new candidate.

## 2. Non-goals

The following are deliberately outside v2:

- a daemon, queue, or automatic continuation after the live Orchestrator
  session exits;
- concurrent candidates, benchmarks, profiler runs, or background local work
  while verification is active;
- platform-enforced per-agent tool, skill, filesystem, or network ACLs;
- token-accounting telemetry when the runtime does not expose authoritative
  per-agent usage;
- KernelWiki publication, retrieval, or cross-project experience transfer;
- backend-neutral lowering, automatic backend selection, or deep profiler
  interpretation.

## 3. State model and terminology

`phase` remains the v1 operational state. v2 adds a workflow-level state that
is not inferred from a role completion notification:

```text
workflow_status = running | stopped | blocked
round_result = accepted | no-improvement | screened-out | design-rejected |
               candidate-failed | aborted
run_epoch = a bounded campaign over one immutable policy revision
```

`round_result` ends one round only. It never ends the workflow by itself.
`workflow_status: stopped` is written only by the global termination evaluator.
`blocked` preserves a safe recovery point and never consumes a round merely
because the environment failed.

Phase 0 produces the baseline and is round `000`; it does not consume any of
the v2 optimization-round budget.

## 4. Continuous control loop and global stopping

At the start of a run epoch, Orchestrator records this immutable policy:

```yaml
termination_policy:
  max_rounds: 20
  valid_no_improvement_limit: 3
  adoption_threshold_pct: 5
  target: null # or the comparable target schema in Section 4.2
```

After each terminal round has passed artifact gates and been committed,
Orchestrator performs this sequence exactly once:

```text
update pointers, counters, role digests, and project overview
evaluate global termination policy
  stopped -> write final transition and report; end the live run
  blocked -> write blocking transition and report; end the live run
  running -> send a three-round checkpoint when due; dispatch the next round
```

No role completion, local result classification, or checkpoint report calls
`end_workflow`. An environment block is handled before this terminal-round path
and writes `workflow_status: blocked` instead.

### 4.1 Hard stopping conditions

The live run stops when any of these conditions applies:

1. `total_rounds` reaches 20;
2. `performance_miss_streak` reaches 3;
3. the optional comparable target is reached;
4. the user requests stop; or
5. an environment problem cannot reach a safe recoverable boundary.

The first four conditions create `workflow_status: stopped`; the fifth creates
`workflow_status: blocked`. User stop is requested immediately but completes at
the current safe command boundary. A user may explicitly request immediate
interruption of a stuck command.

`performance_miss_streak` follows the valid-attempt rule:

- a complete authoritative `no-improvement` increments it;
- `accepted` resets it; and
- `screened-out`, `design-rejected`, `candidate-failed`, and `aborted` neither
  increment nor reset it.

This means three valid, comparable, correct attempts with no accepted gain stop
the run even when implementation failures occurred between them. A Designer may
not override this hard stop by proposing a fourth hypothesis.

### 4.2 Optional target / upbound

The user may provide a target after Phase 0 establishes a comparable baseline:

```yaml
target:
  mode: absolute_latency_ms | speedup_vs_baseline
  value: <positive number>
  metric: wall_time_ms
  direction: minimize
  measurement_fingerprint: <baseline fingerprint>
  source: user
```

The target is evaluated only from baseline or accepted-candidate authoritative
measurements under the recorded fingerprint. It is an early success condition,
not a substitute for the 5% adoption threshold. A baseline that already reaches
the target stops before Round 001. A target amendment is append-only, becomes
effective at the next safe terminal boundary, and never reclassifies past
results.

### 4.3 Three-round checkpoints

After rounds 3, 6, 9, and so on, while the run remains `running`, Orchestrator
sends one non-blocking progress message containing:

- completed rounds out of 20 and the valid miss streak;
- best latency and speedup versus baseline;
- distance to target when present;
- classifications from the most recent three rounds;
- current bottleneck, backlog status, and next hypothesis; and
- environment or watchdog incidents.

The checkpoint is derived from committed state and recent reports. It is not a
new durable artifact. `team-state.md` records `last_checkpoint_round` so a
resume does not resend it. A stopping third round emits only the final report.

## 5. Measurement, repair, and machine exclusivity

### 5.1 Compile-smoke and repair bounds

Before Verifier receives a candidate, Coder must prove it parses, the actual
harness loader can load it, and it completes a current-regime warm-up / compile
smoke execution. Coder may make the initial implementation plus at most two
local fixes driven by this gate. If it still fails, the result is
`candidate-failed`.

Verifier may request one implementation repair. The repaired candidate must
pass the same compile-smoke gate and then repeat the complete verification path.
A further implementation defect ends the round as `candidate-failed`.

### 5.2 Two-stage verification

The fast stage prevents obviously bad candidates from consuming profiler time:

1. correctness and two short interleaved candidate/reference timing pairs;
2. classify `screened-out` only when both pairs show the candidate at least 10%
   slower than the current accepted canonical implementation; and
3. send every other candidate to authoritative verification.

Authoritative verification uses the v1 comparable timing protocol and the 5%
adoption threshold. Only it may produce `accepted` or the valid
`no-improvement` that affects `performance_miss_streak`.

Scoped profiling is mandatory for the baseline and each accepted candidate. It
is also required when a result is close to the adoption boundary or current
bottleneck evidence is insufficient; it is skipped for `screened-out`
candidates. The exact close-boundary range and timing/watchdog multipliers are
target-profile measurement parameters, not role-local prompt guesses.

### 5.3 Measurement-exclusive machine policy

On a shared machine, `verifying` and `measuring` are exclusive phases. Only
Verifier may execute local commands in these phases. Designer, Coder, and
Orchestrator remain idle: they may not start processes, scan files, write
candidates, compile, profile, or warm caches. The lock releases only after the
Verifier command exits and its result is durably recorded.

Verifier uses a liveness watchdog derived from the equivalent baseline command,
with a target-profile multiplier and minimum guard value. It is not a short
performance deadline. A watchdog event is `environment-blocked`, consumes no
round or performance miss, and preserves the incident for recovery.

### 5.4 Terminal-result accounting

| Result | `total_rounds` | Performance miss | Failed-attempt streak | Canonical |
|---|---:|---:|---:|---|
| `accepted` | +1 | reset | reset | advance |
| `no-improvement` | +1 | +1 | unchanged | unchanged |
| `screened-out` | +1 | unchanged | unchanged | unchanged |
| `design-rejected` | +1 | unchanged | +1 | unchanged |
| `candidate-failed` | +1 | unchanged | +1 | unchanged |
| `aborted` | +1 | unchanged | +1 | unchanged |

`screened-out` reports correctness, both short timing pairs, the regression
magnitude, and the reason it did not enter authoritative verification. It is a
real budgeted attempt, but not evidence of a verified performance plateau.

## 6. Hypothesis agenda and role context

### 6.1 Rolling hypothesis backlog

After Phase 0, Designer maintains a ranked backlog of three to five candidate
hypotheses. Each item records a bottleneck, expected mechanism, expected gain,
risk, validation cost, evidence source, and change family. A formal round still
contains one immutable decision and one candidate.

An accepted result reprioritizes the backlog around the new canonical. A valid
`no-improvement` marks its hypothesis family as disproven; the next decision
must use a different change family unless new profiler or measurement evidence
explains why it can clear the 5% threshold. The three-round checkpoint is the
backlog review point. No background preparation occurs while Verifier owns the
machine.

### 6.2 Persistent identities and compact context

When supported by the runtime, Orchestrator creates at most one Designer, Coder,
and Verifier identity and reuses an idle identity for later turns. A role task
ending means `idle`, not destroyed. Agent IDs remain session-local and are never
written into project artifacts.

Each role maintains a compact, durable context state containing:

- role-contract hash and context epoch;
- last completed round and accepted report/kernel pointers;
- current bottleneck and the recent three-round evidence summary;
- open hypotheses or checks; and
- an artifact read-hash ledger.

A cold start or rehydrate reads the complete role contract and required
authoritative artifacts. A continuation reads only the compact digest, changed
artifacts, and current task inputs. The digest is invalidated by role
replacement, contract/profile hash change, base/harness/fingerprint/policy
change, canonical-pointer mismatch, or failed three-round reconciliation.

Runtimes that cannot preserve identities must retain the same artifact semantics
and rehydrate from this compact state. Their adapters declare
`persistent_role_session: false` and `effective_context_mode: rehydrate`; a
supporting runtime declares `true` and `continuation`.

## 7. Policy revision and resume

Within one run epoch, the measurement fingerprint, 5% adoption threshold,
20-round budget, three-miss limit, screening threshold, and repair budgets are
immutable. The Section 4.2 target is the only policy item that may be appended
at a safe terminal boundary. Changing any other item after stop creates a new
run epoch rather than rewriting history.

`recover` is reserved for an uncommitted interruption or `blocked` environment:
it resumes the same safe step without allocating a round or changing counters.

`new run epoch` is required after `stopped`. It retains all prior evidence and
the accepted canonical implementation but resets epoch-local round and miss
counters. It requires an explicit reason:

- a budget extension after 20 rounds;
- new evidence, a new bottleneck hypothesis, or new capability after three
  valid misses;
- a stretch target after target reached; or
- explicit user intent after manual stop.

## 8. Git evidence ledger and run branch

Each optimization project uses a dedicated run branch by default:

```text
kernel-opt/<operator>-<run-epoch-or-timestamp>
```

Phase 0 records the clean `base_branch`, `base_commit`, and `run_branch` in
`team-state.md`. The workflow never silently writes its automatic commits to
`main`, `master`, or `dev`. A user may explicitly reuse an already dedicated
optimization branch. All roles share the same run worktree; they do not switch
or create independent branches.

The Git history is an evidence ledger. It tracks `baseline_adapter.py`, project
and team state, decisions, Coder results, reports, final round statuses,
candidate source snapshots when they exist, role context digests, environment
incidents, and final summaries. Raw profiler traces, benchmark stdout/stderr,
build caches, virtual environments, session IDs, bootstrap messages, and
secrets remain gitignored.

Normalized reports reference any raw log through its relative path, SHA-256,
and exact command. Large traces may later use Git LFS or external object storage,
but neither is required by v2.

One atomic commit is written after successful Phase 0, after each terminal
round, after each blocking incident, and at final stop. Intermediate phase
updates may exist for crash recovery but do not receive their own commits.

## 9. Acceptance requirements

The v2 implementation must cover these contract fixtures without an MLU device:

1. a non-stopping terminal result dispatches the next round instead of ending
   the workflow;
2. only authoritative `no-improvement` changes the valid performance-miss
   streak, and `accepted` resets it;
3. the third valid miss and twentieth terminal optimization round stop exactly,
   while Phase 0 does not count;
4. baseline or accepted performance meeting the optional target stops before a
   subsequent round;
5. checkpoints occur once at rounds 3, 6, 9, and so on, remain non-blocking,
   and do not duplicate a final report;
6. `verifying` and `measuring` reject non-Verifier local dispatches;
7. same-session roles use continuation, while each documented invalidation path
   forces compact-state rehydrate;
8. compile-smoke and Verifier-repair limits produce the defined terminal
   classifications; and
9. terminal commits contain the tracked evidence set and exclude raw logs.

## 10. Future work

This section combines the still-unimplemented v1 future directions with v2
deferrals. Implemented v1 work is intentionally absent: the runtime-neutral
role contracts, concrete `triton_mlu` profile, durable artifacts, measurement
helpers, adapters, and their contract tests are not future work.

### 10.1 Portable design and multi-target lowering

Evolve v1's target-bound Sketch into a backend-neutral Core IR with explicit
target hints, a capability matrix, and deterministic lowering. Automatic
backend/DSL selection, capability-miss fallback routing, and multi-target
candidate generation are eligible only after at least two complete target
profiles, cross-backend examples, stable semantics, and conformance/lowering
tests exist.

### 10.2 Evidence intelligence

Add backend-aware profiler interpretation beyond v1/v2 scoped summaries:
critical-path reconstruction, host/device timelines, kernel-signature clustering,
launch and synchronization attribution, allocator attribution, and target
metrics such as bandwidth, occupancy, registers, and shared memory. Any
automatic intent-to-mechanism conclusion needs evidence-quality controls rather
than heuristic narration.

### 10.3 Cross-task experience transfer (KernelWiki)

Design a bounded, Git-versioned knowledge layer so verified experience from one
optimization task, such as `groupedtopk`, can inform another, such as
`fused_moe`. It must transfer only qualified mechanisms, preconditions,
counterexamples, and provenance; it must never transfer absolute performance,
exact tuning parameters, code, or acceptance decisions without fresh local
verification. The future design should define experience-card schemas, retrieval
ranking, promotion approval, and stale-evidence handling.

### 10.4 Durable autonomous execution and enforcement

Add cross-session scheduling only with an explicit daemon/queue ownership model
and recovery lease. Add genuine per-agent tool, skill, filesystem, and network
ACLs only when the runtime can enforce them rather than merely carrying them in
a bootstrap contract. Runtime-provided per-agent token telemetry can later feed
a usage ledger; proxy estimates must remain visibly non-authoritative.

### 10.5 Isolated parallel search

Explore parallel candidates only with hardware and measurement isolation strong
enough to preserve comparable evidence. It is incompatible with v2's
single-machine measurement-exclusive policy.

## 11. Governance

This document is the v2 architectural source of truth. The next implementation
plan must cite its sections, assign file ownership, and add the Section 9
fixtures. It must not introduce Future Work as implementation scope without a
new approved specification revision.
