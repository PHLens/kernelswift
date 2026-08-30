---
name: kernel-opt-loop
description: Coordinate a bounded, continuous Triton kernel or operator optimization campaign against an auto_bench-style harness using durable Designer, Coder, and Verifier artifacts.
---

# Kernel Optimization Loop

This skill is the runtime-neutral Orchestrator contract. It owns workflow state,
artifact gates, deterministic routing, canonical pointers, project overview
updates, Git commits, global termination, and recovery. Designer, Coder, and
Verifier own only the files declared in their role contracts. One live campaign
has one active candidate and a measurement-exclusive shared machine.

## Deliverable requirement

The deliverable is a Triton implementation. A campaign must always produce a
correctness-PASS Triton candidate as its committed deliverable. Never leave a
target empty because its Triton candidate does not beat the baseline: a correct
but non-winning Triton implementation is still the required submission, and the
final summary must record it as the canonical deliverable rather than an empty
result. `aborted`, `no-improvement`, and `screened-out` are workflow-terminal
states, not a license to omit the candidate source.

## When to use

Use this skill when a project has an immutable PyTorch-style `base.py`, an
`auto_bench.py`-style harness, and a request for iterative Triton operator or
kernel optimization. It preserves correctness, benchmark wall time, and
attributable profiler evidence across bounded continuous rounds.

Do not use it for a one-shot bug fix, a non-iterative refactor, a workflow that
may modify reference or harness semantics, or a target without a complete
profile. Existing optimization projects are not migrated automatically.

## Required inputs

Resolve these before mutation:

1. Absolute project root and immutable `base.py` path.
2. Absolute harness path and its actual module-loading behavior.
3. Absolute interpreter path, selected device, and required environment.
4. Shape, dtype, correctness tolerances, warmup, repeat, profiling mode,
   profiling warmup, and profiling iteration settings.
5. An optional user target only when supplied as `absolute_latency_ms` or
   `speedup_vs_baseline`; otherwise leave it null.
6. Absolute skill root containing this file, the target-profile registry, one
   matching complete profile, role contracts, templates, validators, helpers,
   evaluator, and one runtime adapter.

Repository layout convention: the operator's immutable reference is a shared,
device-neutral `base.py` at `<operator>/base.py`, and a campaign root is
`<operator>/<backend>/` without its own base copy. Record the base in
`project.md` as a relative path such as `../base.py`; resolve it to an absolute
path before mutation and record its starting bytes and SHA-256. Never copy or
edit the shared base.

Ask only for undiscoverable user-owned values. Never infer a device,
interpreter, concurrency promise, target, or semantic tolerance from a
candidate.

## Runtime selection

Choose in this exact order: Codex collaboration when exposed, Claude Code agent teams when enabled, then sequential fallback.

Load exactly one runtime adapter and use its common operations. Multi-agent
availability changes orchestration mechanics, not artifacts, ownership, routing,
or state semantics. Sequential fallback executes each role contract in the main
session without nested agent processes and still respects every ownership and
handoff gate.

## Agent bootstrap contract

Orchestrator resolves every placeholder to an absolute path, supplies only the
current phase inputs and required outputs, and sends this compact bootstrap. Do
not paste the complete role contract into a role start message.

```text
You are the <role> for kernel-opt-loop.

Before taking any action, read these files completely and follow them:
- Role contract: <absolute-skill-root>/prompts/<role>.md
- Runtime adapter: <absolute-skill-root>/adapters/<runtime>.md

Skill root: <absolute-skill-root>
Project root: <absolute-project-root>
Current phase: <phase-or-round>
Inputs:
- <absolute-input-path>
Required outputs:
- <absolute-output-path>

Do not rely on parent conversation history. Do not write files outside your
declared ownership. Report completion through the runtime adapter.
```

The role reads its full contract. State-changing role responses are advisory
until the required durable artifact exists and passes its gate.

## Phase 0

Perform initialization in this order:

1. Resolve absolute skill, project, base, harness, interpreter, and device
   paths. Record starting SHA-256 and bytes of `base.py` and the harness.
2. Record a clean Git base branch and commit. Create the dedicated run branch
   `kernel-opt/<operator>-<run-epoch-or-timestamp>` unless the user explicitly
   authorizes an existing dedicated branch. Reject automatic execution on
   `main`, `master`, or `dev`.
3. Create `rounds/`, `state/`, and gitignored `log/`. Materialize `project.md`,
   `team-state.md`, and `state/designer_context.md`, `state/coder_context.md`,
   and `state/verifier_context.md` from their templates. Only Orchestrator
   writes the manifest and project overview.
4. Discover implementation language, backend, target profile, Triton
   distribution/version, active backend target/version when available, and
   device architecture. Select exactly one matching complete profile from
   `prompts/coder_targets/`; never fall back across backends. A missing runtime,
   missing profile, or identity mismatch is an environment block.
5. Dispatch Designer for Phase 0 semantic analysis. Request only unknown
   user-owned values and validate its writable-file boundary before continuing.
6. Run `scripts/make_baseline_adapter.py` to create `baseline_adapter.py` by
   renaming the one top-level `Model` to `ModelNew`. Verify recorded `base.py`
   bytes are unchanged.
7. Dispatch Verifier for baseline correctness, benchmark wall samples, and the
   required baseline profiler evidence. Require `rounds/report_000.md` with
   result `baseline`, fingerprints, normalized evidence, and exact reproduction
   commands.
8. Compute the measurement fingerprint as SHA-256 over base bytes, NUL, harness bytes, NUL, and canonical JSON measurement settings. Serialize with
   `sort_keys=True` and `separators=(',', ':')`. The JSON object has exactly
   these keys: `"shape"`, `"dtype"`, `"device"`, `"warmup"`, `"repeat"`,
   `"profile_mode"`, `"profile_warmup"`, and `"profile_iterations"`.
9. After all gates pass, set `last_completed_round: "000"`,
   `last_accepted_round: "000"`, `last_accepted_kernel: baseline_adapter.py`,
   `last_accepted_report: rounds/report_000.md`, completed report pointers,
   `last_result: baseline`, `phase: ready`, and `workflow_status: running`.
   Append Phase 0 transition and overview rows, then commit the artifacts.

On a Phase 0 environment failure, create an incident, set `phase: blocked` and
`workflow_status: blocked`, leave accepted pointers null, append and commit the
blocking transition, and report remediation. Phase 0 is not a completed round.

## Round N

Round number is `total_rounds + 1`, formatted as three digits. Never allocate a
second active round. Resolve `last_accepted_kernel` and
`last_accepted_report` before dispatch.

1. Set `phase: designing`. Dispatch Designer with canonical evidence, recent
   completed evidence, exact profile, project invariants, and anti-pattern
   guidance. Designer writes one `decision_NNN.md`.
2. Run `scripts/validate_decision.py` with the manifest target profile. Record
   decision hash. A proceeding decision is immutable before coding.
3. A valid abort form produces terminal result `aborted` without Coder or
   Verifier. An `aborted` result does not erase the deliverable requirement: if
   a correctness-PASS Triton candidate already exists, keep it as the canonical
   deliverable; if none exists, the Orchestrator must still dispatch a minimal
   correctness-PASS Triton implementation before ending the run.
4. Set `phase: coding`. Dispatch Coder with immutable decision and canonical
   source. Require `coder_result_NNN.md` and, for `candidate-ready`, matching
   candidate hashes and compile-smoke evidence.
5. Apply the routing table. Never dispatch Coder directly from Designer or
   Verifier directly from Coder; all handoffs pass through Orchestrator.
6. Set `phase: verifying` and `measurement_exclusive: true` only when Verifier
   owns local commands. Require durable status updates and route at most one
   same-round repair or missing-evidence request through Orchestrator.
7. Validate all artifacts and hashes before calculating one terminal result.
   Set `last_completed_round`, `last_completed_decision`,
   `last_completed_coder_result`, `last_completed_report`, and `last_result`,
   using null where no artifact exists. Append exactly one overview row and one
   terminal transition row.
8. Clear `measurement_exclusive` at durable completion. The terminal artifact
   gate is the only input to the continuous run controller below.

`round_status_NNN.md` is updated at verification start, after correctness, after
each timing pair, and at verification end. Completion notifications are
preferred when the adapter supports them; Orchestrator does not poll a runtime
that already delivers completion.

## Routing and state transitions

Only Orchestrator transitions the manifest. Allowed phases are
`initializing|ready|designing|coding|verifying|repairing|measuring|blocked|stopped`.

| Producer classification | Orchestrator action |
|---|---|
| Designer abort | Complete `aborted`; do not dispatch Coder or Verifier. |
| Coder `candidate-ready` | Dispatch Verifier after recorded Coder gate. |
| Coder `major-deviation` or `capability-miss` | Complete `design-rejected`; preserve evidence and keep canonical unchanged. |
| Coder `implementation-failed` | Complete `candidate-failed`; keep canonical unchanged. |
| Any `environment-blocked` | Write/preserve incident, set `blocked`, and report remediation; no terminal round. |
| Verifier `implementation-repair-required` | Set `repairing` and return to Coder exactly once in the same round. |
| Verifier requires a design change | Complete `design-rejected`; never edit the current decision. |
| Verifier correctness fails after repair | Complete `candidate-failed`. |
| Verifier `measurement-incomplete` | Set `measuring`, collect named probe, then return to Verifier; classify design or environment if impossible. |
| Verifier terminal evidence | Complete `accepted`, `no-improvement`, or `screened-out` from the role contract. |

Environment incidents update neither counter nor `total_rounds`. Rejected
candidates remain auditable and never become the next source baseline.

## Continuous run controller

A round_result is not workflow termination. At every terminal artifact gate,
commit the terminal artifacts before evaluating the pure helper. The required
routing is:

```text
terminal artifact gate -> terminal commit -> evaluate_run_policy.py
  workflow_status=running -> optional checkpoint -> continue idle Designer
  workflow_status=stopped -> final summary commit -> end_workflow
  workflow_status=blocked -> incident commit -> blocking report -> end live run
```

Build a JSON input projection from the manifest; it is evaluator input, not a
second persisted state store. For example:

```bash
python3 <skill-root>/scripts/evaluate_run_policy.py \
  --state-json '{"total_rounds":2,"performance_miss_streak":2,"failed_attempt_streak":0,"last_checkpoint_round":null,"max_rounds":20,"valid_no_improvement_limit":3}' \
  --result no-improvement
```

Apply returned counters, `workflow_status`, `phase`, `stop_reason`, and
`last_checkpoint_round` atomically to `team-state.md` and its transition log.
When `workflow_status=running`, commit, then and only then dispatch the next round through the idle Designer.
When stopped, write and commit the final summary before ending the live run.
When blocked, commit the incident and send the blocking report before ending the
live run.

## Global termination policy

The frozen default policy is:

```yaml
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
```

Every terminal result increments `total_rounds` exactly once. Counter effects
are exact:

| Result | `performance_miss_streak` | `failed_attempt_streak` | Canonical |
|---|---:|---:|---|
| `accepted` | reset to 0 | reset to 0 | Advance candidate and report. |
| `no-improvement` | increment by 1 | unchanged | Unchanged. |
| `screened-out` | unchanged | unchanged | Unchanged. |
| `design-rejected` | unchanged | increment by 1 | Unchanged. |
| `candidate-failed` | unchanged | increment by 1 | Unchanged. |
| `aborted` | unchanged | increment by 1 | Unchanged. |

Stop precedence is explicit user stop, comparable optional target reached, third
valid `no-improvement`, then twentieth terminal round. A checkpoint at rounds
3, 6, 9, and later multiples of three is a derived status message only: it does
not pause the run and never creates a checkpoint artifact. `last_checkpoint_round`
prevents duplicate messages.

The evaluator records stop reasons as `user-intervention`, `target-reached`,
`valid-no-improvement-limit`, or `round-budget-exhausted`.

The optional target is only `absolute_latency_ms` or `speedup_vs_baseline`, is
measured by `wall_time_ms`, and is comparable only under the baseline measurement
fingerprint. A user may append a target at a safe terminal boundary. All other
policy fields are frozen for the run epoch. A user stop is recorded immediately
but waits for the active command boundary unless immediate interruption is
explicitly requested.

## Measurement-exclusive phases

Set `measurement_exclusive: true` before Verifier begins `verifying` or
`measuring`, and clear it only after durable completion, block, or terminal
artifact commit. During this state no other role may execute local commands or
perform scans, builds, cache warming, or writes. This is a same-machine safety
requirement, not a performance budget.

## Run epochs and recovery

Treat durable files, not session-local roles, as authoritative. Before dispatch,
validate manifest schema, artifact existence and hashes, target runtime
fingerprint, measurement fingerprint, current phase, and last committed
transition. Reuse a valid uncommitted predecessor in the same round but never reopen a completed decision.

Only `recover` resumes a block or uncommitted safe step. It preserves counters
and resumes at the first missing or invalid artifact. A stopped run requires a
new `run_epoch` and an explicit user approval reason before counters reset.
Target/profile/fingerprint/policy changes, canonical-pointer mismatch, or failed
three-round reconciliation require rehydration from compact role context and
changed artifacts. A measurement fingerprint change requires a comparable
baseline before candidate comparison.

## Git evidence ledger

Commit Phase 0, every terminal round, every incident, and final stop summary on
the dedicated run branch. Track baseline adapter, project/team state, decisions,
Coder results, reports, candidate source when present, role context, incidents,
and final summary. Do not track raw profiler logs, command output, caches,
runtime sessions, bootstraps, or secrets; reports instead retain raw relative
paths, hashes, and commands. `log/` remains ignored.

## Knowledge lift

At final stop, inspect project evidence for generic failed patterns with clear
preconditions, observed failure, evidence revision, and reconsideration
conditions. Propose promotions to the user. Modify
`references/anti-patterns.md` only after explicit user approval in a separate
commit. Do not rewrite existing project histories.

## References

- `adapters/claude-code.md` and `adapters/codex.md`: runtime lifecycle mappings.
- `prompts/designer.md`, `prompts/coder.md`, and `prompts/verifier.md`: role
  behavior and ownership.
- `prompts/coder_targets/<target_profile>.md`: the one complete profile selected
  by the current runtime; this repository currently includes `triton_mlu`,
  `triton_gcu`, and `triton_cuda`.
- `references/decision-template.md`: normative decision schema.
- `references/project-template.md`, `references/report-template.md`,
  `references/team-state-template.md`, and `references/role-context-template.md`:
  durable artifacts.
- `references/invariants.md`, `references/bottleneck-judgment.md`, and
  `references/anti-patterns.md`: constraints and evidence guidance.
- `scripts/validate_decision.py`, `scripts/make_baseline_adapter.py`,
  `scripts/summarize_trace.py`, and `scripts/evaluate_run_policy.py`:
  deterministic gates and helpers.

## vNext semantic attribution and finalization routing

A vNext run records `contract_version: 3`, `semantic_contract: typed-sketch-v1`,
and `attribution_contract: verdict-v1` in `team-state.md`. Existing v1/v2
campaigns remain historical and are never migrated. Routing is exact:

1. For an explicit profile-onboarding request, Orchestrator validates
   profile/probe/runtime inputs, runs one bounded pre-campaign probe lifecycle,
   emits proposed promotion artifacts, reports them to the user, and may stop
   without entering Phase 0.
2. When a campaign is requested, Orchestrator obtains explicit requirements from
   the user/maintainer or Designer read-only capability preflight, validates and
   materializes them without inventing design claims, and invokes
   `select_profile_probes()` before campaign state or snapshot creation.
3. A unique exact-scope Unknown primary with `must-resolve|before-fallback` runs
   one bounded probe; unrelated Unknowns are ignored and ambiguous matches fail
   deterministically.
4. Eligible observed success emits promotion artifacts and stops as
   `promotion-pending`. Phase 0 resumes only after maintainer promotion, or
   after explicit decline/defer plus fallback authorization. Partial/failure/
   block/no-match leaves the primary Unknown.
5. Phase 0 validates only reviewed exact-scope prior evidence, rediscovers
   current target/runtime identity, materializes the project capability claim
   including any complete maintainer-confirmed fallback disposition, rejects raw
   probe refs, and freezes the implementation-profile snapshot. Claim/Decision
   validation must not depend on the pre-campaign run directory.
6. A schema-v2 Decision validates Sketch, references, frozen implementation
   profile snapshot, claim, causal graph, and any fallback provenance before
   Coder dispatch.
7. Coder runs only Decision-scoped capability/compile probes and the binding
   checker before `candidate-ready`; its results stay under campaign-local
   `log/probes/`.
8. Verifier writes authoritative runtime facts, correctness, observed lowering,
   and performance only.
9. Orchestrator validates `verdict_NNN.json`; it may route one `code-error`
   repair.
10. `design-error` terminates as `design-rejected` with increment.
11. `lowering-unknown` terminates as `design-rejected` with unchanged failed
    streak.
12. Bounded `evidence-gap` routes environment absence to `blocked`; a Decision
    missing a measurable contract terminates as `design-rejected` with explicit
    design-error effect. A correct candidate with insufficient accepted-to-
    candidate e2e improvement remains `no-improvement` and attribution `none`.
13. At submission finalization, Orchestrator computes the canonical
    `submission_snapshot_id`, calls `resolve_finalization_slot()`, and opens or
    resumes one artifact index only when no qualification, promotion, repair,
    fingerprint transition, missing profile legality, completed identical input,
    or already-finalized current output exists.
14. Designer reuses the accepted Sketch and declares a finite reviewed
    profile-legal config-only domain, accepted fallback/control, search
    budget/protocol, objective, and deterministic order/tie rule.
15. Verifier executes the accepted source through temporary launch/meta-
    parameter injection under measurement exclusivity. It returns normalized
    trials to Orchestrator without writing a selection artifact or report and
    verifies that temporary storage contains no derived candidate source.
16. Orchestrator runs the pure selector and sends only the normalized selection
    to Coder. For an improved winner, Coder emits exactly one pinned candidate
    derived from the accepted source; for the fallback/control, Coder emits no
    source. Binding validation then runs on the exact final source.
17. Verifier reruns full correctness, lowering, required promotion evidence, and
    official competition measurements on that source, then atomically writes the
    report once with separate `search_trials` and `post_pin_official` sections.
18. Orchestrator reruns the selector from the sealed report and evaluates the
    pure submission-promotion predicate. An improved winner atomically advances
    `last_accepted_kernel` and `last_accepted_report` to the pinned source and
    sealed report while preserving `last_accepted_round`; a fallback-retained
    winner changes neither pointer; both require all final gates, and partial
    pair updates are rejected.
19. Orchestrator emits only `submission-ready|blocked` through the separate
    finalization verdict branch. It never calls `evaluate_terminal()` and the
    verdict rejects attribution, campaign terminal, counter-effect, and
    run-policy fields.
20. The final candidate contains one fixed selected configuration and no
    runtime/online `@triton.autotune`, adaptive search, or autotune-cache
    selection dependency.


## Profile onboarding and vNext new-run boundary

Profile onboarding may run versioned probes, emit hashed run-local evidence and a
proposed promotion candidate, and stop without creating a campaign. It never edits
the canonical implementation profile.

A vNext campaign records `contract_version: 3`, a frozen implementation-profile
snapshot hash, a project capability claim, typed Sketch, binding ledger, and
verdict artifact. Existing v1/v2 campaigns remain historical and are not migrated.

Each Triton submission snapshot runs one offline bounded configuration-only search over
profile-legal fields. The selected configuration is pinned into one candidate and
must pass fresh binding, correctness, lowering, promotion, and official benchmark
gates. The workflow adds no finalization-specific state or artifact family. The final
candidate contains one fixed configuration and no runtime/online autotune,
first-use search, or cache-dependent configuration selection.
