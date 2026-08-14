---
name: kernel-opt-loop
description: Coordinate iterative Triton kernel or operator optimization against an auto_bench-style harness using durable Designer, Coder, and Verifier artifacts. Use for multi-round optimization from an immutable base.py reference, not for one-shot fixes.
---

# Kernel Optimization Loop

This skill is the runtime-neutral Orchestrator contract. It owns workflow state,
artifact gates, deterministic routing, canonical pointers, project overview
updates, Git commits, stop decisions, and resume. Designer, Coder, and Verifier
own the files declared in their role contracts. v1 runs exactly one active round
and one candidate at a time against the complete `triton_mlu` target profile.

## When to use

Use this skill when a project has an immutable PyTorch-style `base.py`, an
`auto_bench.py`-style harness, and a request for iterative Triton operator or
kernel optimization. The workflow applies when correctness, benchmark wall
time, and attributable profiler evidence must be preserved across multiple
rounds or sessions.

Do not use it for a one-shot bug fix, a non-iterative refactor, a workflow that
may modify the reference or harness semantics, or a target that has no complete
profile. Existing optimization projects are not migrated automatically.

## Required inputs

Resolve these before mutation:

1. Absolute project root and immutable `base.py` path.
2. Absolute harness path and its actual module-loading behavior.
3. Absolute interpreter path, selected device, and environment needed by it.
4. Shape, dtype, correctness tolerances, warmup, repeat, profiling mode,
   profiling warmup, and profiling iteration settings.
5. User-owned upbound or safety limits when these cannot be discovered.
6. Absolute skill root containing this file, one target profile, role contracts,
   templates, validators, helpers, and one runtime adapter.

Ask only for undiscoverable user-owned values. Never infer a device, interpreter,
upbound, concurrency promise, or semantic tolerance from a candidate.

## Runtime selection

Choose in this exact order: Codex collaboration when exposed, Claude Code agent teams when enabled, then sequential fallback.

Load exactly one runtime adapter and use its common operations. Multi-agent
availability changes orchestration mechanics, not artifacts, ownership, routing,
or state semantics. Portable default agents are sufficient; specialized role
types are optional. Sequential fallback executes each role contract in the main
session without starting nested agent processes and still respects all ownership
and handoff gates.

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

The role reads its own full contract. Every state-changing role response goes to
Orchestrator. Messages are advisory until the required durable artifact exists
and passes its gate.

## Phase 0

Perform initialization in this order:

1. Resolve absolute skill, project, base, harness, interpreter, and device paths.
   Record the starting SHA-256 and bytes of `base.py` and the harness.
2. Create `rounds/`, `state/`, and gitignored `log/`. Materialize `project.md`
   from `references/project-template.md`, `team-state.md` from
   `references/team-state-template.md`, and empty role state files. Only
   Orchestrator writes the manifest and overview.
3. Discover implementation language, backend, target profile, Triton
   distribution/version, active backend target/version when available, and
   device architecture. Load only the exact matching complete profile. A missing
   runtime or mismatch is an environment block.
4. Dispatch Designer for Phase 0 semantic analysis. Request only unknown
   user-owned values and validate its writable-file boundary before continuing.
5. Run `scripts/make_baseline_adapter.py` to create `baseline_adapter.py` by
   renaming the one top-level `Model` to `ModelNew`. Verify the recorded
   `base.py` bytes are unchanged.
6. Dispatch Verifier for baseline correctness, benchmark wall samples, and a
   Level 1 separately scoped profiler summary. Require `rounds/report_000.md`
   with result `baseline`, fingerprints, normalized evidence, and exact
   reproduction commands.
7. Compute the measurement fingerprint as SHA-256 over base bytes, NUL, harness bytes, NUL, and canonical JSON measurement settings. Serialize with
   `sort_keys=True` and `separators=(',', ':')`. The JSON object has exactly
   these keys: `"shape"`, `"dtype"`, `"device"`, `"warmup"`, `"repeat"`,
   `"profile_mode"`, `"profile_warmup"`, and `"profile_iterations"`.
8. After all gates pass, set `last_completed_round: "000"`,
   `last_accepted_round: "000"`, `last_accepted_kernel: baseline_adapter.py`,
   `last_accepted_report: rounds/report_000.md`, both completed report pointers
   to `rounds/report_000.md`, `last_result: baseline`, and `phase: ready`.
   Append the Phase 0 overview and transition rows, then commit the artifacts.

On any Phase 0 environment failure, create a timestamped incident, set
`phase: blocked`, leave accepted pointers null, append and commit the blocking
transition, and report the exact remediation required. The incident is not a
completed round.

## Round N

Round number is `total_rounds + 1`, formatted as three digits. Never allocate a
second active round. Resolve both `last_accepted_kernel` and
`last_accepted_report` before dispatch.

Execute this state machine in order:

1. Set `phase: designing`. Dispatch Designer with accepted canonical evidence,
   recent completed failure evidence, the exact profile, project invariants, and
   anti-pattern guidance. Designer writes one `decision_NNN.md`.
2. Run `scripts/validate_decision.py` with the manifest target profile. Record
   the decision hash. A proceeding decision becomes immutable before coding.
3. A valid abort form completes `aborted` without Coder or Verifier.
4. Set `phase: coding`. Dispatch Coder with the immutable decision and canonical
   source. Require `coder_result_NNN.md` and, for `candidate-ready`, the declared
   candidate and matching hashes.
5. Apply the routing table below. Never dispatch Coder directly from Designer or
   Verifier directly from Coder; all handoffs pass through Orchestrator.
6. Before `phase: verifying`, parse the candidate and run the actual harness
   loader. A local defect follows the bounded Coder path; an environment defect
   blocks without completing the round.
7. Dispatch Verifier with the candidate, decision, accepted reference/report,
   project regime, and Verifier state. Require durable status updates and route
   one repair or missing-evidence request through Orchestrator.
8. Validate all artifacts and hashes before calculating exactly one terminal
   result.
9. Set `last_completed_round`, `last_completed_decision`,
   `last_completed_coder_result`, `last_completed_report`, and `last_result`.
   Use null where the selected path does not produce an artifact.
10. Append exactly one project overview row and one terminal transition row.
11. Increment counters exactly once, advance accepted pointers only on
    `accepted`, commit, then and only then dispatch the next round.

`round_status_NNN.md` is updated at verification start, after correctness, after
each timing pair, and at verification end. Completion notifications are
preferred when the adapter supports them; Orchestrator does not poll a runtime
that already delivers completion.

## Routing and state transitions

Only Orchestrator transitions the manifest. The allowed phases are
`initializing|ready|designing|coding|verifying|repairing|measuring|blocked|stopped`.

| Producer classification | Orchestrator action |
|---|---|
| Designer abort | Complete `aborted`; do not dispatch Coder or Verifier. |
| Coder `candidate-ready` | Validate candidate with the harness loader, then dispatch Verifier. |
| Coder `major-deviation` | Complete `design-rejected`; the next unused round returns to Designer after commit. |
| Coder `capability-miss` | Complete `design-rejected`; preserve capability evidence and keep canonical unchanged. |
| Coder `implementation-failed` | Complete `candidate-failed`; keep canonical unchanged. |
| Any `environment-blocked` | Write/preserve incident, set `blocked`, and report remediation; no terminal round. |
| Verifier `implementation-repair-required` | Set `repairing` and return to Coder exactly once in the same round. |
| Verifier requires a design change | Complete `design-rejected`; never edit the current decision. |
| Verifier correctness fails after repair | Complete `candidate-failed`. |
| Verifier `measurement-incomplete` | Set `measuring`, collect the named probe, then return to Verifier; if impossible, classify design or environment first. |
| Verifier terminal evidence | Complete `accepted` or `no-improvement` from correctness, guardrails, and unrounded median wall time. |

Terminal results and counter effects are exact:

| Result | `performance_miss_streak` | `failed_attempt_streak` | Canonical |
|---|---:|---:|---|
| `accepted` | 0 | 0 | Advance to candidate and report. |
| `no-improvement` | +1 | 0 | Unchanged. |
| `design-rejected` | 0 | +1 | Unchanged. |
| `candidate-failed` | 0 | +1 | Unchanged. |
| `aborted` | 0 | +1 | Unchanged. |

`total_rounds` increments once for each terminal row. Environment incidents update neither counter nor `total_rounds`. The canonical implementation and report
advance together only after an `accepted` report passes all gates. Rejected
candidates remain auditable and are never copied as the next starting point.

## Stop criteria

After every terminal commit, evaluate all five criteria:

1. `measurement-bound`: normalized device ratio is below 5% and targeted
   evidence attributes the remaining host time to fixed harness work.
2. `diminishing returns`: either progress streak reaches three.
3. `upbound reached`: accepted performance enters the declared comparable bound.
4. `resource exhausted`: configured round or time safety limit is reached.
5. `user intervention`: the user requests stop.

Verifier recommends with evidence; Orchestrator decides and records reason,
timestamp, eligibility, and constraints. User intervention is unconditional.
Designer may reject another non-user stop only with a concrete next hypothesis
expected to clear 5%. On stop, set `phase: stopped`, append a transition, and
commit before ending the runtime workflow.

## Resume

Treat durable files, not session-local roles, as authoritative. Before dispatch,
validate manifest schema and skill version, artifact existence and hashes,
target runtime fingerprint, measurement fingerprint, current phase, and the
last committed transition.

Resume an interrupted uncommitted phase at the first missing invalid artifact.
Reuse a valid uncommitted predecessor in the same round, but never reopen a completed decision. Resume an environment block at the same safe step without allocating a
new round or changing streaks. A target or runtime change invalidates a
target-bound decision. A measurement fingerprint change requires a comparable
baseline before candidate comparison.

Stop-specific conditions are: a new shape/regime for `measurement-bound`; new
hypothesis, evidence, or skill capability for `diminishing returns`; an explicit
stretch goal for `upbound reached`; user acknowledgment for `resource exhausted`;
and an explicit resume request for `user intervention`.

## Knowledge lift

At stop, inspect completed project evidence for generic failed patterns with
clear preconditions, observed failure, evidence revision, and reconsideration
conditions. Propose promotions to the user. Modify
`references/anti-patterns.md` only after explicit user approval and in a
separate commit. Do not publish to an external knowledge base in v1 and do not
rewrite existing project histories.

## References

- `adapters/claude-code.md` and `adapters/codex.md`: runtime lifecycle mappings.
- `prompts/designer.md`, `prompts/coder.md`, and `prompts/verifier.md`: role
  behavior and ownership.
- `prompts/coder_targets/triton_mlu.md`: sole complete v1 capability profile.
- `references/decision-template.md`: normative decision schema.
- `references/project-template.md`, `references/report-template.md`, and
  `references/team-state-template.md`: durable project artifacts.
- `references/invariants.md`, `references/bottleneck-judgment.md`, and
  `references/anti-patterns.md`: constraints and evidence guidance.
- `scripts/validate_decision.py`, `scripts/make_baseline_adapter.py`, and
  `scripts/summarize_trace.py`: deterministic gates and helpers.
