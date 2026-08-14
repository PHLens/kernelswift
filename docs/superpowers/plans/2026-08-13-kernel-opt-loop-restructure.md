# kernel-opt-loop Skill Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `kernel-opt-loop` into a cross-runtime Designer/Coder/Verifier workflow with target-bound decisions, attributable runtime evidence, immutable rounds, and deterministic resume behavior.

**Architecture:** `SKILL.md` owns a runtime-neutral state machine and compact role bootstrap. Skill-local role contracts communicate through durable artifacts; Claude Code and Codex adapters only translate orchestration operations. A validated `Optimization Intent + Unified Sketch/Host Plan + Evaluation Contract` binds each round to one discovered Triton-MLU profile, and Verifier mirrors the evaluation contract with authoritative runtime evidence.

**Tech Stack:** Markdown Agent Skill files; Python 3 standard library; `unittest`; existing `auto_bench.py`; torch profiler JSON; Claude Code agent teams; Codex collaboration tools; Git.

**Authoritative spec:** `docs/superpowers/specs/2026-08-13-kernel-opt-loop-restructure-design.md` at commit `a76d5d371d582b251036e990b4295d4a3d9959f3`.

## Global Constraints

- Source of truth is `skills/kernel-opt-loop/` in this repository.
- Begin implementation in an isolated worktree created with `superpowers:using-git-worktrees`.
- Runtime-neutral files contain no active Claude Code- or Codex-specific tool syntax.
- The user does not write role prompts. `SKILL.md` resolves the shared bootstrap; roles read `prompts/*.md` themselves.
- Portable default/general-purpose agents are sufficient; runtime-local specialized agent types are optional.
- v1 has exactly one active round and one candidate. Roles may persist, but competing candidates do not run in parallel.
- `base.py` is user-owned and immutable. Phase 0 generates `baseline_adapter.py`.
- `last_accepted_kernel` and `last_accepted_report` are the only canonical comparison pointers.
- A validated `decision_NNN.md` is immutable. A design change completes the current round as `design-rejected` and starts a new round only after commit.
- Coder never returns `accepted`; only Verifier can produce the `accepted` terminal result.
- v1 supports only the complete `triton_mlu` target profile. Do not add inactive target stubs, candidate lists, automatic DSL fallback, or capability-miss history for future routing.
- Environment failures block without completing a round or changing either progress streak.
- Profiler totals are normalized per forward call. Reference and candidate scopes are never combined.
- Benchmark wall time, not profiler time, controls adoption. Adoption requires correctness, guardrails, and at least 5% unrounded median wall improvement.
- Backend-neutral IR, deterministic lowering, KernelWiki, and deep automatic profiler analysis are future directions and must not produce v1 files or tasks.
- Existing `groupedtopk`, `fused_moe`, and other project logs are not migrated.
- Each task ends in a focused commit after its tests pass.

---

## File Structure

```text
skills/kernel-opt-loop/
  SKILL.md                              # runtime-neutral orchestration contract
  adapters/
    claude-code.md                      # Claude Code lifecycle mapping only
    codex.md                            # Codex lifecycle mapping only
  prompts/
    designer.md                         # decision ownership and hypothesis contract
    coder.md                            # target realization and result taxonomy
    verifier.md                         # runtime execution and evidence contract
    coder_targets/
      triton_mlu.md                     # sole v1 target profile
  references/
    anti-patterns.md                    # reusable, evidence-backed failed approaches
    bottleneck-judgment.md              # retained and updated measurement guidance
    decision-template.md                # normative decision schema
    invariants.md                       # source/harness/ownership invariants
    project-template.md                 # Phase 0 project and overview schema
    report-template.md                  # mirrored evaluation report schema
    team-state-template.md              # manifest and transition log schema
  scripts/
    make_baseline_adapter.py            # immutable-base adapter generator
    summarize_trace.py                  # normalized scoped profiler summary
    validate_decision.py                # deterministic decision/schema validator
  tests/
    fixtures/
      decisions/
      traces/
    test_validate_decision.py
    test_helpers.py
    test_contracts.py
```

Delete `skills/kernel-opt-loop/references/log-template.md`; `project.md` plus
round artifacts replace it.

The workflow creates:

```text
<op>/
  base.py
  baseline_adapter.py
  project.md
  team-state.md
  triton_<op>_<NNN>.py
  rounds/
    decision_NNN.md
    coder_result_NNN.md
    report_NNN.md
    round_status_NNN.md
    incident_NNN_<UTC-timestamp>.md
  state/
    designer_state.md
    coder_state.md
    verifier_state.md
  log/
    *.pt.trace.json
```

`log/` is gitignored. All normalized evidence and state files are durable.

## Traceability

| Spec section | Implementation tasks | Acceptance evidence |
|---|---|---|
| §4 Runtime-neutral architecture | Tasks 4, 5, 9 | adapter contract tests and runtime smoke scenarios |
| §5 Target identity/profile | Tasks 3, 7, 9 | profile match, unknown capability, and no-stub assertions |
| §6 Decision contract | Tasks 1, 3, 6 | kernel/host/mixed validator fixtures |
| §7 Roles and routing | Tasks 6–10 | routing matrix and one-repair pressure scenarios |
| §8 Runtime evidence/profiler | Tasks 2, 8, 10 | 50-call normalization, scoped comparison, mirrored observables |
| §9 Round state machine | Tasks 3, 9, 10 | canonical/counter/env-fail transition fixtures |
| §10 Phase 0 | Tasks 2, 3, 6, 8, 9 | immutable base, baseline report, fingerprint fixtures |
| §11 Stop/resume | Tasks 3, 8–10 | resume and stop-transition scenarios |
| §12 Durable structure/knowledge lift | Tasks 3, 9 | ownership/static checks and approval gate |
| §13 Acceptance strategy | Task 10 | all automated and guided checks pass |
| §14 Migration | Global constraints, Task 10 | negative assertion that existing logs are not rewritten |
| §15 Future directions | Task 10 | negative assertions: no future-scope implementation |
| §16 Spec/plan governance | This traceability table, Task 10 | pinned spec and no plan-only architecture |

---

### Task 1: Implement the Decision Schema and Validator

**Files:**
- Create: `skills/kernel-opt-loop/references/decision-template.md`
- Create: `skills/kernel-opt-loop/scripts/validate_decision.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/decisions/kernel-valid.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/decisions/host-valid.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/decisions/mixed-valid.md`
- Create: `skills/kernel-opt-loop/tests/test_validate_decision.py`

**Interfaces:**
- `validate_decision(path: pathlib.Path, expected_profile: str | None = None) -> dict[str, object]` returns parsed normalized data or raises `DecisionValidationError`.
- CLI: `python3 scripts/validate_decision.py DECISION --expected-profile triton_mlu` prints one JSON object and exits 0; errors are `path:line: error-code: message` on stderr with exit 2.
- Later tasks consume the exact headings, JSON fields, Sketch grammar, and CLI defined here.

- [ ] **Step 1: Write the valid kernel fixture and first failing test**

Use these exact normative section names and representative values in
`kernel-valid.md`:

````markdown
# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"mlu","target_profile":"triton_mlu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"fuse the routing reduction into the target kernel","allowed_changes":["kernel dataflow"],"invariants":["ModelNew public contract","output dtype and shape"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor scores shape=[T,E] dtype=fp32 layout=row_major memory=global
tile row shape=[BLOCK_E] dtype=fp32 memory=register

# O Operations
load row <- scores[token,0:E]
compute probs = softmax(row)
store output[token,0:K] <- topk(probs,K)

# C Control
parallel token over T
guard token < T

# H Target Hints
target=triton_mlu
num_warps=1
num_stages=2
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the routing reduction into the target kernel","expected_causal_chain":["external routing kernels disappear","device time decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"external_kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; no matching failure invalidates this path.

## Rationale and Evidence

The accepted trace contains separate routing kernels that are inside the candidate's change boundary.
````

In `test_validate_decision.py`, import the module by adding `scripts/` to
`sys.path`, call `validate_decision()`, and assert `change_scope == "kernel"`,
`target_profile == "triton_mlu"`, and four Sketch section names are returned.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_validate_decision.py' -v
```

Expected: FAIL because `validate_decision.py` does not exist.

- [ ] **Step 3: Add host-only and mixed fixtures plus invalid cases**

`host-valid.md` uses `change_scope: "host"`, the exact text
`N/A: host-only change` under Unified Sketch, and this Host Plan object:

```json
{"applicability":"required","affected_scope":["ModelNew.forward","output cache"],"state_owner":"ModelNew instance","lifetime":"model lifetime","allocation_reuse":"reuse when shape, dtype, and device match","cache_key":["shape","dtype","device"],"invalidation":"replace on cache-key change","concurrency":"one model instance is not shared across concurrent forwards","device_stream_behavior":"caller-selected device and current stream are preserved","unchanged_behavior":["returned shape","returned dtype","numerical semantics"]}
```

`mixed-valid.md` uses a valid Sketch and the same required Host Plan schema.
Add tests that mutate fixtures in memory and assert these error codes:

```python
self.assertValidationError(missing_host_plan, "host-plan-required")
self.assertValidationError(profile_mismatch, "target-profile-mismatch")
self.assertValidationError(two_hints_on_one_line, "sketch-h-one-directive-per-line")
self.assertValidationError(missing_sketch_fence, "sketch-fence-missing")
self.assertValidationError(missing_observable, "evaluation-observable-required")
```

- [ ] **Step 4: Implement the minimal validator**

Implement these symbols with only the standard library:

- `DecisionValidationError(code, message, line=1)` stores all three values and
  renders the CLI diagnostic format declared in Interfaces;
- `extract_sections(text)` returns each H2 heading's starting line and body and
  rejects duplicate headings;
- `parse_single_json_block(section)` accepts exactly one fenced JSON object;
- `parse_sketch(section, target_profile)` returns four ordered lists of normalized
  Sketch statements;
- `validate_decision(path, expected_profile=None)` returns normalized Metadata,
  Optimization Intent, Sketch, Host Plan, and Evaluation Contract dictionaries;
- `main(argv=None)` prints normalized JSON on success and returns 2 for
  `DecisionValidationError`.

Validation rules are exact:

- required H2 headings are the seven headings in `kernel-valid.md`;
- Metadata enums are `decision = proceed|abort` and
  `change_scope = kernel|host|mixed|none`;
- a proceeding kernel or mixed decision has exactly one `sketch` fence and the
  four ordered headers shown above;
- nonblank D/O/C lines start respectively with
  `tensor|tile|scalar`, `alloc|load|compute|store`, and
  `parallel|for|if|else|guard|end`;
- H begins with exactly `target=<metadata target_profile>` and subsequent lines
  are one `name=value` directive; whitespace-separated second directives fail;
- host-only uses the exact N/A marker; mixed requires both Sketch and required
  Host Plan; kernel requires Host Plan `not-applicable`;
- an abort uses `change_scope: none`, Unified Sketch `N/A: aborted`, Host Plan
  `{"applicability":"not-applicable","reason":"aborted"}`, and an Evaluation
  Contract with `{"applicability":"not-applicable","reason":"aborted"}`;
- proceeding Evaluation Contracts require a nonempty causal chain, at least one
  observable, `wall_time`, threshold `5.0`, nonempty guardrails, and profiling
  level `summary|targeted|deep-on-demand`.

Make the validator executable:

```bash
chmod +x skills/kernel-opt-loop/scripts/validate_decision.py
```

- [ ] **Step 5: Write the reusable decision template**

Copy the seven headings and field schemas from `kernel-valid.md` into
`decision-template.md`. Add a field-requirements table for
`change_scope = kernel|host|mixed|none`, the exact host-only and aborted N/A
markers, the Host Plan required-field list, the one-directive-per-line H rule,
the three requested profiling modes, and the mandatory Level 0/1 evidence. Include complete kernel,
host-only, and abort examples; each example must pass `validate_decision.py`
after its example paths are materialized in a temporary project.

- [ ] **Step 6: Run the validator tests and CLI fixture checks**

Run:

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_validate_decision.py' -v
python3 skills/kernel-opt-loop/scripts/validate_decision.py \
  skills/kernel-opt-loop/tests/fixtures/decisions/kernel-valid.md \
  --expected-profile triton_mlu
```

Expected: all tests PASS; CLI prints JSON with `"valid": true`.

- [ ] **Step 7: Commit**

```bash
git add skills/kernel-opt-loop/references/decision-template.md \
  skills/kernel-opt-loop/scripts/validate_decision.py \
  skills/kernel-opt-loop/tests/fixtures/decisions \
  skills/kernel-opt-loop/tests/test_validate_decision.py
git commit -m "skills: add kernel decision contract"
```

---

### Task 2: Add Baseline and Profiler Helpers

**Files:**
- Create: `skills/kernel-opt-loop/scripts/make_baseline_adapter.py`
- Create: `skills/kernel-opt-loop/scripts/summarize_trace.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/traces/scoped-50-calls.json`
- Create: `skills/kernel-opt-loop/tests/test_helpers.py`

**Interfaces:**
- `make_baseline_adapter(source: Path, destination: Path) -> None` renames exactly one top-level `Model` class to `ModelNew` without changing the source.
- `summarize_trace(path: Path, iterations: int, scope: str | None, wall_ms: float | None) -> dict[str, object]` returns normalized totals, counts, top kernels, and optional `device_ratio`.
- CLIs use positional source/destination or trace paths and produce actionable nonzero errors.

- [ ] **Step 1: Write failing baseline-adapter tests**

Test a temporary module containing imports, one `Model`, `get_inputs`, and
`get_init_inputs`. Assert:

```python
source_before = source.read_bytes()
make_baseline_adapter(source, destination)
self.assertEqual(source.read_bytes(), source_before)
tree = ast.parse(destination.read_text())
self.assertIn("ModelNew", [n.name for n in tree.body if isinstance(n, ast.ClassDef)])
self.assertNotIn("Model", [n.name for n in tree.body if isinstance(n, ast.ClassDef)])
```

Also assert zero or two top-level `Model` definitions raise a precise error and
an existing destination is not overwritten unless CLI `--force` is supplied.

- [ ] **Step 2: Write failing trace-normalization tests**

The committed trace fixture contains two `record_function` X-events named
`accepted_reference` and `candidate`, each spanning 50 kernel events. Candidate
kernels total 1,000 us and its wall time is 0.1 ms/call. Assert:

```python
summary = summarize_trace(trace, 50, "candidate", 0.1)
self.assertEqual(summary["device_total_us"], 1000.0)
self.assertEqual(summary["device_us_per_call"], 20.0)
self.assertEqual(summary["kernel_count_per_call"], 1.0)
self.assertEqual(summary["device_ratio"], 0.2)
```

Add cases for independent reference scope, missing scope, zero iterations, and a
kernel event outside both scopes.

- [ ] **Step 3: Run tests and verify RED**

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_helpers.py' -v
```

Expected: FAIL because both helper modules are absent.

- [ ] **Step 4: Implement `make_baseline_adapter.py`**

Use `ast.parse`, change only the selected `ClassDef.name`, call
`ast.fix_missing_locations`, and write `ast.unparse(tree) + "\n"`. Expose
`find_model_class(tree)`, `make_baseline_adapter(source, destination,
force=False)`, and `main(argv=None)` with the signatures in Interfaces. The CLI
returns 2 for source, AST, or destination validation errors and 0 after a
successful write.

- [ ] **Step 5: Implement `summarize_trace.py`**

Use X-event timestamp containment for the named `record_function` scope. Include
only `cat == "kernel"` events whose `[ts, ts + dur]` interval is inside that
scope. Return this exact shape:

```python
{
    "scope": "candidate",
    "iterations": 50,
    "device_total_us": 1000.0,
    "device_us_per_call": 20.0,
    "kernel_count_total": 50,
    "kernel_count_per_call": 1.0,
    "device_ratio": 0.2,
    "kernels": [
        {"name": "kernel_a", "count_total": 50, "count_per_call": 1.0,
         "total_us": 1000.0, "us_per_call": 20.0}
    ]
}
```

Sort kernels by `total_us` descending. Reject overlapping duplicate scopes or a
scope with no kernel events rather than silently mixing data.

- [ ] **Step 6: Run tests and command-line checks**

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_helpers.py' -v
python3 skills/kernel-opt-loop/scripts/summarize_trace.py \
  skills/kernel-opt-loop/tests/fixtures/traces/scoped-50-calls.json \
  --iterations 50 --scope candidate --wall-ms 0.1
```

Expected: PASS and JSON reports `device_us_per_call: 20.0` and
`device_ratio: 0.2`.

- [ ] **Step 7: Commit**

```bash
git add skills/kernel-opt-loop/scripts/make_baseline_adapter.py \
  skills/kernel-opt-loop/scripts/summarize_trace.py \
  skills/kernel-opt-loop/tests/fixtures/traces/scoped-50-calls.json \
  skills/kernel-opt-loop/tests/test_helpers.py
git commit -m "skills: add reproducible measurement helpers"
```

---

### Task 3: Add Durable Templates, Invariants, and Anti-patterns

**Files:**
- Create: `skills/kernel-opt-loop/references/project-template.md`
- Create: `skills/kernel-opt-loop/references/report-template.md`
- Create: `skills/kernel-opt-loop/references/team-state-template.md`
- Create: `skills/kernel-opt-loop/references/invariants.md`
- Create: `skills/kernel-opt-loop/references/anti-patterns.md`
- Modify: `skills/kernel-opt-loop/references/bottleneck-judgment.md`
- Delete: `skills/kernel-opt-loop/references/log-template.md`
- Create: `skills/kernel-opt-loop/tests/test_contracts.py`

**Interfaces:**
- Templates define the field names consumed by all roles and `SKILL.md`.
- `team-state.md` is Markdown with YAML-compatible frontmatter and an append-only transition table.
- `report_NNN.md` mirrors `Evaluation Contract` observables by name.

- [ ] **Step 1: Write template contract assertions**

Start `test_contracts.py` with helpers that read skill files and assert exact
strings. Add failing assertions for:

```python
for field in (
    "last_accepted_kernel", "last_accepted_report", "last_completed_round",
    "performance_miss_streak", "failed_attempt_streak", "measurement_fingerprint",
    "target_profile", "runtime_fingerprint_ref", "blocked_incident",
):
    self.assertIn(field, team_state_template)
self.assertIn("evidence_for_next_round", report_template)
self.assertNotIn("target_dsl_candidates", combined_templates)
self.assertNotIn("capability_miss_log", combined_templates)
```

- [ ] **Step 2: Write `team-state-template.md`**

Use these initial frontmatter values:

```yaml
---
schema_version: 1
skill_version: 2.0.0
runtime: unset
phase: initializing
project_started_at: null
current_round: "000"
last_completed_round: null
last_accepted_round: null
last_accepted_kernel: null
last_accepted_report: null
last_completed_decision: null
last_completed_coder_result: null
last_completed_report: null
last_result: null
performance_miss_streak: 0
failed_attempt_streak: 0
total_rounds: 0
measurement_fingerprint: null
implementation_language: triton
implementation_backend: mlu
target_profile: triton_mlu
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: null
stop_reason: null
stop_timestamp: null
resume_eligible: always
resume_constraints: []
---
```

Below it add `## Transition Log` with columns Timestamp, Phase, Round, Result,
Canonical, Incident, Commit. Document manifest phases exactly as
`initializing|ready|designing|coding|verifying|repairing|measuring|blocked|stopped`.

- [ ] **Step 3: Write project and report templates**

`project-template.md` contains fixed sections for semantics, invariants,
runtime fingerprint (`triton_distribution`, `triton_version`, `backend_target`,
`backend_version`, `device_arch`), measurement regime, measurement fingerprint, upbound,
reproduction, and this overview table:

```markdown
| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
```

`report-template.md` contains:

```text
Result: baseline | accepted | no-improvement | design-rejected | candidate-failed
Round
Decision, candidate, accepted reference, and source hashes
Correctness and guardrail matrix
Reference/candidate interleaved raw samples and unrounded medians
Improvement percentage
Evaluation Contract mirror: observable, expectation, observation, verdict
Hypothesis verdict: confirmed | partially-confirmed | falsified | inconclusive
Profiler level, iterations, device totals/per-call, ratio, counts, top kernels
Retry history with before/after candidate hashes
Upbound gap
evidence_for_next_round
Stop recommendation and evidence
Exact reproduction commands
```

Include the formulas:

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

For `Result: baseline`, require correctness, baseline wall samples, Level 1
profiler summary, runtime/measurement fingerprints, and reproduction commands;
mark the Evaluation Contract mirror `not-applicable: Phase 0` because no round
decision exists.

- [ ] **Step 4: Write invariants and seed anti-patterns**

`invariants.md` separates immutable reference/harness rules, AST loader behavior,
canonical pointer rules, buffer/device/stream lifecycle, role ownership, and
measurement attribution. Preserve verified current-skill pitfalls without
turning them into universal target claims.

Extract the historical failure descriptions from the pinned object:

```bash
git show bd80f49^:groupedtopk/log.md > /tmp/kernel-opt-loop-groupedtopk-history.md
```

Write `anti-patterns.md` entries with Evidence revision, Preconditions, Attempt,
Observed failure, and Reconsider when. Include the documented winner-tree,
sort-32+sort-64, `tl.gather` compaction, and cumsum approaches; do not copy
shape-specific claims into a generic rule without the Preconditions field.

- [ ] **Step 5: Correct bottleneck guidance**

Update `bottleneck-judgment.md` so all examples normalize device totals per call,
distinguish benchmark time from profiler time, require separate reference and
candidate scopes, and state that detailed host decomposition is Level 2 evidence
requested by Evaluation Contract rather than unconditional work.

- [ ] **Step 6: Delete the legacy template and run tests**

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_contracts.py' -v
test ! -e skills/kernel-opt-loop/references/log-template.md
```

Expected: PASS and the legacy template is absent.

- [ ] **Step 7: Commit**

```bash
git add skills/kernel-opt-loop/references skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: add durable kernel loop contracts"
```

---

### Task 4: Add the Claude Code Runtime Adapter

**Files:**
- Create: `skills/kernel-opt-loop/adapters/claude-code.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

**Interfaces:**
- Maps `start_role`, `continue_idle_role`, `send_advisory`, `wait_for_completion`, `inspect_roles`, and `end_workflow` to Claude Code.
- Consumes the resolved bootstrap from `SKILL.md`; it does not duplicate role behavior.

- [ ] **Step 1: Add failing adapter assertions**

Assert the adapter contains all six common operations, requires
`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`, names version floor `2.1.178`, and does
not contain active `TeamCreate(`, `TeamDelete(`, or `team_name=` syntax.

- [ ] **Step 2: Write `adapters/claude-code.md` from current official semantics**

Record these exact rules:

- preflight Claude Code `>= 2.1.178` and the experimental flag;
- spawn named teammates `designer`, `coder`, and `verifier` directly with the
  resolved bootstrap; the lead conversation history is not inherited;
- use `SendMessage` to continue an idle teammate or steer a running teammate;
- rely on automatic message delivery and idle notifications instead of polling;
- route every state-changing response back to the lead/Orchestrator;
- ask teammates to shut down before ending; session cleanup is automatic;
- never edit Claude's generated team config/task directories;
- if the preflight fails, use the sequential main-session fallback.

Link the compatibility evidence directly in the adapter:

- `https://code.claude.com/docs/en/agent-teams`
- `https://code.claude.com/docs/en/tools-reference`

- [ ] **Step 3: Run contract tests**

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_contracts.py' -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add skills/kernel-opt-loop/adapters/claude-code.md \
  skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: add Claude Code orchestration adapter"
```

---

### Task 5: Add the Codex Runtime Adapter

**Files:**
- Create: `skills/kernel-opt-loop/adapters/codex.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

**Interfaces:**
- Implements the same six common operations using current Codex collaboration tools.
- Uses one persistent agent identity per role when collaboration is available.

- [ ] **Step 1: Add failing Codex adapter assertions**

Require the strings `spawn_agent`, `followup_task`, `send_message`, `wait_agent`,
`list_agents`, `interrupt_agent`, and `fork_turns="none"`. Reject instructions
that shell out to `codex exec`.

- [ ] **Step 2: Write `adapters/codex.md`**

Define this operation mapping:

| Common operation | Codex action |
|---|---|
| `start_role` | `spawn_agent` with deterministic task name and `fork_turns="none"` |
| `continue_idle_role` | `followup_task` |
| `send_advisory` | `send_message` only when the role is already running |
| `wait_for_completion` | `wait_agent` with a multi-minute timeout |
| `inspect_roles` | `list_agents` for diagnostics only |
| `end_workflow` | let completed roles finish; `interrupt_agent` only for a stuck role |

The adapter must say the bootstrap, not optional agent type, defines the role.
Prefer `architect`, `developer`, and `qa` only when exposed; otherwise use
`default`. If collaboration is unavailable, select sequential fallback.

- [ ] **Step 3: Run contract tests**

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_contracts.py' -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add skills/kernel-opt-loop/adapters/codex.md \
  skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: add Codex orchestration adapter"
```

---

### Task 6: Write the Designer Contract

**Files:**
- Create: `skills/kernel-opt-loop/prompts/designer.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

**Interfaces:**
- Phase 0 consumes base/harness/environment inputs and authors the semantic portions of `project.md` plus `state/designer_state.md`.
- Round N consumes accepted canonical evidence, recent rejected evidence, profile, invariants, and anti-patterns; produces one immutable `decision_NNN.md`.

- [ ] **Step 1: Add failing Designer contract checks**

Assert Designer explicitly reads `last_accepted_kernel` and
`last_accepted_report`, writes all decision sections, never writes runtime
numbers, never edits `team-state.md`, and never revises a decision after coding
starts.

- [ ] **Step 2: Write the Phase 0 behavior**

Require Designer to extract semantics, shapes, dtypes, invariants, and a defensible
upbound from source evidence. It fills `project-template.md` but leaves measured
fields for Verifier. Unknown user-owned device/interpreter/upbound choices are
reported to Orchestrator instead of invented.

- [ ] **Step 3: Write Round N behavior**

Require this exact order:

1. Resolve `last_accepted_kernel` and `last_accepted_report` from manifest.
2. Read the latest completed failure evidence as history, never as canonical.
3. Consult target profile, bottleneck guidance, invariants, and anti-patterns.
4. Choose one bottleneck and one falsifiable intervention.
5. Fill Optimization Intent, conditional Sketch/Host Plan, and Evaluation Contract.
6. Run `validate_decision.py --expected-profile triton_mlu`.
7. Hand the path to Orchestrator; do not contact Coder directly.

When stable improvement of at least 5% cannot be justified, produce the complete
abort form defined in Task 1. On `major-deviation`, `capability-miss`, or failed
measurement-design evidence, record the rejected idea and write a new decision
only at the next unused round after the current round commits.

- [ ] **Step 4: Run contract tests**

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_contracts.py' -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/kernel-opt-loop/prompts/designer.md \
  skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: add Designer decision contract"
```

---

### Task 7: Write the Coder Contract and Triton-MLU Profile

**Files:**
- Create: `skills/kernel-opt-loop/prompts/coder.md`
- Create: `skills/kernel-opt-loop/prompts/coder_targets/triton_mlu.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

**Interfaces:**
- Consumes immutable decision, `last_accepted_kernel`, base contract, invariants, runtime fingerprint, and one exact profile.
- Produces candidate when possible and always produces `rounds/coder_result_NNN.md`.
- Result enum is `candidate-ready|design-revision-required|implementation-failed|environment-blocked`; design revision reason is `major-deviation|capability-miss`.

- [ ] **Step 1: Add failing taxonomy and ownership checks**

Assert Coder contains all four result names and both design-revision reasons,
states “Coder never returns accepted,” starts from `last_accepted_kernel`, runs
the decision validator, writes `coder_result_NNN.md`, and never edits decision,
profile, manifest, project overview, or Verifier report.

- [ ] **Step 2: Write `prompts/coder.md`**

Require this sequence:

1. Validate decision and compare project runtime fingerprint to profile match.
2. Classify missing runtime/profile mismatch as `environment-blocked`.
3. Check every Sketch primitive and H hint against profile Supported,
   Constrained, Unsupported, and Unknown tables.
4. Return `design-revision-required(reason=capability-miss)` for Unsupported or
   unprovable Unknown constructs; do not omit them silently.
5. Copy only `last_accepted_kernel`, implement the normative Optimization Intent,
   Unified Sketch, and Host Plan, and preserve base/public invariants.
6. Run `ast.parse` and the actual harness loader. Repair non-semantic syntax,
   import, or load defects at most twice.
7. Write structured result with source canonical path/hash, decision hash,
   profile/fingerprint, conformance notes, attempts, candidate path/hash, and
   reason code.
8. Return the artifact path to Orchestrator; never send directly to Verifier.

Small target-language adjustments are conformance notes under `candidate-ready`.
Any algorithm, dataflow, lifecycle, or Evaluation Contract change is
`major-deviation`.

- [ ] **Step 3: Write the self-contained `triton_mlu.md` profile**

Use these sections:

```markdown
# Target Profile: triton_mlu
## Identity and Match
## Runtime and Launcher Conventions
## Supported Primitives
## Constrained Primitives
## Unsupported Primitives
## Unknown Primitives
## Allowed Fallbacks
## Target-specific Pitfalls
## Evidence Ledger
```

Seed only repository-backed claims:

- mark `tl.load`, `tl.store`, `tl.arange`, `tl.program_id`, `tl.dot`,
  `tl.argmax`, and `tl.reshape` Supported with exact evidence paths from
  `groupedtopk/triton_grouped_topk_004.py`,
  `fused_moe/triton_fused_moe_005.py`,
  `flexattention/triton_flexattention_003.py`, and their project logs;
- describe `tl.zeros` only as a value-producing tensor operation, with no storage
  placement claim;
- mark `tl.make_block_ptr` Unknown until a local MLU probe establishes semantics
  and support;
- mark `num_warps` Constrained: value 1 is locally used, value 2 failed in the
  `flexattention/log.md` experiment, and every other value is Unknown;
- mark `num_stages=2` Constrained with
  `flexattention/triton_flexattention_004.py` and `flexattention/log.md` as
  evidence;
- mark `vectorize` and `async_copy` Unknown rather than fabricating support;
- document both observed `fast_libentry` import forms and require runtime
  introspection before choosing one;
- record device-context removal and output caching as Host Plan patterns, not
  Sketch primitive translations.

Each capability row has Primitive, Status, Constraint, Evidence, and Failure
classification columns. Do not include a “Sketch construct → generated code”
table.

- [ ] **Step 4: Add negative no-stub/no-fake-lowering tests**

Assert no files named `triton_cuda.md`, `triton_hip.md`, `triton_ascend.md`, or
`tilelang.md` exist. Assert the MLU profile does not contain the phrases
`tl.make_block_ptr.*register tile` or `tl.zeros.*SMEM` using case-insensitive
regular expressions.

- [ ] **Step 5: Run contract tests**

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_contracts.py' -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add skills/kernel-opt-loop/prompts/coder.md \
  skills/kernel-opt-loop/prompts/coder_targets/triton_mlu.md \
  skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: add target-bound Coder contract"
```

---

### Task 8: Write the Verifier Runtime and Evidence Contract

**Files:**
- Create: `skills/kernel-opt-loop/prompts/verifier.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

**Interfaces:**
- Consumes decision, candidate, accepted reference/report, project regime, and verifier state.
- Produces one `report_NNN.md` after actual execution, or a timestamped environment incident.
- Nonterminal classifications are `implementation-repair-required` and `measurement-incomplete`; terminal evidence leads to `accepted|no-improvement|candidate-failed|design-rejected` through Orchestrator.

- [ ] **Step 1: Add failing Verifier contract checks**

Assert Verifier is the sole authoritative runtime owner, compares to accepted
reference, runs correctness before timing, allows one Coder repair, mirrors
Evaluation Contract observables, normalizes profiler iterations, distinguishes
scopes, and never edits candidate, decision, manifest, or project overview.

- [ ] **Step 2: Write correctness and repair behavior**

Require Verifier to run the exact project reproduction command. A local
implementation defect returns `implementation-repair-required` through
Orchestrator with candidate hash, command, exit code, and diff/trace summary.
There is exactly one same-round Coder repair. A second correctness failure becomes
`candidate-failed`; a required algorithm/dataflow/lifecycle change becomes
`design-revision-required` and therefore `design-rejected`.

- [ ] **Step 3: Write authoritative timing behavior**

Run three interleaved pairs in one Verifier turn:

```text
accepted reference, candidate,
accepted reference, candidate,
accepted reference, candidate
```

Use identical warmup/repeat flags and store all six raw candidate-side wall
measurements. Compare the unrounded median of three accepted-reference samples to
the unrounded median of three candidate samples. Correctness and guardrails plus
`improvement_pct >= 5.0` produce `accepted`; otherwise produce
`no-improvement` after the measurement regime's repeat/noise check.

Each sample uses the existing harness, changing only `--v1_file`:

```bash
python3 auto_bench.py --v0_file operator/base.py \
  --v1_file operator/baseline_adapter.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file operator/base.py \
  --v1_file operator/triton_operator_001.py --warmup 50 --repeat 100
```

At runtime, substitute the manifest's accepted and candidate paths; keep the
other flags byte-for-byte identical across all six invocations.

- [ ] **Step 4: Write Evaluation Contract and profiler behavior**

Implement the four evidence levels from spec §8:

- Level 0 for every candidate: correctness and paired wall timing;
- Level 1 after correctness PASS: scoped reference/candidate device us per call,
  kernel count per call, and top-k kernels via `summarize_trace.py`;
- Level 2 only for named mechanism observables;
- Level 3 only for conflicting, unattributed, noise-bound, or stop-boundary cases.

For every observable write expected, observed, and verdict. Set hypothesis verdict
to one of `confirmed|partially-confirmed|falsified|inconclusive`. If a required
observable is absent, emit `measurement-incomplete`; collect the missing probe or
classify the cause as design/environment before any adoption decision.

Always populate `evidence_for_next_round` with facts and remaining bottleneck;
never prescribe the next implementation.

Use the harness's existing dual-scope profiling interface rather than editing the
harness:

```bash
python3 auto_bench.py --v0_file operator/base.py \
  --v1_file operator/triton_operator_001.py \
  --profile --profile-reference-file operator/baseline_adapter.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output operator/log/round_001_forward_50iter.pt.trace.json
```

Summarize scopes `reference_baseline_adapter` and
`candidate_triton_operator_001` separately with `summarize_trace.py`.

- [ ] **Step 5: Write environment and stop behavior**

An import failure, missing dependency, OOM unrelated to candidate design, device
loss, or indistinguishable required profiler scopes produces a timestamped
incident with command, exit code, stderr, fingerprint, and remediation need. It
does not write a terminal result.

Stop recommendations include evidence for measurement-bound, diminishing
returns, upbound reached, resource exhausted, or user intervention. Verifier
recommends; Orchestrator transitions.

- [ ] **Step 6: Run contract tests**

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_contracts.py' -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add skills/kernel-opt-loop/prompts/verifier.md \
  skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: add attributable Verifier contract"
```

---

### Task 9: Rewrite `SKILL.md` as the Orchestrator

**Files:**
- Modify: `skills/kernel-opt-loop/SKILL.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

**Interfaces:**
- Detects one runtime adapter, resolves the common bootstrap, validates artifacts, applies transitions, commits rounds, and handles stop/resume.
- Is the sole writer of manifest, project overview, canonical pointers, counters, and Git commits.

- [ ] **Step 1: Add failing orchestrator structure checks**

Require these H2 sections:

```text
When to use
Required inputs
Runtime selection
Agent bootstrap contract
Phase 0
Round N
Routing and state transitions
Stop criteria
Resume
Knowledge lift
References
```

Assert the old monolithic “write kernel directly” workflow and `log.md` references
are absent.

- [ ] **Step 2: Preserve trigger metadata and add runtime selection**

Keep frontmatter `name: kernel-opt-loop`. Update description to trigger on
iterative kernel/operator optimization with Triton and an auto_bench-style
harness. Selection order is Codex collaboration when exposed, Claude Code agent
teams when enabled, then sequential fallback. Load exactly one adapter.

- [ ] **Step 3: Add the exact common bootstrap from spec §4.2**

Copy it verbatim into `SKILL.md`. Require absolute resolved paths, no parent
conversation dependency, declared file ownership, and adapter-mediated
completion. Do not paste full role contracts into spawn prompts.

- [ ] **Step 4: Implement Phase 0 orchestration**

Specify these exact actions:

1. Resolve absolute skill/project/base/harness/interpreter/device paths.
2. Initialize `rounds/`, `state/`, `log/`, project template, manifest, and role state.
3. Discover language/backend/profile and runtime fingerprint; validate exact profile match.
4. Dispatch Designer Phase 0 and request only undiscoverable user-owned values.
5. Generate `baseline_adapter.py`; verify `base.py` bytes are unchanged.
6. Dispatch Verifier for baseline correctness, wall timing, and Level 1 profile.
7. Compute SHA-256 over base bytes, NUL, harness bytes, NUL, and canonical JSON
   measurement settings (`sort_keys=True`, separators `(',', ':')`). The object
   has exactly `shape`, `dtype`, `device`, `warmup`, `repeat`, `profile_mode`,
   `profile_warmup`, and `profile_iterations` keys.
8. Set `last_completed_round: "000"`, `last_accepted_round: "000"`,
   `last_accepted_kernel: baseline_adapter.py`, both report pointers to
   `rounds/report_000.md`, `last_result: baseline`, `phase: ready`, append
   overview/transition rows, and commit.

On failure, write timestamped incident, set `blocked`, leave accepted pointers
null, commit the blocking transition, and require remediation.

- [ ] **Step 5: Implement Round N orchestration**

Round number is `total_rounds + 1`. Enforce this ordered state machine:

1. `designing`: Designer writes a decision.
2. Validate decision before `coding`.
3. On abort, complete `aborted` without Coder/Verifier.
4. Coder writes `coder_result_NNN.md` and optional candidate from canonical.
5. Route Coder result according to spec §7.4.
6. Validate candidate with the actual harness loader before `verifying`.
7. Verifier writes report, requests one repair, requests missing evidence, or blocks.
8. Validate required artifacts and calculate exactly one terminal transition.
9. Update `last_completed_round`, `last_completed_decision`,
   `last_completed_coder_result`, `last_completed_report`, and `last_result`, using
   null for artifacts that the selected path does not produce.
10. Append one project overview row and one transition row.
11. Increment counters once, update accepted pointers only for accepted, commit,
    then and only then dispatch the next round.

Use terminal results exactly:

| Result | Performance streak | Failed-attempt streak | Canonical |
|---|---:|---:|---|
| `accepted` | 0 | 0 | candidate |
| `no-improvement` | +1 | 0 | unchanged |
| `design-rejected` | 0 | +1 | unchanged |
| `candidate-failed` | 0 | +1 | unchanged |
| `aborted` | 0 | +1 | unchanged |

Environment incidents update neither counter nor `total_rounds`.

- [ ] **Step 6: Implement stop, resume, visibility, and knowledge lift**

Verifier writes `round_status_NNN.md` at verification start, after correctness,
after each timing pair, and at verification end. Designer and Coder report
completion through the runtime adapter. Orchestrator does not poll when the
runtime provides completion notifications.

After every terminal transition, evaluate all five stop criteria. User stop is
unconditional. Designer may reject another recommendation only with a concrete
next hypothesis expected to clear 5%.

Resume validates artifact hashes, skill version, target fingerprint, and
measurement fingerprint. Reuse a valid uncommitted predecessor artifact in the
same round; never reopen a completed decision. A target change invalidates the
decision, and a measurement change requires a comparable baseline.

At stop, propose generic anti-pattern promotions to the user. Edit skill-level
knowledge only after explicit approval and in a separate commit.

- [ ] **Step 7: Run orchestrator contract tests**

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_contracts.py' -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add skills/kernel-opt-loop/SKILL.md skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: rewrite kernel loop orchestrator"
```

---

### Task 10: Complete Cross-file and Pressure Validation

**Files:**
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`
- Modify only when a failing test identifies a defect: files created in Tasks 1–9

**Interfaces:**
- One automated suite validates names, ownership, state/result enums, forbidden future scope, and cross-file consistency.
- Guided runtime scenarios validate behavior that static Markdown checks cannot prove.

- [ ] **Step 1: Add cross-file consistency tests**

Assert:

- every final-structure file exists and is nonempty;
- decision headings and result/status enums match across templates, roles, and orchestrator;
- all manifest phases and counter names are consistent;
- role contracts forbid edits outside ownership;
- only Verifier uses `accepted` as a produced result;
- all profiler formulas and Evaluation Contract fields appear in both Verifier and report template;
- Claude-specific syntax occurs only in its adapter and Codex-specific syntax only in its adapter;
- no target candidate list, capability-miss routing history, inactive profile stub,
  deterministic lowering, KernelWiki API, or deep-profiler implementation exists;
- no Markdown fence is unclosed;
- `validate_decision.py` is executable.

- [ ] **Step 2: Run the full automated suite**

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_*.py' -v
git diff --check
```

Expected: all tests PASS and `git diff --check` prints nothing.

- [ ] **Step 3: Run five guided pressure scenarios**

Use `/tmp/kernel-opt-loop-pressure/<runtime>/<scenario>/` and copy only fixture
inputs there. Run each available runtime through these exact cases:

1. **Rejected previous candidate:** Round 2 exists and failed while canonical is
   baseline. Designer and Coder must use baseline.
2. **Capability miss:** decision requests an Unknown profile primitive. Coder
   writes `design-revision-required(reason=capability-miss)`; current round commits
   `design-rejected`; Designer creates the next round, not a revision in place.
3. **Verifier repair:** first candidate has a local accuracy defect. Verifier
   returns it to Coder exactly once; second failure completes `candidate-failed`.
4. **Profiler normalization:** 50-call trace totals 1,000 us at 0.1 ms/call.
   Report must state 20 us/call and 20%, with separate accepted/candidate scopes.
5. **Environment block and resume:** missing interpreter creates a timestamped
   incident without changing round/streak; after restoring the same fingerprint,
   workflow resumes the same safe step.

Save produced durable artifacts and a one-line PASS/FAIL ledger under each
scenario directory. A verbal role response without artifacts is FAIL.

- [ ] **Step 4: Run one scenario through sequential fallback**

Disable collaboration/team capability and run scenario 1. Confirm identical
artifact names, ownership, canonical choice, and state transition.

- [ ] **Step 5: Fix the smallest contract for every observed loophole and rerun**

For each failure, add a regression assertion to `test_contracts.py`, edit the one
owning contract, then rerun Step 2 and the failed scenario. Do not weaken fixture
expectations.

- [ ] **Step 6: Verify spec traceability and placeholder absence**

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('docs/superpowers/plans/2026-08-13-kernel-opt-loop-restructure.md').read_text()
for section in range(4, 17):
    assert f'§{section}' in p, section
for forbidden in ('T' + 'BD', 'TO' + 'DO', 'fill in' + ' details', 'Similar' + ' to Task'):
    assert forbidden not in p, forbidden
print('plan traceability: PASS')
PY
```

Expected: `plan traceability: PASS`.

- [ ] **Step 7: Commit**

```bash
git add skills/kernel-opt-loop
git commit -m "skills: verify kernel loop contracts"
```

---

### Task 11: Sync and Verify Runtime Installations

**Files:**
- Source: `skills/kernel-opt-loop/`
- Sync destination: `${HOME}/.claude/skills/kernel-opt-loop/`
- Sync destination: `${HOME}/.codex/skills/kernel-opt-loop/`

**Interfaces:**
- Installed copies must byte-match the verified repository source.
- The exact two skill directories are the only deletion scopes.

- [ ] **Step 1: Re-run all repository checks before installation**

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_*.py' -v
git diff --check
git status --short
```

Expected: tests PASS, diff check is empty, and only intended implementation files
are staged/modified.

- [ ] **Step 2: Back up existing installed skill directories**

```bash
install_backup_root="$(mktemp -d)"
mkdir -p "$install_backup_root/claude" "$install_backup_root/codex"
test ! -d "${HOME}/.claude/skills/kernel-opt-loop" || \
  cp -a "${HOME}/.claude/skills/kernel-opt-loop" "$install_backup_root/claude/"
test ! -d "${HOME}/.codex/skills/kernel-opt-loop" || \
  cp -a "${HOME}/.codex/skills/kernel-opt-loop" "$install_backup_root/codex/"
echo "$install_backup_root"
```

Record the printed backup directory in the execution log.

- [ ] **Step 3: Sync the two exact destinations**

```bash
mkdir -p "${HOME}/.claude/skills/kernel-opt-loop" \
  "${HOME}/.codex/skills/kernel-opt-loop"
rsync -a --delete skills/kernel-opt-loop/ "${HOME}/.claude/skills/kernel-opt-loop/"
rsync -a --delete skills/kernel-opt-loop/ "${HOME}/.codex/skills/kernel-opt-loop/"
```

`--delete` is intentionally limited to the two exact skill directories and
removes legacy files such as `references/log-template.md`. The backup makes this
recoverable.

- [ ] **Step 4: Compare installed trees**

```bash
diff -ru skills/kernel-opt-loop "${HOME}/.claude/skills/kernel-opt-loop"
diff -ru skills/kernel-opt-loop "${HOME}/.codex/skills/kernel-opt-loop"
```

Expected: both commands produce no output.

- [ ] **Step 5: Run one installed-path bootstrap smoke check per runtime**

For Claude Code, verify the adapter selects agent-team behavior only when its
version/flag preflight succeeds. For Codex, verify a disposable role with
`fork_turns="none"` reads its installed role contract and echoes the absolute
skill root, project root, phase, inputs, and required outputs. Do not run an
optimization or modify a project in this smoke check.

- [ ] **Step 6: Record installation verification**

```bash
git status --short --branch
git log --oneline --max-count=12
```

Expected: repository worktree is clean and the task commits appear in order.

---

## Final Review Checklist

- [ ] Every spec requirement maps to a task and an executable/static acceptance check.
- [ ] No v1 task implements §15 Future Directions.
- [ ] No decision is revised after coding begins.
- [ ] Coder statuses and Verifier terminal results are not conflated.
- [ ] Environment incidents are nonterminal and counter-neutral.
- [ ] Host-only optimization is representable without a fake kernel Sketch.
- [ ] Triton vendor differences are captured through profile plus runtime fingerprint.
- [ ] Verifier's report maps every runtime observation to the round's Evaluation Contract.
- [ ] Only accepted candidates advance canonical state.
- [ ] Automated tests, guided pressure scenarios, installed-tree diffs, and Git status all pass.
