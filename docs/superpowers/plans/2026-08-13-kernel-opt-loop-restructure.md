# kernel-opt-loop Skill Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `kernel-opt-loop` into a cross-runtime Designer/Coder/Verifier optimization workflow with explicit file contracts, deterministic round state, resumability, and tested Claude Code and Codex orchestration adapters.

**Architecture:** `SKILL.md` owns the runtime-neutral state machine and the common agent bootstrap contract. Role behavior lives in `prompts/{designer,coder,verifier}.md`; runtime-specific dispatch behavior lives in `adapters/{claude-code,codex}.md`. Project state is persisted in `project.md`, `team-state.md`, `rounds/`, and `state/`, while small Python helpers make baseline creation and profiler normalization reproducible.

**Tech Stack:** Markdown Agent Skill files; Python 3 standard library helpers; existing `auto_bench.py`; Claude Code agent-team teammates; Codex multi-agent collaboration tools; bash for validation.

**Spec:** `docs/superpowers/specs/2026-08-13-kernel-opt-loop-restructure-design.md`. This plan resolves implementation defects discovered after the design was approved. Where the plan is more specific than the spec about runtime APIs, Phase 0 baseline, accepted-candidate state, or metric normalization, this plan is authoritative for implementation.

## Global Constraints

- Skill source of truth is `skills/kernel-opt-loop/` in this repository.
- The skill must work in both Claude Code and Codex. Runtime-neutral files must not contain tool-call syntax from either runtime.
- Claude Code compatibility target is v2.1.178 or newer with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`. Do not use removed `TeamCreate` or `TeamDelete` tools; do not rely on `team_name`.
- Codex requires the `multi_agent` capability. Use `spawn_agent` for initial role creation, `followup_task` for later turns, `send_message` only for steering a running role, and `wait_agent` for completion.
- Use portable general-purpose/default agents plus the skill-local role contracts. Runtime-local `architect`, `developer`, and `qa` definitions are optional optimizations, not required dependencies.
- Every spawned role receives the common bootstrap message defined in `SKILL.md`; do not inline the full role contract into the spawn message.
- v1 is linear: exactly one active optimization round and one candidate at a time. The three roles may stay alive, but they do not work on competing candidates in parallel.
- `base.py` is user-owned and immutable. Phase 0 generates `baseline_adapter.py`; it never rewrites `base.py`.
- `last_accepted_kernel` is the only canonical implementation pointer. A candidate that is slower, improves by less than 5%, fails accuracy, or aborts must not become the next Coder starting point.
- Every terminal round result is exactly one of `accepted`, `no-improvement`, `accuracy-fail`, `abort`, or `env-fail`. Phase 0 uses `baseline`.
- Profiler totals must be normalized to one forward call before computing `device_ratio`.
- Existing `groupedtopk/log.md` and `fused_moe/log.md` are not migrated. Historical anti-pattern extraction uses the pinned Git object `bd80f49^:groupedtopk/log.md` because the working-tree log no longer contains Entries 005–024.
- Runtime state and role state are committed at every completed state transition so a fresh session can resume from disk.
- Manifest phases are `initializing`, `ready`, `designing`, `coding`, `verifying`, `blocked`, and `stopped`. Role dispatch is allowed only when the predecessor artifact is complete and schema-valid.
- Implementation must begin in an isolated worktree created through `superpowers:using-git-worktrees`; do not create or merge branches from inside the skill workflow itself.

---

## Final File Structure

```text
skills/kernel-opt-loop/
  SKILL.md
  adapters/
    claude-code.md
    codex.md
  prompts/
    designer.md
    coder.md
    verifier.md
  references/
    anti-patterns.md
    bottleneck-judgment.md
    decision-template.md
    invariants.md
    project-template.md
    report-template.md
    team-state-template.md
  scripts/
    make_baseline_adapter.py
    summarize_trace.py
  tests/
    check_contracts.sh
    test_helpers.py
```

Project structure produced by the skill:

```text
<op>/
  base.py
  baseline_adapter.py
  project.md
  team-state.md
  triton_<op>_<NNN>.py
  rounds/
    decision_NNN.md
    report_NNN.md
    round_status_NNN.md
  state/
    designer_state.md
    coder_state.md
    verifier_state.md
  log/
    *.pt.trace.json
```

`log/` remains gitignored. All other files are durable project artifacts.

## State Machine Contract

The orchestrator is the sole writer of `team-state.md` and `project.md` overview rows. Roles write only their own state file and their declared decision/report/kernel output.

| Result | Required artifacts | Canonical pointer | Counter update | Next action |
|---|---|---|---|---|
| `baseline` | `baseline_adapter.py`, `report_000.md` | `last_accepted_kernel: baseline_adapter.py` | both streaks = 0 | start Round 1 |
| `accepted` | decision, candidate, report | advance to candidate | both streaks = 0 | next round |
| `no-improvement` | decision, candidate, report | unchanged | no-improvement +1; failed-attempts = 0 | next round or stop at 3 |
| `accuracy-fail` | decision, candidate, report | unchanged | failed-attempts +1; no-improvement = 0 | next round or stop at 3 |
| `abort` | decision only | unchanged | failed-attempts +1; no-improvement = 0 | next round or stop at 3 |
| `env-fail` | Phase 0: environment report; Round N: decision, candidate, environment report | unchanged | streaks unchanged | set phase `blocked`, surface to user |

Round comparison is always against `last_accepted_report`, whose implementation must equal `last_accepted_kernel`, never merely Round N-1. `last_completed_report` tracks the newest audit result independently. Audit commits may contain rejected or broken candidates, but `team-state.md` prevents them from becoming canonical.

---

## Task 1: Add deterministic helper scripts and tests

**Files:**
- Create: `skills/kernel-opt-loop/scripts/make_baseline_adapter.py`
- Create: `skills/kernel-opt-loop/scripts/summarize_trace.py`
- Create: `skills/kernel-opt-loop/tests/test_helpers.py`

**Interfaces:**
- `make_baseline_adapter.py SOURCE DEST` reads a Python operator file, renames its one top-level `Model` class to `ModelNew`, and writes an AST-equivalent adapter without modifying the source.
- `summarize_trace.py TRACE --iterations N [--scope RECORD_FUNCTION]` prints JSON with `iterations`, `device_total_us`, `device_us_per_call`, and a per-kernel breakdown. `--scope` restricts events to kernels nested under one profiler `record_function` span when the trace contains both accepted-reference and candidate calls.

- [ ] **Step 1: Write failing tests for baseline adapter generation**

Add `unittest` cases that create a temporary source containing `Model`, `get_inputs`, and `get_init_inputs`, invoke the helper, and assert:

```python
self.assertIn("class ModelNew", output)
self.assertNotIn("class Model(", output)
self.assertIn("def get_inputs", output)
self.assertEqual(source_path.read_text(), original_source)
```

Also add failure cases for zero or two top-level `Model` classes; both must exit non-zero with a clear error.

- [ ] **Step 2: Write failing tests for trace normalization**

Use a synthetic trace with three kernel events:

```json
{"traceEvents": [
  {"cat": "kernel", "name": "k1", "dur": 20},
  {"cat": "kernel", "name": "k1", "dur": 30},
  {"cat": "kernel", "name": "k2", "dur": 50},
  {"cat": "cpu_op", "name": "ignored", "dur": 999}
]}
```

For `--iterations 10`, assert total `100.0`, per-call `10.0`, `k1.total_us == 50.0`, and `k1.per_call_us == 5.0`. Add a synthetic trace with `reference_*` and `candidate_*` record-function spans, CPU launch events inside each span, and asynchronous kernels linked by `External id`/`correlation`; assert `--scope candidate_test` excludes reference kernels even when kernel timestamps extend beyond the CPU span. Add rejection tests for `--iterations 0`, an unknown scope, ambiguous duplicate scopes, missing correlation metadata, and missing `traceEvents`.

- [ ] **Step 3: Run the helper tests and verify RED**

Run:

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_*.py' -v
```

Expected: failures because both scripts are absent.

- [ ] **Step 4: Implement `make_baseline_adapter.py`**

Use `argparse`, `ast.parse`, an `ast.NodeTransformer`, `ast.fix_missing_locations`, and `ast.unparse`. Rename only a top-level `ClassDef` named `Model`; preserve every other definition. Refuse ambiguous input rather than guessing.

- [ ] **Step 5: Implement `summarize_trace.py`**

Use only the Python standard library. Without `--scope`, sum numeric `dur` values where `cat == "kernel"`. With `--scope`, resolve exactly one complete or begin/end record-function span; collect correlation identifiers from CPU launch descendants on the same thread, then include only kernel events whose `External id`/`correlation` links to those descendants. Timestamp containment may be used only when the trace explicitly lacks async accelerator events; otherwise refuse un-attributable or ambiguous data. Group by `name`, divide totals and counts by the supplied iteration count, sort breakdown rows by descending total duration, and emit stable JSON via `json.dumps(..., indent=2, sort_keys=True)`.

- [ ] **Step 6: Run tests and verify GREEN**

Run the command from Step 3. Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add skills/kernel-opt-loop/scripts skills/kernel-opt-loop/tests/test_helpers.py
git commit -m "skills: add reproducible baseline and trace helpers"
```

---

## Task 2: Add invariant and anti-pattern references

**Files:**
- Create: `skills/kernel-opt-loop/references/invariants.md`
- Create: `skills/kernel-opt-loop/references/anti-patterns.md`

**Interfaces:**
- Consumes current `skills/kernel-opt-loop/SKILL.md`, `references/bottleneck-judgment.md`, and pinned historical log `bd80f49^:groupedtopk/log.md`.
- Produces references read by Designer and Coder.

- [ ] **Step 1: Verify the pinned anti-pattern source exists**

```bash
git show bd80f49^:groupedtopk/log.md | grep -q "### Entry 024"
```

Expected: exit 0. If the Git object is unavailable, stop this task; do not silently use the truncated working-tree log.

- [ ] **Step 2: Write `invariants.md`**

Include concrete safe/unsafe examples for all of these:

1. `_filter_module_ast` strips non-literal module assignments; use the class-body `globals()` pattern for `fast_libentry`.
2. The `argmax sentinel` for masked lanes is zero, not `E`, when values will be summed.
3. `tl.dot` inputs are 2D and inner dimensions match; transpose `[2I,H]` before multiplying by `[1,H]`.
4. Drop `torch.mlu.device()` only when the caller establishes the device.
5. Cache output buffers on `ModelNew` with `torch.empty_like` or an equivalent allocation when shape/device/dtype are stable.
6. `fast_libentry` reduces launcher overhead but does not remove harness synchronization.
7. Fuse routing softmax/top-k/cast when they appear as separate device kernels.
8. Treat `set_seed`, multi-accelerator `sync_devices`, and `build_case/load_state_dict` as harness-fixed costs.
9. Coder starts from `team-state.md:last_accepted_kernel`, never from the numerically previous candidate.

- [ ] **Step 3: Extract anti-patterns from the pinned log**

Read the historical log without modifying the worktree:

```bash
git show bd80f49^:groupedtopk/log.md > /tmp/kernel-opt-loop-groupedtopk-history.md
```

Abstract Entries 004, 005, 006, 007, 011, 012, 013, 014, 015, 016, 017, 019, 021, 022, 023, and 024. Each catalog entry must contain:

- source entry numbers;
- hypothesis;
- structural failure reason;
- recognition signs;
- evidence boundary explaining when the pattern may not apply.

- [ ] **Step 4: Validate reference coverage**

Run:

```bash
for token in _filter_module_ast fast_libentry "argmax sentinel" tl.dot torch.mlu.device torch.empty_like set_seed sync_devices last_accepted_kernel; do
  grep -qF "$token" skills/kernel-opt-loop/references/invariants.md || exit 1
done
for entry in 004 005 006 007 011 012 013 014 015 016 017 019 021 022 023 024; do
  grep -q "Entry $entry" skills/kernel-opt-loop/references/anti-patterns.md || exit 1
done
```

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add skills/kernel-opt-loop/references/invariants.md skills/kernel-opt-loop/references/anti-patterns.md
git commit -m "skills: add kernel invariants and historical anti-patterns"
```

---

## Task 3: Replace the monolithic log template with explicit contracts

**Files:**
- Create: `skills/kernel-opt-loop/references/project-template.md`
- Create: `skills/kernel-opt-loop/references/decision-template.md`
- Create: `skills/kernel-opt-loop/references/report-template.md`
- Create: `skills/kernel-opt-loop/references/team-state-template.md`
- Delete: `skills/kernel-opt-loop/references/log-template.md`

**Interfaces:**
- Produces the exact schemas used by every role and by the orchestrator.
- `team-state.md` is Markdown with YAML frontmatter and a short human-readable transition log.

- [ ] **Step 1: Write `team-state-template.md` with the complete manifest schema**

The template frontmatter must contain these keys and initial values:

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
last_completed_decision: null
last_completed_report: null
last_accepted_report: null
last_result: null
consecutive_no_improvement: 0
consecutive_failed_attempts: 0
total_rounds: 0
measurement_fingerprint: null
stop_reason: null
stop_timestamp: null
skill_version_at_stop: null
measurement_fingerprint_at_stop: null
kb_revision_at_stop: null
resume_eligible: always
resume_constraints: []
---
```

Below the frontmatter, include `## Transition Log`, with one append-only row per orchestrator update.

Define `measurement_fingerprint` deterministically as `sha256(base_bytes + b"\0" + harness_bytes + b"\0" + settings_json_bytes)`, where settings JSON uses `sort_keys=True` and separators `(',', ':')` and contains shape, dtype, device, warmup, repeat, profile mode, profile warmup, and profile iterations. Resume validation must recompute exactly this value.

- [ ] **Step 2: Write `decision-template.md`**

Require these fields:

```text
Decision: proceed | abort
Round
Reference implementation
Reference report
Bottleneck class and normalized device_ratio
Falsifiable hypothesis
One optimization means
Expected wall improvement percentage
Pitfall warnings
Anti-pattern consultation hit/miss
Acceptance rule
```

An abort decision still fills Round, Reference, rejection rationale, and anti-pattern consultation; it is not a one-line file.

- [ ] **Step 3: Write `report-template.md`**

Require these fields:

```text
Result: baseline | accepted | no-improvement | accuracy-fail | env-fail
Round
Candidate and reference implementation paths
Correctness and diff summary
Accepted-reference and candidate trial medians, with aggregate median for each
Reference median wall time
Improvement percentage versus last accepted
profile_iterations
device_total_us
device_us_per_call
device_ratio
Per-kernel per-call breakdown
Upbound gap
Retry history
Stop recommendation and evidence
```

Define the unit formula in the template:

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
device_ratio = device_us_per_call / (candidate_wall_ms * 1000)
```

- [ ] **Step 4: Write `project-template.md`**

Preserve the old project-level sections for semantics, environment, measurement regime, upbound, reproduction, and checkpoint. Change the overview table to:

```markdown
| Round | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Relative to base | Canonical after round |
|---:|---|---|---|---:|---:|---:|---:|---|
```

Document that rejected candidates remain audit rows but do not change the canonical column.

- [ ] **Step 5: Delete `log-template.md` and validate all schemas**

```bash
git rm skills/kernel-opt-loop/references/log-template.md
for file in project-template decision-template report-template team-state-template; do
  test -s "skills/kernel-opt-loop/references/${file}.md" || exit 1
done
for key in last_accepted_kernel last_completed_report last_accepted_report consecutive_no_improvement consecutive_failed_attempts resume_constraints; do
  grep -qF "$key" skills/kernel-opt-loop/references/team-state-template.md || exit 1
done
```

Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add skills/kernel-opt-loop/references
git commit -m "skills: define project round and resume contracts"
```

---

## Task 4: Write the Claude Code runtime adapter

**Files:**
- Create: `skills/kernel-opt-loop/adapters/claude-code.md`

**Interfaces:**
- Consumes the bootstrap template from `SKILL.md` and role contracts from `prompts/`.
- Produces runtime-specific instructions for spawning, messaging, waiting, and shutdown.

- [ ] **Step 1: Capture an unguided control**

In a disposable Claude Code session, ask only: “run kernel-opt-loop with designer, coder, verifier agents.” Record whether it tries `TeamCreate`, relies on a custom agent type, or omits role-contract paths. This establishes the RED behavior; do not alter repository files from the control session.

- [ ] **Step 2: Write `adapters/claude-code.md`**

The adapter must require:

1. Verify Claude Code version is at least 2.1.178.
2. Verify `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`; if absent, use the sequential fallback in `SKILL.md` or ask the user to enable it.
3. Spawn three **agent-team teammates** named `designer`, `coder`, and `verifier`; explicitly say “agent-team teammate” so Claude does not choose ordinary one-shot subagents.
4. Use a portable general-purpose agent unless an equivalent runtime-local role is already available.
5. Let Claude choose its currently supported teammate-spawn primitive; the adapter specifies desired roles and lifecycle, not a hardcoded creation call.
6. Pass only the common bootstrap message with absolute role-contract, adapter, project, input, and output paths. Do not paste the full role prompt into the spawn request.
7. Use `SendMessage` for teammate-to-teammate and teammate-to-lead communication.
8. On finish, the lead sends a shutdown request to each teammate. Do not call `TeamCreate`, `TeamDelete`, or depend on `team_name`; session cleanup is automatic.

- [ ] **Step 3: Run a guided smoke test**

Spawn one disposable teammate with a bootstrap pointing to `prompts/designer.md` and a temporary project fixture. Require it to echo the resolved contract path, adapter path, project root, phase, inputs, and outputs before exiting. Verify all six fields appear.

- [ ] **Step 4: Commit**

```bash
git add skills/kernel-opt-loop/adapters/claude-code.md
git commit -m "skills: add Claude Code orchestration adapter"
```

---

## Task 5: Write the Codex runtime adapter

**Files:**
- Create: `skills/kernel-opt-loop/adapters/codex.md`

**Interfaces:**
- Maps runtime-neutral role lifecycle operations to Codex collaboration tools.
- Must work when only the `default` agent type is portable.

- [ ] **Step 1: Capture an unguided control**

In a disposable Codex session with multi-agent enabled, ask only: “run kernel-opt-loop with designer, coder, verifier agents.” Record whether the workers receive role-contract and artifact paths. This is the RED baseline.

- [ ] **Step 2: Write `adapters/codex.md`**

Specify this mapping:

| Lifecycle operation | Codex action |
|---|---|
| Create role first time | `spawn_agent` with `fork_turns="none"` and the common bootstrap message |
| Start a later idle turn | `followup_task` with current phase and new input/output paths |
| Steer a running turn | `send_message` |
| Wait for completion | `wait_agent` with a multi-minute timeout |
| Inspect live roles | `list_agents` only for diagnostics |
| End workflow | let completed roles finish; interrupt only a stuck role |

Prefer `architect`, `developer`, and `qa` when the current Codex runtime exposes them; otherwise use `default`. The role contract, not the optional agent type, defines behavior.

Require a preflight check for multi-agent collaboration. If unavailable, invoke the sequential fallback; do not shell out to nested `codex exec` processes.

- [ ] **Step 3: Run a guided smoke test**

Spawn a disposable default agent with `fork_turns="none"` and a bootstrap pointing to one role contract. Verify that it reads the contract and returns all bootstrap fields without relying on parent conversation history.

- [ ] **Step 4: Commit**

```bash
git add skills/kernel-opt-loop/adapters/codex.md
git commit -m "skills: add Codex orchestration adapter"
```

---

## Task 6: Write the Designer role contract

**Files:**
- Create: `skills/kernel-opt-loop/prompts/designer.md`

**Interfaces:**
- Phase 0 consumes `base.py`, `project-template.md`, and environment inputs; produces `project.md` and `state/designer_state.md`.
- Round N consumes `team-state.md`, `last_accepted_report`, `last_completed_decision`, `last_completed_report`, anti-patterns, bottleneck judgment, and invariants; produces `decision_NNN.md` and updates only `designer_state.md`.

- [ ] **Step 1: Write an unguided Designer pressure fixture**

Create a temporary project whose Round 2 immediately follows a slower Round 1, while `team-state.md` still points to the accepted baseline. Ask an unguided agent to choose its reference. Record the RED failure if it chooses `triton_test_001.py` merely because it is numerically previous.

- [ ] **Step 2: Write `prompts/designer.md`**

The contract must state:

- Read the bootstrap-named files completely before acting.
- In Phase 0, write `project.md` and initialize only `designer_state.md`; this is the only phase in which Designer may create `project.md`. Do not invent baseline timings.
- In Round N, read `team-state.md:last_accepted_kernel` with `last_accepted_report` as the performance reference. Read `last_completed_decision` and `last_completed_report` only as recent evidence; a rejected Round N-1 result never becomes the reference.
- Use exactly one bottleneck and one falsifiable optimization.
- Quantify expected improvement relative to the accepted reference.
- If a stable 5% improvement cannot be justified, write a complete abort decision from `decision-template.md` and notify the orchestrator; do not contact Coder.
- Consult `anti-patterns.md` every round and record hits/misses plus why a superficially matching attempt differs.
- Outside Phase 0, own only `state/designer_state.md` and the current decision; never write `team-state.md` or `project.md` overview rows.
- For a Coder major-deviation request, revise the same decision at most twice before aborting.
- Never overwrite a completed prior-round decision. A rejected shutdown after Round N creates `decision_<N+1>.md`.

- [ ] **Step 3: Run the guided pressure test**

Use the same fixture and bootstrap the Designer with its contract. Expected: the produced decision names the accepted baseline from `team-state.md`, not the slower previous candidate, and contains every `decision-template.md` field.

- [ ] **Step 4: Commit**

```bash
git add skills/kernel-opt-loop/prompts/designer.md
git commit -m "skills: add Designer role contract"
```

---

## Task 7: Write the Coder role contract

**Files:**
- Create: `skills/kernel-opt-loop/prompts/coder.md`

**Interfaces:**
- Consumes `decision_NNN.md`, `team-state.md:last_accepted_kernel`, `base.py`, invariants, and `coder_state.md`.
- Produces `triton_<op>_<NNN>.py` and updates only `coder_state.md`.

- [ ] **Step 1: Write an unguided Coder pressure fixture**

Create a fixture where `triton_test_001.py` exists but is marked `no-improvement`, while `team-state.md:last_accepted_kernel` points to `baseline_adapter.py`. Ask an unguided agent what to copy. Record RED if it chooses the numerically previous file.

- [ ] **Step 2: Write `prompts/coder.md`**

Require Coder to:

- Read the common bootstrap inputs and `invariants.md`.
- Copy `last_accepted_kernel` as the starting point, even when later rejected files exist.
- Implement only the decision; no unrelated refactor.
- Preserve the `ModelNew`, `get_inputs`, and `get_init_inputs` contract.
- Classify deviations as minor or major. Minor deviations are logged in `coder_state.md`; major deviations trigger a Designer revision request, with at most two round trips.
- Self-check with both `ast.parse` and the actual harness loader:

```bash
<python> -c "from pathlib import Path; import auto_bench as b; m=b.load_ks_module(Path('<candidate>')); assert hasattr(m, 'ModelNew'); assert hasattr(m, 'get_inputs'); assert hasattr(m, 'get_init_inputs')"
```

- Attempt at most two self-fixes for syntax/import/load failures before escalating.
- Hand the candidate path, source accepted-kernel path, and deviations to Verifier through the active runtime adapter.
- Own only `state/coder_state.md` and the current candidate; never edit `team-state.md`, manifest counters, or canonical pointers.

- [ ] **Step 3: Run the guided pressure test**

Expected: Coder copies `baseline_adapter.py`, produces an AST-parseable candidate, and records its source path in `coder_state.md`.

- [ ] **Step 4: Commit**

```bash
git add skills/kernel-opt-loop/prompts/coder.md
git commit -m "skills: add Coder role contract"
```

---

## Task 8: Write the Verifier role contract

**Files:**
- Create: `skills/kernel-opt-loop/prompts/verifier.md`

**Interfaces:**
- Phase 0 consumes `base.py`, `project.md`, helper scripts, `auto_bench.py`, and the interpreter path; produces `baseline_adapter.py`, `report_000.md`, and `verifier_state.md`.
- Round N consumes the candidate, `last_accepted_kernel`, `last_accepted_report`, project measurement regime, and verifier state; produces `report_NNN.md`, optional status file, and updates only `verifier_state.md`.

- [ ] **Step 1: Write an unguided metric pressure fixture**

Give an unguided agent a trace totaling 1,000 us across 50 forwards and a wall time of 0.1 ms/call. Record RED if it reports `device_ratio = 1000 / 100 = 1000%` instead of `(1000/50)/100 = 20%`.

- [ ] **Step 2: Write the Phase 0 contract**

Require Verifier to:

1. Generate `baseline_adapter.py` with `make_baseline_adapter.py`.
2. Run `auto_bench.py --v0_file base.py --v1_file baseline_adapter.py` using project warmup/repeat settings.
3. Profile baseline forward calls using the declared iteration count.
4. Run `summarize_trace.py TRACE --iterations N --scope candidate_baseline_adapter` (or the exact emitted candidate record-function label).
5. Write `report_000.md` with `Result: baseline`, normalized device metrics, and the exact reproduction commands.
6. Initialize `state/verifier_state.md` with environment, measurement regime, and baseline samples.

- [ ] **Step 3: Write the Round N verification contract**

Require:

- Correctness first. One Coder retry is allowed; a second failure produces `Result: accuracy-fail` and a full audit report.
- For a passing candidate, compare to `last_accepted_kernel` and `last_accepted_report`, not simply Round N-1.
- Run three interleaved accepted-reference/candidate timing pairs with identical flags in the same Verifier turn, in the order reference, candidate, reference, candidate, reference, candidate. Each run uses `auto_bench.py --v0_file base.py --v1_file <implementation>`. Record all six v1 measurements and compare the median of the three reference samples with the median of the three candidate samples; do not compare the candidate to a stale baseline-only sample.
- Compute `improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100` without using rounded display values.
- `accepted` requires correctness PASS and median improvement of at least 5%.
- Any slower result or improvement below 5% is `no-improvement`.
- Environment failure gets zero retries and produces `Result: env-fail` with command, exit code, stderr summary, and remediation requirement.
- Profile the accepted reference and candidate in the same trace with `--profile-reference-file <last_accepted_kernel> --profile-mode forward`; summarize each emitted record-function scope separately. If the harness cannot produce distinguishable scopes, emit `env-fail` instead of combining both implementations' kernels.
- All device metrics use `summarize_trace.py`; calculate `device_ratio` only from the candidate or canonical-after-round `device_us_per_call`, never from a mixed trace total.
- Status updates go to `round_status_NNN.md` at start, after correctness, after each timing sample, and at end. Main does not poll.
- Verifier owns the current report/status files and `state/verifier_state.md`; it never edits `team-state.md` or `project.md`.

- [ ] **Step 4: Define stop recommendations**

The report recommends stop when any criterion is met:

1. `measurement-bound`: the canonical-after-round implementation has normalized device ratio below 5% and measured remaining host overhead is entirely harness-fixed. If the current candidate was rejected, use the paired accepted-reference data.
2. `diminishing-returns`: manifest already shows two consecutive matching streak events and this report would make the third.
3. `upbound-reached`: accepted cumulative performance enters the declared band.
4. `resource-exhausted`: this transition would exceed 30 overview entries, 40 total rounds, or 24 hours since `project_started_at`.
5. `user-intervention`: lead forwarded a user stop request.

Verifier sends the recommendation and evidence to both Designer and orchestrator. The orchestrator remains final authority.

- [ ] **Step 5: Run the guided metric pressure test**

Expected: `device_us_per_call = 20`, `device_ratio = 20%`, and all report fields are present.

- [ ] **Step 6: Commit**

```bash
git add skills/kernel-opt-loop/prompts/verifier.md
git commit -m "skills: add Verifier role contract"
```

---

## Task 9: Rewrite `SKILL.md` as the runtime-neutral orchestrator

**Files:**
- Modify: `skills/kernel-opt-loop/SKILL.md`

**Interfaces:**
- Loads exactly one runtime adapter, then applies the shared bootstrap and state-machine contracts.
- Main session is the only writer of `team-state.md`, project overview rows, and Git commits.

- [ ] **Step 1: Preserve valid trigger metadata**

Keep frontmatter with `name: kernel-opt-loop`. Rewrite the description to trigger on iterative operator/kernel optimization involving Triton and an auto_bench-style harness, without summarizing the full workflow.

- [ ] **Step 2: Add runtime detection and sequential fallback**

Define this selection order:

1. If Codex collaboration tools are available, read `adapters/codex.md`.
2. Else if Claude Code agent-team support is enabled, read `adapters/claude-code.md`.
3. Else execute Designer, Coder, and Verifier contracts sequentially in the main session, preserving the same file ownership and state transitions.

The fallback is required so missing experimental collaboration support does not make the skill unusable.

- [ ] **Step 3: Add the common bootstrap contract**

Include this exact template in `SKILL.md`:

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

State that every placeholder is resolved before dispatch and that the full role contract is never pasted into the bootstrap message.

- [ ] **Step 4: Implement Phase 0 orchestration**

Specify these ordered actions:

1. Resolve absolute paths for skill root, project root, `base.py`, `auto_bench.py`, interpreter, and device.
2. Initialize `rounds/`, `state/`, `log/`, `team-state.md`, and empty role state files.
3. Record `runtime`, `project_started_at`, and `phase: initializing` in the manifest.
4. Run Designer Phase 0 to produce `project.md`.
5. Ask the user only for environment/upbound fields that cannot be discovered locally.
6. Run Verifier Phase 0 to generate `baseline_adapter.py` and `report_000.md`.
7. Compute `measurement_fingerprint` as SHA-256 over `base.py`, `auto_bench.py`, and the canonicalized shape/dtype/warmup/repeat/profile settings recorded in `project.md`, then update the project base row and manifest:

```yaml
phase: ready
current_round: "000"
last_completed_round: "000"
last_accepted_round: "000"
last_accepted_kernel: baseline_adapter.py
last_completed_decision: null
last_completed_report: rounds/report_000.md
last_accepted_report: rounds/report_000.md
last_result: baseline
total_rounds: 0
measurement_fingerprint: <sha256>
```

8. If Phase 0 benchmarking fails because of the environment, preserve the evidence as `rounds/report_000_envfail_<UTC-timestamp>.md`, set `last_completed_report` to that path while leaving `last_completed_round` and every accepted pointer null, set `last_result: env-fail` and `phase: blocked`, commit the failure transition, and require remediation before rerunning Phase 0. Never overwrite the preserved failure report.
9. Otherwise commit Phase 0 artifacts, including manifest and all initialized state files.

- [ ] **Step 5: Implement Round N orchestration**

At a new round start, the lead derives `NNN = total_rounds + 1`, sets `current_round` and `phase: designing`, and bootstraps Designer. Then:

1. Designer writes decision.
2. On abort, skip Coder and Verifier and apply the abort transition.
3. Otherwise the lead validates the decision, sets `phase: coding`, and Coder writes the candidate from `last_accepted_kernel`.
4. The lead validates the candidate with the harness loader, sets `phase: verifying`, and Verifier writes one terminal report.
5. Lead validates artifact paths and result enum.
6. Lead appends exactly one project overview row.
7. Lead increments `total_rounds` exactly once and updates `last_completed_round`, `last_completed_decision`, `last_completed_report`, and `last_result` for every terminal result; an abort sets the completed round/decision/result and leaves `last_completed_report` null. Only `accepted` updates `last_accepted_round`, `last_accepted_kernel`, and `last_accepted_report`. A non-stopping, non-environment terminal transition returns to `phase: ready`.
8. Lead commits decision, candidate when present, report/status when present, project, manifest, and all changed role states.
9. Only after commit does the lead dispatch the next round.

- [ ] **Step 6: Implement stop-decision and resume semantics**

A Verifier stop recommendation is evidence, not a state transition. The lead asks Designer to review it: `measurement-bound` and `upbound-reached` are approved; `diminishing-returns` and `resource-exhausted` may be rejected only with a concrete next-round hypothesis expected to deliver at least 5%; `user-intervention` is unconditional. The lead remains final authority. Record the review in `designer_state.md`; never edit or replace the completed round decision. If work continues, Designer creates a new decision at the next unused round number.

After every terminal transition, the lead independently evaluates counter- and resource-based criteria. This is required for an abort path where Verifier did not run: reaching three consecutive failed attempts must still produce a `diminishing-returns` recommendation. An approved stop sets `phase: stopped`, `stop_reason`, `stop_timestamp`, `resume_eligible`, and `resume_constraints` atomically before the final commit. An environment failure instead sets `phase: blocked` and `stop_reason: env-fail`.

Use these resume rules:

| stop reason | eligibility | required change |
|---|---|---|
| measurement-bound | blocked | new shape or measurement regime |
| diminishing-returns | conditional | new hypothesis, new anti-pattern evidence, or skill upgrade |
| upbound-reached | conditional | explicit stretch goal |
| resource-exhausted | always | user acknowledges safety stop |
| user-intervention | conditional | user explicitly resumes |
| env-fail | conditional | environment remediation evidence |

At stop, snapshot `skill_version_at_stop`, `measurement_fingerprint_at_stop`, and the current anti-pattern Git revision into the manifest. On resume from `stopped` or `blocked`, recompute the measurement fingerprint, validate all constraints before spawning roles, then run a Designer sanity-check decision for the next unused round number. Never reuse or overwrite a completed decision/report.

For session interruption in `designing`, `coding`, or `verifying`, do not allocate a new round. Reopen `current_round`, validate all predecessor artifacts, and resume the missing role. A schema-valid decision or harness-loadable candidate is reused; an incomplete current-round artifact may be repaired in place because it was never committed as terminal. `total_rounds` remains unchanged until the round reaches a terminal result.

- [ ] **Step 7: Implement shutdown and knowledge lift**

At an approved stop:

1. Commit the final transition.
2. Ask runtime roles to shut down using the active adapter.
3. Read `designer_state.md` for generic failure patterns.
4. Present proposed anti-pattern additions to the user.
5. Modify skill-level `anti-patterns.md` only after explicit user approval and in a separate commit.

- [ ] **Step 8: Validate `SKILL.md` structure**

```bash
for section in "When to use" "Required inputs" "Runtime selection" "Agent bootstrap contract" "Phase 0" "Round N" "State transitions" "Stop criteria" "Resume" "Knowledge lift" "References"; do
  grep -qF "## $section" skills/kernel-opt-loop/SKILL.md || exit 1
done
! grep -Eq 'TeamCreate|TeamDelete|team_name[[:space:]]*=' skills/kernel-opt-loop/SKILL.md
```

Expected: exit 0.

- [ ] **Step 9: Commit**

```bash
git add skills/kernel-opt-loop/SKILL.md
git commit -m "skills: rewrite kernel-opt-loop as cross-runtime orchestrator"
```

---

## Task 10: Add a static contract checker and run role pressure tests

**Files:**
- Create: `skills/kernel-opt-loop/tests/check_contracts.sh`

**Interfaces:**
- Validates cross-file names, forbidden obsolete APIs, state fields, role ownership, and result enums.

- [ ] **Step 1: Write `check_contracts.sh`**

Use `set -euo pipefail`. The checker must assert:

- every final-structure file exists and is non-empty;
- frontmatter contains `name` and `description`;
- all six state-machine values (`baseline`, `accepted`, `no-improvement`, `accuracy-fail`, `abort`, `env-fail`) appear in the orchestrator, while the five report-producing values appear in the report template;
- `last_accepted_kernel`, `last_completed_report`, and `last_accepted_report` appear in the relevant role, template, and orchestrator contracts;
- all seven phase names and the deterministic `measurement_fingerprint` inputs appear in the state/orchestrator contracts;
- `profile_iterations`, `device_us_per_call`, and the unit formula appear in Verifier and report template;
- each role contract says it must not edit `team-state.md`;
- bootstrap includes absolute role-contract, adapter, project, input, and output paths;
- no skill file contains active invocation syntax for the retired Claude team APIs (`TeamCreate(`, `TeamDelete(`, or `team_name=`); adapters may name them only in compatibility warnings;
- no sync verification hardcodes a user home directory.

- [ ] **Step 2: Run all static tests**

```bash
bash skills/kernel-opt-loop/tests/check_contracts.sh
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_*.py' -v
```

Expected: both commands pass.

- [ ] **Step 3: Run three combined-pressure scenarios**

Run each scenario once through Claude Code and once through Codex where available:

1. Previous candidate is slower but has the highest round number: Designer and Coder must use `last_accepted_kernel`.
2. Profiler contains 50 calls: Verifier must divide before computing ratio.
3. Session resumes after `accuracy-fail`: orchestrator must allocate a new round number and preserve the old decision/report.

Capture outputs under `/tmp/kernel-opt-loop-contract-tests/<runtime>/<scenario>/`. A scenario passes only when the produced artifacts satisfy `check_contracts.sh`-equivalent assertions; do not accept a verbal claim.

- [ ] **Step 4: Fix any discovered loopholes and rerun**

Edit only the smallest relevant contract, then rerun Step 2 and the failing scenario until all pass.

- [ ] **Step 5: Commit**

```bash
git add skills/kernel-opt-loop/tests/check_contracts.sh skills/kernel-opt-loop
git commit -m "skills: add kernel-opt-loop contract validation"
```

---

## Task 11: Sync and verify both runtime installations

**Files:**
- Sync source: `skills/kernel-opt-loop/`
- Claude Code destination: `${HOME}/.claude/skills/kernel-opt-loop/`
- Codex destination: `${HOME}/.codex/skills/kernel-opt-loop/`

**Interfaces:**
- Produces identical runtime copies without hardcoded usernames.

- [ ] **Step 1: Sync to Claude Code and Codex**

```bash
rsync -av --delete skills/kernel-opt-loop/ "${HOME}/.claude/skills/kernel-opt-loop/"
rsync -av --delete skills/kernel-opt-loop/ "${HOME}/.codex/skills/kernel-opt-loop/"
```

`--delete` intentionally removes legacy `log-template.md` and stale files inside these two exact skill directories.

- [ ] **Step 2: Compare installed trees with the source**

```bash
diff -ru skills/kernel-opt-loop "${HOME}/.claude/skills/kernel-opt-loop"
diff -ru skills/kernel-opt-loop "${HOME}/.codex/skills/kernel-opt-loop"
```

Expected: no output and exit 0 for both commands.

- [ ] **Step 3: Verify runtime discovery in fresh sessions**

In fresh Claude Code and Codex sessions, ask each runtime to list or resolve `kernel-opt-loop` and report the skill description. Verify that invocation loads the new `SKILL.md` and selects the matching adapter.

- [ ] **Step 4: Record the sync result without committing home-directory files**

The repository already contains the source. Do not `git add` anything under `${HOME}/.claude` or `${HOME}/.codex`.

---

## Task 12: Run an isolated end-to-end Phase 0 and Round 1 smoke test

**Files:**
- Read only from implementation branch: `fused_moe/base.py`, `auto_bench.py`, `skills/kernel-opt-loop/`
- Produce only inside a temporary worktree: `smoke_fused_moe/` artifacts and smoke commits

**Interfaces:**
- Verifies real workflow mechanics without overwriting existing `fused_moe/triton_fused_moe_001.py` through `006.py`.

- [ ] **Step 1: Create a temporary smoke-test worktree and branch**

```bash
smoke_root=$(mktemp -d /tmp/kernel-opt-loop-smoke.XXXXXX)
smoke_branch="kernel-opt-loop-smoke-$(basename "$smoke_root")"
git worktree add -b "$smoke_branch" "$smoke_root" HEAD
mkdir -p "$smoke_root/smoke_fused_moe"
cp "$smoke_root/fused_moe/base.py" "$smoke_root/smoke_fused_moe/base.py"
```

Record the resolved `smoke_root`; do not use an unresolved environment variable in later cleanup.

- [ ] **Step 2: Verify accelerator prerequisites**

Use the intended interpreter:

```bash
<python> -c "import torch, torch_mlu, triton; print('env OK')"
```

If unavailable, mark only the hardware E2E as deferred. Tasks 1–11 must still pass; do not claim hardware acceptance.

- [ ] **Step 3: Run Phase 0 through the installed skill**

Invoke `kernel-opt-loop` for project root `$smoke_root/smoke_fused_moe`, explicitly authorizing its three-role workflow. Verify creation of:

```text
baseline_adapter.py
project.md
team-state.md
rounds/report_000.md
state/designer_state.md
state/coder_state.md
state/verifier_state.md
```

Verify `report_000.md` has normalized per-call device time and `team-state.md` points to `baseline_adapter.py`.

- [ ] **Step 4: Run one optimization round**

Verify creation of `decision_001.md`, then inspect its terminal path. For `accepted`, `no-improvement`, or `accuracy-fail`, also require `triton_smoke_fused_moe_001.py` and `report_001.md`. For `abort`, require neither candidate nor report. Accept any terminal result except `env-fail` for mechanics testing. Check that manifest behavior matches the result:

- `accepted`: canonical pointer advances.
- `no-improvement` or `accuracy-fail`: canonical pointer remains `baseline_adapter.py`.
- `abort`: no candidate or report is required and failed-attempt count becomes 1.

- [ ] **Step 5: Verify commits and source-worktree isolation**

Inside the smoke worktree, confirm Phase 0 and Round 1 commits exist. In the implementation worktree, run:

```bash
git status --short
```

Expected: no new `fused_moe/` files and no modifications caused by the smoke workflow.

- [ ] **Step 6: Shut down runtime roles and clean the disposable worktree**

Use the active adapter to stop live roles. Then, from outside the smoke worktree, rediscover and validate the exact worktree path before removal:

```bash
smoke_root=$(git worktree list --porcelain | awk '/^worktree \/tmp\/kernel-opt-loop-smoke\./ {print $2}')
test -n "$smoke_root"
test "$(printf '%s\n' "$smoke_root" | wc -l)" -eq 1
case "$smoke_root" in /tmp/kernel-opt-loop-smoke.*) ;; *) exit 1 ;; esac
smoke_branch=$(git -C "$smoke_root" branch --show-current)
case "$smoke_branch" in kernel-opt-loop-smoke-kernel-opt-loop-smoke.*) ;; *) exit 1 ;; esac
git worktree remove "$smoke_root"
git branch -D "$smoke_branch"
```

The branch is disposable and contains only smoke artifacts. Do not merge it.

---

## Final Verification

Run from the implementation worktree:

```bash
bash skills/kernel-opt-loop/tests/check_contracts.sh
python3 -m unittest discover -s skills/kernel-opt-loop/tests -p 'test_*.py' -v
git diff --check
git status --short
```

Expected:

- contract checker passes;
- helper tests pass;
- `git diff --check` emits no errors;
- status contains only intentional implementation changes, or is clean after commits;
- no existing `fused_moe/triton_fused_moe_*.py` file was modified by testing.

## Self-Review

- Runtime API coverage: Claude Code adapter uses post-2.1.178 teammate semantics; Codex adapter uses collaboration tools and has a sequential fallback.
- Bootstrap coverage: the common message is owned by `SKILL.md` and passes paths rather than duplicating role prompt contents.
- Baseline coverage: `baseline_adapter.py` makes the existing v0/v1 harness contract executable and Phase 0 produces `report_000.md` before Round 1.
- State coverage: every result maps to one manifest transition; only accepted results advance the canonical pointer.
- Resume coverage: manifest and role states are initialized and committed from Phase 0 onward.
- Measurement coverage: total profiler duration is divided by the declared number of forward calls before ratio calculation.
- Historical data coverage: anti-pattern extraction uses a pinned Git revision containing Entries 004–024.
- Safety coverage: E2E runs in a disposable worktree and cannot overwrite the existing fused_moe version files.
- Portability coverage: installation paths use `${HOME}` and verify both runtime copies without hardcoded usernames.
