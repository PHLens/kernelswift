# kernel-opt-loop v2 Continuous-Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the v1 kernel optimization contracts into a single-session,
autonomous, bounded v2 campaign that continues across terminal rounds, stops at
the declared global conditions, protects single-machine measurements, and keeps
role context compact and recoverable.

**Architecture:** Keep `team-state.md` and round artifacts as the durable source
of truth. Add one standard-library Python run-policy evaluator for deterministic
terminal counter, stop, and checkpoint outcomes; Orchestrator invokes it with a
JSON projection of the manifest rather than introducing a second state store.
Express role behavior, artifact ownership, and runtime lifecycle in the existing
Markdown contracts, templates, and adapters, then protect every new cross-file
invariant with Python `unittest` fixtures.

**Tech Stack:** Python 3 standard library, `unittest`, Markdown contract files,
Git, existing `validate_decision.py` and skill helpers.

---

## Preconditions

The v1 implementation must be present before starting this plan. It is currently
the `skill-restructure-plan-v2` commit line and supplies the files under
`skills/kernel-opt-loop/` that v2 modifies. After v1 merges to `dev`, rebase this
branch and verify the baseline before editing:

```bash
git fetch origin
git rebase origin/dev
test -f skills/kernel-opt-loop/SKILL.md
test -f skills/kernel-opt-loop/references/team-state-template.md
test -f skills/kernel-opt-loop/tests/test_contracts.py
python3 -m unittest discover -s skills/kernel-opt-loop/tests -v
```

Expected: the v1 contract suite passes. Do not copy v1 implementation files from
another branch manually, and do not implement any Future Work from the v2 spec.

## File Structure

| Path | Responsibility |
|---|---|
| `skills/kernel-opt-loop/scripts/evaluate_run_policy.py` | Pure, CLI-accessible evaluation of one terminal result or environment block from a JSON manifest projection. |
| `skills/kernel-opt-loop/tests/test_run_policy.py` | Unit tests for counters, stop precedence, checkpoint cadence, and counter-neutral blocks. |
| `skills/kernel-opt-loop/references/team-state-template.md` | v2 workflow policy, run epoch, checkpoint, Git branch, and workflow-status fields. |
| `skills/kernel-opt-loop/references/project-template.md` | Comparable optional target and dedicated run-branch project facts. |
| `skills/kernel-opt-loop/references/report-template.md` | Screening-tier evidence, `screened-out`, profiler applicability, and global-stop observations. |
| `skills/kernel-opt-loop/references/role-context-template.md` | Compact durable role context and read-hash ledger materialized for Designer, Coder, and Verifier. |
| `skills/kernel-opt-loop/references/decision-template.md` | Required decision `change_family` field and examples. |
| `skills/kernel-opt-loop/scripts/validate_decision.py` | Deterministic validation of nonempty, normalized `change_family`. |
| `skills/kernel-opt-loop/tests/fixtures/decisions/*.md` | Valid decision fixtures updated with `change_family`. |
| `skills/kernel-opt-loop/tests/test_validate_decision.py` | Positive and negative `change_family` validation fixtures. |
| `skills/kernel-opt-loop/prompts/designer.md` | Rolling hypothesis backlog, context updates, and change-family switching rules. |
| `skills/kernel-opt-loop/prompts/coder.md` | Current-regime compile-smoke evidence and bounded repair handoff. |
| `skills/kernel-opt-loop/prompts/verifier.md` | Screening versus authoritative verification, scoped profiling, watchdog, and exclusive measurement behavior. |
| `skills/kernel-opt-loop/adapters/codex.md` | Session-persistence capability declaration and continuation/rehydrate behavior for Codex. |
| `skills/kernel-opt-loop/adapters/claude-code.md` | Equivalent session-persistence declaration and lifecycle behavior for Claude Code. |
| `skills/kernel-opt-loop/SKILL.md` | Runtime-neutral v2 controller, Git ledger rules, policy freeze, resume, and routing. |
| `skills/kernel-opt-loop/tests/test_contracts.py` | Cross-file v2 contract tests and absence checks for Future Work. |
| `docs/superpowers/specs/2026-08-14-kernel-opt-loop-v2-continuous-run-design.md` | Update status to `Approved` only after the user approves this plan; do not change Future Work scope. |

No source is added for a daemon, kernel wiki, per-agent ACL, token accounting,
new target profile, or parallel candidate runner.

## Acceptance Criteria

- AC-1: A non-stopping terminal result advances the run to the next round; only
  the twentieth terminal round, third valid `no-improvement`, target, or user
  stop produces `workflow_status: stopped`.
- AC-2: `screened-out` is a budgeted terminal result that does not change either
  streak and can only follow correctness plus two clearly regressive short pairs.
- AC-3: Phase 0 remains outside the 20-round budget, and an environment block
  leaves round and streak counters unchanged.
- AC-4: A target is comparable only under the baseline measurement fingerprint;
  target amendments are append-only at a safe boundary, while other policy values
  are frozen per run epoch.
- AC-5: Designer, Coder, and Verifier contract changes preserve one candidate,
  one immutable decision, one same-round Verifier repair, and a measurement-
  exclusive shared machine.
- AC-6: A supported runtime reuses idle role identities with deltas; an
  unsupported runtime rehydrates only from compact role state and durable
  artifacts, with identical workflow semantics.
- AC-7: Each terminal/block/final commit tracks evidence artifacts and excludes
  raw profiler logs, command output, caches, session IDs, and secrets.
- AC-8: All existing v1 tests and the new v2 policy/contract tests pass without
  MLU hardware.

## Path Boundaries

### Upper Bound

Implement every v2 behavior in the approved spec through the listed skill files
and tests. The evaluator may read only the supplied JSON state projection and
write JSON to stdout; it must not parse, rewrite, or commit Markdown manifests.

### Lower Bound

At minimum, implement the global controller semantics, v2 manifest/template
fields, two-stage verifier contract, compact role state contract, runtime
continuation declaration, run-branch evidence ledger, and all Section 9 v2
fixtures.

### Allowed Choices

- Use Python standard-library modules only for the policy evaluator.
- Keep v1 Markdown style and `unittest` conventions.
- Add tests to existing files when they exercise an existing contract; create a
  dedicated test file only for the pure policy evaluator.
- Use `subprocess.run(..., check=True, capture_output=True, text=True)` in tests
  only when exercising the evaluator CLI.

### Prohibited Choices

- Do not add PyYAML, a database, a background scheduler, network calls, or a
  second persisted state file.
- Do not modify `base.py`, harness semantics, target profile capabilities, or
  v1 measurement helper output formats.
- Do not persist agent IDs, raw conversations, token estimates, raw traces, or
  secrets in tracked artifacts.
- Do not implement KernelWiki, automatic backend selection, deep profiler
  analysis, or concurrent candidate measurement.

## Dependencies and Sequence

1. Deterministic run policy before contracts that invoke it.
2. Manifest and context templates before role and Orchestrator contract changes.
3. Decision-family validation before Designer backlog routing can rely on it.
4. Role contracts before the runtime-neutral controller and adapters.
5. Cross-file tests and full-suite verification last.

## Implementation Tasks

### Task 1: Add the Deterministic Run-Policy Evaluator

**Files:**
- Create: `skills/kernel-opt-loop/scripts/evaluate_run_policy.py`
- Create: `skills/kernel-opt-loop/tests/test_run_policy.py`

- [ ] **Step 1: Write failing unit tests for terminal accounting and stop precedence**

Create `test_run_policy.py` with direct function tests. Import the script by
adding its `scripts/` directory to `sys.path`, matching `test_helpers.py`.

```python
from evaluate_run_policy import RunPolicyError, evaluate_block, evaluate_terminal


def state(**overrides):
    value = {
        "total_rounds": 0,
        "performance_miss_streak": 0,
        "failed_attempt_streak": 0,
        "last_checkpoint_round": None,
        "max_rounds": 20,
        "valid_no_improvement_limit": 3,
    }
    value.update(overrides)
    return value


class RunPolicyTests(unittest.TestCase):
    def test_third_valid_no_improvement_stops_without_checkpoint(self):
        result = evaluate_terminal(
            state(total_rounds=2, performance_miss_streak=2),
            "no-improvement",
        )
        self.assertEqual(result["total_rounds"], 3)
        self.assertEqual(result["performance_miss_streak"], 3)
        self.assertEqual(result["workflow_status"], "stopped")
        self.assertEqual(result["stop_reason"], "valid-no-improvement-limit")
        self.assertFalse(result["dispatch_next_round"])
        self.assertFalse(result["emit_checkpoint"])
```

Add separate tests for: `accepted` resetting both streaks; `screened-out` only
incrementing `total_rounds`; a twentieth terminal result stopping with
`round-budget-exhausted`; target and user-stop precedence; a running third round
emitting one checkpoint; duplicate checkpoint suppression; `evaluate_block`
preserving all counters; unknown result and invalid numeric state raising
`RunPolicyError`; and CLI JSON output/error status.

- [ ] **Step 2: Run the new test module to verify it fails**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_run_policy.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate_run_policy'`.

- [ ] **Step 3: Implement the pure policy module and CLI**

Create `evaluate_run_policy.py` with this public surface:

```python
#!/usr/bin/env python3
"""Evaluate v2 terminal-run policy from a JSON manifest projection."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping, Sequence


TERMINAL_RESULTS = frozenset({
    "accepted", "no-improvement", "screened-out", "design-rejected",
    "candidate-failed", "aborted",
})
FAILED_RESULTS = frozenset({"design-rejected", "candidate-failed", "aborted"})


class RunPolicyError(ValueError):
    pass


def _positive_int(state: Mapping[str, Any], name: str) -> int:
    value = state.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunPolicyError(f"{name} must be a non-negative integer")
    return value


def evaluate_terminal(
    state: Mapping[str, Any],
    result: str,
    *,
    target_reached: bool = False,
    user_stop_requested: bool = False,
) -> dict[str, Any]:
    if result not in TERMINAL_RESULTS:
        raise RunPolicyError(f"unknown terminal result: {result}")
    total_rounds = _positive_int(state, "total_rounds") + 1
    performance_miss_streak = _positive_int(state, "performance_miss_streak")
    failed_attempt_streak = _positive_int(state, "failed_attempt_streak")
    max_rounds = _positive_int(state, "max_rounds")
    miss_limit = _positive_int(state, "valid_no_improvement_limit")
    if max_rounds <= 0 or miss_limit <= 0:
        raise RunPolicyError("max_rounds and valid_no_improvement_limit must be positive")

    if result == "accepted":
        performance_miss_streak = 0
        failed_attempt_streak = 0
    elif result == "no-improvement":
        performance_miss_streak += 1
    elif result in FAILED_RESULTS:
        failed_attempt_streak += 1

    stop_reason = None
    if user_stop_requested:
        stop_reason = "user-intervention"
    elif target_reached:
        stop_reason = "target-reached"
    elif performance_miss_streak >= miss_limit:
        stop_reason = "valid-no-improvement-limit"
    elif total_rounds >= max_rounds:
        stop_reason = "round-budget-exhausted"

    running = stop_reason is None
    last_checkpoint = state.get("last_checkpoint_round")
    emit_checkpoint = (
        running and total_rounds % 3 == 0 and last_checkpoint != total_rounds
    )
    return {
        "total_rounds": total_rounds,
        "performance_miss_streak": performance_miss_streak,
        "failed_attempt_streak": failed_attempt_streak,
        "workflow_status": "running" if running else "stopped",
        "phase": "ready" if running else "stopped",
        "stop_reason": stop_reason,
        "dispatch_next_round": running,
        "emit_checkpoint": emit_checkpoint,
        "last_checkpoint_round": total_rounds if emit_checkpoint else last_checkpoint,
    }


def evaluate_block(state: Mapping[str, Any], incident: str) -> dict[str, Any]:
    if not isinstance(incident, str) or not incident:
        raise RunPolicyError("incident must be a non-empty string")
    return {
        "total_rounds": _positive_int(state, "total_rounds"),
        "performance_miss_streak": _positive_int(state, "performance_miss_streak"),
        "failed_attempt_streak": _positive_int(state, "failed_attempt_streak"),
        "workflow_status": "blocked",
        "phase": "blocked",
        "blocked_incident": incident,
        "dispatch_next_round": False,
        "emit_checkpoint": False,
    }
```

Add `main(argv: Sequence[str] | None = None) -> int` with mutually exclusive
`--result` and `--block-incident`, required `--state-json`, optional
`--target-reached` and `--user-stop-requested`, sorted JSON stdout, and a
single-line stderr error with exit status `2`. Its parser and dispatch should be
implemented exactly as follows after validating that decoded state is an object:

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-json", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result", choices=sorted(TERMINAL_RESULTS))
    group.add_argument("--block-incident")
    parser.add_argument("--target-reached", action="store_true")
    parser.add_argument("--user-stop-requested", action="store_true")
    args = parser.parse_args(argv)
    try:
        state = json.loads(args.state_json)
        if not isinstance(state, dict):
            raise RunPolicyError("state JSON must be an object")
        outcome = (
            evaluate_block(state, args.block_incident)
            if args.block_incident is not None
            else evaluate_terminal(
                state, args.result,
                target_reached=args.target_reached,
                user_stop_requested=args.user_stop_requested,
            )
        )
    except (json.JSONDecodeError, RunPolicyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(outcome, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Do not parse or write `team-state.md` in this script.
Mark the new script executable with `chmod +x
skills/kernel-opt-loop/scripts/evaluate_run_policy.py`.

- [ ] **Step 4: Run the policy tests and existing helper tests**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_run_policy.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_helpers.py -v
```

Expected: PASS. The helper tests are unchanged and confirm no regression to
baseline adapter or trace normalization.

- [ ] **Step 5: Commit the policy seam**

```bash
git add skills/kernel-opt-loop/scripts/evaluate_run_policy.py skills/kernel-opt-loop/tests/test_run_policy.py
git commit -m "skills: add v2 run policy evaluator"
```

### Task 2: Extend Durable State, Project, Report, and Role-Context Templates

**Files:**
- Create: `skills/kernel-opt-loop/references/role-context-template.md`
- Modify: `skills/kernel-opt-loop/references/team-state-template.md`
- Modify: `skills/kernel-opt-loop/references/project-template.md`
- Modify: `skills/kernel-opt-loop/references/report-template.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

- [ ] **Step 1: Add failing template assertions**

Extend `DurableContractTests` with one test that requires the v2 frontmatter
fields and one that requires the new role-context template:

```python
def test_team_state_contains_v2_workflow_policy(self):
    template = read_reference("team-state-template.md")
    for field in (
        "workflow_status: running", "run_epoch: 1", "max_rounds: 20",
        "valid_no_improvement_limit: 3", "adoption_threshold_pct: 5",
        "last_checkpoint_round: null", "base_branch: null", "run_branch: null",
    ):
        self.assertIn(field, template)

def test_role_context_template_has_rehydrate_fields(self):
    template = read_reference("role-context-template.md")
    for field in (
        "role_contract_sha256", "context_epoch", "last_completed_round",
        "recent_three_round_evidence", "open_hypotheses", "artifact_read_hashes",
    ):
        self.assertIn(field, template)
```

Also require `screened-out`, `verification_tier`, `screening_pairs`,
`target_mode`, and `target_measurement_fingerprint` in the relevant templates.

- [ ] **Step 2: Run the contract test to verify it fails**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: FAIL because v2 fields and `role-context-template.md` do not exist.

- [ ] **Step 3: Materialize the v2 template fields**

Add these initial frontmatter fields after v1 identity fields in
`team-state-template.md`:

```yaml
workflow_status: running
run_epoch: 1
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
target_mode: null
target_value: null
target_measurement_fingerprint: null
target_source: null
last_checkpoint_round: null
base_branch: null
base_commit: null
run_branch: null
measurement_exclusive: false
```

Update the allowed workflow description to distinguish `workflow_status` from
`phase`, list `screened-out` as a terminal result, and add an append-only
`## Policy Revisions` table with columns `Timestamp`, `Run epoch`, `Field`,
`Old value`, `New value`, `Reason`, and `Commit`.

Replace the old free-form `## Upbound` in `project-template.md` with
`## Optional Target`, documenting only `absolute_latency_ms` and
`speedup_vs_baseline`, the `wall_time_ms` metric, `source: user`, and the
baseline measurement-fingerprint requirement. Add `## Git Run Identity` with
the three run-branch values mirrored from team state.

Add this role-context template, leaving no agent ID or conversation content:

```markdown
# <Role> Context State

- role_contract_sha256: `<sha256>`
- context_epoch: `<integer>`
- last_completed_round: `<NNN-or-null>`
- accepted_kernel: `<relative path-or-null>`
- accepted_report: `<relative path-or-null>`

## Current Bottleneck

- `<Verifier-backed fact only>`

## Recent Three-round Evidence

- `<round, result, evidence pointer, and change family>`

## Open Hypotheses or Checks

- `<bounded next work item>`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `<relative path>` | `<sha256>` | `<NNN>` |
```

Extend `report-template.md` as follows:

- include `screened-out` in the Result enum;
- add `verification_tier: baseline | screening | authoritative` in Identity;
- add `## Screening Evidence` with two ordered reference/candidate short-pair
  rows and a `screened-out` rule of both pairs at least 10% slower;
- make profiler evidence `required | not-run: screened-out | not-run: not-needed`;
- replace old stop recommendations with `continue | target-reached |
  valid-no-improvement-limit | round-budget-exhausted | user-intervention`;
- state that only authoritative timing can yield `accepted` or
  `no-improvement`.

- [ ] **Step 4: Run the template contract suite**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: PASS for all existing durable-template assertions plus the new v2
fields. Confirm every Markdown file has balanced fences:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -k markdown_fences_close_and_validator_is_executable -v
```

- [ ] **Step 5: Commit durable v2 templates**

```bash
git add skills/kernel-opt-loop/references skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: add v2 durable run templates"
```

### Task 3: Validate Decision Change Families for Backlog Routing

**Files:**
- Modify: `skills/kernel-opt-loop/references/decision-template.md`
- Modify: `skills/kernel-opt-loop/scripts/validate_decision.py`
- Modify: `skills/kernel-opt-loop/tests/fixtures/decisions/kernel-valid.md`
- Modify: `skills/kernel-opt-loop/tests/fixtures/decisions/host-valid.md`
- Modify: `skills/kernel-opt-loop/tests/fixtures/decisions/mixed-valid.md`
- Modify: `skills/kernel-opt-loop/tests/test_validate_decision.py`

- [ ] **Step 1: Write failing tests for `change_family`**

Add these tests to `ValidateDecisionTests`:

```python
def test_change_family_is_normalized(self):
    result = validate_decision(FIXTURES / "kernel-valid.md")
    self.assertEqual(result["metadata"]["change_family"], "kernel-fusion")

def test_change_family_is_required_and_slug_shaped(self):
    text = (FIXTURES / "kernel-valid.md").read_text(encoding="utf-8")
    self.assertValidationError(
        text.replace(',"change_family":"kernel-fusion"', "", 1),
        "metadata-field-required",
    )
    self.assertValidationError(
        text.replace("kernel-fusion", "Kernel fusion", 1),
        "metadata-change-family-invalid",
    )
```

Update the complete-template-example assertion so all examples must include the
new metadata field.

- [ ] **Step 2: Run the validator tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_decision.py -v
```

Expected: FAIL because the field is not required or present in v1 fixtures.

- [ ] **Step 3: Add the field to the template, fixtures, and validator**

Add `change_family` to `METADATA_FIELDS` and require a lower-case slug using
this exact check immediately after other metadata enum validation:

```python
change_family = metadata["change_family"]
if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", change_family):
    raise DecisionValidationError(
        "metadata-change-family-invalid",
        "change_family must be a lower-case hyphenated slug",
        sections["Metadata"].line,
    )
```

Document `change_family` in the metadata table as the policy-visible mechanism
class, such as `kernel-fusion`, `allocation-reuse`, or `launch-overhead`. Add it
to every complete example and v1 fixture. Use `kernel-fusion` for the kernel
fixture, `allocation-reuse` for host, and `mixed-routing-fusion` for mixed.
Do not make a fixed global enum: each operator needs its own truthful family
name, but the normalized spelling keeps comparison deterministic.

- [ ] **Step 4: Run the decision validator suite**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_decision.py -v
```

Expected: PASS, including all three complete template examples.

- [ ] **Step 5: Commit change-family validation**

```bash
git add skills/kernel-opt-loop/references/decision-template.md skills/kernel-opt-loop/scripts/validate_decision.py skills/kernel-opt-loop/tests/fixtures/decisions skills/kernel-opt-loop/tests/test_validate_decision.py
git commit -m "skills: validate v2 decision change families"
```

### Task 4: Update Designer, Coder, and Verifier Contracts

**Files:**
- Modify: `skills/kernel-opt-loop/prompts/designer.md`
- Modify: `skills/kernel-opt-loop/prompts/coder.md`
- Modify: `skills/kernel-opt-loop/prompts/verifier.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

- [ ] **Step 1: Add failing role-contract assertions**

Add static contract checks for these exact requirements:

```python
def test_v2_role_contracts_define_context_and_measurement_boundaries(self):
    designer = (PROMPTS / "designer.md").read_text(encoding="utf-8")
    coder = (PROMPTS / "coder.md").read_text(encoding="utf-8")
    verifier = (PROMPTS / "verifier.md").read_text(encoding="utf-8")
    for text in ("three to five", "change_family", "different change family", "role-context-template.md"):
        self.assertIn(text, designer)
    for text in ("warm-up / compile smoke", "at most twice", "attempt ledger"):
        self.assertIn(text, coder)
    for text in ("screened-out", "two short interleaved", "10%", "measurement-exclusive", "liveness watchdog"):
        self.assertIn(text, verifier)
```

Add negative assertions that Designer/Coder do not run local work while Verifier
owns the shared measurement phase, and that Verifier does not promote a screen
result to `accepted` or `no-improvement`.

- [ ] **Step 2: Run the role-contract tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: FAIL on missing v2 strings and the old mandatory Level 1 wording.

- [ ] **Step 3: Implement the v2 role behavior**

In `designer.md`:

- materialize and update the role-context template rather than an unstructured
  state note;
- create a ranked three-to-five hypothesis backlog after Phase 0;
- record bottleneck, expected gain, risk, evidence, validation cost, and
  `change_family` for each item;
- after a valid `no-improvement`, select a different `change_family` unless a
  new Verifier-backed observation names why the same family can clear 5%; and
- on a continuation, read only the context state, changed artifacts, and current
  inputs unless a documented invalidation condition applies.

In `coder.md`, preserve the existing two local non-semantic repairs but make the
gate explicit: `ast.parse`, real harness loader, and one current-regime
warm-up/compile-smoke execution must succeed before `candidate-ready`. Record
each command, exit status, and candidate hash in the attempt ledger. A Verifier
repair remains exactly one and must repeat that gate.

In `verifier.md`:

- enforce correctness, then two short interleaved pairs;
- emit `screened-out` only when both pairs are at least 10% slower than the
  accepted reference and include its report evidence;
- send every other correct candidate through existing authoritative timing;
- profile baseline and accepted candidates, profile boundary/insufficient-
  evidence cases, and skip profiler for `screened-out`;
- treat a watchdog derived from baseline-equivalent elapsed time as an
  environment incident, not a performance result; and
- state that only Verifier may issue local commands during `verifying` or
  `measuring`; other roles remain idle until durable completion.

Remove v1 statements that require Level 1 profiling after every correct
candidate or let Designer override the third valid miss.

- [ ] **Step 4: Run role and Markdown contract tests**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: PASS. Verify every existing ownership assertion still passes; no v2
wording may authorize a role to edit `team-state.md` or canonical pointers.

- [ ] **Step 5: Commit role-contract changes**

```bash
git add skills/kernel-opt-loop/prompts skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: add v2 role measurement contracts"
```

### Task 5: Rewrite the Runtime-Neutral Orchestrator Contract

**Files:**
- Modify: `skills/kernel-opt-loop/SKILL.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

- [ ] **Step 1: Add failing orchestrator contract tests**

Extend `OrchestratorContractTests` to require all v2 sections and remove stale
v1 stop behavior:

```python
for heading in (
    "## Continuous run controller", "## Global termination policy",
    "## Measurement-exclusive phases", "## Run epochs and recovery",
    "## Git evidence ledger",
):
    self.assertIn(heading, self.skill)

for text in (
    "evaluate_run_policy.py", "max_rounds: 20",
    "valid_no_improvement_limit: 3", "screened-out",
    "round_result is not workflow termination", "last_checkpoint_round",
    "kernel-opt/<operator>-<run-epoch-or-timestamp>",
):
    self.assertIn(text, self.skill)

self.assertNotIn("Designer may reject another non-user stop", self.skill)
self.assertNotIn("normalized device ratio is below 5%", self.skill)
```

Add a cross-file test that all references to terminal results use the exact six
v2 values and that `evaluate_run_policy.py` is a nonempty executable file.

- [ ] **Step 2: Run orchestrator tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: FAIL on missing v2 sections and stale v1 stop text.

- [ ] **Step 3: Implement the controller, lifecycle, and ledger contract**

Rewrite the v1 Round/Stop/Resume portions of `SKILL.md` with these exact rules:

```text
terminal artifact gate -> terminal commit -> evaluate_run_policy.py
  workflow_status=running -> optional checkpoint -> continue idle Designer
  workflow_status=stopped -> final summary commit -> end_workflow
  workflow_status=blocked -> incident commit -> blocking report -> end live run
```

Require Orchestrator to invoke the helper using a JSON object made from the
manifest fields, for example:

```bash
python3 <skill-root>/scripts/evaluate_run_policy.py \
  --state-json '{"total_rounds":2,"performance_miss_streak":2,"failed_attempt_streak":0,"last_checkpoint_round":null,"max_rounds":20,"valid_no_improvement_limit":3}' \
  --result no-improvement
```

State that the JSON is an evaluator input projection, not persisted state. Apply
the returned fields atomically to `team-state.md` and transition log.

Add all of the following contract changes:

- Phase 0 records a clean Git base and creates the dedicated run branch unless
  the user explicitly authorizes an existing dedicated branch; reject automatic
  execution on `main`, `master`, or `dev`.
- Phase 0 materializes three role-context files from the new template.
- A terminal round increments exactly once and uses the v2 accounting table.
- Checkpoints at 3/6/9 are derived messages, never pause the run, and never
  create a checkpoint artifact.
- The optional target has exactly the v2 comparable modes and can be appended at
  a safe terminal boundary; all other policy fields are frozen per epoch.
- User stop waits for the active command boundary unless immediate interruption
  is explicitly requested.
- Only `recover` resumes a block/uncommitted safe step. A stopped run requires a
  new epoch with the specified reason before counters reset.
- Track baseline adapter, project/team state, decisions, results, reports,
  candidate source when present, role context, incidents, and final summary;
  ignore raw logs, command output, caches, sessions, and secrets.
- Keep KernelWiki, ACLs, token telemetry, multi-target lowering, deep profiler
  analysis, daemon scheduling, and parallel search out of the implementation.

Keep all runtime-specific tool names out of `SKILL.md`.

- [ ] **Step 4: Run the full contract and policy suites**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_run_policy.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: PASS. Ensure `test_runtime_syntax_is_adapter_local_and_future_scope_is_absent`
still rejects runtime-specific tool syntax in neutral files.

- [ ] **Step 5: Commit the controller contract**

```bash
git add skills/kernel-opt-loop/SKILL.md skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: add v2 continuous run controller"
```

### Task 6: Declare Runtime Persistence and Rehydrate Semantics

**Files:**
- Modify: `skills/kernel-opt-loop/adapters/codex.md`
- Modify: `skills/kernel-opt-loop/adapters/claude-code.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

- [ ] **Step 1: Add failing adapter capability assertions**

Add assertions that each adapter declares exactly one supported behavior:

```python
for adapter_name in ("codex.md", "claude-code.md"):
    adapter = read_adapter(adapter_name)
    self.assertIn("persistent_role_session:", adapter)
    self.assertIn("effective_context_mode:", adapter)
    self.assertIn("role-context-template.md", adapter)
    self.assertIn("cold rehydrate", adapter)
```

For Codex also retain the existing `spawn_agent` / `followup_task` assertions.
For Claude retain agent-team / `SendMessage` assertions. Add a negative check
that neither adapter promises a daemon or cross-session autonomous continuation.

- [ ] **Step 2: Run adapter tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: FAIL on the missing v2 capability declarations.

- [ ] **Step 3: Implement adapter-local lifecycle declarations**

Add this exact capability block near the top of both adapters, adapting only
the value for the actual runtime capability:

```yaml
runtime_capabilities:
  persistent_role_session: true
  effective_context_mode: continuation
  autonomous_scope: one-live-orchestrator-session
```

For Codex, specify that idle roles are reactivated with `followup_task`; for
Claude, specify `SendMessage`. Both adapters must require one compact bootstrap
delta after the first full bootstrap, and a cold rehydrate when Orchestrator
reports identity loss, contract/profile/fingerprint/policy change, canonical
pointer mismatch, or failed three-round reconciliation.

Document sequential fallback as:

```yaml
persistent_role_session: false
effective_context_mode: rehydrate
```

It must recreate only from role context plus changed artifacts and preserve the
same terminal routing. Do not add per-agent tool/skill enforcement claims.

- [ ] **Step 4: Run adapter and cross-file tests**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: PASS, including the existing portability rule that active runtime
syntax appears only in adapter files.

- [ ] **Step 5: Commit adapter lifecycle changes**

```bash
git add skills/kernel-opt-loop/adapters skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: define v2 role continuation semantics"
```

### Task 7: Perform the Cross-File v2 Acceptance Pass

**Files:**
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`
- Modify: `docs/superpowers/specs/2026-08-14-kernel-opt-loop-v2-continuous-run-design.md`

- [ ] **Step 1: Add final failing acceptance tests**

Add one cross-file test per v2 acceptance concern that is not already covered:

```python
def test_v2_contracts_share_terminal_and_future_scope_boundaries(self):
    report = read_reference("report-template.md")
    verifier = (PROMPTS / "verifier.md").read_text(encoding="utf-8")
    orchestrator = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    for result in (
        "accepted", "no-improvement", "screened-out", "design-rejected",
        "candidate-failed", "aborted",
    ):
        self.assertIn(result, report)
        self.assertIn(result, verifier)
        self.assertIn(result, orchestrator)
    for future_only in ("KernelWiki API", "token-accounting telemetry", "daemon"):
        self.assertNotIn(future_only, "\n".join(
            path.read_text(encoding="utf-8")
            for path in (SKILL_ROOT / "references").glob("*.md")
        ))
```

Also add a test that raw trace extensions and `log/` appear in the Git-ignore
contract rather than the tracked evidence list, and that every required Markdown
file including `role-context-template.md` has balanced fences.

- [ ] **Step 2: Run the final test to verify it fails before its assertions are implemented**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: FAIL only on newly added v2 cross-file assertions.

- [ ] **Step 3: Reconcile contracts and approve the v2 specification**

Resolve every failing assertion by correcting the smallest owning contract or
template. Do not weaken tests merely to accept contradictory wording. When all
acceptance requirements are represented and the user approves the written plan,
change only this spec metadata:

```markdown
**Status**: Approved
```

Do not edit its Future Work sections or add implementation tasks for them.

- [ ] **Step 4: Run the complete v1 + v2 suite and document checks**

Run:

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -v
git diff --check
```

Expected: every test passes and `git diff --check` prints no whitespace errors.

Run this explicit static guard as well:

```bash
rg -n "KernelWiki API|deterministic lowering implementation|deep-profiler implementation" skills/kernel-opt-loop
```

Expected: no matches in the implemented skill tree.

- [ ] **Step 5: Commit the v2 acceptance pass**

```bash
git add skills/kernel-opt-loop/tests/test_contracts.py docs/superpowers/specs/2026-08-14-kernel-opt-loop-v2-continuous-run-design.md
git commit -m "test: cover kernel opt loop v2 contracts"
```

## Final Verification

- [ ] Run the full contract suite from a clean worktree:

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -v
```

Expected: all v1 and v2 tests pass.

- [ ] Review the implementation boundary:

```bash
git diff origin/dev...HEAD --check
git diff --name-only origin/dev...HEAD
```

Expected: only the v2 spec, plan, kernel-opt-loop contract files, evaluator, and
their tests change. No KernelWiki, daemon, ACL, token ledger, target-profile, or
raw-log implementation files are present.

- [ ] Inspect commit sequence:

```bash
git log --oneline origin/dev..HEAD
```

Expected: small commits matching the seven tasks above and no unrelated churn.
