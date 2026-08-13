# kernel-opt-loop Skill Restructure Design

**Date**: 2026-08-13
**Status**: Approved (design phase), pending spec review → implementation plan
**Scope**: Architectural restructure of `~/.claude/skills/kernel-opt-loop/`

## 1. Motivation

Current `kernel-opt-loop` skill couples decision, implementation, and verification into a single linear workflow run by the main Claude session. Evidence from two real projects shows structural problems:

- **fused_moe** (6 rounds) — workflow tractable but main session context bloat across rounds.
- **groupedtopk** (24 rounds, 18 marked 失败 = 75% failure rate) — stop criterion ("5 rounds") invalidated by reality; repeated failure patterns (winner tree, sort-32+sort-64, `tl.gather` compact, cumsum collect) recurred across entries with no capture mechanism; per-role reasoning had no home and was lost between sessions.

Root cause: the skill treats kernel optimization as one monolithic task. It is actually three distinct roles — **Designer** (path analysis & decision), **Coder** (kernel implementation), **Verifier** (runtime feedback) — that should be embodied as isolated subagents with explicit file contracts, plus an orchestrator (main session) handling round boundaries, commits, and user decisions.

## 2. Non-goals

- **Knowledge base integration** in v1. The skill will leave a Designer-side hook ("consult `references/anti-patterns.md` if present") but no external KB lookup. KB integration is a follow-up.
- **Changing the operator-level workflow semantics** (auto_bench harness, profiler JSON analysis, 5% improvement threshold). The methodology content moves into subagent prompts; the rules themselves are preserved.
- **Per-project state outside the kernel-opt domain**. This design is specific to kernel optimization; other skill domains are out of scope.

## 3. Architecture

### 3.1 Components

- **Main skill** `~/.claude/skills/kernel-opt-loop/SKILL.md` — becomes an **orchestrator guide**. Defines phases (Phase 0 setup, Phase N round, stop), how to spawn the team, round-boundary actions (stop/continue/commit), and user-decision points. No longer contains "how to analyze a bottleneck" or "how to write a Triton kernel" — that content moves to subagent definitions.
- **Three subagent definitions** under `~/.claude/skills/kernel-opt-loop/agents/`:
  - `kernel-designer.md` — uses `Plan`-style profile. Reads `base.py` + previous `report_NNN.md` + `references/anti-patterns.md`. Produces `decision_NNN.md`.
  - `kernel-coder.md` — uses `developer`-style profile. Reads `decision_NNN.md` + `references/invariants.md`. Produces `triton_<op>_<NNN>.py`.
  - `kernel-verifier.md` — uses `qa`-style profile. Reads `triton_<op>_<NNN>.py` + `project.md` measurement regime. Runs `auto_bench.py` + profiler. Produces `report_NNN.md`, sends `shutdown_request` signal when stop criteria hit.
- **References** under `~/.claude/skills/kernel-opt-loop/references/` (see §7.1).

### 3.2 Communication architecture — hybrid (team-internal + main session boundary)

- **Phase 0** — main session uses `TeamCreate(team_name: "<op>-opt")` once. Spawns Designer, Coder, Verifier via `Agent(team_name=..., name="designer"/"coder"/"verifier", subagent_type=...)`. The three roles stay alive across rounds (idle between turns, wake on `SendMessage`).
- **Inside a round** — roles communicate P2P via `SendMessage`:
  - Designer → Coder: sends `decision_NNN.md` path + 1-paragraph briefing.
  - Coder → Verifier: sends `triton_<op>_<NNN>.py` path + AST-filter/invariant notes.
  - Verifier → Designer: sends `report_NNN.md` path + runtime numbers + optional `shutdown_request` (CC main session).
- **Round boundary** — main session intervenes. Two paths:
  - **Normal path** (Designer produced a kernel): Verifier's `report_NNN.md` lands → main session reads the runtime numbers. If Verifier sent `shutdown_request`: wait for Designer's `shutdown_response`. If `approve=true`, main session does final commit + `TeamDelete`. If `approve=false`, main session confirms continuation, team persists. If no shutdown signal: main session appends one row to `project.md` overview table + commits the round (`triton_<op>_<NNN>.py` + `rounds/decision_NNN.md` + `rounds/report_NNN.md` + `project.md`) + signals Designer to start next round.
  - **Abort path** (Designer output `decision: abort`): no Coder or Verifier run this round. Main session commits `rounds/decision_NNN.md` alone (audit chain shows why no kernel) + updates `team-state.md` to reflect "round N aborted, abort streak = K". If K ≥ 3 → emit shutdown (diminishing-returns hard stop). Otherwise signal Designer to try a different direction next round.
- **User decision points** — main session surfaces to user: stop/continue, shape change, direction override, manual abort.

### 3.3 Why subagents, not single-session role phases

- Each role has a fresh, focused context window — Designer holds the bottleneck-analysis context without Coder's code or Verifier's trace numbers polluting it.
- Failure cost is contained: a wrong Coder implementation doesn't poison Designer's next-round reasoning; Designer just reads Verifier's report and tries again.
- Cross-round context: team-internal, Designer stays alive and remembers its last-round hypothesis queue. (Main session would have to re-establish this every round if it were orchestrator-mediated.)

## 4. Role contracts

### 4.1 Designer (`kernel-designer.md`)

**Inputs** (per round):
- `base.py` (round 0 only, or when shape changes)
- `rounds/report_<NNN-1>.md` (previous round's runtime numbers + bottleneck judgment)
- `state/designer_state.md` (own hypothesis queue, role-local)
- `references/anti-patterns.md` (if present — scan before picking path)
- `references/bottleneck-judgment.md` (procedure)
- `references/invariants.md` (what's safe to assume about Coder)

**Output**: `rounds/decision_NNN.md` containing:
- **Bottleneck class** — device-bound / host-bound / measurement-bound, with `device_ratio` and justification.
- **Hypothesis** — what the proposed change should do (concrete, falsifiable).
- **Optimization means** — concrete pattern (e.g. "fuse routing into kernel", "use `fast_libentry`"). Specific enough that Coder doesn't have to re-decide.
- **Expected improvement** — quantitative (e.g. "≥5% wall drop, device_ratio moves from 15% → 35%"). If Designer cannot justify ≥5%, output `decision: abort` and stop the round instead of writing a kernel.
- **Pitfall warnings** — anti-patterns relevant to this round (referencing `anti-patterns.md` + previous `designer_state.md` rejections).
- **KB hook** — section recording "consulted anti-patterns.md, hit/miss: X, Y" (v1 placeholder for future KB integration).

**Round 0 special case**: Designer also writes `project.md` (problem spec, measurement regime, upbound) instead of `decision_000.md`. This is the one-time project initialization.

### 4.2 Coder (`kernel-coder.md`)

**Inputs**:
- `rounds/decision_NNN.md` (the contract)
- `references/invariants.md` (AST filter handling, `fast_libentry` pattern, output caching, etc.)
- `state/coder_state.md` (invariants carried from previous round, unresolved smells)
- Previous round's `triton_<op>_<NNN-1>.py` (copy as starting point — change only what decision requires)

**Output**: `triton_<op>_<NNN>.py` with:
- `ModelNew` exposing `__init__` / `forward` / `get_inputs` / `get_init_inputs` matching `base.py`.
- Only the changes `decision_NNN.md` specifies — no unrelated refactoring.
- AST-filter-safe patterns (per `invariants.md`).

**Communication**: SendMessage to Verifier with the file path + any invariants Coder established this round (which Verifier needs to know if they affect measurement, e.g. "switched to preallocated output").

### 4.3 Verifier (`kernel-verifier.md`)

**Inputs**:
- `triton_<op>_<NNN>.py` (the implementation to test)
- `project.md` (measurement regime, repro command)
- `state/verifier_state.md` (env snapshot, noise baseline from previous rounds)
- `auto_bench.py` path + python interpreter

**Output**: `rounds/report_NNN.md` containing:
- **Correctness** — PASS/FAIL + diff summary (no commit on FAIL).
- **Wall time** — auto_bench numbers (`v0`, `v1`, `speedup`).
- **Device time** — sum of `dur` for `cat == "kernel"` events in profiler JSON, plus per-kernel breakdown table.
- **device_ratio** — `device_time / wall_time`, with bottleneck class label.
- **Upbound gap** — quantitative distance to declared upbound.
- **Noise check** — if wall is within 5% of previous round, flag as noise; require re-run.
- **Stop recommendation** — if any of the 5 stop criteria (§5) hit, emit `shutdown_request` via SendMessage to Designer (CC main session) with `reason` + `data`.

## 5. Team termination

### 5.1 Five stop criteria (Verifier emits if any hits)

1. **measurement-bound** (hard stop) — `device_ratio < 5%` AND remaining host overhead is entirely harness fixed costs (`set_seed` + `sync_devices`). No wall-time payoff remains.
2. **diminishing returns** (hard stop) — 3 consecutive successful rounds each < 5% wall improvement. OR 3 consecutive failed attempts where Designer couldn't justify a ≥5% path. (Replaces the current skill's "5 rounds" — groupedtopk proved that threshold loose.)
3. **upbound reached** (soft stop) — cumulative speedup enters declared upbound's X% band. Verifier emits soft stop; Designer decides whether to chase stretch goal.
4. **resource exhausted** (hard stop) — `project.md` overview table > 30 entries, OR total rounds > 40, OR team has run > 24h. Safety against runaway.
5. **user intervention** (override) — main session forwards user stop signal to Verifier; Verifier emits shutdown immediately; Designer MUST approve=true (user is final authority).

### 5.2 Mechanical flow

```
Verifier.run() →
  if stop_criteria_hit:
    SendMessage(to: "designer",
                message: {type: "shutdown_request",
                           request_id: <uuid>,
                           reason: <one of 5>,
                           data: {round, numbers, ...}})
    SendMessage(to: "main", message: <same, CC>)
    wait for Designer's shutdown_response

Designer.recv(shutdown_request) →
  if reason == "user-intervention": approve = true (forced)
  elif reason == "measurement-bound" or "upbound-reached": approve = true
  elif reason == "diminishing-returns" or "resource-exhausted":
    if has_viable_next_round: approve = false, attach new decision_NNN.md
    else: approve = true
  SendMessage(to: "verifier",
              message: {type: "shutdown_response",
                        request_id: <echo>,
                        approve: bool,
                        feedback: <if reject, reasoning>})

Main session:
  if approve == true:
    commit final round + write final entry to project.md
    TeamDelete
  elif approve == false:
    confirm continuation, team persists, Designer picks up from new decision
```

**Timeouts / anomalies**:
- Designer doesn't respond to `shutdown_request` within 30s → main session treats as stop (conservative).
- Team unresponsive → main session force `TeamDelete` (recorded in `references/invariants.md` pitfalls as forceful-shutdown).

## 6. Resume design

### 6.1 The resume problem

Claude Code teams are session-bound. When the session ends, the team is gone. "Resume" = spawn a fresh team that reads on-disk state and picks up. The state must capture three things:
1. Where the team stopped (round number, last decision, last report).
2. Why it stopped (which of the 5 stop criteria — they have different resume semantics).
3. What changed before resume (skill version, KB presence, shape, harness) — without this, stale state gets reused.

### 6.2 State files

**`<op>/team-state.md`** (single manifest, written by main session at every shutdown):
```yaml
current_round: 024
last_decision: rounds/decision_024.md
last_report: rounds/report_024.md
stop_reason: diminishing-returns     # one of 5
stop_timestamp: 2026-08-13T15:42Z
resume_eligible: conditional         # always | conditional | blocked
resume_constraints:
  - "skill version >= 2.0"
  - "knowledge_base consulted"
  - "no shape/harness change since stop"
skill_version_at_stop: 1.3
kb_version_at_stop: null             # v1 not integrated
```

**Per-role state files** (role-local, append per round, capture reasoning not流水账):
- `state/designer_state.md` — next-round candidate hypothesis queue (directions Designer considered but didn't pick, with rejection reasons) + current leaning.
- `state/coder_state.md` — invariants established this round (AST-filter pattern chosen, `fast_libentry` variant, output caching state) + unresolved code smells.
- `state/verifier_state.md` — env snapshot (python path, `--warmup --repeat`, device state) + noise baseline (so next round's numbers are comparable).

### 6.3 Resume semantics by stop_reason

| stop_reason | resume_eligible | Resume precondition |
|---|---|---|
| measurement-bound | blocked | Requires new shape or new harness declaration — same regime can't progress. |
| diminishing-returns | conditional | Requires new input (KB consulted, new idea, skill improved). |
| upbound-reached | conditional | Requires new stretch-goal upbound declaration. |
| resource-exhausted | always | Direct resume; safety stop, not real stop. |
| user-intervention | conditional | Requires user reconfirmation + reason for resuming. |

### 6.4 Resume flow

1. Main session reads `team-state.md` → validates all `resume_constraints` satisfied (skill version, KB presence, shape/harness unchanged).
2. If `stop_reason == measurement-bound` or `upbound-reached` → reject auto-resume; require user to declare new shape / new upbound first.
3. If `stop_reason == diminishing-returns` / `resource-exhausted` / `user-intervention` → spawn fresh team.
4. Each role reads: `team-state.md` (latest pointers) + own `state/<role>_state.md` + last `report_NNN.md` (Designer) or `decision_NNN.md` (Coder/Verifier).
5. Designer's first action: emit "resume sanity check" decision (is the last-round bottleneck still present? is the hypothesis queue still viable?) before picking the real next round.

## 7. Multi-project structure & memory tiers

### 7.1 Tier 1 — skill-level (cross-project, all subagents read)

```
~/.claude/skills/kernel-opt-loop/
  SKILL.md                          # orchestrator guide (main session)
  agents/
    kernel-designer.md
    kernel-coder.md
    kernel-verifier.md
  references/
    bottleneck-judgment.md          # preserved from current skill
    project-template.md             # replaces log-template.md
    invariants.md                    # AST filter, fast_libentry, output caching, etc.
    anti-patterns.md                 # cross-project failure modes; v1 empty placeholder
```

Subagent definitions live under `agents/` (not global `~/.claude/agents/`) so the skill is self-contained — easier to version, share, migrate.

### 7.2 Tier 2 — project-level (per operator)

```
<op>/
  base.py                           # user-provided reference
  project.md                        # problem spec + measurement + upbound + overview table + repro + checkpoint
  team-state.md                     # resume manifest
  triton_<op>_<NNN>.py              # Coder output (one per round)
  rounds/
    decision_001.md                 # Designer output
    report_001.md                   # Verifier output
    ...
  state/                            # role-local (Tier 3)
    designer_state.md
    coder_state.md
    verifier_state.md
  log/                              # gitignored profiler JSON
    *.pt.trace.json
```

### 7.3 Tier 3 — role-level (per-role, per-project)

`state/<role>_state.md` — written and read only by the corresponding subagent. Main session never touches. Reconstructable from `rounds/` if lost.

### 7.4 Knowledge lift (project → skill)

- Designer writes failed-attempt patterns into `state/designer_state.md` with rejection reason each round.
- At project stop, main session scans `designer_state.md`, extracts **generic** (non-shape-specific) failure patterns.
- Main session asks user: "promote these to `references/anti-patterns.md`?"
- On user confirm, main session appends to skill-level `anti-patterns.md` (git commit on the skill repo).
- Future projects' Designer reads updated `anti-patterns.md` automatically.

## 8. log.md fate

Current `log.md` (8 sections) is split:
- Sections 1-3, 7, 8 → `project.md` (project-level, written once in Phase 0, overview table grows by one row per round).
- Section 4 (entries) → `rounds/decision_NNN.md` + `rounds/report_NNN.md` (per-round).
- Section 5 (current bottleneck) → `team-state.md` (and `designer_state.md` for the queue).
- Section 6 (next direction) → `designer_state.md`.

The file `log.md` no longer exists. Audit chain is `project.md` overview table + `rounds/` directory listing.

## 9. KB hook (v1)

Designer's prompt includes a mandatory step:
> "Before picking the optimization path, scan `references/anti-patterns.md`. Record hit/miss in `decision_NNN.md` under the **KB hook** section. If `anti-patterns.md` is empty or absent, note 'KB unavailable' and proceed using `bottleneck-judgment.md` + previous `report_NNN.md`."

No external KB lookup in v1. The hook is structural — when KB is later integrated, the `decision_NNN.md` already records what was consulted, so old rounds remain interpretable.

## 10. Migration from current skill

Existing worked examples (`groupedtopk/log.md`, `fused_moe/log.md`) are reference material; they stay as-is in the repo for now. The new skill doesn't auto-migrate them to the new structure. Future projects use the new structure from Phase 0.

For active projects that may resume: they remain on the old `log.md` format until the user explicitly asks to migrate. Migration is out of scope for v1 (postponed until the new structure proves itself).

## 11. Open questions for implementation plan

These are deferred to the writing-plans phase:
- Exact `subagent_type` for each role (likely `Plan` for Designer, `developer` for Coder, `qa` for Verifier — but final choice happens in implementation).
- Whether to provide a `kernel-opt-loop-init` slash command for Phase 0, or rely on user typing "optimize operator X".
- Whether `project.md` should be markdown or YAML+markdown hybrid for the overview table (machine-parseable vs human-readable tradeoff).
- Concrete `invariants.md` content — needs to be populated from the current skill's "Pitfalls log" + `bottleneck-judgment.md` compressible-vs-fixed table.
