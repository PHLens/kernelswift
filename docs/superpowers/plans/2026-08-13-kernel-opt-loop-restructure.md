# kernel-opt-loop Skill Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the monolithic `kernel-opt-loop` skill into a Designer/Coder/Verifier team-orchestrated skill with file-based role contracts, hybrid team communication, 5-criterion team termination, resume design, and 3-tier memory with knowledge lift.

**Architecture:** Main skill `SKILL.md` becomes an orchestrator guide; role behavior lives in `prompts/{designer,coder,verifier}.md` injected as spawn-prompt prefixes for built-in `subagent_type` (architect/developer/qa); project-level artifacts (`project.md`, `rounds/decision_NNN.md`, `rounds/report_NNN.md`, `team-state.md`, `state/*.md`) replace the old monolithic `log.md`.

**Tech Stack:** Markdown skill files; Claude Code `Agent` tool with `team_name` + built-in `subagent_type`; Claude Code `TeamCreate`/`SendMessage`/`TeamDelete`; bash + `jq` for profiler JSON analysis (inherited from current skill).

**Spec:** `docs/superpowers/specs/2026-08-13-kernel-opt-loop-restructure-design.md`

## Global Constraints

- Skill source of truth: `skills/kernel-opt-loop/` in this repo (git-tracked). The `~/.claude/skills/kernel-opt-loop/` is a synced copy — implementation writes to the repo path; sync happens in Task 9.
- Subagent embodiment: built-in `subagent_type` only (`architect`, `developer`, `qa`, `general-purpose`). Custom agent files under skill `agents/` are NOT auto-resolved (verified 2026-08-13 via probe). Role behavior lives in `prompts/*.md` and is injected as the spawn-prompt prefix.
- Communication architecture: hybrid. Three roles form a long-lived team via `TeamCreate(team_name: "<op>-opt")` + `Agent(team_name=..., name=..., subagent_type=...)`. Round-internal handoffs use `SendMessage`. Main session intervenes only at round boundaries (commit, stop/continue, user decisions).
- File contracts (spec §4.x):
  - `rounds/decision_NNN.md` — Designer output, Coder input.
  - `triton_<op>_<NNN>.py` — Coder output, Verifier input.
  - `rounds/report_NNN.md` — Verifier output, Designer input (next round).
  - `project.md` — project-level spec + overview table (replaces old `log.md` Sections 1-3, 7, 8).
  - `team-state.md` — resume manifest.
  - `state/{designer,coder,verifier}_state.md` — role-local reasoning state.
  - `rounds/round_status_NNN.md` — opt-in visibility for long Verifier runs.
- Round operation rules (spec §4.3):
  - Coder→Designer revision: minor deviation = proceed + log; major = refuse + SendMessage; max 2 round-trips per round.
  - Failure retry budgets: Coder code broken = 2 self-fix; Verifier env failure = 0 (escalate to user); Verifier accuracy FAIL = 1 Coder retry; Verifier < 5% noise = 2 Verifier re-runs.
  - User visibility: passive idle notifications at each role's turn end + opt-in `round_status_NNN.md`. No polling.
- Stop criteria (spec §5.1): measurement-bound, diminishing-returns (3 consecutive < 5% OR 3 aborts), upbound-reached, resource-exhausted (> 30 entries OR > 40 rounds OR > 24h), user-intervention. Verifier emits `shutdown_request`; Designer approves/rejects; main session is final authority.
- Resume (spec §6): `team-state.md` manifest records stop_reason + resume_constraints; main session validates constraints before spawning fresh team; per `stop_reason` resume eligibility table drives behavior.
- v1 supports only linear workflow (no parallel attempts).
- Knowledge base: v1 leaves only a Designer-side hook. `references/anti-patterns.md` is seeded from groupedtopk's failed attempts; no external KB lookup.
- Don't auto-migrate existing `groupedtopk/log.md` or `fused_moe/log.md` to the new structure (spec §10).

---

## File Structure

After this plan, the skill directory will look like:

```
skills/kernel-opt-loop/
  SKILL.md                          # orchestrator guide (main session) — REWRITTEN
  prompts/                          # NEW subdirectory
    designer.md                     # Designer role behavior
    coder.md                        # Coder role behavior
    verifier.md                     # Verifier role behavior
  references/
    bottleneck-judgment.md          # PRESERVED (no changes)
    project-template.md             # RENAMED from log-template.md, content adapted
    invariants.md                   # NEW — code invariants (AST filter, fast_libentry, etc.)
    anti-patterns.md                # NEW — cross-project failure patterns from groupedtopk
```

Removed:
- `references/log-template.md` (replaced by `project-template.md`).

Project-level (per operator) structure produced when the skill runs:

```
<op>/
  base.py
  project.md
  team-state.md
  triton_<op>_<NNN>.py
  rounds/
    decision_001.md
    report_001.md
    round_status_001.md
    ...
  state/
    designer_state.md
    coder_state.md
    verifier_state.md
  log/                              # gitignored
    *.pt.trace.json
```

---

## Task 1: Branch, backup, and skeleton

**Files:**
- Create: `skills/kernel-opt-loop/SKILL.md.legacy` (backup of current SKILL.md)
- Create: `skills/kernel-opt-loop/prompts/` (directory)

**Interfaces:**
- Consumes: current `skills/kernel-opt-loop/SKILL.md` (source material for later extraction).
- Produces: backup file `SKILL.md.legacy` (read by Task 2 for pitfalls extraction); `prompts/` directory (written by Tasks 5-7).

- [ ] **Step 1: Create feature branch**

```bash
git checkout -b kernel-opt-loop-restructure
```

Verify: `git branch --show-current` outputs `kernel-opt-loop-restructure`.

- [ ] **Step 2: Backup current SKILL.md**

```bash
cp skills/kernel-opt-loop/SKILL.md skills/kernel-opt-loop/SKILL.md.legacy
```

Verify: `ls skills/kernel-opt-loop/SKILL.md.legacy` succeeds.

- [ ] **Step 3: Create prompts/ directory**

```bash
mkdir -p skills/kernel-opt-loop/prompts
```

Verify: `ls -d skills/kernel-opt-loop/prompts` succeeds.

- [ ] **Step 4: Commit backup + skeleton**

```bash
git add skills/kernel-opt-loop/SKILL.md.legacy
git commit -m "skills: backup current kernel-opt-loop SKILL.md before restructure"
```

Verify: `git log --oneline -1` shows the backup commit.

---

## Task 2: Write `references/invariants.md`

Extract code invariants from the current `SKILL.md.legacy` pitfalls log + `bottleneck-judgment.md` compressible-vs-fixed table. Target audience: Coder (reads this to know what patterns are safe) and Designer (reads this to know what's already solved, so doesn't re-propose).

**Files:**
- Create: `skills/kernel-opt-loop/references/invariants.md`

**Interfaces:**
- Consumes: `skills/kernel-opt-loop/SKILL.md.legacy` (Pitfalls log + Step 3 AST filter section + Step 4 accuracy failure causes); `skills/kernel-opt-loop/references/bottleneck-judgment.md` (Compressible vs fixed table).
- Produces: `references/invariants.md` referenced by `prompts/coder.md` (Task 6) and `prompts/designer.md` (Task 5).

**Content checklist (each must appear in the file):**
1. `_filter_module_ast` stripping — auto_bench's filter drops non-literal module-level assigns. Pattern: `fast_libentry()(_kernel)` at module scope → NameError. Fix: class-body `globals()` trick.
2. argmax sentinel — `tl.where(is_best, e_idx, E)` corrupts the sum. Use `tl.where(is_best, e_idx, 0)`.
3. `tl.dot` shape — requires 2D inputs and matching inner dims. `[1, H] @ [2I, H]` is wrong; transpose first.
4. `torch.mlu.device()` context manager — has host enter/exit overhead. If caller sets device, drop it.
5. `torch.empty_like` per forward — allocator overhead. Cache output tensor on ModelNew instance.
6. `torch.cuda.is_available()` on MLU box — returns True (CUDA stub loaded), so `sync_devices()` syncs BOTH cuda and mlu → double sync cost per iter. Harness overhead, not fixable in kernel.
7. `fast_libentry` — compresses Triton launcher default path. (From compressible-vs-fixed table.)
8. Routing PyTorch ops (softmax/topk/cast) — fuse into kernel. (From compressible-vs-fixed table.)
9. `set_seed` per forward — fixed (harness). Don't try to optimize.
10. `sync_devices` syncing multiple accelerators — fixed (harness). Don't try to optimize.
11. `build_case` + `load_state_dict` state diff — fixed (harness). Don't try to optimize.

- [ ] **Step 1: Read source material**

Read `skills/kernel-opt-loop/SKILL.md.legacy` (full file, ~200 lines) and `skills/kernel-opt-loop/references/bottleneck-judgment.md` (full file).

- [ ] **Step 2: Write the failing test (content checklist)**

Create a temporary checklist file `skills/kernel-opt-loop/references/.invariants-checklist.tmp` with the 11 items above (one per line). This is the test — each item must appear in the final `invariants.md`.

- [ ] **Step 3: Verify test fails**

```bash
test -f skills/kernel-opt-loop/references/invariants.md && echo "exists" || echo "missing"
```

Expected: `missing` (file doesn't exist yet).

- [ ] **Step 4: Write `invariants.md`**

Structure:
```markdown
# Kernel Implementation Invariants

[1-paragraph intro: this file is read by Coder (to know safe patterns) and Designer (to know what's already solved). Updates here propagate to all projects.]

## Code patterns (apply when writing kernels)

### _filter_module_ast stripping
[explanation + the class-body globals() trick with code block, copied from SKILL.md.legacy Step 3]

### argmax sentinel
[explanation + correct tl.where pattern]

### tl.dot shape requirements
[explanation + correct shape pattern]

### fast_libentry
[explanation + usage]

### Output tensor caching
[explanation + ModelNew instance cache pattern]

### torch.mlu.device() context manager
[explanation + when to drop it]

## Harness fixed costs (do not try to optimize)

### set_seed per forward
[explanation]

### sync_devices multi-accelerator sync
[explanation, especially torch.cuda.is_available() returning True on MLU box]

### build_case + load_state_dict state diff
[explanation]

## Compressible vs fixed quick reference

[Table copied from bottleneck-judgment.md "Compressible vs fixed host overhead" section]
```

- [ ] **Step 5: Verify test passes (content checklist)**

For each of the 11 items in `.invariants-checklist.tmp`, grep the new file:

```bash
while IFS= read -r item; do
  if grep -qF "$item" skills/kernel-opt-loop/references/invariants.md; then
    echo "OK: $item"
  else
    echo "MISSING: $item"
  fi
done < skills/kernel-opt-loop/references/.invariants-checklist.tmp
```

Expected: all 11 lines start with `OK:`. If any `MISSING`, edit `invariants.md` to add the missing item.

- [ ] **Step 6: Cleanup checklist + commit**

```bash
rm skills/kernel-opt-loop/references/.invariants-checklist.tmp
git add skills/kernel-opt-loop/references/invariants.md
git commit -m "skills: add invariants reference for kernel-opt-loop"
```

Verify: `git log --oneline -1` shows the commit.

---

## Task 3: Write `references/anti-patterns.md`

Seed the cross-project failure-pattern catalog from groupedtopk's 16 failed attempts. Target audience: Designer (scans before picking optimization path; records hit/miss in `decision_NNN.md` KB hook section).

**Files:**
- Create: `skills/kernel-opt-loop/references/anti-patterns.md`

**Interfaces:**
- Consumes: `groupedtopk/log.md` (entries 004, 005, 006, 007, 011, 012, 013, 014, 015, 016, 017, 019, 021, 022, 023, 024 — all marked 失败).
- Produces: `references/anti-patterns.md` referenced by `prompts/designer.md` (Task 5).

**Anti-pattern extraction approach:**
Each failed entry's "踩坑" or "状态" + "结果" sections reveal a structural failure mode. Abstract the *pattern* (not the shape-specific code) — e.g. "winner tree" not "winner tree with 32 experts". Each anti-pattern entry should answer: (a) what was the hypothesis, (b) why it structurally failed (not just "didn't speed up"), (c) how to recognize it before trying.

- [ ] **Step 1: Read failed entries from groupedtopk/log.md**

Read `groupedtopk/log.md` lines covering entries 004, 005, 006, 007 (around lines 267-369 based on grep output). Use `Read` tool with offset/limit to capture all 16 failed entries (004-024, skipping 008/010/018 which succeeded). Total ~700 lines.

- [ ] **Step 2: Catalog patterns to scratch file**

Write `skills/kernel-opt-loop/references/.anti-patterns-catalog.tmp` with one line per failed entry:

```
entry_004 | <one-line pattern name> | <one-line structural reason>
entry_005 | ...
...
entry_024 | ...
```

Aim for 16 lines. If two entries share a pattern, note both entry numbers on one line (e.g. `entry_021,022,023 | U1 batch tile attempts | <reason>`).

- [ ] **Step 3: Verify catalog has 16 entries**

```bash
wc -l skills/kernel-opt-loop/references/.anti-patterns-catalog.tmp
```

Expected: between 10 and 16 lines (some entries may merge). If fewer than 10, re-read entries — you missed some.

- [ ] **Step 4: Write `anti-patterns.md`**

Structure:
```markdown
# Kernel Optimization Anti-Patterns

[1-paragraph intro: this file is read by Designer before picking an optimization path. Each entry is a structural failure mode observed in past projects. When considering a path that matches an anti-pattern, record the hit in decision_NNN.md KB hook section and justify why this attempt is different.]

## Catalog

### <Pattern name> (seen in: entry_XXX, entry_YYY)

**Hypothesis it tried to validate**: <one line>

**Structural reason it failed**: <2-3 lines — why this approach can't work for this class of problem, not just "didn't measure up">

**Recognition signs**: <how to tell a future proposal is the same pattern>

---

[repeat for each pattern]
```

- [ ] **Step 5: Verify each catalog entry appears in the file**

```bash
while IFS='|' read -r entries name reason; do
  if grep -qF "$(echo "$name" | xargs)" skills/kernel-opt-loop/references/anti-patterns.md; then
    echo "OK: $entries"
  else
    echo "MISSING: $entries ($name)"
  fi
done < skills/kernel-opt-loop/references/.anti-patterns-catalog.tmp
```

Expected: all lines `OK`. Fix any `MISSING` by adding the pattern to `anti-patterns.md`.

- [ ] **Step 6: Cleanup + commit**

```bash
rm skills/kernel-opt-loop/references/.anti-patterns-catalog.tmp
git add skills/kernel-opt-loop/references/anti-patterns.md
git commit -m "skills: seed anti-patterns reference from groupedtopk failures"
```

Verify: `git log --oneline -1` shows the commit.

---

## Task 4: Write `references/project-template.md` (replaces log-template.md)

Adapt the current `log-template.md` to the new structure: only project-level sections (1-3, 7, 8 from the old template). Per-round entries are no longer in this file — they live in `rounds/decision_NNN.md` + `rounds/report_NNN.md`.

**Files:**
- Create: `skills/kernel-opt-loop/references/project-template.md`
- Delete: `skills/kernel-opt-loop/references/log-template.md`

**Interfaces:**
- Consumes: `skills/kernel-opt-loop/references/log-template.md` (source material to adapt from).
- Produces: `references/project-template.md` referenced by `prompts/designer.md` (Task 5, used in Phase 0 to initialize project.md).

- [ ] **Step 1: Read current log-template.md**

Read `skills/kernel-opt-loop/references/log-template.md` (full file, ~160 lines).

- [ ] **Step 2: Write `project-template.md`**

Keep only Sections 1 (problem + measurement), 2 (upbound), 3 (overview table), 7 (repro), 8 (checkpoint) from the old template. Drop Sections 4, 5, 6 — they're per-round and now live in `rounds/`. Structure:

```markdown
# <Operator Name> Triton Kernel Optimization Project

[1-paragraph intro: what this operator does, what device, what the optimization goal is. Per-round entries live in rounds/decision_NNN.md + rounds/report_NNN.md. This file is the project-level contract written once in Phase 0; the overview table grows by one row per round.]

## 1. 固定问题与测试口径

### 1.1 算子语义
[copy from log-template.md Section 1.1]

### 1.2 环境
[copy from log-template.md Section 1.2]

### 1.3 测量规则
[copy from log-template.md Section 1.3, but adapt rule 5 to reference the new abort/noise handling from spec §4.3(b)]

## 2. Upbound 定义
[copy from log-template.md Section 2]

## 3. 当前结果总览

| 实现 | Wall time/call (auto_bench) | Kernel device time | 相对上一阶段 | 相对 base |
|---|---:|---:|---:|---:|
| `base.py` eager | <X> ms | ~<Y> ms / 50 iter | - | 1.00x |

[Note: each round adds one row. Updated by main session at round commit time, not by subagents.]

## 4. 复现命令
[copy from log-template.md Section 7]

## 5. Checkpoint
[copy from log-template.md Section 8, but adapt: v1–v<NNN> Triton refers to files in triton_<op>_<NNN>.py; per-round decisions/reports in rounds/]
```

- [ ] **Step 3: Verify template covers required sections**

```bash
for section in "1. 固定问题与测试口径" "2. Upbound 定义" "3. 当前结果总览" "4. 复现命令" "5. Checkpoint"; do
  grep -qF "$section" skills/kernel-opt-loop/references/project-template.md && echo "OK: $section" || echo "MISSING: $section"
done
```

Expected: all 5 `OK`. Fix any `MISSING`.

- [ ] **Step 4: Delete old log-template.md**

```bash
git rm skills/kernel-opt-loop/references/log-template.md
```

Verify: `ls skills/kernel-opt-loop/references/log-template.md 2>&1` shows "No such file or directory".

- [ ] **Step 5: Commit**

```bash
git add skills/kernel-opt-loop/references/project-template.md
git commit -m "skills: replace log-template with project-template for kernel-opt-loop"
```

Verify: `git log --oneline -1` shows the commit.

---

## Task 5: Write `prompts/designer.md`

Designer role behavior. Spawned via `Agent(subagent_type: "architect", team_name: "<op>-opt", name: "designer", prompt: <contents of this file> + <round briefing>)`. Per spec §4.1.

**Files:**
- Create: `skills/kernel-opt-loop/prompts/designer.md`

**Interfaces:**
- Consumes: `references/bottleneck-judgment.md`, `references/invariants.md` (Task 2), `references/anti-patterns.md` (Task 3), `references/project-template.md` (Task 4), `base.py` (round 0 only), `rounds/report_<NNN-1>.md` (previous round), `state/designer_state.md` (own state).
- Produces: `rounds/decision_NNN.md` (round N≥1) or `project.md` (round 0 special case).

- [ ] **Step 1: Write `prompts/designer.md`**

Content must cover (per spec §4.1):

```markdown
# Designer Role

You are the Designer in a kernel-opt-loop team. Your job: analyze the previous round's runtime report, pick ONE bottleneck, write a decision that Coder can implement without re-deciding.

## Inputs you read each round

- Round 0: `base.py` (operator reference) + `references/project-template.md` (to write `project.md`).
- Round N≥1: `rounds/report_<NNN-1>.md` (Verifier's previous report), `state/designer_state.md` (your hypothesis queue), `references/anti-patterns.md`, `references/bottleneck-judgment.md`, `references/invariants.md`.

## Output: `rounds/decision_NNN.md`

Required sections, in order:

### Bottleneck class
- Compute `device_ratio = sum(kernel dur) / wall_time` from the previous `report_NNN.md`.
- Label: device-bound (>80%) / mixed (20-80%) / host-bound (<20%) / measurement-bound (<5% AND wall stuck).
- 2-3 sentence justification with the numbers.

### Hypothesis
- What the proposed change should do. Concrete, falsifiable: "switching from elementwise outer product to `tl.dot` for the GEMM should drop device time from 21us to ~12us by using tensor cores".

### Optimization means
- Concrete pattern (e.g. "fuse routing into kernel", "use `fast_libentry`"). Specific enough that Coder doesn't re-decide. Reference `references/invariants.md` for the safe pattern.

### Expected improvement
- Quantitative: "≥5% wall drop, device_ratio moves from 15% → 35%".
- If you cannot justify ≥5%, output `decision: abort` as the only content and stop. Do NOT write a kernel. Record the rejection reason in `state/designer_state.md` hypothesis queue.

### Pitfall warnings
- Anti-patterns relevant to this round. Reference `references/anti-patterns.md` entries by name + the structural reason they failed. If this attempt matches an anti-pattern, justify why this attempt is different.

### KB hook
- Record "consulted anti-patterns.md, hit/miss: X, Y" — even if anti-patterns.md is empty, write "consulted anti-patterns.md, no hits (file empty or no matching patterns)".

## Round 0 special case

In Round 0, you do NOT write `decision_000.md`. Instead, you write `project.md` from `references/project-template.md`:
1. Read `base.py` — extract operator semantics, shape, dtype, routing/scoring math.
2. Fill in Section 1.1 (operator semantic), 1.2 (environment — ask user for device/python/triton versions if not in `base.py`), 1.3 (measurement rules — use defaults from project-template).
3. Section 2 (upbound): ask user for the upbound reference (e.g. CNNL `Op<half>` latency). If user gives none, write "no upbound declared; stop on measurement-bound or diminishing-returns only".
4. Section 3 (overview table): write the base.py row only.
5. Section 4 (repro): write the auto_bench command using the python interpreter + path.
6. Section 5 (checkpoint): today's date, "v1 Triton: 待补".
7. Initialize `state/designer_state.md` with empty hypothesis queue + current leaning "start round 1: identify biggest bottleneck in base.py".

## Communicating with Coder

When your `decision_NNN.md` is written, SendMessage to `coder`:
- `to: "coder"`
- `message`: one-paragraph briefing + the path `rounds/decision_NNN.md`. Example: "Round 5 decision ready at rounds/decision_005.md. Bottleneck: host-bound, launcher dominates. Optimization: switch to fast_libentry. Expected ≥10% wall drop."

## Handling Coder revision requests

If Coder SendMessage you saying the decision is unimplementable (major deviation), you have two options:
- Revise `decision_NNN.md` in place (overwrite — no version bump) with the corrected approach. Max 2 revisions per round.
- Mark `decision: abort` and stop the round.

## Handling Verifier shutdown_request

When Verifier sends `shutdown_request`, respond within 30s:
- If reason is `user-intervention`, `measurement-bound`, or `upbound-reached`: approve=true.
- If reason is `diminishing-returns` or `resource-exhausted`: check your hypothesis queue. If you have a viable ≥5% path, approve=false and attach a new `decision_NNN.md` (overwrite). If not, approve=true.

Response format: `SendMessage(to: "verifier", message: {type: "shutdown_response", request_id: <echo>, approve: bool, feedback: <if reject, reasoning>})`.

## State file: `state/designer_state.md`

Append per round:
- Next-round candidate hypothesis queue (directions you considered but didn't pick, with rejection reasons).
- Current leaning for next round.

This file is your role-local memory. Main session does not read it; you read it at start of each round to recall what you considered last time.
```

- [ ] **Step 2: Smoke test — spawn Designer with minimal fixture**

Spawn an architect subagent with this prompt + a minimal fixture (synthetic base.py + a fake previous report) to verify the prompt is coherent and produces a decision.md with required sections.

Prepare fixture at `/tmp/kernel-opt-loop-test/designer-smoke/`:
```bash
mkdir -p /tmp/kernel-opt-loop-test/designer-smoke
cat > /tmp/kernel-opt-loop-test/designer-smoke/base.py <<'EOF'
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, N=64):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N, N))
    def forward(self, x):
        return torch.matmul(x, self.weight)

def get_inputs():
    return (torch.randn(1, 64),)

def get_init_inputs():
    return (64,)
EOF

cat > /tmp/kernel-opt-loop-test/designer-smoke/report_000.md <<'EOF'
# Report 000

**Correctness**: PASS
**Wall time**: v0=1.2 ms, v1=1.2 ms, speedup=1.0x
**Device time**: 980 us / iter (50 iters total 49 ms)
**device_ratio**: 81% (device-bound)
**Upbound gap**: base row, no comparison
**Stop recommendation**: none

## Per-kernel breakdown
matmul: count=50, total=49000us, avg=980us
EOF
```

Spawn:
```
Agent(
  subagent_type: "architect",
  description: "Designer smoke test",
  prompt: <<contents of skills/kernel-opt-loop/prompts/designer.md>>

Fixture base for this smoke test:
- base.py: /tmp/kernel-opt-loop-test/designer-smoke/base.py
- previous report: /tmp/kernel-opt-loop-test/designer-smoke/report_000.md
- output path: /tmp/kernel-opt-loop-test/designer-smoke/decision_001.md

Round 1 task: read base.py + report_000.md, produce decision_001.md per your role contract. Do NOT actually optimize — this is a smoke test to verify your prompt is coherent. Write the decision with placeholder content if you can't determine real numbers.
)
```

- [ ] **Step 3: Verify smoke test output has required sections**

```bash
for section in "Bottleneck class" "Hypothesis" "Optimization means" "Expected improvement" "Pitfall warnings" "KB hook"; do
  grep -qF "$section" /tmp/kernel-opt-loop-test/designer-smoke/decision_001.md && echo "OK: $section" || echo "MISSING: $section"
done
```

Expected: all 6 `OK`. If any `MISSING`, fix `prompts/designer.md` and re-spawn.

- [ ] **Step 4: Cleanup fixture + commit**

```bash
rm -rf /tmp/kernel-opt-loop-test
git add skills/kernel-opt-loop/prompts/designer.md
git commit -m "skills: add Designer role prompt for kernel-opt-loop"
```

Verify: `git log --oneline -1` shows the commit.

---

## Task 6: Write `prompts/coder.md`

Coder role behavior. Spawned via `Agent(subagent_type: "developer", team_name: ..., name: "coder", prompt: <contents of this file> + <round briefing>)`. Per spec §4.2.

**Files:**
- Create: `skills/kernel-opt-loop/prompts/coder.md`

**Interfaces:**
- Consumes: `rounds/decision_NNN.md` (Designer's output), `references/invariants.md` (Task 2), `state/coder_state.md` (own state), previous round's `triton_<op>_<NNN-1>.py`.
- Produces: `triton_<op>_<NNN>.py` exposing `ModelNew` with `__init__`/`forward`/`get_inputs`/`get_init_inputs` matching `base.py`.

- [ ] **Step 1: Write `prompts/coder.md`**

Content must cover (per spec §4.2 + §4.3(a) revision loop + §4.3(b) failure handling):

```markdown
# Coder Role

You are the Coder in a kernel-opt-loop team. Your job: read Designer's `decision_NNN.md`, produce `triton_<op>_<NNN>.py` that implements the decision. Change ONLY what the decision requires — no unrelated refactoring.

## Inputs you read each round

- `rounds/decision_NNN.md` (the contract — what to implement).
- `references/invariants.md` (safe patterns: AST filter, fast_libentry, output caching, etc.).
- `state/coder_state.md` (invariants from previous round, unresolved smells).
- Previous round's `triton_<op>_<NNN-1>.py` (copy as starting point — change only what decision requires).
- `base.py` (for the `ModelNew` contract: `__init__` signature matching `get_init_inputs()`, `forward` matching `get_inputs()`).

## Output: `triton_<op>_<NNN>.py`

Must expose `ModelNew` with:
- `__init__(self, ...)` matching `base.py`'s `get_init_inputs()` signature.
- `forward(self, ...)` matching `base.py`'s `get_inputs()` argument list.
- Same dtype/shape semantics as `base.py` for correctness to pass.

Apply invariants from `references/invariants.md`:
- AST-filter-safe patterns (class-body `globals()` trick if `fast_libentry` is needed).
- Cache output tensor on `ModelNew` instance (no `torch.empty_like` per forward).
- Drop `torch.mlu.device()` context if caller sets device.

## Self-check before handoff

Before SendMessage to Verifier:
1. AST parse your own file: `python -c "import ast; ast.parse(open('<path>').read())"`. If it fails, fix and retry (max 2 self-fix attempts).
2. Verify `ModelNew` class exists and has `__init__` + `forward` matching `base.py`'s contract.
3. Verify no module-level `fast_libentry()(_kernel)` (would be stripped by `_filter_module_ast`).

## Communicating with Verifier

When your file is ready + self-check passes, SendMessage to `verifier`:
- `to: "verifier"`
- `message`: file path `triton_<op>_<NNN>.py` + any invariants established this round that affect measurement (e.g. "switched to preallocated output — Verifier should expect lower host overhead").

## Handling Designer revision requests (minor vs major deviation)

When reading `decision_NNN.md`, classify any implementation-level inconsistency:

- **Minor deviation** (clearly implied by decision's intent, e.g. adding `tl.trans` to make `tl.dot` shape-legal): proceed + log deviation in `state/coder_state.md` under "Deviations this round". No Designer round-trip.

- **Major deviation** (decision's core path is unimplementable as specified, e.g. decision says "use `tl.dot` for GEMM" but the shapes fundamentally don't allow it without restructuring): refuse + SendMessage `designer` requesting revision. Format:
  - `to: "designer"`
  - `message`: "Decision NNN has major implementation blocker: <one-paragraph reason>. Requesting revision or abort."
  
Budget: max 2 Coder→Designer revision round-trips per round. Beyond that, Designer marks `decision: abort`.

## State file: `state/coder_state.md`

Append per round:
- Invariants established this round (AST-filter pattern chosen, fast_libentry variant, output caching state).
- Unresolved code smells (for next round's Coder to know what's pending).
- Deviations from decision (minor ones, with reasoning).

This file is your role-local memory. Main session does not read it.
```

- [ ] **Step 2: Smoke test — spawn Coder with minimal fixture**

Prepare fixture at `/tmp/kernel-opt-loop-test/coder-smoke/`:
```bash
mkdir -p /tmp/kernel-opt-loop-test/coder-smoke
cp /tmp/kernel-opt-loop-test/designer-smoke/base.py /tmp/kernel-opt-loop-test/coder-smoke/ 2>/dev/null || cat > /tmp/kernel-opt-loop-test/coder-smoke/base.py <<'EOF'
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, N=64):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N, N))
    def forward(self, x):
        return torch.matmul(x, self.weight)

def get_inputs():
    return (torch.randn(1, 64),)

def get_init_inputs():
    return (64,)
EOF

cat > /tmp/kernel-opt-loop-test/coder-smoke/decision_001.md <<'EOF'
# Decision 001

### Bottleneck class
device-bound (device_ratio ~81% from report_000).

### Hypothesis
Switching from torch.matmul to a triton kernel with tiling should reduce device time.

### Optimization means
Write a Triton kernel for matmul using tl.dot with 2D tiling. Expose ModelNew wrapping it.

### Expected improvement
≥5% wall drop, device_ratio stays similar but device time drops.

### Pitfall warnings
- tl.dot shape: requires 2D inputs and matching inner dims. See invariants.md.

### KB hook
consulted anti-patterns.md, no hits (matmul is too basic to match groupedtopk patterns)
EOF
```

Spawn:
```
Agent(
  subagent_type: "developer",
  description: "Coder smoke test",
  prompt: <<contents of skills/kernel-opt-loop/prompts/coder.md>>

Fixture:
- base.py: /tmp/kernel-opt-loop-test/coder-smoke/base.py
- decision: /tmp/kernel-opt-loop-test/coder-smoke/decision_001.md
- output path: /tmp/kernel-opt-loop-test/coder-smoke/triton_test_001.py

Round 1 task: write triton_test_001.py per decision_001.md. This is a smoke test — the kernel need not be correct, just structurally valid (ModelNew class, AST-parseable).
)
```

- [ ] **Step 3: Verify smoke test output**

```bash
python -c "import ast; ast.parse(open('/tmp/kernel-opt-loop-test/coder-smoke/triton_test_001.py').read()); print('AST OK')"
grep -q "class ModelNew" /tmp/kernel-opt-loop-test/coder-smoke/triton_test_001.py && echo "ModelNew OK" || echo "MISSING ModelNew"
grep -q "def forward" /tmp/kernel-opt-loop-test/coder-smoke/triton_test_001.py && echo "forward OK" || echo "MISSING forward"
grep -q "def get_inputs" /tmp/kernel-opt-loop-test/coder-smoke/triton_test_001.py && echo "get_inputs OK" || echo "MISSING get_inputs"
grep -q "def get_init_inputs" /tmp/kernel-opt-loop-test/coder-smoke/triton_test_001.py && echo "get_init_inputs OK" || echo "MISSING get_init_inputs"
```

Expected: all 5 `OK`. If any `MISSING`, fix `prompts/coder.md` and re-spawn.

- [ ] **Step 4: Cleanup + commit**

```bash
rm -rf /tmp/kernel-opt-loop-test
git add skills/kernel-opt-loop/prompts/coder.md
git commit -m "skills: add Coder role prompt for kernel-opt-loop"
```

Verify: `git log --oneline -1` shows the commit.

---

## Task 7: Write `prompts/verifier.md`

Verifier role behavior. Spawned via `Agent(subagent_type: "qa", team_name: ..., name: "verifier", prompt: <contents of this file> + <round briefing>)`. Per spec §4.4 + §4.3(b) failure handling + §4.3(c) user visibility + §5 stop criteria.

**Files:**
- Create: `skills/kernel-opt-loop/prompts/verifier.md`

**Interfaces:**
- Consumes: `triton_<op>_<NNN>.py` (Coder's output), `project.md` (measurement regime, repro command), `state/verifier_state.md` (env snapshot, noise baseline), `auto_bench.py` (harness), python interpreter path.
- Produces: `rounds/report_NNN.md`; `rounds/round_status_NNN.md` (status updates during long runs); `shutdown_request` SendMessage to Designer when stop criteria hit.

- [ ] **Step 1: Write `prompts/verifier.md`**

Content must cover (per spec §4.4 + §4.3(b) + §4.3(c) + §5):

```markdown
# Verifier Role

You are the Verifier in a kernel-opt-loop team. Your job: run `auto_bench.py` + torch.profiler on Coder's `triton_<op>_<NNN>.py`, produce `report_NNN.md` with the numbers, and emit `shutdown_request` when stop criteria hit.

## Inputs you read each round

- `triton_<op>_<NNN>.py` (the implementation to test).
- `project.md` Section 1.3 (measurement regime: warmup, repeat, sync rules) + Section 4 (repro command).
- `state/verifier_state.md` (env snapshot + noise baseline from previous rounds).
- `auto_bench.py` path + python interpreter (from project.md Section 4).

## Output: `rounds/report_NNN.md`

Required sections:

### Correctness
- PASS/FAIL + diff summary. On FAIL: do NOT commit; proceed to failure-handling flow below.

### Wall time
- auto_bench numbers: `v0`, `v1`, `speedup`. Use project.md's measurement regime (warmup 50, repeat 100 by default — adapt if project.md declares otherwise).

### Device time
- Sum of `dur` for `cat == "kernel"` events in profiler JSON.
- Per-kernel breakdown table (top kernels by total dur).

### device_ratio
- `device_time / wall_time`. With bottleneck class label (device-bound / mixed / host-bound / measurement-bound).

### Upbound gap
- Quantitative distance to project.md's declared upbound. If no upbound declared, write "no upbound; gap analysis skipped".

### Noise check
- If wall is within 5% of previous round (from `state/verifier_state.md` noise baseline), flag as noise. Re-run 2 times. If still < 5%, accept as no-improvement.

### Stop recommendation
- If any of the 5 stop criteria (below) hit, emit `shutdown_request` via SendMessage to `designer` (CC main session).

## Stop criteria — emit shutdown_request if any hits

1. **measurement-bound** (hard): `device_ratio < 5%` AND remaining host overhead is entirely harness fixed costs (set_seed + sync_devices).
2. **diminishing-returns** (hard): 3 consecutive successful rounds each < 5% wall improvement. OR 3 consecutive failed attempts (Designer aborts).
3. **upbound-reached** (soft): cumulative speedup enters project.md's declared upbound X% band.
4. **resource-exhausted** (hard): project.md overview table > 30 entries, OR total rounds > 40, OR team has run > 24h.
5. **user-intervention**: main session forwards user stop signal; emit immediately.

For each, include `reason` + `data` (round number, current numbers, streak counts) in the shutdown_request.

## Failure handling (per spec §4.3(b))

| Failure type | Action |
|---|---|
| Verifier env failure (auto_bench won't run, OOM, missing dep) | 0 retries. SendMessage main session "env failure: <details>". Do NOT write report_NNN.md. |
| Verifier accuracy FAIL | Coder gets 1 retry. SendMessage `coder` with the failure + diff. If 2nd attempt fails, write report_NNN.md with FAIL + commit for audit; round aborts. |
| Verifier PASS but < 5% (noise) | Re-run 2 times. If still < 5%, accept as no-improvement; write report_NNN.md with the median + noise flag; counts as failed attempt. |

## Status updates (per spec §4.3(c))

Write `rounds/round_status_NNN.md` at:
- Turn start: 1 line "verifier started at <ISO time>, round <NNN>".
- Mid-run (for long auto_bench): update with progress, e.g. "warmup done, 30/100 repeats".
- Turn end: 1 line "verifier done at <ISO time>, result: PASS/FAIL, wall=X ms".

This file is opt-in visibility for the user; main session does not poll it.

## Communicating with Designer

After report_NNN.md is written, SendMessage `designer`:
- `to: "designer"`
- `message`: "Report NNN ready at rounds/report_NNN.md. Wall=X ms, device_ratio=Y%, bottleneck class=Z. [stop recommendation if any]"

If shutdown_request: use the structured format `{type: "shutdown_request", request_id: <uuid>, reason: <one of 5>, data: {...}}`. CC main session.

## State file: `state/verifier_state.md`

Append per round:
- Env snapshot (python path, warmup/repeat, device state).
- Noise baseline (last 3 rounds' wall times, so next round can compare).
- Any harness quirks discovered.

## Repro command (always)

Use the command from project.md Section 4. Adapt `--v1_file` to the current `triton_<op>_<NNN>.py`. Standard:
```bash
<python> auto_bench.py \
  --v0_file <op>/base.py \
  --v1_file <op>/triton_<op>_<NNN>.py \
  --warmup 50 --repeat 100
```

For profiler:
```bash
<python> -c "
import torch, json
from torch.profiler import profile, ProfilerActivity
... (per project.md Section 4 if specified, else default 50-iter forward profile)
" > <op>/log/<NNN>.pt.trace.json
```
```

- [ ] **Step 2: Smoke test — spawn Verifier with minimal fixture**

Prepare fixture at `/tmp/kernel-opt-loop-test/verifier-smoke/`:
```bash
mkdir -p /tmp/kernel-opt-loop-test/verifier-smoke
cat > /tmp/kernel-opt-loop-test/verifier-smoke/project.md <<'EOF'
# Test Project

## 1.3 测量规则
1. auto_bench.py --warmup 50 --repeat 100
2. python: /usr/bin/python3

## 4. 复现命令
python3 auto_bench.py --v0_file base.py --v1_file triton_test_001.py --warmup 5 --repeat 10
EOF

cat > /tmp/kernel-opt-loop-test/verifier-smoke/triton_test_001.py <<'EOF'
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, N=64):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N, N))
    def forward(self, x):
        return torch.matmul(x, self.weight)

def get_inputs():
    return (torch.randn(1, 64),)

def get_init_inputs():
    return (64,)
EOF

cat > /tmp/kernel-opt-loop-test/verifier-smoke/base.py <<'EOF'
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, N=64):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(N, N))
    def forward(self, x):
        return torch.matmul(x, self.weight)

def get_inputs():
    return (torch.randn(1, 64),)

def get_init_inputs():
    return (64,)
EOF

# create a stub auto_bench.py that just prints fake numbers
cat > /tmp/kernel-opt-loop-test/verifier-smoke/auto_bench.py <<'EOF'
import sys, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--v0_file", required=True)
parser.add_argument("--v1_file", required=True)
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--repeat", type=int, default=100)
args = parser.parse_args()
print(f"PASS accuracy")
print(f"v0=1.20 ms, v1=1.00 ms, speedup=1.20x")
EOF
```

Spawn:
```
Agent(
  subagent_type: "qa",
  description: "Verifier smoke test",
  prompt: <<contents of skills/kernel-opt-loop/prompts/verifier.md>>

Fixture:
- project.md: /tmp/kernel-opt-loop-test/verifier-smoke/project.md
- base.py: /tmp/kernel-opt-loop-test/verifier-smoke/base.py
- triton_test_001.py: /tmp/kernel-opt-loop-test/verifier-smoke/triton_test_001.py
- auto_bench.py: /tmp/kernel-opt-loop-test/verifier-smoke/auto_bench.py
- output: /tmp/kernel-opt-loop-test/verifier-smoke/report_001.md
- python: python3
- op_name: test
- round: 1

This is a smoke test with a stub auto_bench. Verify you can produce a structurally-valid report_001.md; profiler step can be skipped (no MLU available in smoke env). Note in report that profiler was skipped.
)
```

- [ ] **Step 3: Verify smoke test output has required sections**

```bash
for section in "Correctness" "Wall time" "Device time" "device_ratio" "Upbound gap" "Noise check" "Stop recommendation"; do
  grep -qF "$section" /tmp/kernel-opt-loop-test/verifier-smoke/report_001.md && echo "OK: $section" || echo "MISSING: $section"
done
```

Expected: all 7 `OK`. If any `MISSING`, fix `prompts/verifier.md` and re-spawn.

- [ ] **Step 4: Cleanup + commit**

```bash
rm -rf /tmp/kernel-opt-loop-test
git add skills/kernel-opt-loop/prompts/verifier.md
git commit -m "skills: add Verifier role prompt for kernel-opt-loop"
```

Verify: `git log --oneline -1` shows the commit.

---

## Task 8: Rewrite `SKILL.md` as orchestrator guide + cleanup

The new `SKILL.md` is the main session's guide: how to spawn the team, run Phase 0 + Phase N rounds, handle round boundaries, commit, stop, resume. Per spec §3 (architecture) + §5 (termination) + §6 (resume) + §7 (multi-project) + §9 (KB hook) + §10 (no auto-migration).

**Files:**
- Modify: `skills/kernel-opt-loop/SKILL.md` (full rewrite — read `SKILL.md.legacy` only if needed for cross-reference, but the new content is the source of truth).
- Delete: `skills/kernel-opt-loop/SKILL.md.legacy` (after SKILL.md is rewritten and verified).

**Interfaces:**
- Consumes: `prompts/{designer,coder,verifier}.md` (Tasks 5-7), `references/{bottleneck-judgment,invariants,anti-patterns,project-template}.md` (Tasks 2-4).
- Produces: the orchestrator guide main session reads when user invokes `kernel-opt-loop` skill.

- [ ] **Step 1: Write the new `SKILL.md`**

Full content structure:

```markdown
---
name: kernel-opt-loop
description: [updated to reflect team-based architecture — keep trigger phrases "optimize operator X" + Triton + auto_bench]
---

# Kernel Opt Loop — Orchestrator Guide

[1-paragraph intro: this skill drives an iterative kernel optimization loop with three team-embodied roles — Designer (decision), Coder (implementation), Verifier (runtime feedback). Main session orchestrates; team-internal P2P for round work; main session intervenes at round boundaries.]

## When to use
[Copy from spec §2 non-goals + original skill's "When to use" section, adapted]

## Required inputs
[From spec §2 + original skill's "Required inputs": base.py, auto_bench.py, python interpreter, target device. Note: base.py is user-provided, do not write yourself.]

## Architecture
[Summarize spec §3.1 + §3.2: three roles + prompts/ + built-in subagent_type + hybrid team communication. Reference prompts/{designer,coder,verifier}.md + references/.]

## Phase 0 — Setup

1. Verify required inputs exist (base.py, auto_bench.py, python, device). Ask user for missing pieces.
2. Create branch: `git checkout -b <op-name>-opt` from master.
3. `TeamCreate(team_name: "<op-name>-opt")`.
4. Spawn three teammates:
   - `Agent(subagent_type: "architect", team_name: "<op-name>-opt", name: "designer", prompt: <contents of skills/kernel-opt-loop/prompts/designer.md> + "Phase 0: read base.py at <path>, write project.md per references/project-template.md. Output project.md path.")`
   - `Agent(subagent_type: "developer", team_name: ..., name: "coder", prompt: <contents of prompts/coder.md>)` — idle until Designer sends a decision.
   - `Agent(subagent_type: "qa", team_name: ..., name: "verifier", prompt: <contents of prompts/verifier.md>)` — idle until Coder sends a kernel.
5. Wait for Designer to write `project.md` + `state/designer_state.md`.
6. Read project.md yourself. Verify Section 1-5 are filled. Ask user to confirm upbound (Section 2).
7. Commit Phase 0: `git add <op>/base.py <op>/project.md && git commit -m "<op>: add eager baseline"`.

## Phase N — Round N (N ≥ 1)

Each round is ONE bottleneck, ONE .py file, ONE decision+report pair, ONE commit. Driven by team-internal P2P; main session intervenes at boundaries.

### Round-internal flow (team P2P, no main session)
1. Designer reads previous `rounds/report_<NNN-1>.md` + `state/designer_state.md`, scans `references/anti-patterns.md`, writes `rounds/decision_NNN.md`, SendMessage Coder.
2. Coder reads decision, writes `triton_<op>_<NNN>.py`, self-checks (AST parse, ModelNew contract), SendMessage Verifier.
3. Verifier runs auto_bench + profiler, writes `rounds/report_NNN.md`, SendMessage Designer (with optional shutdown_request).

### Round-boundary actions (main session)
1. Receive Verifier's idle notification (report_NNN.md written).
2. Read report_NNN.md.
3. If Verifier sent `shutdown_request`:
   - Wait up to 30s for Designer's `shutdown_response`.
   - If `approve=true`: commit final round, write final entry to project.md overview table, `TeamDelete`. Skill ends.
   - If `approve=false`: confirm continuation (Designer has a new decision); team persists; loop back to step 1 of next round.
4. If no shutdown_request:
   - Append one row to project.md overview table (`| triton_<op>_<NNN> | <wall> ms | <device> us | <rel-to-prev>x | <rel-to-base>x |`).
   - Commit: `git add <op>/triton_<op>_<NNN>.py <op>/rounds/decision_NNN.md <op>/rounds/report_NNN.md <op>/project.md && git commit -m "<op>: v<NNN> <short method>, <rel-to-base>x"`.
   - SendMessage Designer "round N complete, start round N+1".

### Abort path (Designer emitted `decision: abort`)
- No Coder or Verifier run.
- Commit `rounds/decision_NNN.md` alone with commit message `<op>: v<NNN> abort (no viable ≥5% path)`.
- Update `team-state.md` "abort streak = K".
- If K ≥ 3: emit shutdown (diminishing-returns hard stop).
- Otherwise: signal Designer to try a different direction next round.

## Stop criteria
[Copy spec §5.1 five criteria table + §5.2 mechanical flow.]

## Resume
[Copy spec §6 resume design: team-state.md manifest fields, per-stop_reason eligibility table, resume flow steps 1-5.]

## Multi-project structure
[Copy spec §7 file structure: skill-level (Tier 1), project-level (Tier 2), role-level (Tier 3) + knowledge lift mechanism.]

## Knowledge base hook (v1)
[Copy spec §9: Designer scans references/anti-patterns.md before picking path; records hit/miss in decision_NNN.md KB hook section. No external KB lookup in v1.]

## Migration
[Copy spec §10: existing groupedtopk/log.md + fused_moe/log.md are NOT auto-migrated. Future projects use new structure from Phase 0.]

## References
- [bottleneck-judgment.md](references/bottleneck-judgment.md) — bottleneck class procedure (preserved from current skill)
- [project-template.md](references/project-template.md) — skeleton for project.md
- [invariants.md](references/invariants.md) — code invariants (AST filter, fast_libentry, output caching, etc.)
- [anti-patterns.md](references/anti-patterns.md) — cross-project failure patterns (seeded from groupedtopk)
- See `groupedtopk/log.md` and `fused_moe/log.md` in this repo for worked examples (legacy format — pre-restructure)
```

- [ ] **Step 2: Verify SKILL.md covers all spec sections**

```bash
for section in "When to use" "Required inputs" "Architecture" "Phase 0" "Phase N" "Stop criteria" "Resume" "Multi-project structure" "Knowledge base hook" "Migration" "References"; do
  grep -qF "## $section" skills/kernel-opt-loop/SKILL.md && echo "OK: $section" || echo "MISSING: $section"
done
```

Expected: all 11 `OK`. Fix any `MISSING`.

- [ ] **Step 3: Verify frontmatter is valid**

```bash
head -5 skills/kernel-opt-loop/SKILL.md
```

Expected: starts with `---`, has `name: kernel-opt-loop`, has `description:` line. The description should retain trigger phrases from the original ("optimize operator X", "Triton + auto_bench").

- [ ] **Step 4: Delete `SKILL.md.legacy`**

```bash
git rm skills/kernel-opt-loop/SKILL.md.legacy
```

Verify: `ls skills/kernel-opt-loop/SKILL.md.legacy 2>&1` shows "No such file or directory".

- [ ] **Step 5: Commit**

```bash
git add skills/kernel-opt-loop/SKILL.md
git commit -m "skills: rewrite kernel-opt-loop SKILL.md as orchestrator guide"
```

Verify: `git log --oneline -1` shows the commit.

---

## Task 9: Sync to `~/.claude/skills/` + verify final structure

Sync the restructured skill from repo to `~/.claude/skills/` so Claude Code can load it. Verify final structure.

**Files:**
- Sync: `skills/kernel-opt-loop/` (repo) → `~/.claude/skills/kernel-opt-loop/` (loaded by Claude Code).

- [ ] **Step 1: Sync repo to ~/.claude/skills/**

```bash
rsync -av --delete skills/kernel-opt-loop/ ~/.claude/skills/kernel-opt-loop/
```

`--delete` removes the old `references/log-template.md` from `~/.claude/skills/` since it's gone from the repo.

- [ ] **Step 2: Verify final structure**

```bash
find ~/.claude/skills/kernel-opt-loop -type f | sort
```

Expected output (exact):
```
/home/lipenghui/.claude/skills/kernel-opt-loop/SKILL.md
/home/lipenghui/.claude/skills/kernel-opt-loop/prompts/coder.md
/home/lipenghui/.claude/skills/kernel-opt-loop/prompts/designer.md
/home/lipenghui/.claude/skills/kernel-opt-loop/prompts/verifier.md
/home/lipenghui/.claude/skills/kernel-opt-loop/references/anti-patterns.md
/home/lipenghui/.claude/skills/kernel-opt-loop/references/bottleneck-judgment.md
/home/lipenghui/.claude/skills/kernel-opt-loop/references/invariants.md
/home/lipenghui/.claude/skills/kernel-opt-loop/references/project-template.md
```

If `log-template.md` appears, the `--delete` didn't take; re-run rsync.

If `SKILL.md.legacy` appears, the `--delete` didn't take; re-run rsync.

- [ ] **Step 3: Verify skill loads**

In a fresh Claude Code session (user does this manually since current session has cached skills), the skill should appear in `/skills` list with the new description. Note: in current session, the skill description in the system reminder will still show the OLD description (cached at session start). That's expected — verification happens in a fresh session.

For this session: verify the file is readable.
```bash
head -10 ~/.claude/skills/kernel-opt-loop/SKILL.md
```

Expected: frontmatter starts with `---`, has `name: kernel-opt-loop`.

- [ ] **Step 4: No commit needed (sync is filesystem-only, not git-tracked)**

`~/.claude/skills/` is outside the repo; no git commit. Repo state already committed in Task 8.

---

## Task 10: End-to-end Phase 0 + 1 round smoke test on fused_moe

Manual acceptance test: run the restructured skill on the existing `fused_moe/base.py` operator. Verify Phase 0 produces `project.md`, and Round 1 produces a valid `decision_001.md` + `triton_fused_moe_001.py` + `report_001.md`. Stop short of full optimization — just verify the workflow mechanics.

**Prerequisites:**
- MLU environment available (python with torch + torch_mlu + triton).
- `fused_moe/base.py` exists (it does — confirmed during investigation).
- `auto_bench.py` exists at repo root (it does).

**Files:**
- Read: `fused_moe/base.py`
- Expect produced: `fused_moe/project.md`, `fused_moe/team-state.md`, `fused_moe/rounds/decision_001.md`, `fused_moe/triton_fused_moe_001.py`, `fused_moe/rounds/report_001.md`, `fused_moe/state/designer_state.md` (+ coder/verifier state).

- [ ] **Step 1: Verify environment**

```bash
ls fused_moe/base.py auto_bench.py
<python from fused_moe/project.md or ask user> -c "import torch, torch_mlu, triton; print('env OK')"
```

If env not available, mark this task as "deferred — requires MLU env" and stop. The skill is structurally complete; this is acceptance testing.

- [ ] **Step 2: Invoke the restructured skill**

In a fresh Claude Code session (so the new skill description loads), invoke `kernel-opt-loop` with input: "optimize operator fused_moe".

Or, in current session, manually drive the workflow per the new SKILL.md:
1. Create branch `fused_moe-opt` (or reuse existing if present).
2. TeamCreate `fused_moe-opt`.
3. Spawn three teammates per Task 8 Phase 0 step 4.
4. Wait for Designer to write `fused_moe/project.md`.
5. Read project.md, verify Sections 1-5 filled, ask user to confirm upbound.
6. Commit Phase 0.
7. SendMessage Designer "start round 1".
8. Wait for Verifier's idle notification (report_001.md written).
9. Read report_001.md, append overview table row, commit round 1.

- [ ] **Step 3: Verify Phase 0 artifacts**

```bash
test -f fused_moe/project.md && echo "project.md OK" || echo "MISSING project.md"
test -f fused_moe/team-state.md && echo "team-state.md OK" || echo "MISSING team-state.md"
test -f fused_moe/state/designer_state.md && echo "designer_state.md OK" || echo "MISSING designer_state.md"
```

Expected: all 3 `OK`.

For project.md, verify required sections:
```bash
for section in "1. 固定问题与测试口径" "2. Upbound 定义" "3. 当前结果总览" "4. 复现命令" "5. Checkpoint"; do
  grep -qF "$section" fused_moe/project.md && echo "OK: $section" || echo "MISSING: $section"
done
```

Expected: all 5 `OK`.

- [ ] **Step 4: Verify Round 1 artifacts**

```bash
test -f fused_moe/rounds/decision_001.md && echo "decision_001.md OK" || echo "MISSING"
test -f fused_moe/triton_fused_moe_001.py && echo "triton_001.py OK" || echo "MISSING"
test -f fused_moe/rounds/report_001.md && echo "report_001.md OK" || echo "MISSING"
python -c "import ast; ast.parse(open('fused_moe/triton_fused_moe_001.py').read()); print('AST OK')"
```

Expected: all 4 `OK`.

For report_001.md, verify required sections:
```bash
for section in "Correctness" "Wall time" "Device time" "device_ratio" "Upbound gap" "Noise check" "Stop recommendation"; do
  grep -qF "$section" fused_moe/rounds/report_001.md && echo "OK: $section" || echo "MISSING: $section"
done
```

Expected: all 7 `OK`.

- [ ] **Step 5: Verify git history**

```bash
git log --oneline -5
```

Expected: see commits for Phase 0 (add eager baseline) and Round 1 (v001 + method + speedup).

- [ ] **Step 6: TeamDelete + final commit**

If the team is still alive:
```
SendMessage(to: "designer", message: {type: "shutdown_request", reason: "smoke test complete"})
```
Wait for approve=true, then `TeamDelete`.

```bash
git add fused_moe/project.md fused_moe/team-state.md fused_moe/rounds/ fused_moe/state/ fused_moe/triton_fused_moe_001.py
git commit -m "skills: kernel-opt-loop smoke test on fused_moe (Phase 0 + Round 1)"
```

Verify: `git log --oneline -1` shows the smoke test commit.

- [ ] **Step 7: Mark skill restructure complete**

```bash
git checkout master
git merge kernel-opt-loop-restructure
```

If merge clean, the restructure is complete. If conflicts, resolve manually (likely only on `docs/superpowers/specs/` and `docs/superpowers/plans/` paths if other work landed on master).

---

## Self-Review

After writing the complete plan, I checked it against the spec:

**1. Spec coverage:**
- §3.1 architecture (3 roles + prompts + built-in subagent_type) → Tasks 5-7 (prompts) + Task 8 (SKILL.md orchestrator).
- §3.2 hybrid team communication → Task 8 SKILL.md (Phase 0 spawn + Phase N round-boundary actions + abort path).
- §4.1 Designer contract → Task 5.
- §4.2 Coder contract → Task 6.
- §4.3 Round operation rules (revision loop, failure handling, visibility) → embedded in Tasks 5/6/7 prompts.
- §4.4 Verifier contract → Task 7.
- §5 termination (5 criteria + mechanical flow) → Task 7 (Verifier emits) + Task 8 (SKILL.md documents flow).
- §6 resume (team-state.md manifest + per-role state + per-stop_reason eligibility) → Task 8 SKILL.md (Resume section).
- §7 multi-project structure (3 tiers + knowledge lift) → Task 8 SKILL.md.
- §8 log.md fate → Tasks 4 (project-template.md replaces log-template.md) + 8 (SKILL.md orchestrates per-round files in rounds/).
- §9 KB hook → Task 5 (Designer prompt has KB hook section) + Task 8 (SKILL.md documents hook).
- §10 migration → Task 8 SKILL.md (no auto-migration).
- §11 resolved decisions (subagent type, etc.) → embedded across Tasks 5-8.
- §11 open questions (slash command, project.md format, invariants content, anti-patterns seed, resume sanity check form) → addressed: invariants content = Task 2 checklist; anti-patterns seed = Task 3; project.md format = Task 4 (markdown, not hybrid); slash command = deferred (SKILL.md relies on user typing "optimize operator X" — matches current skill's invocation pattern); resume sanity check = embedded in Task 5 Designer prompt ("Round 0 special case" handles initial, resume sanity check is implicit in reading state files).

**2. Placeholder scan:** No "TBD", "TODO", "implement later", "similar to Task N" patterns found. All steps have concrete content. Test code is real (smoke spawn with fixtures).

**3. Type consistency:**
- `decision_NNN.md` referenced consistently in Tasks 5/6/7/8.
- `report_NNN.md` referenced consistently.
- `team-state.md` referenced consistently.
- `state/{designer,coder,verifier}_state.md` referenced consistently.
- File paths: `skills/kernel-opt-loop/...` (repo) for implementation; `<op>/...` (project-level) for skill output. Both consistent.
- subagent_type values: `architect` (Designer), `developer` (Coder), `qa` (Verifier) — consistent across Tasks 5/6/7/8.

No issues found. Plan is complete.
