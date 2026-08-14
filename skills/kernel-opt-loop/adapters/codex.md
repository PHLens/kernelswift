# Codex Runtime Adapter

This adapter translates the six kernel-opt-loop orchestration operations into
Codex collaboration tools. `SKILL.md` owns workflow semantics and supplies a
fully resolved bootstrap. The bootstrap defines the role; an optional Codex
agent type never changes the role contract, ownership, inputs, or outputs.

```yaml
runtime_capabilities:
  persistent_role_session: true
  effective_context_mode: continuation
  autonomous_scope: one-live-orchestrator-session
```

## Availability and Identity

At workflow start, confirm that Codex collaboration tools are exposed. When
they are unavailable, Orchestrator selects the sequential main-session fallback
and executes the same role contracts and artifact gates itself. The fallback
does not launch a nested Codex CLI process.

With collaboration available, create at most one persistent agent identity for
each role: `designer`, `coder`, and `verifier`. Keep the returned agent target in
session-local orchestration state and reuse it for later rounds. Durable project
artifacts, not agent identity or conversation memory, remain authoritative for
resume in a new session.

## Continuation and Cold Rehydrate

The first dispatch uses the complete resolved bootstrap. For a continuing role,
Orchestrator sends a compact bootstrap delta: the role's section from
`role-context-template.md`, the new phase and inputs, changed canonical
pointers, and the terminal evidence that caused the transition. This preserves
the role's persistent identity without treating its context as the source of
truth.

A cold rehydrate is required when the target is unavailable, the role reports
lost context, a canonical pointer or run fingerprint no longer matches, or a
policy/run-epoch change invalidates the prior bootstrap. Recreate the role from
the durable role context and changed artifacts only, then resume through the
same artifact gate as a normal continuation.

At each three-round reconciliation, Orchestrator compares the role's compact
state against `team-state.md` and the terminal artifact chain. A mismatch
requires a cold rehydrate before the role receives more work.

## Sequential Fallback

When collaboration tools are unavailable, use this effective runtime profile:

```yaml
runtime_capabilities:
  persistent_role_session: false
  effective_context_mode: rehydrate
  autonomous_scope: one-live-orchestrator-session
```

The main session rehydrates the relevant role from `role-context-template.md`
and changed artifacts at each dispatch. It preserves the same ownership,
artifact gates, terminal routing, and global-stop evaluation.

## Operation Mapping

| Common operation | Codex action |
|---|---|
| `start_role` | Use `spawn_agent` with the role's deterministic task name, the resolved bootstrap, and `fork_turns="none"`. |
| `continue_idle_role` | Use `followup_task` for the existing idle role identity. |
| `send_advisory` | Use `send_message` only when that role is already running. |
| `wait_for_completion` | Use `wait_agent` with a multi-minute timeout. |
| `inspect_roles` | Use `list_agents` for diagnostics only. |
| `end_workflow` | Let completed roles finish; use `interrupt_agent` only for a stuck role. |

### `start_role`

Call `spawn_agent` once for a role, using `task_name="designer"`,
`task_name="coder"`, or `task_name="verifier"`. Pass the exact resolved
bootstrap as `message` and set `fork_turns="none"` so parent conversation
history is not inherited. The bootstrap carries absolute skill/project paths,
phase, inputs, outputs, ownership, and completion routing.

If the runtime exposes specialized types, Orchestrator may prefer `architect`
for Designer, `developer` for Coder, and `qa` for Verifier. Otherwise it uses
`default`. This is only a scheduling hint: the bootstrap, not `agent_type`,
defines the role.

### `continue_idle_role`

Call `followup_task` with the existing role target and the next resolved
bootstrap or bounded repair task. This starts a new turn on the same persistent
identity. Do not create a replacement agent merely because the role is idle.

### `send_advisory`

Call `send_message` only for a role known to be running. Advisory context can
clarify an input or point to a durable artifact; it cannot change file ownership
or bypass an artifact gate. To start work on an idle role, use
`continue_idle_role` instead.

Every state-changing response returns to the root agent acting as Orchestrator.
Roles do not update canonical pointers, counters, phases, or round allocation.

### `wait_for_completion`

Call `wait_agent` with a multi-minute timeout, such as
`timeout_ms=180000`. Wait for mailbox delivery rather than busy polling. A role
is complete only after Orchestrator receives its classification and validates
the required durable artifact.

### `inspect_roles`

Call `list_agents` only for occasional diagnostics, such as resolving whether a
target is running or idle after an unexpected timeout. Do not use it as a poll
loop and do not treat agent status as an artifact gate.

### `end_workflow`

Allow completed or idle roles to finish naturally. Call `interrupt_agent` only
when a running role is stuck and cannot reach a safe boundary. An interruption
does not authorize deletion or rewriting of that role's durable artifacts.

## Invariants

- Reuse one agent target per role for the lifetime of the collaboration session.
- Dispatch only the resolved bootstrap or a bounded continuation derived from
  it; do not duplicate role behavior in this adapter.
- Orchestrator alone validates artifacts, commits, and mutates workflow state.
- Resume from `team-state.md` and the artifact chain, never assumed agent memory.
