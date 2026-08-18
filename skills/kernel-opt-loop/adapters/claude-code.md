# Claude Code Runtime Adapter

This adapter translates the six kernel-opt-loop orchestration operations into
Claude Code agent-team behavior. `SKILL.md` owns workflow semantics and resolves
the complete role bootstrap before any operation here runs. Do not paste role
contracts into this adapter or add role-specific behavior here.

```yaml
runtime_capabilities:
  persistent_role_session: true
  effective_context_mode: continuation
  autonomous_scope: one-live-orchestrator-session
```

Compatibility evidence:

- [Agent teams](https://code.claude.com/docs/en/agent-teams)
- [Tools reference](https://code.claude.com/docs/en/tools-reference)

## Preflight and Fallback

Before starting a role, confirm all of the following:

1. Claude Code is version `>= 2.1.178`.
2. Agent teams are enabled with
   `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in the environment or settings.
3. The current session exposes agent-team spawning and `SendMessage`.

If any check fails, Orchestrator selects the sequential main-session fallback.
The main session then executes the same role contract with the same ownership
and artifact gates. The fallback never starts a nested Claude CLI process.

Agent teams are session-local. Durable project artifacts remain authoritative
for restart and resume.

## Continuation and Cold Rehydrate

The initial prompt carries the full resolved bootstrap. A continuing teammate
receives a compact bootstrap delta: its section from
`role-context-template.md`, the phase and new inputs, changed canonical
pointers, and the terminal evidence that caused the next transition. The same
teammate identity is reused while the lead session lives, but durable artifacts
remain the only authority.

Perform a cold rehydrate if the teammate is absent, reports lost context, has a
canonical pointer or run-fingerprint mismatch, or if a policy/run-epoch change
invalidates its bootstrap. Spawn a replacement from the durable role context and
changed artifacts only, then put it through the ordinary artifact gate.

During three-round reconciliation, the lead compares each role's compact state
with `team-state.md` and the terminal artifact chain. A mismatch requires a
cold rehydrate before new work is sent.

## Sequential Fallback

When agent teams are not available, use this effective runtime profile:

```yaml
runtime_capabilities:
  persistent_role_session: false
  effective_context_mode: rehydrate
  autonomous_scope: one-live-orchestrator-session
```

The main session rebuilds the relevant role from `role-context-template.md` and
changed artifacts at every dispatch, while retaining the same ownership,
artifact gates, terminal routing, and global-stop evaluation.

## Operation Mapping

### `start_role`

Spawn a named teammate directly through Claude Code's Agent capability. Use the
deterministic teammate name `designer`, `coder`, or `verifier`, and pass the
resolved bootstrap from `SKILL.md` as that teammate's initial prompt. Spawn at
most one teammate identity for each role and reuse it while the session lives.

The teammate starts with an independent context window: lead conversation
history is not inherited. The resolved bootstrap therefore contains absolute
skill/project paths, current phase, inputs, outputs, ownership, and reporting
instructions. Do not replace it with a user-authored role prompt.

### `continue_idle_role`

Use `SendMessage` to send the next resolved bootstrap or bounded follow-up task
to that role's existing idle teammate. Continue the same identity; do not spawn
a second teammate for the role.

### `send_advisory`

Use `SendMessage` to steer a teammate that is already running. Advisory context
may clarify inputs or point to a durable artifact, but it cannot change the
role's ownership or bypass a workflow gate. Every response that requests a state
change is routed to the lead acting as Orchestrator.

### `wait_for_completion`

Rely on automatic teammate message delivery and idle notifications. Do not poll
the generated task state or repeatedly query teammate status. A role is complete
only when its required artifact exists, passes the gate in `SKILL.md`, and the
lead has received its classification.

### `inspect_roles`

Use Claude Code's agent panel, delivered teammate messages, idle notifications,
and shared task view for occasional diagnostics. Inspection does not replace
artifact validation and must not become a polling loop.

### `end_workflow`

Ask each live teammate through `SendMessage` to shut down after its current safe
boundary. Wait for acknowledgements when available, then end the lead session.
Session cleanup is automatic. Do not manually edit or remove Claude-generated
team configuration or task directories.

## Invariants

- Only the lead/Orchestrator commits and changes canonical pointers, counters,
  phase, or round state.
- Teammates write only the files assigned by their resolved bootstrap.
- Automatic messages are advisory until the lead validates the referenced
  durable artifact.
- Never edit Claude-generated team configuration or task directories.
- Resume from durable artifacts through `SKILL.md`, never from assumed teammate
  memory.
