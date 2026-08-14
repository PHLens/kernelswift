# kernel-opt-loop Skill Restructure Design

**Created**: 2026-08-13

**Last updated**: 2026-08-14

**Status**: Approved design, written spec pending user review

**Scope**: Architectural restructure of `skills/kernel-opt-loop/`

## 1. Motivation

The current `kernel-opt-loop` skill combines optimization design, implementation,
runtime verification, state transitions, and user decisions in one long session.
Evidence from `fused_moe` and `groupedtopk` shows four recurring failures:

- role-specific context accumulates in one conversation and becomes difficult to
  reason about;
- rejected candidates can be mistaken for the next implementation baseline;
- runtime numbers are recorded without a strong causal link to the optimization
  hypothesis;
- failed approaches are preserved as prose but are not exposed through a stable,
  reusable contract.

Kernel optimization is therefore modeled as four responsibilities:

- **Designer** selects one falsifiable optimization and specifies it;
- **Coder** realizes that specification for one concrete target profile;
- **Verifier** executes the candidate and produces attributable runtime evidence;
- **Orchestrator** owns state transitions, routing, commits, stopping, and user
  interaction.

The workflow must run under both Claude Code and Codex without changing its
artifact or state-machine semantics.

## 2. Goals and non-goals

### 2.1 Goals

- Provide runtime-neutral Designer/Coder/Verifier contracts with Claude Code and
  Codex adapters.
- Persist enough state to resume safely in a fresh session.
- Make the last accepted implementation the only canonical implementation.
- Represent kernel/dataflow intent in a validated Unified Sketch while retaining
  explicit support for host, launcher, wrapper, cache, and lifecycle changes.
- Bind every decision to the actual Triton backend, distribution, version, and
  device architecture used by the project.
- Make Verifier results answer the current round's hypothesis rather than merely
  reporting aggregate time.
- Preserve every completed round, including rejected designs and candidates, as
  an auditable artifact chain.

### 2.2 Non-goals for v1

- A backend-neutral compiler IR or deterministic Sketch-to-kernel compiler.
- Automatic backend/DSL selection or fallback routing.
- More than one active candidate or optimization round at a time.
- External KernelWiki lookup or publication.
- Deep automatic profiler interpretation such as critical-path recovery,
  occupancy analysis, or cross-project trace mining.
- Migration of existing project logs unless explicitly requested.
- Changes to user-owned `base.py` or benchmark semantics.

The backend-neutral IR, KernelWiki, and deep-profiler directions are retained in
§15 as explicit future work, not as v1 placeholders or implementation tasks.

## 3. Design principles

1. **One canonical pointer.** `last_accepted_kernel` is the only implementation
   from which a new candidate may start.
2. **One round, one intervention.** A decision targets one bottleneck with one
   falsifiable causal hypothesis. A mixed kernel/host change is allowed only when
   the pieces are inseparable and separately observable.
3. **Immutable decisions.** Once a validated decision is handed to Coder, design
   changes require a new round.
4. **Unified schema, target-bound instance.** The Sketch format is shared, but
   every concrete Sketch is interpreted against one recorded target profile and
   runtime fingerprint.
5. **Evidence before transition.** A state transition is valid only after its
   required artifacts pass deterministic checks.
6. **Runtime measurements have one owner.** Verifier alone produces authoritative
   correctness, benchmark, and profiler results.
7. **Classification is separate from routing.** Roles classify their results;
   Orchestrator applies a deterministic routing table.
8. **Environment failures are not optimization failures.** They block and preserve
   evidence without consuming a round or modifying progress streaks.

## 4. Runtime-neutral architecture

### 4.1 Source of truth

`skills/kernel-opt-loop/SKILL.md` is the runtime-neutral orchestrator contract. It
defines Phase 0, Round N, artifact gates, transitions, stop/resume behavior, and a
common role bootstrap. It contains no active Claude Code- or Codex-specific tool
syntax.

Runtime-specific orchestration lives in:

- `adapters/claude-code.md`
- `adapters/codex.md`

An adapter maps the common operations—start a role, continue an idle role, send
advisory context, wait for completion, inspect status, and end the workflow—to the
capabilities of the active runtime. Portable default/general-purpose agents are
sufficient; runtime-local `architect`, `developer`, or `qa` types are optional
optimizations, not dependencies.

If multi-agent support is unavailable, the main session executes the same role
contracts sequentially with identical file ownership and transitions. The
fallback must not start nested CLI agent processes.

### 4.2 Common bootstrap

The user does not provide a per-role prompt. Orchestrator selects the role and
constructs a compact bootstrap from the current state. The bootstrap template is
defined in `SKILL.md`, while the full role behavior remains in `prompts/*.md`:

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

Every placeholder is resolved before dispatch. The full role contract is not
pasted into the bootstrap message.

### 4.3 Ownership

| Component | Sole writable responsibility |
|---|---|
| Orchestrator | `team-state.md`, project overview rows, round transitions, commits |
| Designer | current uncommitted decision and `state/designer_state.md` |
| Coder | current candidate, `rounds/coder_result_NNN.md`, `state/coder_state.md` |
| Verifier | current report/status, profiler outputs, `state/verifier_state.md` |

Roles may exchange advisory information, but every state-changing response is
sent to Orchestrator. No role may update canonical pointers, counters, or start a
new round.

## 5. Target identity and profile contract

### 5.1 Target identity

`target_dsl: triton_mlu` conflates a language family with a backend. v1 records
them separately:

```yaml
implementation:
  language: triton
  backend: mlu
  target_profile: triton_mlu
runtime_fingerprint:
  triton_distribution: <discovered in Phase 0>
  triton_version: <discovered in Phase 0>
  backend_target: <reported by the active driver/compiler>
  backend_version: <when discoverable>
  device_arch: <discovered in Phase 0>
```

The fingerprint is an observed project fact, not a value assumed by the skill.
A missing runtime or profile mismatch is an environment/configuration problem. A
construct unsupported by a correctly matched profile is a capability miss.

### 5.2 v1 target profile

v1 ships one complete profile: `prompts/coder_targets/triton_mlu.md`. It is a
self-contained capability and realization contract containing:

- identity and runtime matching conditions;
- supported, constrained, unsupported, and unknown primitives;
- dtype, shape, alignment, architecture, and tuning restrictions;
- runtime, import, launcher, stream, and buffer-lifecycle conventions;
- target-specific pitfalls and allowed fallbacks;
- evidence or a local reproduction method for every backend-specific claim.

The profile describes accurate semantics and known capabilities; it is not a
line-by-line lowering table. In particular, it must not equate pointer
construction with register allocation or a value-producing tensor operation
with shared-memory placement.

Unknown capabilities are not treated as supported. Coder either proves them with
the declared environment or returns a capability miss.

v1 does not ship inactive CUDA, HIP, Ascend, or TileLang stubs. A future target
lands with a complete profile plus contract fixtures. Profile discovery must not
hardcode a list of hypothetical files.

## 6. Decision contract

### 6.1 Normative and explanatory content

`rounds/decision_NNN.md` contains the following sections:

1. **Metadata** — round, accepted reference and report, bottleneck evidence,
   implementation identity, target profile, and runtime fingerprint reference.
2. **Optimization Intent** — normative objective, allowed change boundary,
   behavior that must remain invariant, and one falsifiable intervention.
3. **Unified Sketch** — normative kernel/dataflow structure when that structure
   changes.
4. **Host Plan** — normative host/wrapper/launcher/cache/lifecycle changes when
   those areas change.
5. **Evaluation Contract** — normative observables that connect the intervention
   to correctness and runtime effects.
6. **Pitfalls and anti-pattern consultation** — relevant known failures and why
   they do or do not apply.
7. **Rationale/Evidence** — explanatory context that cannot silently introduce
   additional implementation requirements.

Coder implements `Optimization Intent`, `Unified Sketch`, and `Host Plan`
together. The old rule that Coder must ignore optimization prose and translate
only the Sketch is invalid because it cannot express host-side optimization
semantics.

### 6.2 Unified Sketch

Unified Sketch uses one fenced `sketch` block with four ordered subsections:

- **D — Declarations/Data domains**: logical tensors, shapes, dtypes, layouts,
  partitions, and memory intent;
- **O — Operations**: primitive computation and explicit data dependencies;
- **C — Control/parallel structure**: loops, guards, program decomposition, and
  synchronization intent;
- **H — Target-scoped hints**: optional tuning directives whose legality and
  meaning come from the selected target profile.

D/O/C share one schema across target profiles, but the concrete instance remains
target-bound. H is never assumed portable. A hint such as a worker or pipeline
count must be checked against the selected profile and current architecture.

Conditional requirements are:

- kernel/dataflow change: Unified Sketch required;
- pure host/wrapper change: Unified Sketch is `N/A` with a reason, Host Plan
  required;
- mixed change: both Unified Sketch and Host Plan required.

The Sketch is a declarative implementation contract for an LLM Coder. It is not
a compiler IR in v1 and must not pretend that abstract memory labels mechanically
map to concrete storage.

### 6.3 Host Plan

Host Plan records at least:

- affected wrapper/launcher/model scope;
- state owner and lifetime;
- allocation and reuse behavior;
- cache key, invalidation, and concurrency assumptions;
- device/stream/context behavior;
- behavior that must remain unchanged.

This covers existing optimization paths such as launcher selection, output
buffer caching, routing fusion, and removal of redundant device context handling.

### 6.4 Evaluation Contract

Every proceeding decision contains a machine-checkable conceptual contract:

```yaml
hypothesis_id: H-<round>
intervention: <one falsifiable change>
expected_causal_chain:
  - <mechanism observable changes>
  - <primary outcome changes>
primary_metric:
  name: wall_time
  expected_improvement: ">= 5%"
mechanism_observables:
  - name: <observable>
    expectation: <direction or bound>
guardrails:
  - correctness: pass
  - <semantic or lifecycle invariant>
profiling_level: summary | targeted | deep-on-demand
```

The exact observables depend on the intervention: kernel count, target-kernel
time, library-kernel removal, allocation behavior, launcher overhead, or another
profile-supported measurement.

### 6.5 Decision validation

Orchestrator runs deterministic `validate_decision.py` before coding. It checks:

- Markdown fences, required sections, and schema types;
- kernel-only, host-only, and mixed conditional requirements;
- D/O/C/H grammar and target-hint syntax;
- reference, profile, and fingerprint links;
- Host Plan ownership/lifecycle fields;
- Evaluation Contract completeness;
- duplicates, unknown sections, and dangling references.

It does not decide whether a backend can realize a primitive. Designer owns
design completeness; Coder owns capability preflight.

## 7. Role contracts and routing

### 7.1 Designer

Designer reads the accepted canonical implementation/report, the most recent
completed evidence, project invariants, anti-patterns, bottleneck guidance, and
the selected target profile. It:

- chooses one bottleneck and one falsifiable intervention;
- quantifies expected improvement against the accepted reference;
- writes the complete decision contract from §6;
- records consulted anti-patterns and rejected alternatives;
- emits a complete abort decision when a stable 5% improvement cannot be
  justified.

Designer does not write code, invent runtime results, or revise a decision after
it has entered coding.

### 7.2 Coder

Coder reads the immutable decision, selected target profile, invariants, base
contract, and `last_accepted_kernel`. It performs profile matching and capability
preflight before implementation.

Coder may make bounded repairs to syntax, imports, or other non-semantic defects.
It may not change the algorithm, dataflow, Host Plan lifecycle, or evaluation
intent. Its structured result is exactly one of:

- `candidate-ready`;
- `design-revision-required`, reason `major-deviation`;
- `design-revision-required`, reason `capability-miss`;
- `implementation-failed`;
- `environment-blocked`.

Small target-language adjustments that do not change normative intent are
recorded as conformance notes under `candidate-ready`; they are not separate
states. Coder never returns `accepted`.

### 7.3 Verifier

Verifier is the sole owner of authoritative runtime execution. It reads the
decision, candidate, accepted reference, project measurement regime, and
verifier state. It runs, in order:

1. conformance and correctness checks;
2. interleaved accepted-reference/candidate timing;
3. the profiling required by the Evaluation Contract;
4. result classification and stop recommendation.

Verifier does not modify candidate code or design. It writes a report that
mirrors every Evaluation Contract field and classifies the hypothesis as
`confirmed`, `partially-confirmed`, `falsified`, or `inconclusive`.

### 7.4 Deterministic routing

| Producer classification | Orchestrator action |
|---|---|
| Coder `candidate-ready` | dispatch Verifier |
| Coder `major-deviation` | complete `design-rejected`; next round to Designer |
| Coder `capability-miss` | complete `design-rejected`; next round to Designer |
| Coder `implementation-failed` | complete `candidate-failed`; next round to Designer |
| Any `environment-blocked` | preserve incident, set workflow `blocked`, report user |
| Verifier local implementation defect | return to Coder once in the same round |
| Verifier requires Sketch/Host Plan change | complete `design-rejected`; next round to Designer |
| Verifier correctness still fails after repair | complete `candidate-failed`; next round to Designer |
| Verifier `measurement-incomplete` | collect the missing declared probe; if impossible, classify the cause as design or environment before transitioning |
| Verifier correctness passes | complete `accepted` or `no-improvement` from measured result |

The one Verifier-to-Coder repair may change only implementation details within
the existing decision. Any design change closes the round first. All feedback is
written to an artifact; agent messages alone are not an audit record.

After a terminal round commit, the next Designer always receives the completed
report or Coder failure evidence. For an accepted result it analyzes the new
canonical; for a rejected result it retains the previous canonical and uses the
failed mechanism as negative evidence.

## 8. Runtime evidence and profiler policy

### 8.1 Adoption and causal verdicts

Candidate adoption requires:

- conformance and correctness PASS;
- every required guardrail PASS;
- median wall improvement of at least 5% against `last_accepted_kernel` under
  the unchanged measurement regime.

The hypothesis verdict is separate from adoption. A repeatable candidate may be
accepted even if the proposed causal explanation is falsified, but the report
must say so and the next Designer must use the corrected explanation. Missing a
required observable yields `measurement-incomplete`; Verifier cannot declare
`accepted` or `no-improvement` until the evidence gap is resolved or classified
as a design/environment block.

Profiler time is diagnostic evidence, never the authoritative wall benchmark.

### 8.2 Profiler levels

- **Level 0 — mandatory for every candidate:** correctness and interleaved paired
  wall timing.
- **Level 1 — mandatory after correctness PASS:** separately scoped reference and
  candidate device time per call, kernel count per call, and top-k kernel
  breakdown.
- **Level 2 — intent-driven:** targeted kernel, host, launcher, allocation,
  synchronization, or backend-specific probes named by Evaluation Contract.
- **Level 3 — deep on demand:** complete trace analysis only when results conflict,
  regression cannot be attributed, noise is decisive, or a stop claim requires
  stronger evidence.

All multi-iteration profiler totals are normalized to one forward call before
computing `device_ratio`. Reference and candidate scopes are summarized
independently and must never be mixed.

Verifier's report contains expected/observed/verdict for each mechanism
observable, retry history, exact reproduction commands, and an
`evidence_for_next_round` section that states observations without selecting the
next optimization.

## 9. Round state machine

### 9.1 Phases and transitions

Manifest phases are `initializing`, `ready`, `designing`, `coding`, `verifying`,
`repairing`, `measuring`, `blocked`, and `stopped`.

```text
ready → designing
  ├─ Designer abort ──────────────────────────> aborted
  └─ decision validated → coding
       ├─ major deviation / capability miss ─> design-rejected
       ├─ implementation retries exhausted ──> candidate-failed
       ├─ environment problem ────────────────> blocked (nonterminal)
       └─ candidate-ready → verifying
            ├─ local implementation defect → repairing → verifying (once)
            ├─ required evidence missing ────> measuring → verifying
            ├─ design change required ────────> design-rejected
            ├─ correctness still fails ───────> candidate-failed
            ├─ improvement < 5% ──────────────> no-improvement
            ├─ correctness + threshold pass ──> accepted
            └─ environment problem ───────────> blocked (nonterminal)
```

Only Orchestrator performs transitions. A new round starts only after the current
terminal transition is validated and committed.

### 9.2 Terminal results

| Result | Canonical effect | Required evidence |
|---|---|---|
| `accepted` | advance to candidate | decision, Coder result, candidate, Verifier report |
| `no-improvement` | unchanged | decision, Coder result, candidate, Verifier report |
| `design-rejected` | unchanged | decision plus structured Coder/Verifier reason |
| `candidate-failed` | unchanged | decision, Coder result, candidate when created, failure report when run |
| `aborted` | unchanged | complete abort decision |

`baseline` is a Phase 0 initialization result, not a normal round terminal.
`env-fail` is a timestamped blocking incident, not a completed round result.

### 9.3 Counters

- `total_rounds` increments exactly once for each completed terminal result.
- `performance_miss_streak` increments only for `no-improvement`.
- `failed_attempt_streak` increments for `design-rejected`, `candidate-failed`,
  and `aborted`.
- Each failure class resets the other class's consecutive counter.
- `accepted` resets both counters.
- Environment incidents change neither round count nor streaks.

### 9.4 Artifact immutability

- A decision is immutable after validation and coding dispatch.
- Every Coder run writes `rounds/coder_result_NNN.md` with status, reason code,
  conformance notes, source canonical, and candidate hash or failure evidence.
- A Verifier repair records before/after hashes and both attempts in one final
  report.
- Environment incidents use timestamped files and never overwrite decisions,
  Coder results, or verification reports.
- Rejected candidates remain auditable but never become a future starting point.

## 10. Phase 0 and measurement identity

Phase 0 performs these ordered actions:

1. Resolve absolute skill, project, base, harness, interpreter, and device paths.
2. Create project directories and initialize the manifest and role state files.
3. Discover and record runtime fingerprint and target profile match.
4. Designer writes the semantic/environment/measurement/upbound portions of
   `project.md`; unknown user-owned values are requested rather than invented.
5. Generate `baseline_adapter.py` without modifying `base.py`.
6. Verifier runs baseline correctness, benchmark, and scoped profiler collection,
   producing `report_000.md`.
7. Compute a deterministic measurement fingerprint from `base.py`, the harness,
   shape, dtype, device, warmup/repeat, and profiler settings.
8. Set `last_accepted_kernel: baseline_adapter.py` and commit Phase 0 artifacts.

If Phase 0 fails because of the environment, it writes a timestamped incident,
sets `phase: blocked`, and leaves accepted pointers null.

Every round compares the candidate to `last_accepted_kernel` and
`last_accepted_report`, never simply to the numerically previous round.

## 11. Stop and resume

### 11.1 Stop criteria

After every terminal transition, Orchestrator evaluates:

1. **measurement-bound** — normalized device ratio is below 5% and targeted
   evidence shows remaining host time is harness-fixed;
2. **diminishing returns** — either progress streak reaches three;
3. **upbound reached** — accepted performance enters the declared bound;
4. **resource exhausted** — configured round/time safety limit is reached;
5. **user intervention** — user requests stop.

Verifier recommends evidence-based stops; Orchestrator owns the transition.
Designer may reject a non-user stop only by supplying a concrete next-round
hypothesis expected to clear the adoption threshold. User intervention is
unconditional.

### 11.2 Resume

Teams are session-local; durable files are authoritative. Resume validates the
manifest, artifact hashes, skill version, target runtime fingerprint, and
measurement fingerprint before dispatch.

- An interrupted, uncommitted phase resumes the same round from the first missing
  valid artifact.
- An environment block resumes the same safe step and does not allocate a round.
- A target/runtime change invalidates silent reuse of a target-bound decision.
- A measurement fingerprint change requires re-establishing a comparable
  baseline before optimization continues.
- Completed decisions and reports are never reused or overwritten.

Stop-specific eligibility remains:

| Reason | Resume condition |
|---|---|
| measurement-bound | new shape or measurement regime |
| diminishing-returns | new hypothesis, evidence, or skill capability |
| upbound-reached | explicit stretch goal |
| resource-exhausted | user acknowledges safety stop |
| user-intervention | user explicitly resumes |

## 12. Durable structure and knowledge lift

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
    coder_targets/
      triton_mlu.md
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
    validate_decision.py
  tests/

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
  state/
    designer_state.md
    coder_state.md
    verifier_state.md
  log/
    *.pt.trace.json
```

`log/` is gitignored; normalized reports are durable. Role-local state captures
hypotheses, invariants, and environment facts, while round artifacts remain the
reconstructable audit source.

At project stop, Orchestrator proposes generic failed patterns from project state
for promotion to `references/anti-patterns.md`. Promotion requires explicit user
approval and a separate commit. v1 performs no external knowledge-base write.

## 13. Acceptance strategy

Implementation is accepted only when contract fixtures cover:

### 13.1 Decision and target fixtures

- valid kernel-only, host-only, and mixed decisions;
- rejected missing sections, target mismatch, invalid H hint, malformed fence,
  duplicate field, and dangling reference;
- target loader finds only complete profiles and does not rely on hardcoded stub
  names.

### 13.2 Routing and state fixtures

- capability miss and major deviation close the current round before Designer is
  dispatched again;
- a local correctness defect returns to Coder at most once;
- performance evidence reaches the next Designer;
- environment failures block without consuming a round or streak;
- each terminal result updates counters once, and only accepted advances the
  canonical pointer.

### 13.3 Measurement fixtures

- multi-iteration profiler totals normalize to per-call values;
- reference and candidate scopes remain separate;
- paired timing uses unrounded values and the accepted reference;
- reports mirror every Evaluation Contract observable;
- a missing required observable produces `measurement-incomplete` rather than a
  false adoption/rejection decision.

### 13.4 Runtime fixtures

- both runtime adapters can bootstrap a role without parent conversation context;
- unavailable multi-agent support selects sequential fallback;
- runtime-neutral files contain no active runtime-specific tool syntax;
- no next round starts before the previous terminal commit.

## 14. Migration

Existing `groupedtopk`, `fused_moe`, and other logs remain reference material.
The new skill does not automatically rewrite them. Active projects remain on the
old format until explicitly migrated.

The historical anti-pattern seed may be extracted from a pinned Git revision so
that deleted working-tree entries remain auditable, but this is implementation
data rather than a change to workflow semantics.

## 15. Future directions

Future work is organized into two independent but connectable planes. Neither
plane is part of the v1 implementation plan.

### 15.1 Portable Design Plane

Evolve target-bound Unified Sketch into a backend-neutral Core IR plus explicit
target hints and deterministic lowering. Multi-target routing becomes eligible
only after there are at least two complete backend profiles, cross-backend
examples, stable operation semantics, and conformance/lowering tests.

Possible later components include:

- normalized operation, type, memory, and schedule semantics;
- a capability matrix across Triton vendor backends;
- deterministic lowering that progressively replaces LLM translation;
- explicit target selection and capability-miss fallback policy.

v1 therefore does not create candidate lists, capability-miss history for future
routing, inactive target stubs, or mock lowering logic.

### 15.2 Evidence Intelligence Plane

Extend Verifier with backend-aware detailed profiler analysis:

- host/device timelines and critical paths;
- kernel-signature clustering and cross-round alignment;
- launch graph, synchronization, and allocator attribution;
- memory bandwidth, occupancy, register/shared-memory, and other target metrics;
- automatic intent → mechanism → observation causal analysis.

Verifier will also own the trusted evidence publication side of a future
KernelWiki integration. After a round is terminal and committed, it may emit a
provenance-rich `KernelEvidenceRecord` containing decision ID, Sketch/code digest,
runtime and measurement fingerprints, correctness, performance, profiler
features, and causal verdict.

Designer owns the knowledge-consumption side: it queries comparable evidence by
operator, shape, dtype, backend, architecture, and bottleneck, then cites evidence
IDs and records why it adopted or rejected prior experience. Orchestrator owns
external authorization, synchronization, and failure handling. KernelWiki
unavailability must not break the local loop.

The two planes can later join through stable Sketch IDs, target profile IDs, and
the evidence schema.

## 16. Specification and plan governance

This document is the architectural source of truth. The implementation plan is a
derived execution document and must not introduce new normative architecture.

The update sequence is:

1. update and commit this spec;
2. self-review it against the approved design dialogue;
3. obtain user review of the written spec;
4. use `superpowers:writing-plans` to replace the implementation plan;
5. add a `spec section → implementation task → acceptance test` traceability
   table;
6. verify spec, plan, fixtures, and Markdown structure before committing the plan.

If implementation planning discovers a new architectural decision, work returns
to this spec before the plan changes. §15 must remain visibly out of scope and
must not silently generate v1 files, tasks, or acceptance requirements.
