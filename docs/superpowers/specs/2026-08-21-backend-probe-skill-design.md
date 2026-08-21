# backend-probe Skill Design

**Created**: 2026-08-21

**Status**: Draft for review

**Scope**: A competition-oriented pre-campaign skill for discovering backend capabilities, evidence quality, and likely optimization routes before starting a full optimization campaign.

## 1. Motivation

The current Triton competition workflow mixes three distinct needs:

1. deciding whether a backend is worth targeting at all;
2. discovering what that backend can actually do; and
3. running a full optimization campaign for one operator on one backend.

`kernel-opt-loop` is designed for the third need. It can discover environment facts during Phase 0, but using a full campaign to answer basic backend questions is expensive and slows horizontal expansion across chips.

The competition scoring model rewards both:

- **horizontal expansion** across more effective backends; and
- **vertical depth** on backends worth deeper optimization work.

A dedicated `backend-probe` skill therefore exists to answer questions such as:

- does this backend have a usable `tl.dot` path for the shapes and dtypes we care about?
- what launcher path is actually available?
- what profiler evidence is available and trustworthy?
- are `num_warps` / `num_stages` tunable or effectively fixed?
- is this backend a good candidate for fusion-first, launch-first, or hybrid optimization?

The skill is intentionally **pre-campaign**. It discovers backend facts and emits promotion candidates; it does not run a formal optimization loop.

## 2. Goals and non-goals

### 2.1 Goals

- Provide a fast, repeatable way to establish a backend's current Triton capability surface.
- Produce machine-consumable, per-family evidence artifacts that later workflows can read.
- Produce a short human-reviewable promotion note describing which observed facts are worth adding to a canonical backend profile.
- Support competition decision-making: whether to start a full campaign, what route to try first, and what backend risks remain.
- Keep canonical backend profiles stable by separating run-local observations from reviewed long-term contract facts.
- Align naturally with future `kernel-opt-loop` vNext profile, claim, and attribution work without depending on vNext implementation.

### 2.2 Non-goals for v1

- Producing a candidate kernel or a competition deliverable.
- Allocating rounds, decisions, reports, verdicts, or canonical implementation pointers.
- Directly mutating canonical backend profiles.
- Replacing `kernel-opt-loop` Phase 0 or campaign verification.
- Proving full backend neutrality or supporting every non-Triton DSL.
- Concluding final competition performance from probes alone.

## 3. Position in the workflow

The skill sits before an optimization campaign:

```text
backend-probe
  -> backend capability observations
  -> route recommendation
  -> optional promotion note
  -> decide whether to start kernel-opt-loop
```

The responsibilities are separated as follows:

- **backend-probe** discovers backend facts and emits promotion candidates.
- **backend profile** stores reviewed, reusable capability constraints and evidence references.
- **kernel-opt-loop** performs operator-specific optimization work within those capability boundaries.

The skill may end without any campaign being started.

## 4. Core design principles

1. **Run-local first.** Probe outputs are initially local observations, not long-term truth.
2. **Family isolation.** Capabilities are probed and recorded in separate families so one weak area does not contaminate others.
3. **Promotion is explicit.** The skill may recommend profile updates, but canonical profiles change only after explicit approval.
4. **Backend facts before optimization plans.** The skill answers whether and how to engage a backend before spending campaign budget.
5. **Conservative profile uplift.** A first successful standard probe may justify `constrained` or `partial` profile facts, not broad `supported` claims.
6. **No second campaign state machine.** The skill emits evidence and recommendations, not a parallel optimization workflow.

## 5. v1 capability families

The first version probes six backend capability families:

1. **`dot`**
   - `tl.dot` availability
   - relevant shape/dtype/layout scope
   - whether observed behavior suggests a real matrix path worth campaign investment

2. **`launch`**
   - direct launch viability
   - fast launcher availability or absence
   - loader/launcher constraints that affect small-shape performance work

3. **`profiler`**
   - device-duration availability
   - runtime-launch-only evidence
   - kernel count and top-k availability
   - acceptable evidence ladder for later campaign use

4. **`tuning`**
   - `num_warps`
   - `num_stages`
   - whether tuning values are established, constrained, or absent

5. **`core-primitives`**
   - basic load/store/index/reshape/broadcast/mask/write capability
   - the minimum stable floor for writing Triton kernels on the backend

6. **`reductions-mixed-precision`**
   - common reductions such as `sum`, `max`, `argmax`
   - fp16/bf16/fp32 regime observations where relevant
   - whether the backend can support performance-relevant reduction/dataflow patterns

These six families are chosen because they directly drive the competition's first two decisions:

- can the backend be targeted at all?
- what style of optimization is likely to pay off?

## 6. Inputs

A probe run requires:

1. **backend target**
   - e.g. `triton_gcu`, `triton_maca`, `triton_ascend`, `triton_cuda`, `triton_mlu`

2. **runtime environment**
   - absolute interpreter path
   - selected device
   - required bootstrap or environment facts

3. **probe bundle**
   - the families to run (`dot`, `launch`, `profiler`, `tuning`, `core-primitives`, `reductions-mixed-precision`)

Optional inputs may include:

- representative shape/dtype/layout sets for competition-relevant paths;
- operator-family hints such as `attention`, `routing`, or `reduction`; and
- an existing canonical backend profile to compare against.

## 7. Outputs

### 7.1 Machine-consumable artifacts

A probe run emits one machine-readable file per family, for example:

- `dot.json`
- `launch.json`
- `profiler.json`
- `tuning.json`
- `core-primitives.json`
- `reductions-mixed-precision.json`

Each file records only its own family and includes, at minimum:

- schema version;
- backend id;
- runtime/toolchain/device identity;
- probe command or execution method;
- shape/dtype/layout scope;
- observed result;
- artifact hashes or referenced evidence;
- a conclusion limited to the observed scope.

The family split is intentional:

- it supports incremental probing;
- it allows selective promotion into canonical profiles; and
- it lets downstream workflows consume only the facts they need.

### 7.2 Human-readable promotion note

The probe run also emits a short Markdown note that states:

1. what was observed;
2. the scope under which it was observed;
3. whether the fact is worth promoting into the canonical profile; and
4. if promoted, whether it should enter as `constrained`, `partial`, or another conservative status.

The promotion note is a recommendation, not a profile patch and not a direct profile mutation.

## 8. Artifact location

Probe artifacts are **project-local or run-local**, not canonical profile files.

The default mental model is:

```text
<project-root>/probes/<backend>/<probe-run-id>/
  dot.json
  launch.json
  profiler.json
  tuning.json
  core-primitives.json
  reductions-mixed-precision.json
  promotion-note.md
```

This keeps the boundaries clean:

- probe artifacts remain local observations;
- canonical profiles remain reviewed shared contracts; and
- no second global mutable state store is introduced.

## 9. Relationship to backend profiles

Backend profiles remain necessary.

The distinction is:

- **probe skill** discovers facts;
- **profile** preserves reviewed capability constraints and evidence references for reuse.

A run-local observation becomes eligible for profile promotion only when it is:

1. produced by a standard probe,
2. reproducible and archived with command/scope/evidence, and
3. phrased conservatively enough for reuse.

### 9.1 Initial promotion rule

A single successful **standard** probe may justify adding a profile fact, but the first promotion should be conservative:

- prefer `constrained` over broad `supported`;
- allow `profile_status: partial` while coverage is incomplete; and
- record the exact scope instead of implying backend-wide support.

### 9.2 Promotion workflow

The intended flow is:

```text
probe run
  -> family artifacts
  -> promotion note
  -> orchestrator or maintainer recommendation
  -> human approval
  -> canonical profile update
```

The skill does not directly edit canonical profiles.

## 10. Relationship to `kernel-opt-loop`

`backend-probe` and `kernel-opt-loop` are complementary.

- `backend-probe` answers: **can and should we engage this backend, and what route is promising?**
- `kernel-opt-loop` answers: **how do we optimize one operator on this backend, and what actually happened?**

The probe skill is therefore a **pre-campaign accelerator**, not a required replacement for Phase 0.

Expected interaction:

- when probe artifacts already exist, Phase 0 may consume them as prior facts;
- when they do not exist, Phase 0 can still perform its own required discovery;
- campaign-local discoveries may later generate new promotion notes.

## 11. Relationship to vNext semantic attribution work

This skill does not depend on vNext implementation, but it aligns with it.

It is a natural upstream input for:

- machine-readable backend profiles;
- project capability claims;
- decision-time capability boundaries;
- later attribution between design error, code error, lowering surprise, and evidence gap.

The intended long-term relationship is:

- `backend-probe` supplies backend facts,
- vNext profile machinery stores reviewed facts,
- vNext campaign machinery consumes those facts during formal optimization work.

## 12. End states

A probe run may complete in any of these normal states:

- **`ready-for-campaign`** — enough evidence exists to start a formal optimization campaign;
- **`not-ready-yet`** — the backend remains incomplete and needs more probing before a campaign is worthwhile;
- **`not-worth-pursuing-now`** — current evidence suggests low competition ROI relative to other backends or routes;
- **`promotion-only`** — the main value of this run is new profile knowledge, not immediate campaign start.

The ability to stop after probing, without opening a campaign, is a feature rather than a failure.

## 13. Acceptance requirements for a v1 implementation

A `backend-probe` v1 implementation is acceptable only when:

1. it runs as an independent skill rather than as a hidden `kernel-opt-loop` mode;
2. it can complete without starting a formal optimization campaign;
3. it emits one machine artifact per probe family;
4. it emits a separate Markdown promotion note;
5. it never mutates a canonical backend profile directly;
6. its outputs are scoped by runtime, device, and relevant shape/dtype/layout context;
7. it can express a conservative recommendation such as `constrained` or `partial` for first-time profile promotion;
8. its route recommendation can distinguish at least `fusion-first`, `launch-first`, `hybrid-first`, and `not-ready-for-gemm` style outcomes;
9. it can be consumed later by `kernel-opt-loop` without assuming vNext has already landed.

## 14. Open design questions for later phases

The following are intentionally deferred beyond this v1 design:

- whether family artifacts should later gain a shared JSON schema registry;
- whether promotion notes should later gain a machine-readable companion file;
- whether a future version should probe operator-specific families in addition to backend-generic families;
- whether track2 / C-like backends should reuse the same skill directly or through a higher-level abstraction;
- whether orchestrator-issued promotion recommendations should later be rendered as profile patch drafts.
