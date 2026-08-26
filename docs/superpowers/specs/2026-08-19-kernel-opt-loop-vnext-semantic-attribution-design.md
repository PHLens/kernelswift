# kernel-opt-loop vNext Semantic Contract and Attribution Design

**Created**: 2026-08-19<br>
**Status**: Proposed<br>
**Scope**: Typed Unified Sketch, executable profile probes, target capability binding, source conformance, deterministic role attribution, and final bounded configuration tuning<br>
**Parent design**: `2026-08-13-kernel-opt-loop-restructure-design.md` and `2026-08-14-kernel-opt-loop-v2-continuous-run-design.md`

## 1. Motivation

Role boundaries and durable artifacts alone cannot distinguish a bad hypothesis from a bad implementation or an unexpected backend lowering.

A structural Unified Sketch validator that checks only section order, statement prefixes, and the first target hint cannot establish coherent declaration, operation, control-domain, mask, effect, alias, or target-primitive semantics. Prose-only target profiles cannot provide machine-readable capability matching, and a self-described Coder summary does not prove candidate-source conformance.

A formal campaign requires a sufficiently complete implementation profile, while a new or weakly understood target needs executable runtime, launcher, primitive, resource, and profiler probes before that profile can be reviewed. Implementation-profile facts require executable, reproducible probes and machine-readable evidence. A separate `backend-probe` skill would duplicate profile identity, evidence, promotion, and lifecycle contracts. Pre-campaign probing therefore operates within the same profile subsystem: it may finish without creating a campaign, but its artifacts remain compatible with the canonical profile and later campaigns.

The combined attribution gap is:

```text
A target capability or expected mechanism is missing
  -> unclear whether the profile was never probed,
     the runtime does not match the profile,
     the Decision was invalid,
     the Coder failed to implement it,
     the backend lowered it differently,
     or the required evidence is unavailable
```

This specification adds executable profile probes, a typed semantic contract, and an auditable attribution chain without turning Unified Sketch into a compiler IR or changing the official measurement protocol.

## 2. Goals

1. Make the Unified Sketch a typed, machine-checkable semantic contract for the complete computation boundary affected by one round.
2. Bind each Sketch statement to source-level target implementation through a deterministic, statement-level ledger; the first implementation remains Python/Triton AST-based.
3. Represent implementation capabilities in machine-readable profiles with scoped evidence and explicit `Unknown` state.
4. Execute versioned pre-campaign profile probes through a deterministic runner that captures runtime identity, command provenance, result artifacts, and evidence hashes without allocating campaign state.
5. Reuse one probe definition/result contract for pre-campaign profile onboarding and bounded campaign-local capability checks while keeping their ownership and authority distinct.
6. Keep concrete target identity separate from implementation profile identity so one API-compatible backend does not imply another device or toolchain.
7. Separate source conformance from observed compiler/backend lowering.
8. Make Orchestrator attribution deterministic and auditable through a standalone verdict artifact.
9. Preserve the existing terminal result vocabulary and continuous-run policy, adding attribution metadata rather than replacing round results.
10. Enable the new campaign contract only for new runs; existing v1/v2 campaigns remain read-only historical artifacts.
11. Run one offline, bounded configuration-tuning gate before each Triton submission snapshot and confirm its final source through config-only binding, correctness, lowering, promotion-evidence, and official-measurement predicates.

## 3. Non-goals

- Redesigning the official competition or benchmark measurement protocol.
- Replacing the existing `auto_bench.py` interface in this specification.
- Automatically proving numerical equivalence, precision propagation, or tolerance behavior from Sketch.
- Building a backend-neutral compiler IR or automatic Sketch-to-target lowering.
- Reconstructing complete compiler IR, assembly, or hardware critical paths for every backend.
- Automatically promoting a run-local probe into the canonical implementation profile.
- Creating a separate `backend-probe` skill, optimization-round ledger, daemon, or second mutable workflow state machine.
- Inferring competition ROI, `fusion-first` versus `launch-first`, or global campaign priority from backend capability facts alone.
- Retaining runtime/online `@triton.autotune` or an autotune-cache dependency in the final candidate.
- Migrating historical campaign artifacts into the vNext schema.
- Allowing Coder or Verifier to update shared profiles, canonical pointers, or workflow state.

## 4. Design principles

### 4.1 Four distinct claims

The workflow keeps these claims separate:

```text
Design claim       What the Decision says should change.
Source claim       What the candidate source implements.
Lowering claim     What the target compiler/backend actually produced.
Performance claim  What the configured benchmark measured.
```

Designer owns the design claim. Coder owns the source claim. Verifier owns lowering and performance observations. Orchestrator classifies the relationship between claims; it does not invent runtime facts.

### 4.2 Evidence before blame

A failed expected mechanism is not automatically a Designer or Coder failure. A deterministic attribution is allowed only when the required preconditions and evidence artifacts exist. Otherwise the result is `evidence-gap`.

### 4.3 Source binding is not final lowering

A target source operation can be fused, expanded, reordered, or eliminated by compilation. The binding ledger proves source-level conformance to the Decision. Verifier evidence separately records observed lowering. No source statement is required to map one-to-one to a final device kernel.

### 4.4 Unknown is a valid profile state, not support

A capability profile may be structurally complete while individual capabilities are `unknown` or `unavailable`. A normative Sketch may not depend on an unproven `unknown` capability.

### 4.5 Unknown must be qualified before a mechanism-changing fallback

Absence from an unrelated operator probe is not negative capability evidence. Before Phase 0 freezes a project claim, Orchestrator must run a deterministic qualification gate for any `unknown` capability marked `must-resolve` or `before-fallback`. When an exact-scope catalog probe exists, that probe is attempted before a fallback that changes mechanism family, such as replacing `matrix.dot` with elementwise multiply plus `reduction.sum`. Optional Unknowns do not trigger an unbounded probe sweep.

### 4.6 One evidence plane, two probe contexts

Pre-campaign and campaign-local probes use the same versioned definitions and result schema. A pre-campaign run establishes reusable target facts and may end after emitting a promotion candidate. A campaign-local probe answers only the current immutable Decision and remains run-local evidence. Neither context directly mutates the canonical profile or writes the Phase 0 project capability claim; Orchestrator may later materialize only reviewed profile evidence or a maintainer-confirmed hash-only fallback disposition under the rules below.

### 4.7 Probe facts are not portfolio strategy

The probe subsystem reports observed capabilities, gaps, profile readiness, and exact scope. Operator-specific campaign eligibility is decided only after matching a project capability claim against a reviewed profile and a current runtime snapshot. A successful or failed matrix probe never decides whether unrelated reduction, elementwise, launcher, or native-language work is worth pursuing.

### 4.8 Configuration tuning preserves the accepted semantic contract

Final configuration tuning searches only Decision-declared, profile-legal implementation parameters. The accepted Sketch, algorithm, precision, effects, aliases, Host Plan, and public interface remain immutable. A change outside that boundary returns to a normal Designer round. Search executes the accepted candidate hash through temporary launch/meta-parameter injection; only the pinned-or-retained source, validated binding, final report facts, and verdict participate in submission routing.

## 5. Version and compatibility model

### 5.1 New-run boundary

A vNext run is identified in `team-state.md` by a schema and contract version, for example:

```yaml
contract_version: 3
semantic_contract: typed-sketch-v1
attribution_contract: verdict-v1
```

New runs use the vNext artifacts and gates. Existing runs remain v1/v2 read-only evidence. They are not retrofitted with invented Sketch, binding, or verdict claims.

### 5.2 Immutable implementation-profile snapshot

Each vNext campaign records the exact canonical implementation profile path/version and freezes the entire root-confined profile dependency closure under `state/implementation_profile_snapshot/`: `profile.yaml`, its vendored profile schema, every probe definition and declared probe input artifact named by `probe_catalog`, and every approved evidence record/attachment referenced by the capability matrix. `implementation_profile_snapshot_ref` points to the copied `profile.yaml`; profile, schema, catalog, input, and evidence hashes authenticate the closure. `load_profile()` must validate the snapshot after the canonical profile directory is changed or deleted. Historical decisions continue to use this closed snapshot. Pre-campaign probe runs freeze their own inputs but do not create a campaign snapshot or `team-state.md`.

### 5.3 Stable artifact identity

Every semantic artifact has a schema version and SHA-256. Candidate source spans are not semantic identity: they may change during formatting or the one permitted repair. `statement_id` is stable within the Decision; every binding and verdict also records the candidate or artifact hash to which it applies.

A probe result records the exact probe-definition hash, implementation-profile hash, runtime fingerprint, command argv, and every evidence artifact's relative path, byte count, and SHA-256. A result produced from another probe revision or runtime is distinct evidence even when its human summary is identical.

`submission_snapshot_id` is `sha256_canonical_json()` over the accepted candidate hash, accepted binding hash, Sketch hash, frozen profile hash, project-claim hash, runtime-snapshot hash, measurement-fingerprint hash, official-harness hash, and base/reference hash. It excludes the final-tuning Decision hash and artifact index. Decision, report, and verdict record this same ID; the verdict also records the final candidate and binding hashes. Before allocation, Orchestrator scans validated finalization verdicts and rejects either the same `submission_snapshot_id` or a current accepted candidate/binding pair already recorded as a final output under the same Sketch/profile/claim/runtime/measurement/harness/base anchors. Any semantic, profile, claim, runtime, harness, base, or measurement change produces a distinct snapshot that requires a normal eligibility check. Finalization completion is derived from validated artifacts rather than mutable workflow state.

### 5.4 Target and implementation identity

The contracts keep these identifiers separate:

- `target_id`: the concrete deployment target being observed, such as `bi150`, `s60`, `ascend910b`, or another device/runtime identity;
- `implementation_profile_id`: the language/backend/toolchain capability contract used by an implementation, such as `triton_cuda`, `triton_ascend`, or `ascendc`;
- `language`, `backend`, and `runner_adapter`: explicit implementation and execution dimensions rather than facts inferred from either identifier.

A canonical profile may match more than one concrete target only through explicit `identity_match` rules. API compatibility, such as exposing `torch.cuda`, never permits capability evidence to move between vendors, devices, architectures, or toolchains without a reviewed match rule.

## 6. Artifact chains

Pre-campaign profile onboarding uses this chain and may stop after it completes:

```text
profiles/<implementation_profile_id>/probes/<probe-id>.json
        |
        v
probes/<target-id>/<probe-run-id>/run.json
        |
        +--> results/<probe-id>.json
        +--> evidence/<probe-id>.*
        |
        v
optional render_profile_promotion.py
        |
        +--> promotion-candidate.json   machine-readable scoped recommendation
        +--> promotion-note.md          human rendering of the same candidate
        |
        v
explicit maintainer review -> optional canonical profile commit
```

No Decision, round number, candidate, benchmark ranking, `team-state.md`, or accepted implementation is allocated by this chain.

A vNext campaign round uses the following durable artifacts:

```text
state/implementation_profile_snapshot/profile.yaml
state/implementation_profile_snapshot/schema/<vendored schema>
state/implementation_profile_snapshot/probes/<definitions and declared inputs>
state/implementation_profile_snapshot/evidence/<approved records and attachments>
state/project_capability_claim.json
        |
        v
rounds/sketch_NNN.json       typed semantic contract
        |
        v
rounds/decision_NNN.md       immutable decision and causal contract
        |
        v
rounds/binding_NNN.json      Coder source-level binding ledger
        |
        v
rounds/report_NNN.md         Verifier facts and performance evidence
        |
        v
rounds/verdict_NNN.json      Orchestrator attribution and route
```

The Decision records the Sketch path and hash. The binding records the Decision and Sketch hashes. The report records all input artifact hashes and observed lowering evidence. Campaign verdicts record every input hash, rule ID, precondition, classification, terminal result, route, and counter effect; finalization verdicts use the separate counter-free branch defined below. Campaign-local Coder probe results may be referenced by the binding and Coder result only; Verifier-owned targeted probes may be referenced by the report. Neither rewrites the frozen project claim or bypasses authoritative Verifier execution.

A final-tuning finalization references the accepted Sketch and uses the same Decision, report, binding, and verdict families. For `decision_kind: final-autotune`, Decision Metadata uses `artifact_index`, while binding, report fact-pack, and verdict metadata use `artifact_kind: submission-finalization` plus the same `artifact_index`; campaign `round` is absent. `NNN` is an artifact-slot index, not a campaign round. Allocation scans all standard indexed artifact families and selects the maximum occupied index plus one. A valid Decision with the same `submission_snapshot_id` and no verdict reserves and resumes its index. A valid sealed report without a verdict resumes verdict creation; an incomplete/invalid report or conflicting Decision hash blocks recovery. Finalization does not update `current_round`, `last_completed_round`, `total_rounds`, streaks, `last_result`, `last_attribution`, or campaign-round pointers, and it adds no manifest pointer. Its presence is discovered by validated artifact scan.

`last_accepted_kernel` and `last_accepted_report` remain one canonical submission pair. An improved finalization atomically advances both to the pinned candidate and sealed finalization report after verdict validation; `last_accepted_round` remains the last accepted campaign round. A fallback-retained selection changes neither pointer, and any partial pair update is invalid.

Temporary tuning data may contain only normalized configuration-value tables, compiler caches/binaries, and raw command output under the existing gitignored `log/final-tuning/` boundary. It may not contain candidate-language source derived from the accepted candidate. The final report fact pack and verdict retain deterministic comparison and exact final-source/binding hashes without introducing a finalization-specific state artifact or artifact family.

## 7. Typed Unified Sketch

### 7.1 Scope

The Sketch describes the complete computation boundary affected by the current round. It does not need to restate an unchanged model or the entire forward method, but it must include every changed kernel/dataflow region and its boundary:

- input and output values;
- declarations and storage spaces;
- operations and data dependencies;
- parallel and loop domains;
- guards, masks, and index bounds;
- writes, mutations, aliases, and externally visible effects;
- target hints and their modality.

A host-only Decision continues to use an explicit host-only marker and a typed Host Plan. A mixed Decision contains two typed sub-contracts, one kernel Sketch and one Host Plan, connected by one shared causal graph.

### 7.2 Canonical JSON shape

`rounds/sketch_NNN.json` is the normative source. Markdown may render a human-readable view but is not independently authoritative.

The initial schema is intentionally declarative, not executable:

```json
{
  "schema_version": 1,
  "sketch_id": "sketch-001",
  "round": "001",
  "scope": {
    "kind": "changed-computation-boundary",
    "entrypoints": ["ModelNew.forward", "_route_kernel"],
    "unchanged_boundary": ["public ModelNew contract"]
  },
  "declarations": [
    {
      "id": "scores",
      "kind": "tensor",
      "shape": ["T", "E"],
      "dtype": "fp32",
      "layout": "row_major",
      "memory": "global"
    },
    {
      "id": "row",
      "kind": "tile",
      "shape": ["BLOCK_E"],
      "dtype": "fp32",
      "layout": "contiguous",
      "memory": "register"
    }
  ],
  "operations": [
    {
      "id": "op.load.row",
      "kind": "load",
      "inputs": ["scores"],
      "outputs": ["row"],
      "index_domain": "token x expert",
      "mask": "expert < E",
      "effects": {"reads": ["scores"], "writes": []}
    }
  ],
  "control": [
    {"id": "ctrl.token", "kind": "parallel", "variable": "token", "domain": "0 <= token < T"},
    {"id": "guard.expert", "kind": "guard", "condition": "expert < E"}
  ],
  "effects": {
    "outputs": ["topk_indices", "topk_values"],
    "mutations": [],
    "aliases": []
  },
  "hints": [
    {"name": "num_warps", "value": 1, "modality": "preferred"},
    {"name": "num_stages", "value": 2, "modality": "exploratory"}
  ],
  "causal_nodes": [
    {"id": "m.reduce-fusion", "kind": "mechanism", "expected": "external reduction kernels decrease"}
  ]
}
```

### 7.3 Required static checks

The Sketch checker must reject a proceeding Decision when any of the following fails:

- schema version, unique IDs, required fields, and reference integrity;
- declaration type consistency for operation inputs and outputs;
- SSA-style def-use: every consumed value is defined, and each value has one authoritative definition;
- shape, dtype, layout, and memory-space compatibility for every operation edge;
- index domains, guards, masks, and load/store bounds are present and connected to the affected operations;
- side effects, output writes, mutations, and aliases are explicitly declared;
- every changed computation region has a control and effect boundary;
- `required` hints match the selected implementation-profile snapshot and are not silently replaced;
- `preferred` and `exploratory` hints have an explicit fallback or observation policy;
- an algorithm-substitution fallback names the primary/fallback contracts and exact signatures, `fallback_kind`, probe policy, and onboarding disposition;
- a proceeding Decision does not silently replace an Unknown primary while its `before-fallback` qualification is unresolved or `promotion-pending`;
- fallback disposition id/hash matches the frozen project claim, whose embedded maintainer authorization has no raw probe-result reference;
- every causal node referenced by the Evaluation Contract exists and is connected to the declared intervention.

The checker does not prove numerical equivalence. Correctness, tolerance, tie rules, and precision behavior remain Verifier guardrails.

### 7.4 Hint modality

Every target hint has one modality:

| Modality | Meaning | Violation route |
|---|---|---|
| `required` | Normative implementation requirement | `code-error` if the source binding violates it; `design-error` if the profile makes it impossible and the Decision proceeded anyway |
| `preferred` | Desired implementation choice with an allowed target-specific accommodation | Coder must record the accommodation; absence alone is not an error |
| `exploratory` | Tuning or probe variable, not a conformance requirement | Verifier records the observed effect; no conformance failure |

`preferred` does not authorize a silent algorithm substitution. If its primary capability is Unknown and the declared policy is `before-fallback`, the pre-campaign qualification disposition must exist before the accommodation can enter a proceeding Decision.

## 8. Implementation capability and probe model

### 8.1 Evidence lifecycle and campaign matching

Profile evidence flows through a reviewed lifecycle before a campaign consumes it:

```text
operator capability requirement
  -> deterministic demand-scoped probe selection
  -> versioned probe definition
  -> run-local probe result and hashed evidence
  -> explicit promotion review
  -> canonical implementation profile
  -> project capability claim
  -> immutable campaign run snapshot
```

The campaign-facing matching model still has three layers:

```text
Canonical implementation profile
  Language, backend, toolchain, target-match rules, launcher, primitive,
  resource, lifecycle, profiler, and versioned probe capabilities.

Project capability claim
  The shape/dtype/layout/harness signatures required by this operator, the
  reviewed evidence scope, and any primary/fallback qualification disposition.

Run snapshot
  The actual Phase 0 target identity, runtime identity, measurement fingerprint,
  profile hash, project-claim hash, and selected interpreter/device/runner.
```

These layers are matched, not merged into one mutable document. Run-local probe results are upstream evidence; they become canonical facts only through an explicit profile commit.

Before those layers are frozen, a read-only preflight may supply an ephemeral, normalized qualification requirement:

```json
{
  "requirement_id": "attention-qk-dot-fp16",
  "primary_contract": "matrix.dot",
  "primary_signature": {"lhs_dtype": "fp16", "rhs_dtype": "fp16", "accumulator_dtype": "fp32", "layout": "blocked", "m": 16, "n": 128, "k": 64},
  "fallback_contract": "reduction.sum",
  "fallback_signature": {"dtype": "fp32", "axis": "k"},
  "fallback_kind": "algorithm-substitution",
  "probe_policy": "before-fallback"
}
```

This tuple originates from an explicit user/maintainer request or Designer's read-only capability preflight over the operator semantics and candidate mechanism families. Designer does not write campaign artifacts in this mode; Orchestrator validates and materializes the normalized selector input. The provisional project claim is not frozen until qualification completes. The tuple is not a second profile or mutable campaign store: it is copied and hashed into the selected pre-campaign run for audit. The selector considers only explicit requirements, never all Unknowns in a profile, and Orchestrator does not invent primary or fallback design claims.

### 8.2 Canonical implementation profile layout

An implementation profile is maintained as:

```text
profiles/<implementation_profile_id>/
  profile.yaml
  schema/
    profile.schema.json       profile-local vendored schema
    shared-profile.schema.json frozen copy of the shared source schema
  probes/
    <probe-id>.json
    <probe-payload files>
  evidence/
    README.md
    <reviewed evidence records>
```

`schema/profile.schema.json` and `schema/shared-profile.schema.json` are byte-for-byte copies of the shared schema, not path-only pointers. `profile.yaml` records both relative paths/hashes and the shared schema version. `load_profile()` compares the two copies entirely inside the profile root. Canonical contract tests additionally compare `shared-profile.schema.json` with `skills/kernel-opt-loop/schemas/profile.schema.json`; a campaign snapshot needs no external path and avoids any self-referential schema hash field.

The profile schema requires these sections, although individual capability entries may be Unknown or unavailable:

- `identity_match`: implementation profile id/version, language, backend, runner adapter, toolchain, permitted target/device architecture scope, and match rules;
- `runtime_launcher`: imports or build bootstrap, launch syntax or ABI, stream/synchronization behavior, and harness/runner constraints;
- `capability_matrix`: primitive or native-operation families and signatures;
- `probe_catalog`: versioned probe-definition IDs and hashes; run-local results are not embedded here;
- `fallback_and_unknown_policy`: normalized primary/fallback contract pairs, `fallback_kind`, `probe_policy`, and waiver requirements;
- `profiler_evidence`: available/unavailable event types, scopes, and normalized fields.

These sections are conditional when declared by a profile:

- `resource_constraints`: warp/stage, core/grid, register, memory, build, or IR constraints;
- `configuration_constraints`: legal configuration fields, finite values, cross-field exclusions, shape/dtype scope, temporary-injection method, and whether each field is a launch option or compile-time meta-parameter;
- `host_lifecycle`: allocation, cache, context, stream, workspace, and concurrency behavior.

`configuration_constraints` records legality and exact scope, not performance preference. A Triton profile is submission-finalization eligible only when reviewed constraints cover the accepted configuration and every enumerated alternative. Missing or Unknown legality blocks as `profile-legality-unavailable`. A singleton domain is valid only when the profile explicitly proves that accepted configuration at the exact scope. Project-local tuning observations do not update the canonical profile.

The schema is language-neutral enough for later C-like profiles such as `ascendc`, while the first binding checker remains Python/Triton-specific. A profile is structurally complete even when its status is:

```yaml
profile_status: partial
```

### 8.3 Capability entry

Each capability entry records:

```yaml
id: memory.load.contiguous-fp32
family: memory
capability_kind: primitive
contract_name: memory.load
implementation_symbol: tl.load
signature:
  dtype: fp32
  layout: contiguous
  shape: ["N"]
status: supported # supported|constrained|unknown|unsupported|prohibited
constraints:
  - masked bounds required
scope:
  target_id: bi150
  runtime: "triton 3.1.0 / CoreX 4.4.0"
  device_arch: "Iluvatar BI-V150 major=7 minor=1"
  shape_signature: "project-defined"
evidence:
  - evidence_id: bi150-basic-memory-001-reviewed
    review_status: approved
    archived_result_ref: evidence/bi150-basic-memory-001.result.json
    archived_result_sha256: "..."
    probe_id: bi150-basic-memory-001
    probe_definition_sha256: "..."
    result_sha256: "..."
    kind: observed
    target_id: bi150
    toolchain_fingerprint: "..."
    device_arch: "Iluvatar BI-V150 major=7 minor=1"
    runner_adapter: python-command
    launcher_context: direct-launch/current-stream
failure_classification:
  mismatch: environment-blocked
  unprovable_required_use: capability-miss
```

`load_profile()` resolves each approved `archived_result_ref` inside the profile's `evidence/` directory, verifies `archived_result_sha256 == result_sha256`, validates the archived file as a probe result, and checks that the approved capability scope is no broader than the archived observation scope. Raw run-directory paths and unreviewed results are invalid canonical evidence.

`unknown` is not support. A normative Sketch cannot require an unknown capability. `prohibited` represents a policy or semantic ban even when the backend may technically support the construct. A fallback policy never changes capability status: it declares whether the fallback is `semantic-accommodation` or `algorithm-substitution`, and whether probing is `optional`, `before-fallback`, or `must-resolve`. The v1 promotion renderer never recommends `supported`: a first or later automated recommendation is at most `constrained`, unchanged status, or additional evidence. Only a maintainer may approve `supported`, with separate coverage rationale, while the overall profile may remain `partial`.

### 8.4 Versioned probe definition

A profile-local probe definition is declarative and immutable by hash. It names the implementation profile but does not claim a successful result. Its scope metadata must be precise enough for demand selection; a matrix probe declares input/accumulator dtypes, layouts, tile or shape regime, target/runtime/toolchain/device-architecture constraints, and launcher context where relevant:

```json
{
  "schema_version": 1,
  "probe_id": "triton-mlu-basic-memory-001",
  "implementation_profile_id": "triton_mlu",
  "family": "core-primitives",
  "purpose": "Compile and execute masked load/store with one-dimensional indexing",
  "capability_ids": [
    "memory.load.contiguous-fp32",
    "memory.store.contiguous-fp32"
  ],
  "input_artifacts": [
    {"path": "probes/basic_memory.py", "sha256": "...", "run_path": "basic_memory.py"}
  ],
  "runner": {
    "kind": "command",
    "argv": [
      "{interpreter}",
      "{probe_inputs_root}/basic_memory.py",
      "--result-json",
      "{result_payload_path}",
      "--target-id",
      "{target_id}",
      "--runtime-snapshot",
      "{runtime_snapshot_path}"
    ],
    "cwd": "{probe_run_dir}",
    "timeout_seconds": 120
  },
  "required_runtime_fields": ["interpreter", "device", "toolchain", "device_arch", "runner_adapter", "bootstrap_modules", "synchronize_api"],
  "scope_template": {
    "dtype": "fp32",
    "layout": "contiguous",
    "shape": ["N"]
  }
}
```

`runner.argv` is an argument array, never an interpolated shell command. Every executable source, build manifest, or other file input is declared under `input_artifacts`, validated by root-confined path and SHA-256, and copied into the run before launch. Only `{interpreter}`, `{probe_inputs_root}`, `{probe_run_dir}`, `{result_payload_path}`, `{runtime_snapshot_path}`, `{target_id}`, and allowlisted runtime fields such as `{device}` may be resolved, all from frozen run inputs or the validated runtime snapshot. A command runner can later invoke a Python probe, compiler, build adapter, native executable, or another approved payload without changing the result schema.

### 8.5 Run-local probe execution

Demand selection is a pure operation:

```python
select_profile_probes(
    profile: Mapping[str, Any],
    requirements: Sequence[Mapping[str, Any]],
    runtime_snapshot: Mapping[str, Any],
) -> QualificationPlan
```

`QualificationPlan` contains ordered `selections` plus one disposition for every explicit requirement, so `no-exact-probe` is represented even when no runner work is selected. It selects only explicit `before-fallback` or `must-resolve` requirements whose exact-scope primary capability is `unknown`. A `before-fallback` selection additionally requires a supported/constrained fallback with `fallback_kind: algorithm-substitution`. Exactly one catalog definition must cover the primary signature and current target/profile/runtime/toolchain/device scope; zero matches produce `no-exact-probe`, multiple matches are a deterministic ambiguity error, and unrelated Unknowns are ignored. Requirement dispositions sort by requirement id; selected work sorts by requirement id and then probe id.

The public execution interfaces are:

```text
python3 <skill-root>/scripts/run_profile_probe.py \
  --profile <absolute-profile.yaml> \
  --probe-id <probe-id> \
  --target-id <concrete-target-id> \
  --runtime-snapshot <absolute-runtime.json> \
  --output-root <absolute-project-local-root> \
  [--qualification-requirement <absolute-normalized-requirement.json>]

python3 <skill-root>/scripts/render_profile_promotion.py \
  --run-dir <absolute-completed-probe-run> \
  --profile <absolute-current-profile.yaml>
```

`target-id` and an explicit `probe-run-id` accept only `[A-Za-z0-9._-]+`. When the CLI omits it, the runner generates UTC `YYYYMMDDTHHMMSSffffffZ`; a collision fails rather than overwriting or retrying under an ambiguous identity.

The runner:

1. validates the canonical profile, probe definition, profile hash, target id, and required runtime fields;
2. creates a new run directory with exclusive creation and rejects path traversal or reuse;
3. copies the exact validated profile, probe definition, runtime snapshot, optional normalized qualification requirement, and every declared input artifact into `inputs/` and records their hashes so the run remains self-validating after canonical files change;
4. executes the argv without a shell and with a bounded timeout;
5. captures stdout, stderr, exit code, duration, payload result, and allowlisted environment/bootstrap facts;
6. hashes every referenced artifact and writes normalized `run.json` and `results/<probe-id>.json` atomically;
7. validates that the payload result names the same probe, profile, target, capability ids, and observed scope.

After the run validates independently from the current canonical file, the optional renderer compares its frozen profile input with the current profile and derives `promotion-candidate.json` plus its Markdown rendering. It accepts only `evidence-ready` or `partial` runs with at least one valid observed fact; blocked or failed runs remain diagnostic evidence only. The renderer never edits `profile.yaml`.

The runner-produced layout is:

```text
<output-root>/probes/<target-id>/<probe-run-id>/
  inputs/
    profile.snapshot.yaml
    probe-definition.json
    runtime-snapshot.json
    qualification-requirement.json  # only for demand-selected runs
    payload/
      <declared input artifacts>
  run.json
  results/
    <probe-id>.json
  evidence/
    <probe-id>.stdout.log
    <probe-id>.stderr.log
    <probe attachments>
```

The renderer then adds:

```text
  promotion-candidate.json
  promotion-note.md
```

`run.json` records `schema_version`, run id, target id, implementation profile id/path/version/hash, runtime fingerprint, requested probe ids, start/end timestamps, and terminal execution summary. Each result records the probe-definition hash, command argv, exit code, timeout state, observed scope, capability observations, evidence references with byte counts and SHA-256, and one result level from `observed|inferred|unknown`.

Normal run summaries are `evidence-ready|partial|environment-blocked|probe-failed`:

- `evidence-ready`: every declared observation completed with numerically checked `observed` success;
- `partial`: execution completed and artifacts validate, but at least one declared observation is `inferred`, `unknown`, or unavailable without invalidating the run;
- `environment-blocked`: the matched interpreter/executable, runtime bootstrap, device, toolchain, or runner cannot be started or does not match before the probe can make a capability observation;
- `probe-failed`: the bounded payload starts but times out, exits nonzero, emits malformed or mismatched output, or reports a compile/execution/correctness failure.

Invalid profile or probe-definition contracts are validation errors and do not create a misleading completed run. These summaries describe probe completion, not competition priority or operator-specific campaign readiness. Raw logs remain project-local and gitignored by default; reviewed result documents may be archived when promotion needs durable evidence.

### 8.6 Demand-qualification outcome routing

`promotion-pending` is an onboarding disposition, not a fifth probe summary or a campaign terminal result. For an Unknown primary selected by `probe_policy: before-fallback`:

- numerically checked `observed` success produces promotion artifacts and the pre-campaign disposition `promotion-pending`; Orchestrator creates no campaign state and does not continue silently with the fallback;
- Phase 0 may use the primary only after a maintainer promotes the exact scope into the canonical profile and preflight revalidates that revision;
- a maintainer may explicitly decline or defer promotion and authorize fallback, but raw success never satisfies the campaign claim;
- `partial`, `probe-failed`, `environment-blocked`, or `no-exact-probe` leaves the primary `unknown`, never auto-promotes it to `unsupported`, and permits an algorithm-substitution fallback only with maintainer authorization; a campaign-global target/profile/runtime identity block still blocks campaign creation;
- `must-resolve` never falls back; an unresolved result blocks campaign creation.

The immutable `state/project_capability_claim.json` is the authority for fallback authorization. It embeds the complete normalized requirement and its hash, onboarding outcome, `promotion_disposition: declined|deferred|not-applicable`, `fallback_authorized: true`, reason, maintainer confirmation identity/time/method, optional probe id/definition/result hashes, and `primary_remains_unknown: true`:

```json
{
  "disposition_id": "s60-attention-dot-fallback-001",
  "requirement": {"requirement_id": "attention-qk-dot-fp16", "primary_contract": "matrix.dot", "primary_signature": {}, "fallback_contract": "reduction.sum", "fallback_signature": {}, "fallback_kind": "algorithm-substitution", "probe_policy": "before-fallback"},
  "requirement_sha256": "...",
  "onboarding_outcome": "probe-failed",
  "promotion_disposition": "not-applicable",
  "fallback_authorized": true,
  "reason": "Use explicit sum substitution for this run epoch while dot remains unproven",
  "maintainer_confirmation": {"confirmed_by": "...", "confirmed_at": "...Z", "method": "explicit-user-instruction"},
  "probe_id": "s60-dot-fp16-001",
  "probe_definition_sha256": "...",
  "probe_result_sha256": "...",
  "primary_remains_unknown": true
}
```

The requirement hash is SHA-256 over UTF-8 JSON of the embedded requirement serialized with `sort_keys=True`, `separators=(',', ':')`, and `ensure_ascii=False`. `qualification_disposition_sha256` uses the same canonical serialization over the complete disposition object exactly as embedded in the project claim, including `disposition_id`, requirement and requirement hash, onboarding outcome, promotion disposition, fallback authorization, reason, maintainer confirmation, optional probe hashes, and `primary_remains_unknown`; the disposition object contains no self-hash field. Confirmation method is exactly `explicit-user-instruction|maintainer-reviewed-commit`, and `confirmed_at` is UTC RFC 3339. The claim must not contain a raw probe-result path or any dependency on the pre-campaign run directory. Phase 0 freezes and hashes this claim beside the implementation-profile snapshot; deleting the entire pre-campaign run must not invalidate campaign history.

An eventual fallback Decision references the disposition id/hash from the frozen project claim and records `fallback_from`, exact primary/fallback signatures, `fallback_kind`, and the expected causal/performance consequence. It does not reference raw probe output. Existing campaign snapshots never rerun this gate or absorb later promotions. A new profile revision requires a new campaign or run epoch.

### 8.7 Promotion and campaign-local consumption

- A pre-campaign probe run may complete without creating a formal optimization campaign.
- `promotion-candidate.json` lists exact observed facts, current and recommended capability statuses, scope, result/evidence hashes, unresolved gaps, rationale, and `onboarding_disposition: promotion-pending` for eligible demand-selected success. `promotion-note.md` is its human-readable rendering; neither artifact is a patch application.
- The candidate status is `proposed` until a user or profile maintainer reviews the result and updates the canonical profile in a separate commit. The committed capability entry references the archived result hash and probe-definition hash; the approved scope may never be broader than the source probe scope.
- Phase 0 may consume prior probe evidence only after it has been reviewed and archived by the canonical profile, and only when the target, implementation profile, runtime/toolchain/device scope, definition hash, result hash, and evidence hashes validate. Raw proposed output never enters a campaign snapshot, and prior evidence never bypasses current runtime identity discovery. Optional probe hashes inside a maintainer-authorized fallback disposition are provenance only, not a capability fact or dependency.
- Coder may execute the same definitions for bounded compile/capability checks required by the current Decision. Those results live under the campaign's run-local `log/probes/` area and may support the binding or Coder result, but they do not rewrite the Phase 0 project claim or mutate the canonical profile.
- Verifier retains ownership of authoritative candidate execution, correctness, lowering, and performance evidence. A Coder probe cannot substitute for Verifier evidence.
- Runtime/profile mismatch is `environment-blocked`, not `capability-miss`.

## 9. Source-level binding ledger

### 9.1 Binding schema

`rounds/binding_NNN.json` is produced by Coder and checked before Verifier dispatch:

```json
{
  "schema_version": 1,
  "round": "001",
  "decision_sha256": "...",
  "sketch_sha256": "...",
  "candidate_path": "triton_operator_001.py",
  "candidate_sha256": "...",
  "source_analyzer": "python-ast-triton",
  "binding_model": "primitive-call",
  "bindings": [
    {
      "statement_id": "op.load.row",
      "relation": "implemented-by",
      "contract_name": "memory.load",
      "implementation_symbol": "tl.load",
      "source_spans": [
        {"path": "triton_operator_001.py", "start": [42, 8], "end": [42, 39]}
      ],
      "status": "implemented",
      "notes": "masked contiguous load",
      "evidence": ["candidate-ast-parse", "profile-capability-check"]
    }
  ]
}
```

Allowed source-level relations include:

- `implemented-by`: one Sketch statement is implemented by one or more source regions;
- `fused-into`: multiple Sketch statements share one source region;
- `expanded-into`: one Sketch statement is represented by multiple source regions;
- `elided-by`: a statement is intentionally removed by a source-level algebraic simplification and must include a reason and replacement relation.

The ledger must cover every required Sketch statement. It may not silently omit or substitute a normative operation.

### 9.2 Deterministic conformance checker

The checker validates:

1. Decision, Sketch, candidate, implementation-profile snapshot, and binding hashes;
2. the profile-selected `source_analyzer` and `binding_model`;
3. stable statement IDs and complete coverage;
4. source paths and source spans against the candidate hash;
5. semantic contract names, profile-mapped implementation symbols, and operation signatures;
6. allowed relation cardinality and explicit many-to-many reasons;
7. implementation-profile capability and hint modality;
8. public entrypoints and declared effects at the candidate boundary;
9. no binding to `base.py`, harness files, or files outside candidate ownership.

The first implementation provides `python-ast-triton`. A C-like profile can validate structurally and declare a future analyzer such as `clang-like` or `symbol-and-call`; until that adapter exists, automatic source conformance is explicitly unavailable rather than misclassified as a Python candidate failure.

The checker proves source conformance only. It cannot prove that the compiler emitted the expected number or shape of final device kernels; that is an observed lowering claim owned by Verifier.

## 10. Evaluation and lowering evidence

The existing Evaluation Contract remains the source of expected causal behavior. vNext adds structural references:

```json
{
  "hypothesis_id": "H-001",
  "intervention": "fuse routing reduction into the target kernel",
  "causal_graph": {
    "nodes": ["m.reduce-fusion", "o.external-kernel-count", "p.wall-time"],
    "edges": [
      ["m.reduce-fusion", "o.external-kernel-count"],
      ["o.external-kernel-count", "p.wall-time"]
    ]
  }
}
```

A causal graph checker verifies structural connectivity and direction declarations. It does not prove that the mechanism is true at runtime. Verifier must report each observable, its observation, verdict, evidence path/hash, and confidence.

Observed lowering uses a target-specific evidence contract declared by the profile. A profile may require one or more of:

- scoped profiler/kernel summary;
- runtime launch summary;
- compiler IR or assembly dump;
- generated-kernel signature;
- targeted microprobe.

The profile need not require compiler dumps when the backend cannot provide them, but it must state what weaker evidence means and what it cannot distinguish.

## 11. Attribution model

### 11.1 Internal classification

The Orchestrator writes `rounds/verdict_NNN.json`. The attribution class is independent from the existing terminal result:

```text
design-error | code-error | lowering-unknown | evidence-gap
```

A successful or ordinary performance miss may have no attribution class or may carry `none`/`not-applicable`.

A campaign verdict contains:

- `classification`;
- `terminal_result`;
- `rule_id`;
- `confidence`;
- input artifact paths and SHA-256 hashes;
- preconditions with pass/fail/missing status;
- supporting observations and evidence references;
- explanation constrained to the selected rule;
- next route and counter effect.

The Orchestrator may use LLM reasoning to select and explain a rule, but only from the structured checker outputs and Verifier facts. A verdict is invalid if its rule preconditions are not satisfied. If no rule is deterministically satisfied, classification must be `evidence-gap`.

For `decision_kind: final-autotune`, the verdict uses a separate schema branch with `route: submission-ready|blocked`. It omits `classification`, `terminal_result`, `failed_attempt_effect`, and every run-policy projection field. It validates deterministic selection, temporary-boundary cleanup, selected-source pinning or accepted-source confirmation, fresh binding validation, and final official evidence. The route bypasses the campaign rule table and run-policy evaluator.

### 11.2 Rule table

| Rule ID | Required preconditions | Classification | Terminal result | Route/counter |
|---|---|---|---|---|
| `DESIGN.SKETCH.INVALID` | Sketch schema or semantic checker fails before coding | `design-error` | `design-rejected` | no Coder dispatch; failed streak +1 |
| `DESIGN.CAUSAL.INVALID` | causal graph is disconnected, contradictory, or not tied to the declared intervention/observable | `design-error` | `design-rejected` | new Decision next round; failed streak +1 |
| `CODE.BINDING.MISSING` | required statement has no binding or binding hash/span is invalid | `code-error` | repair, then `candidate-failed` if repair fails | one local repair; failed streak only after terminal failure |
| `CODE.BINDING.VIOLATION` | source primitive/signature/effect violates validated Sketch/profile | `code-error` | repair, then `candidate-failed` if repair fails | one local repair; failed streak only after terminal failure |
| `CODE.CORRECTNESS.FAIL` | candidate fails correctness or declared structural guardrail | `code-error` | `candidate-failed` after allowed repair | failed streak +1 |
| `LOWERING.EXPECTED.ABSENT` | Sketch and binding pass, profile matches, targeted lowering evidence shows expected mechanism absent | `lowering-unknown` | `design-rejected` | next Designer hypothesis; no failed streak |
| `EVIDENCE.OBSERVABLE.MISSING` | required observable cannot be obtained after bounded probe | `evidence-gap` | `blocked` or `design-rejected` | environment block or invalid Decision |
| `ENV.PROFILE.MISMATCH` | runtime identity does not match implementation-profile snapshot | no attribution or `evidence-gap` | `blocked` | counter-neutral recovery |

A correct candidate whose mechanism is observed but whose accepted-to-candidate e2e result does not clear the adoption threshold is `no-improvement`, not `design-error` or `code-error`. Its evidence is passed to Designer for the next hypothesis.

### 11.3 Lowering-unknown policy

`lowering-unknown` is intentionally not treated as a failed implementation. It means the current experiment did not establish whether the source-level contract can induce the expected backend lowering. The round is terminal for auditability, but it does not increment `failed_attempt_streak`.

The run-policy evaluator must therefore consume the attribution counter effect rather than inferring all `design-rejected` rounds are equivalent.

## 12. Role contracts and routing changes

Before campaign state exists, Orchestrator owns pre-campaign profile-onboarding execution. Orchestrator validates the profile/definition/runtime inputs, creates and writes only the isolated probe root, invokes the runner/validator/renderer, and reports the proposed promotion artifacts. It may stop there. Orchestrator does not create campaign state or update the canonical profile; the user or profile maintainer exclusively owns any approval and canonical profile commit.

### 12.1 Designer

Designer must:

- in read-only capability preflight, identify explicit optimization-critical primary/fallback pairs from operator semantics without writing a Decision, Sketch, or campaign state;
- author the typed Sketch and causal graph references after Phase 0 begins;
- ensure the intervention is structurally connected to declared observables;
- select only profile capabilities that are supported or constrained for the project claim;
- record required, preferred, and exploratory hints;
- distinguish semantic accommodation from algorithm substitution and record primary/fallback signatures plus onboarding disposition;
- never describe an Unknown primary as unavailable or select its `before-fallback` substitution while qualification is unresolved;
- not claim that an expected lowering is guaranteed when the profile only supports a probe or inference;
- for eligible final tuning, reuse the accepted Sketch and declare only finite profile-legal `preferred|exploratory` configuration fields, immutable input hashes, budget, comparison objective, and deterministic winner rule.

Designer is accountable for `design-error`, not for an unconfirmed backend lowering or a correct candidate that misses the e2e threshold.

### 12.2 Coder

Coder must:

- run only Decision-scoped compile/capability probes required to establish source conformance against the frozen implementation-profile snapshot;
- record those results as campaign/round/Decision-local evidence and never claim backend-wide support or mutate the profile;
- implement the immutable Decision from `last_accepted_kernel`;
- produce the complete binding ledger;
- pass the deterministic conformance checker before `candidate-ready`;
- record target-specific accommodations for preferred hints;
- for final tuning, receive the normalized Verifier-selected configuration from Orchestrator, emit at most one pinned candidate derived from the accepted source, and emit a fresh binding ledger for the exact final source; when the accepted fallback wins, emit no candidate source and bind the accepted source hash;
- perform at most one local repair for binding or implementation defects.

Coder is accountable for `code-error` only when the validated source contract is violated. Coder is not accountable for an observed lowering surprise after source conformance passes.

### 12.3 Verifier

Verifier must:

- consume validated Decision, Sketch, frozen implementation-profile snapshot, candidate, and binding;
- produce runtime facts, correctness results, mechanism observables, lowering observations, confidence, and missing-evidence declarations;
- use the profile's target-specific evidence contract;
- under final-tuning measurement exclusivity, evaluate injected configurations and return normalized trials to Orchestrator without a persisted selection artifact; after Coder pinning or fallback confirmation, rerun complete verification and atomically write the sealed report once;
- never assign `design-error` or `code-error` as the final responsibility classification;
- never modify the candidate, Decision, binding, canonical pointers, or manifest.

### 12.4 Orchestrator

Orchestrator must:

- before Phase 0, obtain explicit requirements from the user/maintainer or Designer read-only capability preflight, normalize them without inventing design claims, run demand-scoped selection for `before-fallback|must-resolve`, attempt only the unique exact-scope probe, and ignore unrelated Unknowns;
- stop without campaign state on `promotion-pending`; resume preflight only after explicit promotion or a maintainer-confirmed fallback disposition that Orchestrator embeds in the project claim without raw probe refs;
- in Phase 0, materialize the operator-specific project capability claim including fallback provenance, validate any exact-scope approved prior probe evidence, and freeze the reviewed implementation-profile snapshot;
- never promote raw pre-campaign, Coder, or Verifier probe evidence into the canonical profile during a campaign;
- validate all hashes and checker outputs before dispatch;
- compute the canonical `submission_snapshot_id`, scan validated finalization artifacts for completed or resumable identity, allocate or resume the deterministic artifact index, and open final tuning only when the accepted snapshot is fingerprint-stable with no unresolved qualification, promotion, repair, or measurement transition;
- validate the finite profile-legal domain, search budget/order, semantic immutability, and absence of derived source in temporary storage; run the pure selector over Verifier's non-persistent normalized trials and route only that configuration to Coder for one-time pinning;
- after the sealed report exists, rerun the selector, validate the exact final source/binding and pure submission-promotion predicate, then emit `submission-ready|blocked` without campaign terminal or counter projection;
- create `verdict_NNN.json` from structured facts and rule preconditions;
- route repair, design rejection, blocked recovery, or next-round Designer work;
- apply the terminal result and attribution-specific counter effect;
- for campaign rounds, update canonical pointers only for `accepted`; for finalization, atomically update the existing `last_accepted_kernel`/`last_accepted_report` submission pair only for an improved winner satisfying the separate pure submission-promotion predicate, without changing `last_accepted_round`;
- keep official base-to-candidate competition evidence separate from accepted-to-candidate promotion evidence.

## 13. Dual performance evidence boundary

This specification does not redefine the measurement runner. It requires the workflow to preserve two distinct result claims:

```text
Promotion evidence
  accepted reference -> candidate
  Controls whether the canonical last_accepted_kernel/last_accepted_report pair advances.

Competition evidence
  official base/reference -> candidate
  Records the candidate's competition-facing e2e result.

Final-tuning search evidence
  accepted configuration -> ephemeral configuration injection
  Ranks the bounded configuration set only; it cannot authorize submission.

Pinned submission evidence
  official base/reference -> exact pinned candidate hash
  Authorizes the final submission verdict.
```

The official e2e result is the competition-facing value. Kernel time, kernel count, and host overhead remain optimization diagnostics and causal evidence. A kernel-time improvement without accepted-to-candidate promotion or e2e benefit is not silently converted into `accepted`. Final-tuning search measurements select a configuration but never substitute for correctness and official measurements on the exact pinned source hash.

## 14. Final bounded configuration tuning

A Triton submission snapshot runs exactly one final configuration-tuning gate after normal optimization has stopped and a validated `last_accepted_kernel` exists. Orchestrator computes the §5.3 `submission_snapshot_id`, scans validated finalization artifacts, and allocates or resumes the §6 artifact-slot index. The stage does not call continuous-run policy evaluation or change campaign-round pointers, terminal fields, or counters. Eligibility requires no open profile qualification, promotion, repair, or measurement-fingerprint transition and requires reviewed exact-scope configuration legality.

Schema-v2 Decision Metadata uses `decision_kind: optimization|final-autotune`. The final-tuning Decision reuses the accepted Sketch and records:

- the accepted candidate path/hash, accepted binding path/hash, Sketch path/hash, frozen implementation-profile identity/hash, project-claim hash, measurement fingerprint, official harness hash, and base/reference hash;
- a finite, deterministically ordered set of tunable fields and values that includes the accepted configuration as fallback/control;
- a public-input shape-key policy when multiple fixed configurations are required;
- the search comparison objective, maximum configuration count, time budget, warmup/repeat settings, mutation-reset requirements, deterministic winner/tie rule, and `pin_selected_config: true`.

The measurement fingerprint continues to identify only the official measurement contract. Search protocol fields are authenticated by the immutable final-tuning Decision hash; search measurements rank configurations only and cannot satisfy any official gate.

Every tunable field must already be declared `preferred` or `exploratory`, remain within a `supported|constrained` exact-scope profile capability, and preserve the accepted Sketch, algorithm, precision contract, effects, aliases, Host Plan, and public interface. Fusion, algorithm substitution, compute-unit substitution, precision changes, lifecycle changes, and other semantic/dataflow changes require a normal Designer round. When the frozen profile admits no alternative configuration, the domain is the singleton accepted configuration and the gate performs the same pin/hash confirmation and final verification without exploratory trials.

The tuning operation is offline and bounded. Verifier evaluates the exact accepted candidate hash through profile-declared temporary launch/meta-parameter injection under measurement exclusivity. Temporary storage may contain configuration tables, compiler caches/binaries, and raw command output, but no candidate-language source copy. Verifier owns compile, correctness screening, mutation reset, comparable search measurements, and confidence facts for every configuration.

Search completion returns normalized trials to Orchestrator as a non-persistent role handoff. Orchestrator runs the pure selector and sends only the normalized selected configuration to Coder. Coder emits exactly one pinned candidate derived from the accepted source when an improved configuration wins, or emits no source when the accepted fallback/control wins; binding validation then runs on the exact final source. If interrupted before the final report exists, the same reserved artifact index resumes and repeats the deterministic search; it does not allocate another finalization.

After pinning or fallback confirmation, Verifier runs complete correctness, lowering, promotion, and official competition evidence on the exact final source hash, then atomically writes the existing report once. Its `final_configuration_tuning` fact pack separates `search_trials` from `post_pin_official`, records the immutable `submission_snapshot_id`, selected configuration, `selection_outcome: improved|fallback-retained`, final candidate/binding hashes, and all final gates. Orchestrator reruns the same pure selector from the sealed report and verifies the pinned values before writing the verdict. The sealed report is the sole persisted finalization evidence before verdict creation; no finalization-specific state or artifact family is introduced.

Finalization uses a pure submission-promotion predicate: config-only pin conformance or accepted-source confirmation, valid binding, correctness, required promotion evidence, and complete official competition evidence. It does not emit an `accepted` terminal result or call campaign run policy. When an improved configuration satisfies the predicate, Orchestrator atomically advances `last_accepted_kernel` to the pinned source and `last_accepted_report` to the sealed finalization report while leaving `last_accepted_round` unchanged; when the accepted fallback/control wins, both canonical pointers remain unchanged. Both outcomes require the same final correctness and official measurement and route as `submission-ready`; any failed gate routes `blocked`. The final candidate contains one fixed selected configuration and no runtime/online `@triton.autotune`, adaptive search, or autotune-cache selection dependency.

## 15. Prerequisite fixes

Before enabling vNext profile onboarding or campaigns, the implementation plan must also:

1. unify `state/*_context.md` and legacy `state/*_state.md` naming;
2. make Decision validation check referenced artifact existence and runtime-fingerprint anchors, not only path shape;
3. synchronize the implementation-profile registry, `SKILL.md`, and repository README support lists;
4. add schema/version/hash checks for probe definitions, probe runs/results, Sketch, implementation-profile snapshot, binding, and verdict artifacts;
5. implement the referenced profile-probe runner rather than leaving declarative probe commands without an executable owner;
6. keep `target_id` distinct from `implementation_profile_id` across profiles, probe artifacts, claims, and snapshots;
7. update run-policy evaluation so attribution-specific counter effects are explicit.

These are compatibility and enforcement prerequisites, not new optimization behavior.

## 16. Implementation plan

### Phase A: schemas and fixtures

- Define JSON Schemas for probe definitions, probe runs, probe results, profile-promotion candidates, Sketch, project capability claim, profile, binding, and verdict.
- Add valid and invalid fixtures for probe identity/hash/evidence failures and shape/type/SSA/index/effect/hint failures.
- Add profile fixtures for supported, constrained, unknown, unsupported, prohibited, partial, target mismatch, implementation-profile mismatch, and finite exact-scope configuration legality cases.

### Phase B: profile onboarding execution

- Add canonical profile and probe-definition validation.
- Add a shell-free, timeout-bounded `run_profile_probe.py` command runner with atomic run-local artifacts and evidence hashing.
- Add deterministic promotion-candidate generation and Markdown rendering that never edit the canonical profile.
- Prove with fixture commands that a pre-campaign probe can complete without campaign state or accelerator hardware.

### Phase C: semantic and attribution checkers

- Extend or replace `validate_decision.py` with typed Sketch validation.
- Add implementation-profile snapshot and capability matching validation.
- Add source binding/conformance checker with source hash and span checks.
- Add causal graph structural checker.
- Add final-tuning Decision, report fact-pack, pinned-binding, and verdict validation without a new artifact family.
- Add verdict schema and rule-precondition validator.

### Phase D: role and routing integration

- Update Designer, Coder, Verifier, and Orchestrator contracts.
- Add `binding_NNN.json` and `verdict_NNN.json` to artifact gates and Git evidence ledger.
- Let Phase 0 and bounded campaign-local probes consume the shared probe contract without granting them canonical profile write authority.
- Add attribution-specific routing and counter behavior.
- Add one-shot submission-finalization routing for finite profile-legal configuration comparison, exact-source confirmation, fresh binding validation, post-pin official verification, atomic accepted kernel/report pair promotion, and a counter-free verdict branch.
- Keep v1/v2 campaigns read-only.

### Phase E: implementation-profile migration

- Convert one partial implementation profile first and keep its unknowns honest.
- Add at least one real versioned probe payload plus fixture-only runner coverage; do not record fabricated hardware success.
- Keep Markdown profile documents as rendered human-facing explanations during migration.
- Promote additional profiles only after schema, runner, result, and promotion fixtures pass.
- Keep the profile/probe schema language-neutral so later `ascendc` or other C-like profiles can supply their own build/runner payloads without copying the lifecycle.

## 17. Acceptance requirements

The vNext implementation is ready for profile onboarding and a new campaign only when tests prove:

1. A pre-campaign profile probe can execute a fixture payload and terminate without creating `team-state.md`, a Decision, a round, a candidate, or a benchmark result.
2. Frozen profile/definition/runtime inputs, run manifests, results, runtime fingerprints, command argv, evidence references, byte counts, and SHA-256 values are validated deterministically.
3. Probe execution uses no shell, enforces a timeout, confines all paths, rejects reused run directories, and records the exact `environment-blocked|probe-failed|partial|evidence-ready` classification without claiming unsupported facts.
4. A validated probe result, proposed promotion candidate, and rendered note do not modify the canonical profile; promotion requires a separate reviewed commit.
5. `target_id` and `implementation_profile_id` mismatches fail explicitly, and API-compatible targets do not inherit one another's evidence.
6. Phase 0 may consume only approved, archived, exact-scope prior evidence and still performs current runtime identity matching; campaign-local Coder probes remain non-authoritative for the project claim, correctness, and lowering.
7. A Sketch with invalid shape/type/SSA/index/effect semantics is rejected before Coder dispatch as `design-error`.
8. A required Unknown or prohibited capability cannot enter a proceeding Sketch.
9. Required/preferred/exploratory hint behavior is distinct and deterministic.
10. A valid Sketch can bind one-to-many, many-to-one, and many-to-many source relations with explicit reasons.
11. Missing or stale source spans fail binding conformance after candidate hash changes.
12. A candidate that violates binding is routed to one local Coder repair and then terminates correctly if repair fails.
13. A candidate that passes Sketch and binding but lacks expected lowering becomes `lowering-unknown`, not `code-error`, and does not increase the failed streak.
14. A disconnected causal graph is `design-error`.
15. A correct candidate with mechanism improvement but insufficient e2e/promotion improvement is `no-improvement` with evidence for Designer.
16. Missing required evidence routes to a bounded probe, then to `blocked` or `design-rejected` according to the missing-evidence rule.
17. Campaign verdict artifacts contain input hashes, rule IDs, preconditions, classification, terminal result, and counter effect.
18. Frozen implementation-profile snapshots preserve historical interpretation after canonical profile updates.
19. Existing v1/v2 campaign artifacts remain readable and are not silently rewritten.
20. Probe, profile, Sketch, binding, causal, and verdict checkers reject malformed or schema-version-incompatible artifacts deterministically.
21. A frozen implementation-profile snapshot contains the complete schema/probe-input/approved-evidence closure and still validates after the canonical profile directory is changed or removed.
22. An S60-shaped attention or MoE requirement with Unknown exact-scope `matrix.dot`, supported `reduction.sum`, and `probe_policy: before-fallback` selects only the unique matching dot probe; unrelated Unknowns are ignored.
23. Zero matching probes produce `no-exact-probe`, while multiple exact matches fail as ambiguous rather than selecting by order.
24. Numerically checked dot success emits proposed promotion artifacts and `promotion-pending`, creates no campaign, and cannot satisfy Phase 0 before explicit maintainer disposition.
25. Partial, failed, blocked, or unavailable qualification leaves `matrix.dot` Unknown; a later sum fallback is valid only with explicit primary/fallback signatures, onboarding disposition, and causal consequence in the Decision.
26. The frozen project claim embeds maintainer fallback authorization and optional hash-only probe provenance without a raw result ref; after deleting the pre-campaign run, the claim, snapshot, and fallback Decision still validate independently, while any disposition-field mutation invalidates its canonical hash.
27. Each Triton submission snapshot runs final tuning exactly once. Decision, report, and verdict carry one canonical `submission_snapshot_id`; validated verdict scanning rejects the same input ID and rejects a current accepted candidate/binding already recorded as a final output under unchanged anchors.
28. The Decision declares a finite deterministic domain covered by reviewed exact-scope profile legality and includes the accepted configuration as fallback/control. Missing or Unknown legality blocks; a singleton is valid only when explicitly covered.
29. Every tuning field is `preferred|exploratory` and configuration-only; any algorithm, dataflow, precision, effect, alias, Host Plan, public-interface, or semantic layout change is rejected and requires a normal Designer round.
30. Search executes one accepted candidate source through temporary launch/meta-parameter injection. The ignored boundary contains no derived candidate source, and normalized facts use the existing artifact families without finalization-specific state.
31. Verifier enforces the declared trial/time budget, order, warmup/repeat settings, mutation reset, search protocol, objective, and deterministic tie rule. Search results are handed to Orchestrator without a persisted selection artifact; the report is written atomically only after final verification.
32. Coder pins only the selected configuration into the sole candidate or confirms the accepted fallback/control, and binding validation runs against the exact final source hash; stale binding hashes fail.
33. Full correctness, lowering evidence, promotion evidence, and official competition measurement run on the exact pinned-or-retained source hash. Search measurements alone cannot authorize submission.
34. `selection_outcome: improved|fallback-retained` is report evidence. Both outcomes require the pure submission-promotion predicate and route only as `submission-ready`; an improved winner atomically advances `last_accepted_kernel` and `last_accepted_report`, a fallback-retained winner changes neither, and a partial pair update or failed gate is rejected.
35. Finalization artifacts use a deterministic recoverable artifact index but do not update `last_accepted_round`, campaign-round pointers, or counters. Their verdict branch omits attribution, terminal-result, counter-effect, and run-policy projection fields and never invokes continuous-run policy evaluation.
36. The final candidate contains one fixed selected configuration and no runtime/online `@triton.autotune`, adaptive search, or autotune-cache selection dependency.
