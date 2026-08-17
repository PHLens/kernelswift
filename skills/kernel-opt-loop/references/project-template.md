# <Operator> Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `<absolute-project-root>`
- base: `base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `<absolute-harness-path>`
- interpreter: `<absolute-interpreter-path>`
- device: `<device>`
- implementation_language: `triton`
- implementation_backend: `<selected runtime backend>`
- target_profile: `<selected complete target profile>`

## Semantics

- operator: `<operator>`
- inputs: `<names, shapes, dtypes, layouts, and devices>`
- outputs: `<names, shapes, dtypes, layouts, and devices>`
- mathematical_behavior: `<reference behavior>`
- tolerance_and_tie_rules: `<exact comparison contract>`
- public_contract: `<ModelNew constructor and forward signature>`

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- `<semantic invariant>`
- `<environment invariant>`
- `<lifecycle invariant>`
- `<measurement invariant>`

The complete workflow-level rules are in `references/invariants.md`.

## Runtime Fingerprint

```yaml
triton_distribution: <discovered-distribution>
triton_version: <discovered-version>
backend_target: <active-driver-or-compiler-target>
backend_version: <discovered-version-or-null>
device_arch: <discovered-device-architecture>
```

- target_profile_match: `pass | fail`
- discovery_commands: `<exact commands>`
- discovered_at: `<UTC timestamp>`

These values are observed in Phase 0. They are not assumed from the profile.

## Measurement Regime

- harness_path: `<absolute-harness-path>`
- harness_sha256: `<sha256>`
- shape: `<shape>`
- dtype: `<dtype>`
- device: `<device>`
- warmup: `<count>`
- repeat: `<count>`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `<count>`
- profiler_scopes: `accepted_reference,candidate`
- correctness_command: `<exact command>`
- benchmark_command: `<exact command>`
- profiler_command: `<exact command>`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `<sha256 of base.py, harness, shape, dtype, device, warmup/repeat, and profiler settings>`
- base_sha256: `<sha256>`
- baseline_adapter_sha256: `<sha256>`
- fingerprint_command: `<exact command>`

A fingerprint change requires a new comparable baseline before optimization can
continue.

## Optional Target

- target_mode: `absolute_latency_ms | speedup_vs_baseline | null`
- target_value: `<user-provided value-or-null>`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `<baseline measurement_fingerprint-or-null>`

An optional target is comparable only when it uses `wall_time_ms` under the
baseline measurement fingerprint. It is not an estimated bound, device-time
goal, or inferred target. Orchestrator records later target amendments in the
append-only team-state policy-revision table at a safe terminal boundary.

## Git Run Identity

- base_branch: `<branch used for Phase 0>`
- base_commit: `<Phase 0 commit>`
- run_branch: `kernel-opt/<operator>-<run-epoch-or-timestamp>`

These fields mirror `team-state.md` and identify the dedicated optimization
branch. The run branch is never `main`, `master`, or `dev`.

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | `<median>` | `<per-call>` | - | not-applicable | `baseline_adapter.py` |

Orchestrator appends one row only after a terminal round transition is validated
and committed. Rejected candidates remain listed but never become the comparison
source.

## Reproduction

```bash
<baseline correctness and benchmark command>
```

```bash
<separately scoped accepted-reference/candidate profiler command>
```
