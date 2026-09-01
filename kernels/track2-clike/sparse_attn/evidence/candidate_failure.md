# Report 006

Result: candidate-failed

## Identity

- Round: `006`
- Decision: `rounds/decision_006.md`
- Candidate: `candidate_006.py`
- Device source: `sparse_attn_full_row_vector_exp.cpp`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `6133024c22cd81ed8f345a9f6cc9975b5db2c82bb8c618d54f8cfd50b51e7404`
- Sketch SHA256: `db493a72f73f93beced9ec2723f44f7852c124e7e5a3e82533a5221e5fe9ff0d`
- Candidate SHA256: `9075804117456fa66a33b010787218dc68a41e11d6665c8fdbf0157f43dbdc39`
- Device source SHA256: `2aa0a0f2627acad12bdf706143d466947a72bba5b6d6e2d163b99c5bdc1e9f6e`
- CMake SHA256: `e681f5f24d925b5487beeca306817f76d8c0dbc95d9294f2bf5703be44754316`
- Shared object SHA256: `a2816f94fe14337970443633049084ad3d423db2a494ebbbf6f7ad9321f18e67`
- Binding SHA256: `b9c98f8ea4e36df35e45cc8db24f2d33706dd28c91dc9d202020c91c5bf753f9`
- Alignment ledger SHA256: `bd8edb57262105944ae3e0bf441f8c59af9f14d9ec1f2ee0ff7eff0fdb25fe98`
- Launcher evidence SHA256: `553f9d5267c2b96595858a6eca868eee4280753ae85c484dc22cdaff75f1a2e5`
- Coder result SHA256: `0f321caf2efefe55b5352b7ea1116557b53aaedbf7fa35368e61ef5f385ad2f8`
- Candidate-set SHA256: `2b05f8a71dd32b2f0a764bff00394794812d5c556b53c11770f70a5a202c167b`
- Accepted reference SHA256: `5922fccb822f18d2472b49706b349033733309d2a7cfd5abe0d2054df71632c2`
- Base SHA256: `64fe0fbd270c0270ed7065dd63cd5a1aabd580fd8791f5c4b4dd7504b63c4a88`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Profile snapshot SHA256: `80a02c285a44f231c899dfa85ec3f75dde10ed9e362beb7621e0d1fbbc11d369`
- Capability claim SHA256: `bdac46da2494af0e6640dd0aff830b966d6fcc2737b907c9bd0bede890ec70e1`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `13104b9a206d67116e524ae41a61cd5fece6b44f012c4f4a99a18d797f42ac5f`
- verification_tier: `correctness-only terminal failure`

All frozen hashes matched before execution and again after the terminal failure. Official Decision, Sketch, and multisource binding validation passed; binding returned `valid=true`, five-operation coverage, nineteen required hints, and canonical candidate-set `2b05f8a71dd32b2f0a764bff00394794812d5c556b53c11770f70a5a202c167b`. Static inspection found no candidate-added synchronization or scalar `expf`. Independent `readelf --dyn-syms --wide` output SHA was `77ac0b7a08efc30344e44a1c2c666f8e370d595f33886e73993746ac1c860789`; `aclrtlaunch_sparse_attn_exact_scope` was the unique defined `FUNC GLOBAL DEFAULT` exact export.

## Fail-fast Correctness Sequence

| Stage | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| load/export smoke | module loads and exact launcher resolves | `aclrtlaunch_sparse_attn_exact_scope` resolved successfully | pass | `log/round_006_load_export_smoke.txt` |
| cheapest minimal correctness | all-invalid returns exact zero with correct shape/dtype/layout, non-aliasing, and immutability | every check passed | pass | `log/round_006_minimal_correctness.txt` |
| official nominal correctness | official `atol=1e-2`, `rtol=1e-2`, `equal_nan=True`, seed 42 | output differed: max absolute difference `3.165039`, mean absolute difference `0.1596805`; `0 passed, 1 failed` | fail; terminal | `log/round_006_nominal_correctness.txt` |
| remaining semantic guardrails | run only after nominal passes | prohibited after nominal failure | not run | fail-fast contract |

The exact launcher loads, launches, and handles the all-invalid zero special case, but general official nominal output is numerically incorrect. This is terminal `CODE.CORRECTNESS.FAIL` and does not license any conclusion about mixed-invalid, duplicates, arbitrary nonzero sink, non-default stream, runtime synchronization sentinel, or performance.

## Screening, Timing, and Profiler

- Semantic guardrails: not run after nominal failure.
- Short screening: not run after nominal failure.
- Authoritative interleaved timing: not run after nominal failure.
- Targeted profiler / observed lowering: not run after nominal failure.
- Candidate median and device duration: unavailable.

## Evaluation Contract Mirror

| Observable | Observation | Verdict | Evidence |
|---|---|---|---|
| `required_exp_duplicate_capability_closure` | frozen compile-only preflight and candidate static/binding gates were present | observed statically only | frozen Decision/binding evidence |
| `exponential_source_closure` | no scalar `expf`; candidate source contains the frozen vector `Duplicate<float>` / `Exp<float>` construction | observed statically; runtime numeric result failed | `sparse_attn_full_row_vector_exp.cpp`; `rounds/binding_006.json` |
| `numeric_semantics_source_closure` | binding/static claims passed, but official numerical output failed | falsified at runtime | `log/round_006_nominal_correctness.txt` |
| `ctypes_launcher_export_closure` | unique exact defined export and load smoke passed | observed | `rounds/launcher_export_evidence_006.json`; `log/round_006_load_export_smoke.txt` |
| `vector_operand_alignment_ledger_coverage` | static ledger and binding gates passed; no runtime alignment exception occurred in executed cases | observed statically | `rounds/alignment_ledger_006.json`; `rounds/binding_006.json` |
| `official_correctness_and_semantic_guardrails` | nominal failed; remaining semantic cases prohibited | failed | `log/round_006_nominal_correctness.txt` |
| `short_screening_wall_ms` | not measured | unavailable | correctness-first contract |
| `device_us_per_call` | not profiled | unavailable | correctness-first contract |

- Hypothesis ID: `H-006`
- Hypothesis verdict: `falsified by terminal official nominal numerical failure`
- Classification: `code-error`
- Rule: `CODE.CORRECTNESS.FAIL`

## Retry History

| Attempt | Stage | Frozen artifact state | Outcome |
|---:|---|---|---|
| 1 | load/export smoke | unchanged | pass |
| 2 | cheapest all-invalid minimal correctness | unchanged | exact-zero pass |
| 3 | first official nominal correctness | unchanged | numerical failure: max abs `3.165039`, mean abs `0.1596805` |

No implementation repair was requested or performed. No candidate execution occurred after the nominal failure.

## evidence_for_next_round

- Loader, exact export resolution, 20800-block launch path, and all-invalid exact-zero behavior work for this frozen artifact.
- General nominal semantics remain incorrect despite static source/binding closure; the observed error is large relative to official tolerance.
- The minimal all-invalid pass cannot substitute for official nominal or the unexecuted mixed-invalid, duplicate, nonzero-sink, stream, immutability, non-alias, and synchronization-sentinel suite.
- Feasibility, wall time, observed lowering, and device performance remain completely unmeasured.
- Canonical stays `baseline_adapter.py` / `rounds/report_000.md`.

## Stop Recommendation

- recommendation: `continue to a new design round`
- disposition: `do not promote candidate_006.py`

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift-dev-4ff2094/kernels/track2-clike/sparse_attn/ascendc && /usr/local/python3.11.15/bin/python3 /workspace/kernelswift-dev-4ff2094/auto_bench.py --v0_file ../base.py --v1_file candidate_006.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 0 --repeat 1
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "9075804117456fa66a33b010787218dc68a41e11d6665c8fdbf0157f43dbdc39",
  "correctness": {
    "status": "fail",
    "evidence": [
      "log/round_006_minimal_correctness.txt",
      "log/round_006_nominal_correctness.txt"
    ]
  },
  "observables": [
    {
      "name": "ctypes_launcher_export_closure",
      "status": "observed",
      "value": "unique exact launcher export and runtime load resolution passed",
      "confidence": "high",
      "evidence": ["rounds/launcher_export_evidence_006.json", "log/round_006_load_export_smoke.txt"]
    },
    {
      "name": "minimal_all_invalid_correctness",
      "status": "observed",
      "value": "exact zero, shape/dtype/contiguity, non-aliasing, and immutability passed",
      "confidence": "high",
      "evidence": ["log/round_006_minimal_correctness.txt"]
    },
    {
      "name": "official_correctness_and_semantic_guardrails",
      "status": "inconclusive",
      "value": "nominal failed numerically; remaining semantic guardrails were not run",
      "confidence": "high",
      "evidence": ["log/round_006_nominal_correctness.txt"]
    },
    {
      "name": "short_screening_wall_ms",
      "status": "unavailable",
      "value": "not measured: correctness failure",
      "confidence": "high",
      "evidence": ["rounds/report_006.md#screening-timing-and-profiler"]
    },
    {
      "name": "device_us_per_call",
      "status": "unavailable",
      "value": "not profiled: correctness failure",
      "confidence": "high",
      "evidence": ["rounds/report_006.md#screening-timing-and-profiler"]
    }
  ],
  "lowering": {
    "status": "unavailable",
    "expected_mechanism": "unmeasured",
    "evidence_contract": "correctness-failure-no-profiler",
    "evidence": ["rounds/report_006.md#fail-fast-correctness-sequence"]
  },
  "evidence_gap_cause": "none"
}
```
