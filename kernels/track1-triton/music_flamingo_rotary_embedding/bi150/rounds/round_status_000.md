# Round Status 000

Result: baseline (Phase 0)

## Phase

`verifying` (Phase 0 baseline establishment; measurement_exclusive not yet enabled)

## Completed Steps

### 1. Frozen-artifact hash verification (before measurement)

| Artifact | Actual SHA-256 | project.md recorded | Match |
|---|---|---|---|
| `base.py` | `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341` | `98be7d25264fd4b2af69c6871f6b01a2f90f8d7e6d449d4c8ee85a55dd5c1c42` | **MISMATCH** |
| `baseline_adapter.py` | `433569bbac3bab158ff211a6de7ecb40ec7236d74a3eb7ab7c2b487e1b41772a` | `433569bbac3bab158ff211a6de7ecb40ec7236d74a3eb7ab7c2b487e1b41772a` | match |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | match |

base.py hash mismatch investigation: `git hash-object` and
`git rev-parse HEAD:<path>` both return `f1998e7fe09f798d5562be44dbc6453263a4a871`,
and `git status --short` shows no modification. The working-tree `base.py` bytes
are identical to the committed bytes; only the `base_sha256` recorded in
`project.md` (and propagated to the Orchestrator task message) disagrees with the
actual file SHA-256. This is a recorded-hash discrepancy, not a file mutation.
Adapter and harness hashes match exactly, so the adapter generation and harness
are consistent.

### 2. Runtime fingerprint check

`torch 2.7.1`, `triton 3.1.0`, `Iluvatar BI-V150`, capability `(7, 1)` — matches
`project.md#runtime-fingerprint`.

### 3. Baseline correctness (base.py vs baseline_adapter.py)

Command return code `0`; `PASS accuracy; v0=0.348312 ms, v1=0.348387 ms, speedup=1.000x`;
`Summary: 1 passed, 0 failed, 1 total.`

### 4. Baseline benchmark (three interleaved pairs)

warmup=50, repeat=100, three independent invocations.

| Invocation | v0 wall ms | v1 wall ms | speedup | return code |
|---|---:|---:|---:|---:|
| 1 | `0.353050` | `0.349781` | `1.009x` | 0 |
| 2 | `0.355387` | `0.354048` | `1.004x` | 0 |
| 3 | `0.353447` | `0.350067` | `1.010x` | 0 |

baseline raw samples (v0 side wall ms): `[0.353050, 0.355387, 0.353447]`
unrounded median: `0.353447 ms`

### 5. Baseline profiler

Command return code `0`. Trace written to
`log/round_000_forward_50iter.pt.trace.json`, SHA-256
`bb0fc81b27401060e945243f3ca9a52b0c6a40b0d9f7a7b897b46c1638892412`.

Summaries (separately scoped, iterations=50):

| Scope | device_total_us | device_us_per_call | kernel_count_total | kernel_count_per_call |
|---|---:|---:|---:|---:|
| `baseline_base` | `3431.821` | `68.636` | `543` | `10.86` |
| `candidate_baseline_adapter` | `3467.314` | `69.346` | `550` | `11.0` |

Top kernels (baseline_base scope, by us/call): MulFunctor elementwise (17.694),
CatArrayNoContiguous (10.954), AUnaryFunctor Mul (7.918), sin (7.277), cos
(7.132), direct_copy (6.064), BUnaryFunctor Mul (4.031), neg (3.952), arange
(3.615).

### 6. Post-measurement hash verification

Identical to pre-measurement values; no frozen artifact changed during measurement.

## Next Safe Action

Write `report_000.md` and `state/verifier_context.md`, then report Result=baseline
to Orchestrator. Flag the base.py `base_sha256` record discrepancy to Orchestrator
for correction (does not block correctness/benchmark/profiler, which all use file
contents directly).
