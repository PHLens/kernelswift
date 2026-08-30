# Round Status 000

## Phase 0 Baseline Verification

- phase: verifying (baseline) — complete
- round: 000
- result: baseline

## Completed Commands

1. correctness (warmup 5 / repeat 10): `PASS accuracy; v0=0.404295 ms, v1=0.407810 ms, speedup=0.991x` → exit 0
2. baseline benchmark (warmup 50 / repeat 100): `PASS accuracy; v0=0.409435 ms, v1=0.410860 ms, speedup=0.997x` → exit 0
3. profiler (forward / warmup 20 / iterations 50): PASS, scopes `baseline_base` and `candidate_baseline_adapter` captured
   - reference device: `148.0188 us/call`, `8.66` kernels/call, total `7400.94 us`
   - candidate device: `137.3944 us/call`, `8.84` kernels/call, total `6869.72 us`

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| base.py | `12f8a77b8f52b50d513800907b6b21ff9c98709647b306793241f6f8da3cb105` |
| baseline_adapter.py | `31c4e9acea7d94ddd97740dbd3d33e6b505cbc3a118ed891b28f9e1ac5c0696f` |
| auto_bench.py (harness) | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |
| measurement_fingerprint | `c1359d456700562802630e66368ce04856d871a993562ce1437e037af82581b8` |

## Raw Samples

- reference_median_ms: `0.409435`
- candidate_median_ms: `0.410860`
- improvement_pct: `-0.3481`

## Next Safe Action

None (Phase 0 baseline complete). Await Orchestrator to apply state transitions.
