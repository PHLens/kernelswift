# Round Status 003

## Round 3 Verification

- phase: verifying — complete
- round: 003
- result: no-improvement
- decision: rounds/decision_003.md (SHA256 c2d0d068f7595bed4aec4e2497b9b390ae875f67dcbcf9de551b448383991b37)
- candidate: triton_flexattention_003.py (SHA256 4faadac6cd0e3bb5d1faeaddafd899f0fd64c275632d2635f1612bf182686546)
- accepted reference: triton_flexattention_002.py (SHA256 b0fe058c5b5336978e89933e8f0fed0d5a0449aede33c6b1b7a97c9c319c100f)

## Completed Commands

1. correctness (warmup 5 / repeat 10): `PASS accuracy; v0=0.418195 ms, v1=0.327605 ms, speedup=1.277x` → exit 0
2. authoritative timing (warmup 50 / repeat 100, three interleaved pairs):
   - reference (002): 0.292920, 0.296535, 0.299435
   - candidate (003): 0.321280, 0.317370, 0.336525
3. profiler (forward / warmup 20 / iterations 50, --profile-reference-file):
   - reference device: `54.4268 us/call`, `1.00` kernels/call, total `2721.34 us`
   - candidate device: `24.0532 us/call`, `1.00` kernels/call, total `1202.66 us`

## Terminal Metrics

- reference_median_ms: `0.296535`
- candidate_median_ms: `0.321280`
- improvement_pct: `-8.344714789147998`
- hypothesis H-003 verdict: `partially-confirmed` (device_us_per_call decreased 54.43→24.05; kernel_count_per_call stayed 1; output_allocations stayed 0; but primary metric wall_time regressed)

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| base.py | `12f8a77b8f52b50d513800907b6b21ff9c98709647b306793241f6f8da3cb105` |
| triton_flexattention_002.py (reference) | `b0fe058c5b5336978e89933e8f0fed0d5a0449aede33c6b1b7a97c9c319c100f` |
| triton_flexattention_003.py (candidate) | `4faadac6cd0e3bb5d1faeaddafd899f0fd64c275632d2635f1612bf182686546` |
| auto_bench.py (harness) | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |

## Next Safe Action

None (Round 3 terminal: no-improvement). Await Orchestrator to apply state transitions (performance_miss_streak +1; canonical pointer unchanged).
