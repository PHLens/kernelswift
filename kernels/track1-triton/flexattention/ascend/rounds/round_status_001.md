# Round Status 001

## Round 1 Verification

- phase: verifying — complete
- round: 001
- result: accepted
- decision: rounds/decision_001.md (SHA256 91cae0bcb4eb0792e59be2c359b21dde2cc038a2d11e25f01e36bb20784bf379)
- candidate: triton_flexattention_001.py (SHA256 53e87eff27457f6268040c64979f99dcf30a809effc562caec3db951b141d4a3)
- accepted reference: baseline_adapter.py (SHA256 31c4e9acea7d94ddd97740dbd3d33e6b505cbc3a118ed891b28f9e1ac5c0696f)

## Completed Commands

1. correctness (warmup 5 / repeat 10): `PASS accuracy; v0=0.403910 ms, v1=0.329930 ms, speedup=1.224x` → exit 0
2. authoritative timing (warmup 50 / repeat 100, three interleaved pairs):
   - pair 1: v0=0.386255, v1=0.330810 (speedup 1.168x)
   - pair 2: v0=0.405655, v1=0.326940 (speedup 1.241x)
   - pair 3: v0=0.411600, v1=0.337995 (speedup 1.218x)
3. profiler (forward / warmup 20 / iterations 50): PASS, scopes `baseline_base` and `candidate_triton_flexattention_001`
   - reference device: `145.4256 us/call`, `8.72` kernels/call, total `7271.28 us`
   - candidate device: `54.04 us/call`, `1.00` kernels/call, total `2702.00 us` (single `_causal_attn_kernel`)

## Terminal Metrics

- reference_median_ms: `0.405655`
- candidate_median_ms: `0.330810`
- improvement_pct: `18.450407365865082`
- hypothesis H-001 verdict: `confirmed`

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| base.py | `12f8a77b8f52b50d513800907b6b21ff9c98709647b306793241f6f8da3cb105` |
| baseline_adapter.py | `31c4e9acea7d94ddd97740dbd3d33e6b505cbc3a118ed891b28f9e1ac5c0696f` |
| triton_flexattention_001.py | `53e87eff27457f6268040c64979f99dcf30a809effc562caec3db951b141d4a3` |
| auto_bench.py (harness) | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |

## Next Safe Action

None (Round 1 accepted). Await Orchestrator to apply state transitions and canonical pointer advance.
