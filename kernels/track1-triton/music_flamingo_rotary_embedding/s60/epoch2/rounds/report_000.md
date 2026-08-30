# Report 000

Result: baseline

## Identity

- Round: `000`
- Candidate: `baseline_adapter.py` @`9fc87abbe0e6268f06c969e94f5400abea51cdf315276a4ac5cef5bd0ad8a26f`
- Accepted reference: `../../base.py` @`99829754f4cdc4bfd2808e051de549f0791414241e7fdbad7a1b8294a15be475`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint` (match)
- verification_tier: `baseline`

## Correctness and Guardrails

| Check | Observation | Verdict |
|---|---|---|
| correctness | `PASS accuracy` in all 3 timing pairs | pass |
| runtime bootstrap | torch_gcu/triton_gcu matched fingerprint | pass |
| immutable base | sha256 unchanged | pass |

## Interleaved Wall Timing

- warmup 50 / repeat 100 / seed 42 / interleaved pairs

| Invocation | Reference ms | Candidate ms | speedup |
|---:|---:|---:|---:|
| 1 | 0.559606 | 0.560442 | 0.999x |
| 2 | 0.447718 | 0.447482 | 1.001x |
| 3 | 0.446869 | 0.442808 | 1.009x |

Baseline reference median ≈ 0.447 ms (identity ~1.00x).

## Profiler Evidence

- base: 13 launches (elementwise chain div/mul/repeat_interleave/broadcast/cat/mul
  + vendor cos/sin), device_time_available=false (GCU launch-only trace)

## evidence_for_next_round

- epoch-1 FULL fusion (tl.cos/tl.sin) = -13% (GCU math-dialect trig slow).
- preflight NEW direction: PARTIAL fusion (freqs elementwise -> single kernel,
  cos/sin kept vendor torch.cos/torch.sin) = 1.49x, correctness exact-match 0.0.

## Stop Recommendation

- recommendation: `continue`
