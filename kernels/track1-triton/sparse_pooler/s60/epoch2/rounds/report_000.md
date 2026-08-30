# Report 000

Result: baseline

## Identity

- Round: `000`
- Candidate: `baseline_adapter.py`
- Accepted reference: `../../base.py` @`46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58`
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
| 1 | 0.900805 | 0.897512 | 1.004x |
| 2 | 0.845576 | 0.835659 | 1.012x |
| 3 | 0.837997 | 0.837257 | 1.001x |

Baseline reference median ≈ 0.838 ms (identity ~1.00x).

## Profiler Evidence (preflight decomposition)

Per forward call:
- dense GEMM [83,768]@[768,768] + GELU + LayerNorm: ~165us
- decoder GEMM [83,768]@[768,30522]: ~316us
- log1p(relu) elementwise [83,30522]: ~110us
- max pooling (4 segments) + D2H sync: ~183us (D2H `seq_lens.tolist()` = 125us, 16%)

11 topsLaunchKernel/call. GEMM-bound (481us, 61%).

## evidence_for_next_round

- All optimization directions falsified (see project.md Key Prior):
  1. epoch-1 fused relu/log1p/max: -26.79%
  2. scatter_reduce segment max: 7x slower
  3. D2H sync elimination requires slower hand-written segment reduction
- GEMM (61%) vendor-bound and untouchable.

## Stop Recommendation

- recommendation: `continue` (Designer to confirm or find an untried direction)
