# Report 000

Result: baseline

## Identity

- Round: `000`
- Candidate: `baseline_adapter.py` @`1532b55e399da3a8404f75d31ee7f2453a32f7baef41d10425f556931400ac0c`
- Accepted reference: `../../base.py` @`dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint` (match)
- verification_tier: `baseline`

## Correctness and Guardrails

| Check | Observation | Verdict |
|---|---|---|
| correctness | `PASS accuracy` in all 3 timing pairs + profile run | pass |
| runtime bootstrap | torch_gcu/triton_gcu matched fingerprint | pass |
| immutable base | sha256 unchanged | pass |

## Interleaved Wall Timing

- warmup 50 / repeat 100 / seed 42 / interleaved pairs

| Invocation | Reference ms | Candidate ms | speedup |
|---:|---:|---:|---:|
| 1 | 0.252467 | 0.251723 | 1.003x |
| 2 | 0.253892 | 0.252403 | 1.006x |
| 3 | 0.252189 | 0.252865 | 0.997x |

Baseline reference median ≈ 0.252 ms (identity ~1.00x).

## Profiler Evidence

- trace: `log/report_000_forward.pt.trace.json`
- runtime_launch_count_per_call: 2.0 (`topsLaunchKernel` 2/call, 19.43 us/call launch-API)
- device_time_available: false (GCU launch-only trace)
- base SDPA dispatch: `F.scaled_dot_product_attention(is_causal=True)` → vendor flash-attention, 2 launches/call

## evidence_for_next_round

- base is 2-launch causal SDPA (vendor flash-attention), wall ~0.252 ms.
- epoch-1 naive = 0.42x (tl.sum scalar-expanded, tl.dot misjudged as Unknown).
- mm_encoder_attention s60 e2 recipe (fp16 QK^T tl.dot + fp32 PV + TP=128 + nw1) is the candidate formula; lead preflight measured 0.94x (2.2x over epoch-1).

## Stop Recommendation

- recommendation: `continue`
