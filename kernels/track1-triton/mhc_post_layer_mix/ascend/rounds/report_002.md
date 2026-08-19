# Report 002 — Kernel Tuning Candidate

- round: `002`
- result: `no-improvement`
- change_family: `kernel-tuning`
- bottleneck_class: `device-bound`

## Decision

- decision: `proceed` (decision_002.md, H-002 kernel-tuning: BLOCK_C 256→1280, num_warps 4→2)
- candidate: `candidate_002.py`
- accepted-reference: `candidate_001.py` (last_accepted_kernel)

## Source Hashes

| Artifact | SHA-256 |
|---|---|
| base `../base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` |
| accepted reference `candidate_001.py` | `b74e407348d424c9265ddf831b245cda90297a48bdbaa576fa7e6b57b5d121f9` |
| candidate `candidate_002.py` | `6a66f302b3cbf2316b99c9d207e32161cb2bc05e4ea327279ce7be3d8955357c` |
| decision `decision_002.md` | `0539d245c659369917660581165e8a332e00a65ca9d56128f7a0fe4fbf4d2a21` |

## Correctness and Guardrails

Correctness PASS in all timing pairs and the profiler run (atol=1e-2 / rtol=1e-2).

| Guardrail | Expectation | Observation | Verdict |
|---|---|---:|---|
| correctness | pass within atol/rtol 1e-2 | PASS (all runs) | pass |
| output dtype/shape | `Tensor[2,4096,4,1280]` bf16 | PASS | pass |
| fp32 accumulation before single bf16 cast | fp32 acc, single cast | unchanged from accepted candidate | pass |
| kernel_count stays 1 | 1.0 | 1.0 | pass |
| no 2D-grid regression | `(8192,)` grid, weights loaded once | preserved exactly | pass |
| ModelNew public contract | `ModelNew().forward(...)` | present | pass |

## Authoritative Timing (3 interleaved pairs)

Each pair compares candidate_002 against accepted candidate_001 (both `ModelNew`,
so each is run as `--v1_file` against the fixed `--v0_file base.py`; warmup 50 /
repeat 100, npu:0, identical flags across runs).

| Pair | candidate_001 (ref) v1 ms | candidate_002 v1 ms |
|---|---:|---:|
| 1 | 0.887920 | 0.875655 |
| 2 | 0.876180 | 0.890115 |
| 3 | 0.880395 | 0.885480 |

- Unrounded median candidate_001 ≈ 0.880395 ms; candidate_002 ≈ 0.885480 ms.
- improvement_pct = (0.880395 − 0.885480) / 0.880395 × 100 ≈ **−0.58%**
  (candidate_002 is marginally slower, within measurement noise).

The ~0.5–1% wall gain reported in coder's smoke is NOT reproduced; the
interleaved authoritative timing shows candidate_002 within noise of (and
slightly behind) the accepted candidate_001. improvement_pct is negative and
far below the 5.0 adoption threshold.

## Profiler Evidence (CANN per-scope, forward, 20 warmup / 50 iter)

| Scope | device_us_per_call | kernel_count_per_call | device_ratio |
|---|---:|---:|---:|
| reference_candidate_001 (accepted) | 620.84 | 1.0 | ~0.71 (vs wall 0.880 ms) |
| candidate_candidate_002 | 596.92 | 1.0 | ~0.68 (vs wall 0.885 ms) |

- Single kernel `mhc_fused_kernel` in both scopes; kernel_count stays 1.0.
- device_us_per_call improved 620.84 → 596.92 us (−3.9%). This is a real but
  small device-time reduction, consistent with the latency-bound hypothesis
  (fewer/larger contiguous c-iterations raise memory-level parallelism), but it
  is smaller than coder's ~7% "min" estimate and did NOT translate into wall
  time.

## Evaluation Contract Mirror (H-002)

| Mechanism observable | Expectation | Observation | Verdict |
|---|---|---:|---|
| device_us_per_call | decrease below 620 us | 596.92 us (< 620) | confirmed (−3.9%) |
| kernel_count_per_call | remain 1.0 | 1.0 | confirmed |
| wall_time | decrease below 0.880 ms | 0.8855 ms (no decrease) | falsified |
| primary wall_time improvement ≥5% | required for adopt | −0.58% | falsified |

## Hypothesis Verdict

`falsified` — the tuning raised device throughput (device_us_per_call fell ~4%),
but the expected wall-time improvement did not materialize: wall time is
dominated by a harness-fixed host/sync gap (~0.26 ms), so a ~24 us device gain
is swamped and candidate_002 measures within noise of (slightly slower than) the
accepted candidate. The causal chain "device_us falls → wall_time falls" broke
at the wall-time link because device_ratio remains ~0.68–0.71.

## Retry History

- No correctness retries; candidate_002 passed correctness on first run.

## evidence_for_next_round

- **BLOCK_C tuning genuinely improves device time**: candidate_002 (BLOCK_C=1280,
  num_warps=2) reaches device_us_per_call 596.92 us vs 620.84 us for the accepted
  candidate_001 (BLOCK_C=256, num_warps=4) — a real ~4% device-side reduction.
  This confirms the kernel is latency/occupancy-bound and the tuning knob is
  directionally correct.
- **Wall time is host-overhead dominated**: candidate_002's wall median (0.8855 ms)
  is within noise of (slightly worse than) candidate_001 (0.8804 ms) despite the
  device gain. device_ratio for both candidates ≈ 0.68–0.71, i.e. ~30% of wall
  time is host-side (per-sample `sync_devices()` + `set_seed` + launch latency),
  which is not touched by a kernel-only change.
- Falsified/remaining mechanism: device_us_per_call no longer controls wall time;
  the dominant remaining wall-time cost is the harness-fixed host/sync gap. Any
  further wall-time gain requires attacking host overhead (unproven `fast_libentry`
  launcher) — outside the current kernel-only change family — or accepting that
  the operator is at its practical floor under this harness.
- Bottleneck: shifted from device-bound (round 000) to host-overhead-bound
  (device_ratio ~0.70) after kernel fusion; further device tuning yields
  diminishing (sub-threshold) wall returns.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/mhc-post-layer-mix-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py \
  --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py \
  --v1_file kernels/track1-triton/mhc_post_layer_mix/ascend/candidate_002.py \
  --warmup 50 --repeat 100

/usr/local/python3.11.15/bin/python3 auto_bench.py \
  --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py \
  --v1_file kernels/track1-triton/mhc_post_layer_mix/ascend/candidate_002.py \
  --profile --profile-reference-file kernels/track1-triton/mhc_post_layer_mix/ascend/candidate_001.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output kernels/track1-triton/mhc_post_layer_mix/ascend/log/round_002_forward_50iter.pt.trace.json
```
