# Report 002

## Decision

- classification: `no-improvement`
- candidate: `kernels/track1-triton/music_flamingo_rotary_embedding/ascend/triton_rotary_002.py`
- candidate_sha256: `98e86e48f00a25254561ad5bf8ef91824c87e5bb9dfa0279f221633d910b07ab`
- accepted reference: `kernels/track1-triton/music_flamingo_rotary_embedding/ascend/triton_rotary_001.py` (Round 1 accepted)
- accepted-reference sha256: `51a9a33b82f550abfd80400bb0748b74fd181d0f3c4fd4b5d70b4ca1f5d6984e`
- harness sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- design: `rounds/design_002.md` (decision `proceed`, hypothesis `H-002`, change_family `row-parallel-vectorization`)

## Correctness

- command: `python3 auto_bench.py --v0_file .../base.py --v1_file .../triton_rotary_002.py --warmup 5 --repeat 10 --full-traceback`
- result: `PASS accuracy; v0=0.637150 ms, v1=0.342245 ms, speedup=1.862x` (1 passed, 0 failed)
- exit status: 0

Tuple output `(cos, sin)` compared element-wise (harness `compare_values` tuple branch recurses
per-tensor `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`). Passed.

## Guardrail matrix

| Guardrail | Expectation | Observation | Verdict |
|---|---|---|---|
| correctness | pass | `PASS accuracy` (all runs) | pass |
| output structure | tuple `(cos, sin)` | returns `(cos_out, sin_out)` | pass |
| output shape | each `[4,32,128]` fp32 | `(B, SEQ, 2*DIM)` = `[4,32,128]` fp32 | pass |
| register_buffer semantics | frequency tables precomputed in `__init__`, not per-forward | `inv_freq`, `position_angles`, and new `batch_freq_base` all `register_buffer`, precomputed once | pass |
| state_dict compatibility | `load_state_dict` (strict) must still match reference | `batch_freq_base` registered `persistent=False` → excluded from `state_dict()`; `{inv_freq, position_angles}` matches reference exactly | pass |
| ModelNew public contract | constructor + `forward(timestamps, seq_len)` + `get_inputs`/`get_init_inputs` | unchanged | pass |
| numerical semantics | `cos/sin(cat(batch_freqs,time_freqs)*(-timestamps*2pi))` elementwise | row kernel reproduces identical elementwise chain | pass |

The `batch_freq_base` register_buffer (interleaved `inv_freq`, `[64]`, `persistent=False`) is the
only new state; it is a pure derived cache of `inv_freq`, correctly excluded from `state_dict()`, so
the harness's strict `load_state_dict` against the reference still passes (verified: correctness
PASS implies state_dict transfer succeeded).

## Authoritative timing

Primary comparison is against the accepted reference (triton_rotary_001.py). Because the harness
always uses `--v0_file base.py` as the immutable reference, I measured both candidates interleaved in
one session, each against `base.py`, and compared their candidate (v1) medians byte-for-byte.

Three interleaved candidate samples (warmup 50 / repeat 100), v1 median per run:

| run | triton_rotary_001 (accepted) v1 ms | triton_rotary_002 (candidate) v1 ms |
|---:|---:|---:|
| 1 | 0.327830 | 0.329350 |
| 2 | 0.334255 | 0.330345 |
| 3 | 0.326790 | 0.336880 |
| **median** | **0.327830** | **0.330345** |

- accepted-reference wall median: **0.327830 ms**
- candidate wall median: **0.330345 ms**
- improvement_pct = (0.327830 - 0.330345) / 0.327830 * 100 = **-0.77%** (marginally slower)

For completeness, candidate vs immutable `base.py` (v0) this session: v0 medians ~0.595-0.623 ms,
candidate ~0.330 ms → ~46% vs the un-fused baseline, consistent with Round 1. But the adoption
comparison is against the accepted reference (triton_rotary_001), against which there is no wall
improvement.

improvement_pct (-0.77%) < adoption_threshold (5.0%) → `no-improvement`.

## Profiler evidence (CANN msprof, 50 iters, device_time_available=true)

| scope | kernel_count_per_call | kernel_count_total | device_us_per_call | device_total_us | device_ratio |
|---|---:|---:|---:|---:|---:|
| baseline_base (reference, 14-kernel) | 14.0 | 700 | 48.8556 | 2442.78 | 0.0817 |
| candidate_triton_rotary_002 | 1.0 | 50 | 12.116 | 605.8 | 0.0376 |

`device_ratio` = device_us_per_call / (wall_ms * 1000), using candidate wall median 0.322360 ms
(this session's earliest 3-pair candidate median; interleaved median 0.330345 yields ratio ~0.0367).

Candidate top kernel:

| kernel | count/call | us/call |
|---|---:|---:|
| `_rotary_embedding_row_kernel` | 1 | 12.116 |

## Evaluation Contract mirror (hypothesis H-002)

| mechanism observable | expectation | observation | verdict |
|---|---|---|---|
| device_us_per_call | decrease | 48.27 → 12.12 us (~75% reduction vs Round 1) | confirmed |
| kernel_count_per_call | unchanged-at-1 | 1.0 | confirmed |
| wall_time_ms | decrease | 0.327830 → 0.330345 ms (no change / -0.77%) | falsified |

Overall hypothesis verdict: **partially-confirmed**.

The device-time causal chain is CONFIRMED and stronger than predicted: removing the per-lane integer
div/mod, the redundant dual frequency load, and the `tl.where` select cut device time ~4x (48.27 →
12.12 us). However, the terminal primary metric (wall time) did NOT improve: the device win did not
translate to wall time because after Round 1's fusion the operator became host/launch-bound again —
device time was already only ~14.5% of wall time, so shrinking it further moved the needle less than
1% on wall. The remaining wall time (~0.33 ms) is dominated by the single kernel's host launch /
dispatch overhead, not by device compute (device_ratio fell to ~3.8%).

## evidence_for_next_round

- Device time is now ~12 us/call (down 4x from Round 1), but wall time is UNCHANGED (~0.33 ms)
  because the operator is again host/launch-bound: the single Triton kernel's host-side launch and
  dispatch now dominate (~96% of wall time is not device compute).
- The Round 2 restructure is strictly superior on device metrics (12.12 vs 48.27 us, same 1 kernel)
  but yields no wall benefit, so it cannot be adopted on the wall-time metric alone.
- Remaining lever is the host launch overhead of the single kernel itself (e.g. grid/occupancy,
  Triton launcher overhead, or avoiding per-call tensor allocation `torch.empty` for outputs). Device
  compute is no longer the bottleneck.

## Reproduction commands

```bash
cd /workspace/kernelswift/.worktrees/music-rotary-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/triton_rotary_002.py --warmup 5 --repeat 10 --full-traceback
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/triton_rotary_002.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/triton_rotary_001.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/triton_rotary_002.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/ascend/log/music_rotary_round002_forward_50iter.pt.trace.json
```
