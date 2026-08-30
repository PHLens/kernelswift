# Report 001

## Decision

- classification: `accepted`
- candidate: `kernels/track1-triton/music_flamingo_rotary_embedding/ascend/triton_rotary_001.py`
- candidate_sha256: `51a9a33b82f550abfd80400bb0748b74fd181d0f3c4fd4b5d70b4ca1f5d6984e`
- accepted reference: `kernels/track1-triton/music_flamingo_rotary_embedding/base.py` (Phase 0 baseline)
- accepted-reference sha256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- harness sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- design: `rounds/design_001.md` (decision `proceed`, hypothesis `H-001`, change_family `kernel-fusion`)

## Correctness

- command: `python3 auto_bench.py --v0_file .../base.py --v1_file .../triton_rotary_001.py --warmup 5 --repeat 10 --full-traceback`
- result: `PASS accuracy; v0=0.585220 ms, v1=0.333110 ms, speedup=1.757x` (1 passed, 0 failed)
- exit status: 0

Tuple output `(cos, sin)` compared element-wise by the harness `compare_values` (tuple branch
recurses `compare_values(item0, item1, "output[i]", ...)`, each tensor checked with
`torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)`). Passed.

## Guardrail matrix

| Guardrail | Expectation | Observation | Verdict |
|---|---|---|---|
| correctness | pass | `PASS accuracy` (all 4 runs) | pass |
| output structure | tuple `(cos, sin)` | `forward` returns `(cos_out, sin_out)` | pass |
| output shape | each `[4,32,128]` fp32 | `cos_out`/`sin_out` = `(B, SEQ, 2*DIM)` = `[4,32,128]` fp32 | pass |
| register_buffer invariants | `inv_freq`, `position_angles` precomputed in `__init__`, unchanged | both `register_buffer`, computed once in `__init__`, passed to kernel as inputs | pass |
| ModelNew public contract | constructor + `forward(timestamps, seq_len)` + `get_inputs`/`get_init_inputs` | unchanged signatures | pass |
| numerical semantics | `freqs = cat(batch_freqs, time_freqs) * (-timestamps*2pi)`, `cos`/`sin` | fused kernel reproduces the same elementwise chain (see coder notes re fp32 2π rounding) | pass |

## Authoritative timing (3 interleaved pairs, warmup 50 / repeat 100)

| Pair | reference v0 (ms) | candidate v1 (ms) | speedup |
|---:|---:|---:|---:|
| 1 | 0.632515 | 0.340050 | 1.860x |
| 2 | 0.604445 | 0.333955 | 1.810x |
| 3 | 0.622330 | 0.331230 | 1.879x |

- reference median (unrounded): **0.622330 ms**
- candidate median (unrounded): **0.333955 ms**
- improvement_pct = (0.622330 - 0.333955) / 0.622330 * 100 = **46.33%**

improvement_pct (46.33%) >= adoption_threshold (5.0%) → `accepted`.

## Profiler evidence (CANN msprof, 50 iters, device_time_available=true)

| scope | kernel_count_per_call | kernel_count_total | device_us_per_call | device_total_us | device_ratio |
|---|---:|---:|---:|---:|---:|
| baseline_base (reference) | 14.0 | 700 | 48.5392 | 2426.96 | 0.0780 |
| candidate_triton_rotary_001 | 1.0 | 50 | 48.2708 | 2413.54 | 0.1445 |

`device_ratio` = device_us_per_call / (wall_ms * 1000), using the authoritative reference
median 0.622330 ms and candidate median 0.333955 ms.

Candidate top kernel:

| kernel | count/call | us/call |
|---|---:|---:|
| `_rotary_embedding_fused_kernel` | 1 | 48.2708 |

Reference top kernels (unchanged from Phase 0): RepeatInterleaveV2 (10.24us), Mul (8.72us),
BroadcastTo (8.10us), ConcatD (4.84us), Cos (3.91us), Sin (3.87us), Arange (3.78us), Muls (2.03us),
Divs (1.03us), Neg (0.95us).

### Profiler note (unchanged from Phase 0)

The harness writes the chrome trace inside the per-scope loop with a single `--profile-output` path,
so the trace JSON retains only the last scope's `record_function`; device attribution is taken from
each scope's own CANN `ai_core_op_summary.db`. Note that this round's `baseline_base` CANN directory
accumulated a SECOND capture (the Phase 0 capture is still present alongside it), so
`summarize_cann_trace.py` was pointed at the newest `*_ascend_pt` subdirectory directly rather than
the parent. This is a profiling-environment bookkeeping nuance, not a correctness issue.

## Evaluation Contract mirror (hypothesis H-001)

| mechanism observable | expectation | observation | verdict |
|---|---|---|---|
| kernel_count_per_call | decrease (14 → 1) | 14.0 → 1.0 | confirmed |
| device_us_per_call | decrease | 48.54 → 48.27 us (≈ unchanged) | falsified (device time did not decrease; compute is the same) |
| wall_time_ms | decrease | 0.622330 → 0.333955 ms | confirmed |

Overall hypothesis verdict: **partially-confirmed**.

The causal chain's mechanism (kernel fragmentation → launch overhead) is CONFIRMED: kernel count
collapsed 14→1 and wall time dropped 46.3%. The sub-expectation that `device_us_per_call` would also
decrease was NOT met — device time is essentially unchanged (~48 us), because the fused kernel does
the same total elementwise compute. This is the expected and correct outcome for a host-bound
fragmentation fix: the win comes entirely from eliminating 13 launch/routing/intermediate-allocation
operations per call, not from reducing device compute. `device_ratio` roughly doubled (0.078 → 0.145),
confirming the host overhead that dominated wall time was removed.

## evidence_for_next_round

- Wall time improved 46.3% (0.622330 → 0.333955 ms) via kernel fusion (14 → 1 kernel).
- Device time is now the dominant remaining cost (~48 us/call) and is ~14.5% of wall time; the
  operator is no longer purely host-bound. Further wall-time gains would now require reducing the
  single fused kernel's device time itself (e.g. larger blocks, more warps, better vectorization, or
  fewer redundant trig operations), not further fusion.
- The fused kernel `_rotary_embedding_fused_kernel` (BLOCK=128, num_warps=1) is the sole kernel;
  its 48 us/call is the new bottleneck if further speedup is sought.

## Reproduction commands

```bash
cd /workspace/kernelswift/.worktrees/music-rotary-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/triton_rotary_001.py --warmup 5 --repeat 10 --full-traceback
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/triton_rotary_001.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/triton_rotary_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/ascend/log/music_rotary_round001_forward_50iter.pt.trace.json
```
