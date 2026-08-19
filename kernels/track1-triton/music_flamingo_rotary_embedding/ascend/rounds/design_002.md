# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_rotary_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"row-parallel-vectorization"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"restructure the fused kernel from a flat-index grid (per-lane integer div/mod index decode, redundant dual frequency load discarded by tl.where, and per-lane recompute of block-uniform scalars ts/angle) into a row-per-program grid over (B,SEQ) that computes b/t/ts/angle once per program as scalars, loads each of the 128 frequencies exactly once as two contiguous 64-wide halves (precomputed interleaved batch_freq_base and position_angles[t]), and writes cos/sin with no tl.where branch","allowed_changes":["kernel dataflow and grid mapping","ModelNew.forward kernel launch","__init__ precompute of interleaved inv_freq buffer"],"invariants":["ModelNew public constructor and forward signature","output structure: tuple (cos, sin)","output shape [4,32,128] fp32 each","register_buffer semantics: all frequency tables precomputed in __init__, not per-forward","numerical semantics: cos/cat(batch_freqs,time_freqs)*(-timestamps*2pi) elementwise","harness entry points ModelNew/get_inputs/get_init_inputs"],"expected_wall_improvement_pct":6.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor timestamps shape=[B,SEQ] dtype=fp32 layout=row_major memory=global
tensor batch_freq_base shape=[DIM] dtype=fp32 layout=contiguous memory=global
tensor position_angles shape=[MAX_SEQ,DIM] dtype=fp32 layout=row_major memory=global
tensor cos_out shape=[B,SEQ,2DIM] dtype=fp32 layout=row_major memory=global
tensor sin_out shape=[B,SEQ,2DIM] dtype=fp32 layout=row_major memory=global
tile half shape=[DIM] dtype=fp32 memory=register
scalar b dtype=int
scalar t dtype=int
scalar ts dtype=fp32
scalar angle dtype=fp32

# O Operations
compute pid = program_id
compute b = pid // SEQ
compute t = pid % SEQ
load ts <- timestamps[b*SEQ+t]
compute angle = -ts * TWO_PI
compute scale = b / max_seq_len
load half_batch <- batch_freq_base[0:DIM]
compute row_batch = half_batch * scale
load half_time <- position_angles[t,0:DIM]
compute val_batch = row_batch * angle
compute val_time = half_time * angle
compute cos_b = cos(val_batch)
compute sin_b = sin(val_batch)
compute cos_t = cos(val_time)
compute sin_t = sin(val_time)
store cos_out[b,t,0:DIM] <- cos_b
store cos_out[b,t,DIM:2DIM] <- cos_t
store sin_out[b,t,0:DIM] <- sin_b
store sin_out[b,t,DIM:2DIM] <- sin_t

# C Control
parallel pid over B*SEQ
guard pid < B*SEQ

# H Target Hints
target=triton_ascend
num_warps=4
num_stages=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; host still passes the precomputed register_buffers (now including an interleaved batch_freq_base buffer) plus timestamps and launches one fused triton kernel returning cos and sin"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"restructure the fused kernel from a flat-index grid (per-lane integer div/mod index decode, redundant dual frequency load discarded by tl.where, and per-lane recompute of block-uniform scalars ts/angle) into a row-per-program grid over (B,SEQ) that computes b/t/ts/angle once per program as scalars, loads each of the 128 frequencies exactly once as two contiguous 64-wide halves (precomputed interleaved batch_freq_base and position_angles[t]), and writes cos/sin with no tl.where branch","expected_causal_chain":["per-lane integer division and modulo disappear","redundant dual frequency load and tl.where select disappear","frequency loads per call drop from 3x16384 to 16384","device_us_per_call decreases","wall_time_ms decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"device_us_per_call","expectation":"decrease"},{"name":"kernel_count_per_call","expectation":"unchanged-at-1"},{"name":"wall_time_ms","expectation":"decrease"}],"guardrails":["correctness:pass","output structure tuple (cos, sin) unchanged","output shape [4,32,128] fp32 unchanged","frequency tables remain register_buffers precomputed in __init__","ModelNew public contract unchanged","numerical semantics unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The four recorded failures (winner-tree expert selection, sort-32/sort-64 selection network, dynamic tl.gather compaction, cumsum compaction) are all reduction/selection-network anti-patterns on `groupedtopk`/MLU590-H8. This operator is pure elementwise with no reduction, no selection network, no dynamic gather, and no prefix/compaction; the preconditions do not match. No matching failure invalidates this path.
- Consulted `references/bottleneck-judgment.md`. After Round 1 fusion, `device_ratio = 48.27 / (0.333955 * 1000) = 0.145` (> 20% threshold crossed is false, still < 20%, but the single 48us kernel is now the dominant compressible cost and the host overhead that previously masked it is gone). The report's `evidence_for_next_round` explicitly names "larger blocks, more warps, better vectorization, or fewer redundant operations" as the next lever. This intervention targets the redundant operations directly (dual load + integer decode + uniform recompute), which is the compressible portion of the 48us.
- Guardrail note: a new `batch_freq_base` register_buffer (interleaved `inv_freq`, `[dim]`) is added in `__init__` purely to eliminate the kernel's `c // 2` integer division. It mirrors the existing `position_angles` precompute (`repeat_interleave(2)` in `__init__`) and preserves the "model state, not per-forward" invariant. `inv_freq` and `position_angles` remain register_buffers unchanged.

## Rationale and Evidence

`report_001.md` shows the Round 1 fusion collapsed 14 kernels to 1 and cut wall time 46.3% (0.622330 → 0.333955 ms), but `device_us_per_call` was essentially unchanged (48.54 → 48.27 us). The hypothesis verdict was `partially-confirmed`: the launch-overhead mechanism held, but the device-time sub-expectation was falsified — the single fused kernel still performs the same total elementwise work, so its ~48us is now the dominant remaining cost (device_ratio ~0.145).

Inspection of `triton_rotary_001.py` reveals concrete redundant work inside the fused kernel that can be removed without changing the numerical result:

1. **Redundant dual frequency load.** Every lane loads BOTH `freq_batch_raw` (`inv_freq[c//2]`) and `freq_time` (`position_angles[t,c-dim]`), then discards one via `tl.where(is_time, ...)`. Only one of the two is ever used per lane, so half of the 128-per-row frequency loads (2×16384 → 16384 per call) are wasted memory traffic.
2. **Per-lane integer division/modulo.** `b = idx // (SEQ*2*DIM)`, `t = (idx // (2*DIM)) % SEQ`, `c = idx % (2*DIM)`, and `half = c // 2` are integer div/mod on Ascend vector cores (no native integer division; software-lowered). With `BLOCK=128 = 2*DIM`, `b` and `t` are block-uniform yet recomputed on every lane.
3. **Per-lane recompute of block-uniform scalars.** `ts` (`timestamps[b*SEQ+t]`) and `angle = -ts*TWO_PI` are identical across all 128 lanes of a program but loaded/computed per-lane (128 redundant loads of one address).

Because `BLOCK=2*DIM=128` already equals exactly one output row, the natural refactor is a row-per-program grid over `(B,SEQ)` (128 programs). Each program computes `b`, `t`, `ts`, and `angle` once as scalars, then loads two contiguous 64-wide frequency halves — `batch_freq_base[0:DIM] * (b/max_seq_len)` and `position_angles[t,0:DIM]` — multiplies by the scalar `angle`, applies `cos`/`sin`, and stores directly to the two 64-wide output column ranges. This eliminates every per-lane integer division, the redundant dual load, and the `tl.where` select; each of the 128 frequencies is loaded exactly once, and the `c//2` division is removed by precomputing the interleaved `batch_freq_base` register_buffer in `__init__` (consistent with how `position_angles` is already precomputed).

The irreducible floor is the 16384 cos plus 16384 sin of 16384 distinct values (the `[4,32,128]` output), which no fusion can remove. The removed work (roughly 2/3 of frequency loads, all integer division, and the uniform-scalar recompute) is expected to cut a meaningful fraction of the 48us device time. Since device time is ~14.5% of wall, a ~30–40% device reduction translates to a ~5–6% wall improvement, above the 5% adoption threshold. `num_warps` is re-tunable (the coder's prior 1/2/4 sweep was against the old div/mod-laden kernel, so the optimum may shift after this restructure).
