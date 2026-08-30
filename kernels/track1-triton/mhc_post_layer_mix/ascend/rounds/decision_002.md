# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"candidate_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-tuning"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"tune the single fused kernel's memory-level parallelism by sweeping BLOCK_C (256 -> 320/512/640/1280) and num_warps (4 -> 1/2/8) so the latency-bound kernel approaches the HBM bandwidth ceiling instead of stalling on serialized c-blocks and under-vectorized bf16 loads","allowed_changes":["kernel block size","kernel warp count","compile-time loop trip count"],"invariants":["ModelNew public contract","output dtype and shape","fp32 accumulation before single bf16 cast","single kernel launch (kernel_count stays 1)","weights loaded once per (a,b) program"],"expected_wall_improvement_pct":15.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor x shape=[8192,1280] dtype=bf16 layout=contiguous memory=global
tensor residual shape=[8192,4,1280] dtype=bf16 layout=contiguous memory=global
tensor post_layer_mix shape=[8192,4,1] dtype=fp32 layout=contiguous memory=global
tensor comb_res_mix shape=[8192,4,4] dtype=fp32 layout=contiguous memory=global
tensor out shape=[8192,4,1280] dtype=bf16 layout=contiguous memory=global
scalar BLOCK_C value=tunable memory=constexpr
tile cm tile_shape=[4,4] dtype=fp32 memory=register
tile pm tile_shape=[4] dtype=fp32 memory=register
tile res tile_shape=[4,BLOCK_C] dtype=fp32 memory=register
tile xv tile_shape=[BLOCK_C] dtype=fp32 memory=register
tile acc tile_shape=[4,BLOCK_C] dtype=fp32 memory=register

# O Operations
load cm <- comb_res_mix[ab,0:4,0:4]
load pm <- post_layer_mix[ab,0:4,0]
load res <- residual[ab,0:4,cb*BLOCK_C:cb*BLOCK_C+BLOCK_C] cast fp32
load xv <- x[ab,cb*BLOCK_C:cb*BLOCK_C+BLOCK_C] cast fp32
compute acc <- sum_m cm[m,:] * res[m,:]
compute term <- broadcast_add(broadcast_mul(xv, pm), acc)
store out[ab,0:4,cb*BLOCK_C:cb*BLOCK_C+BLOCK_C] <- cast(term, bf16)

# C Control
parallel ab over 8192
guard ab < 8192
for cb over 1280 // BLOCK_C

# H Target Hints
target=triton_ascend
num_warps=2
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"tune the single fused kernel's memory-level parallelism by sweeping BLOCK_C (256 -> 320/512/640/1280) and num_warps (4 -> 1/2/8) so the latency-bound kernel approaches the HBM bandwidth ceiling instead of stalling on serialized c-blocks and under-vectorized bf16 loads","expected_causal_chain":["larger contiguous loads and fewer serial c-block iterations raise memory-level parallelism","device_us_per_call falls below 620 us","wall_time falls","device_ratio moves toward the harness-fixed host floor"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"device_us_per_call","expectation":"decrease below 620 us"},{"name":"kernel_count_per_call","expectation":"remain 1.0"},{"name":"wall_time","expectation":"decrease below 0.880 ms"}],"guardrails":["correctness:pass","output dtype and shape unchanged","fp32 accumulation before single bf16 cast","kernel_count stays 1","weights loaded once per (a,b) program"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No catalog entry matches a latency-bound memory streaming kernel; the grouped-topk selection/compaction failures are reduction-network pathologies on MLU, unrelated to this Ascend streaming loop.
- Consulted `prompts/coder_targets/triton_ascend.md`. `num_stages`, `vectorize`, `tl.make_block_ptr`, and `async_copy` are `Unknown` on this runtime — the sketch does not depend on them. `num_warps` in {1,2,4} is proven (probe), and `tl.arange` extents 64/128/256 are proven. `BLOCK_C` is a compile-time constexpr, so the sweep is a compile-time grid/loop change, not a new primitive.
- **Prior-round regression guard**: coder_result_001 attempt #2 already measured a 2D grid `(40960,)` formulation at 0.926x (slower) due to redundant per-program weight loads. This decision therefore does NOT flatten the c-loop into a 2D grid; it keeps the `(8192,)` grid and the load-weights-once c-loop structure from the accepted candidate, changing only `BLOCK_C` and `num_warps`.
- **Rejected tl.dot**: the contraction dim is `m=4`, below the probed `(16,16)@(16,16)` `tl.dot` shape; padding to 16 would multiply MACs by 8-16x on a memory-bound kernel. `tl.dot` is not part of this decision.
- **Host-overhead is not targeted**: the ~260 us wall-device gap is dominated by harness-fixed per-sample `sync_devices()` + `set_seed`; candidate-side host cost is already one launch + one caching-allocator-amortized output alloc. `fast_libentry` is `Unknown`, so no launcher change is proposed.

## Rationale and Evidence

report_001 shows the fused kernel is now the only device-side cost (620 us, 1 kernel), and the operator is no longer purely device-bound: `device_ratio` fell to 0.70 because fusion removed ~80% of device work. Two facts bound the remaining space. First, the ~260 us wall-device gap is largely harness-fixed (per-sample device synchronization and seed setup) — it is not a compressible candidate-side target without an unproven launcher (`fast_libentry` is `Unknown`). Second, the kernel itself is far from its memory ceiling: it moves ~190 MB per call in ~620 us, i.e. ~306 GB/s against an ~1.6 TB/s-class HBM, roughly 19% of peak. That gap is the signature of a latency/occupancy-bound kernel — under-vectorized bf16 loads and five serialized c-block iterations per program — not a bandwidth- or compute-bound one.

This decision therefore targets occupancy with proven, safe knobs: sweep `BLOCK_C` upward (fewer, larger contiguous c-iterations) and `num_warps` across the proven set {1,2,4,8}. Raising memory-level parallelism can plausibly recover 1.3-2x of device throughput (620 -> ~400-310 us), moving wall from ~0.88 ms toward ~0.70-0.57 ms, i.e. ~15-35% improvement — above the 5% threshold. The prior 2D-grid regression is explicitly avoided (weights stay loaded once per `(a,b)` program), and correctness is guarded by the unchanged fp32-accumulation-then-bf16-cast contract. If the sweep cannot clear 5%, the round will conclude `no-improvement` and the design will pivot to a different family, since the remaining host cost is harness-fixed and `tl.dot` is ruled out by the `m=4` shape.
