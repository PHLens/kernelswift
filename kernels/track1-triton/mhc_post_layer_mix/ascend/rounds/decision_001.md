# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"fuse the six baseline kernels (3 bf16/fp32 casts, batched matmul, broadcast mul, broadcast add, final bf16 cast) into a single Triton kernel that loads bf16 x/residual and fp32 post_layer_mix/comb_res_mix once, computes the per-(a,b) contraction and the broadcast add/mul in fp32 registers, casts once to bf16, and stores","allowed_changes":["kernel dataflow","matmul lowering","cast folding"],"invariants":["ModelNew public contract","output dtype and shape","fp32 accumulation before single bf16 cast","input tensor immutability"],"expected_wall_improvement_pct":40.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor x shape=[2,4096,1280] dtype=bf16 layout=contiguous memory=global
tensor residual shape=[2,4096,4,1280] dtype=bf16 layout=contiguous memory=global
tensor post_layer_mix shape=[2,4096,4,1] dtype=fp32 layout=contiguous memory=global
tensor comb_res_mix shape=[2,4096,4,4] dtype=fp32 layout=contiguous memory=global
tensor out shape=[2,4096,4,1280] dtype=bf16 layout=contiguous memory=global
tile cm tile_shape=[4,4] dtype=fp32 memory=register
tile res tile_shape=[4,BLOCK_C] dtype=fp32 memory=register
tile pm tile_shape=[4,1] dtype=fp32 memory=register
tile xv tile_shape=[1,BLOCK_C] dtype=fp32 memory=register
tile acc tile_shape=[4,BLOCK_C] dtype=fp32 memory=register

# O Operations
load cm <- comb_res_mix[a,b,0:4,0:4]
load res <- residual[a,b,0:4,c0:c0+BLOCK_C] cast fp32
load pm <- post_layer_mix[a,b,0:4,0:1]
load xv <- x[a,b,c0:c0+BLOCK_C] cast fp32
compute acc <- matmul(cm, res) contract over m
compute term <- broadcast_add(broadcast_mul(xv, pm), acc)
store out[a,b,0:4,c0:c0+BLOCK_C] <- cast(term, bf16)

# C Control
parallel ab over 2*4096
guard ab < 2*4096
parallel c0 over 1280
guard c0 < 1280

# H Target Hints
target=triton_ascend
num_warps=4
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the six baseline kernels (3 bf16/fp32 casts, batched matmul, broadcast mul, broadcast add, final bf16 cast) into a single Triton kernel that loads bf16 x/residual and fp32 post_layer_mix/comb_res_mix once, computes the per-(a,b) contraction and the broadcast add/mul in fp32 registers, casts once to bf16, and stores","expected_causal_chain":["intermediate fp32/bf16 tensors and their materializing cast kernels disappear","kernel count per call falls from 6 to 1","device_us_per_call falls","wall_time falls"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease from 6.0 to 1.0"},{"name":"device_us_per_call","expectation":"decrease"},{"name":"cast_kernel_eliminated","expectation":"aclnnInplaceCopy_CastAiCore_Cast disappears from top-k"}],"guardrails":["correctness:pass","output dtype and shape unchanged","fp32 accumulation before single bf16 cast","input tensors unmodified"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The catalog's grouped-topk selection/compaction failures (hierarchical argmax, full bitonic sort, dynamic `tl.gather`, cumsum compaction) all concern reduction/selection networks on an MLU runtime; none of them applies to this Ascend elementwise+small-GEMM fusion path.
- Consulted `prompts/coder_targets/triton_ascend.md`. Two target-specific facts bound the sketch:
  - The contraction dim `m` is only 4, below the probed `tl.dot` shape `(16,16)@(16,16)`. The sketch therefore treats the matmul lowering as an explicit weighted reduction over the 4 `m` indices (fp32 FMA accumulation) and leaves a `tl.dot`-with-padding form as a non-normative fallback. Coder must not introduce a `tl.dot` shape that the profile has not verified; an unprovable required `tl.dot` shape is a capability-miss, not a silent workaround.
  - `num_stages`, `vectorize`, `tl.make_block_ptr`, and `async_copy` are Unknown on this runtime; the sketch uses none of them. `num_warps=4` is a proven value (probe verified `1/2/4` compile and run).
- The prior flexattention `tl.dot` +55us host penalty is host-bound evidence from a different operator; here the operator is device-bound (device_ratio 0.96), so Cube-side acceleration and kernel fusion directly attack wall time. The decision still keeps the matmul lowering non-normative so Coder/Verifier can measure the explicit-reduction vs `tl.dot` choice against the real harness.

## Rationale and Evidence

Verifier report_000 establishes the operator is device-bound: median wall ≈ 3.21 ms, `device_us_per_call` ≈ 3083 us, `device_ratio` ≈ 0.96, and 6 kernels per call. The six kernels are `aclnnBatchMatMul` (~1111 us, the einsum Cube matmul), `aclnnAdd` (~892 us), `aclnnInplaceCopy_Cast` ×3 (~801 us total, the bf16→fp32 casts of `residual`/`x` and the final fp32→bf16 cast), and `aclnnMul` (~290 us). The Vector-path aggregate (Cast×3 + Add + Mul ≈ 1983 us) exceeds the Cube matmul (~1111 us), so the dominant compressible cost is the materialization of intermediate fp32/bf16 tensors and the six separate kernel launches, not the matmul itself.

Fusing all six kernels into one Triton kernel removes: (1) the three cast kernels, whose cost is pure data movement that folds into the load/compute cast; (2) the separate broadcast Mul and Add kernels, whose arithmetic folds into the in-register compute; and (3) the intermediate `residual.float()`, `x.float()`, and fp32 `term2` tensor traffic. The matmul is a per-(a,b) [4,4]@[4,1280] contraction over `m`=4 batched across 8192 (a,b) pairs; in a fused kernel this becomes an in-register weighted reduction that reuses the already-loaded `comb_res_mix` and `residual` tiles without a separate Cube launch. Because the operator is device-bound, this fusion is expected to reduce device time from ~3083 us toward the matmul-only ~1100–1500 us floor, supporting the 40% expected wall improvement (well above the 5% adoption threshold). Correctness is guarded by preserving fp32 accumulation before the single bf16 cast, matching the reference within atol=1e-2/rtol=1e-2.
