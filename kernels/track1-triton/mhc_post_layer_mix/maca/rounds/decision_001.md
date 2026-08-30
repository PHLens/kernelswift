# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"maca","target_profile":"triton_maca","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"tiny-k-gemm-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"fuse the K=4 einsum contraction and the x*post_layer_mix + term2 elementwise tail into a single hand-written Triton kernel that does 4 explicit fp32 multiply-accumulates instead of a 64x64x128-tiled tf32 GEMM","allowed_changes":["kernel dataflow"],"invariants":["ModelNew public contract","output shape dtype device","fp32 accumulate then bf16 cast","input non-mutation","caller-selected device and current stream"],"expected_wall_improvement_pct":40.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor x shape=[2,4096,1280] dtype=bf16 layout=contiguous memory=global
tensor residual shape=[2,4096,4,1280] dtype=bf16 layout=contiguous memory=global
tensor post_layer_mix shape=[2,4096,4,1] dtype=fp32 layout=contiguous memory=global
tensor comb_res_mix shape=[2,4096,4,4] dtype=fp32 layout=contiguous memory=global
tensor out shape=[2,4096,4,1280] dtype=bf16 layout=contiguous memory=global
scalar n1 size=4096
scalar mhc_mult size=4
scalar h size=1280

# O Operations
load mix_c <- comb_res_mix[b,p,n,0:4]
load res0 <- residual[b,p,0,hblk]
load res1 <- residual[b,p,1,hblk]
load res2 <- residual[b,p,2,hblk]
load res3 <- residual[b,p,3,hblk]
load xv <- x[b,p,hblk]
load plm <- post_layer_mix[b,p,n,0]
compute acc = mix_c[0]*res0 + mix_c[1]*res1 + mix_c[2]*res2 + mix_c[3]*res3
compute acc = xv*plm + acc
store out[b,p,n,hblk] <- bf16(acc)

# C Control
parallel p over n1
parallel n over mhc_mult
parallel hblk over h
guard hblk < h

# H Target Hints
target=triton_maca
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; no host state, allocation reuse, caching, or stream changes"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the K=4 einsum contraction and the x*post_layer_mix + term2 elementwise tail into a single hand-written Triton kernel that does 4 explicit fp32 multiply-accumulates instead of a 64x64x128-tiled tf32 GEMM","expected_causal_chain":["the tf32 GEMM (K=128 tile doing only 4/128 useful work) disappears and is replaced by a K=4 explicit fp32 multiply-accumulate kernel","the elementwise mul/add/bf16-cast kernels are folded into the same kernel, so 6 kernels collapse to 1","device time per call drops sharply from 7559 us","wall time drops accordingly"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"candidate_kernel_count_per_call","expectation":"decrease to 1.0"},{"name":"tf32_gemm_us_per_call","expectation":"decrease to 0"},{"name":"fused_triton_kernel_count_per_call","expectation":"equal 1.0"},{"name":"candidate_device_us_per_call","expectation":"decrease from 7559"}],"guardrails":["correctness:pass","output bf16 allclose atol=1e-2 rtol=1e-2","output shape dtype device unchanged","input non-mutation","fp32 accumulate then bf16 cast"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md` and `references/invariants.md`. The four recorded anti-pattern entries (winner tree, sort-32+sort-64 network, dynamic `tl.gather` compaction, cumsum compaction) concern MLU grouped top-k selection and do not match this operator's shape, dtype, or lowering: there is no index selection, no dynamic gather, no sort network here. The contraction is a dense K=4 multiply-accumulate over contiguous bf16 data; no recorded failure invalidates this path.
- The target profile marks `tl.dot`, `tl.make_block_ptr`, `tl.zeros`, `tl.full`, `tl.async_copy`, and `num_stages` as Unknown (no qualifying C500 probe). The sketch therefore does **not** rely on `tl.dot`: the K=4 contraction is expressed as four explicit scalar multiply-accumulates over `tl.load`ed fp32 tiles, using only the proven `tl.load`, `tl.store`, `tl.arange`, `tl.reshape`, `tl.broadcast_to`, and `tl.static_range` primitives. `num_warps=1` is the only observed-warp-count constraint and is honored.
- fp32 accumulation is mandatory: `residual`/`x` are bf16 and must be upcast to fp32 before compute, then the final result is cast to bf16, mirroring `base.py` (`residual.float()`, `x.float()`, `.bfloat16()`). The baseline tf32 GEMM and fp32 elementwise path already accumulate in fp32, so the candidate must reproduce within atol=rtol=1e-2 without introducing bf16 intermediate rounding.

## Rationale and Evidence

The Phase 0 report (`rounds/report_000.md`) shows this operator is device-bound: `device_ratio ≈ 0.99` and 6 kernels per forward call. The dominant kernel `mcblas__Mck_tf32gemm_nt_64x64x128_4m4n1k_256t_fp32_fp32_tf32_sb_0_0` accounts for ~6071 us/call ≈ 80% of the 7559 us/call device time in a single launch. This kernel is the lowering of `torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())` — a batched `[4,4] x [4,1280]` matmul with contraction dimension `K = mhc_mult = 4`.

The GEMM uses a 64x64x128 tile, so for K=4 only 4 of 128 K-lanes do useful work; roughly 97% of the tile's K-work is wasted on a general-purpose GEMM sized for far larger contractions. This is a textbook tiny-K matmul: four explicit fp32 multiply-accumulates (`sum_m comb_res_mix[b,p,n,m] * residual[b,p,m,h]`) replace the entire GEMM, needing no GEMM tile and no `tl.dot`. The elementwise tail (mul ~741 us, add ~340 us, two bf16 cast kernels ~230 + ~178 us ≈ 20%) is folded into the same kernel, collapsing 6 kernels to 1 and eliminating the intermediate `[2,4096,4,1280]` fp32 `term2` round-trip through global memory.

Because ~80% of a device-bound 7.64 ms wall is a badly-sized GEMM, replacing it with a K=4-unrolled fused kernel is expected to remove most of the GEMM cost and the full elementwise tail, for a wall reduction well above the 5% adoption threshold (realistically 30–60%). The `expected_wall_improvement_pct` of 40.0% is a conservative central estimate; the adoption gate remains the unrounded 5.0% median wall improvement in the Evaluation Contract.

The kernel-only `change_scope` means no Host Plan is required. The fallback for non-benchmark shapes is to keep the unchanged PyTorch einsum path; the Triton kernel targets the benchmark shape `n0=2, n1=4096, h=1280, mhc_mult=4` with `mhc_mult=4` hard-unrolled, preserving exact public semantics and input non-mutation on the caller-selected device/stream.
