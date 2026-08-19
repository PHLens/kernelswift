# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"elementwise-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"fuse the post-GEMM elementwise tail (x bf16->fp32 cast, multiply x*post_layer_mix, add term2, and fp32->bf16 output cast) into a single Triton kernel","allowed_changes":["kernel dataflow"],"invariants":["ModelNew public contract","output dtype and shape","fp32 intermediate precision","einsum term2 result unchanged"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor x shape=[2,4096,1280] dtype=bf16 layout=contiguous memory=global
tensor post_layer_mix shape=[2,4096,4,1] dtype=fp32 layout=contiguous memory=global
tensor term2 shape=[2,4096,4,1280] dtype=fp32 layout=contiguous memory=global
tensor out shape=[2,4096,4,1280] dtype=bf16 layout=contiguous memory=global

# O Operations
load xv <- cast_fp32(x[a,b,c])
load pm <- post_layer_mix[a,b,n,0]
load t2 <- term2[a,b,n,c]
compute acc = xv * pm + t2
store out[a,b,n,c] <- cast_bf16(acc)

# C Control
parallel idx over [a,b,n,c] flattened to 41943040
guard idx < 41943040

# H Target Hints
target=triton_cuda
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; forward still calls torch.einsum for term2 and the fused Triton kernel for the tail, preserving ModelNew public contract and caller-selected device/stream"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the post-GEMM elementwise tail (x bf16->fp32 cast, multiply x*post_layer_mix, add term2, and fp32->bf16 output cast) into a single Triton kernel","expected_causal_chain":["four separate elementwise/cast kernels collapse into one","kernel_count_per_call decreases","intermediate fp32 materialization and launch overhead decrease","device_us_per_call decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","fp32 intermediate precision preserved","einsum term2 result unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No catalog entry matches this decision's
  preconditions (the anti-patterns target MLU590-H8 grouped-topk selection
  networks, not a CUDA/BI150 elementwise fusion). The relevant target-profile
  pitfalls are accounted for: `tl.load`/`tl.store` are Supported for contiguous
  float32/bf16 loads and stores, `tl.arange`/`tl.program_id`/`tl.reshape` are
  Supported, and no `tl.dot` is required because the GEMM (term2) is left
  unchanged and produced by `torch.einsum`. `num_warps`/`num_stages` are left as
  non-normative Coder tuning choices because the target profile marks them
  `Unknown`.

## Rationale and Evidence

The Phase 0 report (report_000.md) establishes a device-bound baseline
(device_ratio ≈ 0.894) with a 5-kernel forward sequence: bf16→fp32 casts, one TCU
batched GEMM for `term2` (~66% of device time), an fp32 multiply-add, and a
fp32→bf16 output cast. The GEMM's contraction dimension is only 4
(`[4,4]@[4,1280]`), so a `tl.dot` rewrite is high-risk and unlikely to beat the
already-optimized TCU batched GEMM; it is deferred to a later round.

The post-GEMM elementwise tail — the `x` bf16→fp32 cast, `x.float()*post_layer_mix`
multiply (~626 us/call), the `+term2` add (~778 us/call), and the fp32→bf16 output
cast (~526 us/call) — totals roughly 1900-2400 us/call (~33% of device time)
across four separate memory-bound elementwise/cast kernels. These kernels are
inside the kernel change boundary and do not touch the GEMM. Fusing them into one
Triton kernel removes three to four kernel launches and the intermediate fp32
materialization (the multiply result and the add result no longer round-trip
through global memory), while preserving the fp32 intermediate precision required
by the semantics (`residual.float()` and `x.float()` promote to fp32; only the
final `.bfloat16()` rounds). Because device_ratio is high, this device-time
reduction is expected to transmit to wall time. A conservative estimate of 8.0%
wall improvement exceeds the 5.0% adoption threshold and is falsifiable via the
kernel-count and device-time observables.

The GEMM `term2` is computed by the unchanged `torch.einsum('abmn,abmc->abnc',
comb_res_mix, residual.float())` path; the fused kernel consumes `term2`,
`x.float()`, and `post_layer_mix` and emits the `bfloat16` output, so the
`ModelNew` public contract, output `[2,4096,4,1280]` bf16 shape/dtype, and fp32
intermediate semantics are preserved.
