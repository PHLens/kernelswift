# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 source reference; no pre-existing canonical implementation)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `8e5c70232e541a02d83343216376ece9127a1c3e6ea6af77dc77a2723783facf`
- Accepted reference SHA256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- Base SHA256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1), 16 SM, 16 GiB)
- Measurement fingerprint: `5c2a51ab3f3ebaab1123b9fa534d4e4b940f3334f80fac00252df780d3900150`
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`

The base, adapter, and harness hashes all match the frozen `project.md` values exactly.
Runtime fingerprint re-verified locally: torch `2.7.1`, triton `3.1.0`, device
`Iluvatar BI-V150` capability `(7, 1)`.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=3.250513 ms, v1=3.262031 ms, speedup=0.996x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| public loader contract | `base.py` exposes `Model/get_init_inputs/get_inputs`; adapter exposes `ModelNew/get_init_inputs/get_inputs` through the actual AST loader | Frozen harness loaded, constructed, moved, and executed both sides without load or constructor error | pass | correctness return code `0` |
| output structure/shape/dtype | single fp16 tensor `out[83,128]` | Harness recursive comparator accepted structure, shape, and dtype | pass | correctness return code `0`; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | No floating mismatch | pass | correctness return code `0` |
| frozen artifact identity | local hashes equal project.md before measurement | base `a0269ac1...`, adapter `8e5c7023...`, harness `71fb3ad0...` all match | pass | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, seed/tolerances defaults, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte | pass | round_status_000.md |

The correctness command's `v0=3.250513 ms` and `v1=3.262031 ms` values are smoke
timing only and do not replace the frozen 50/100 baseline samples.

## Screening Evidence

Not applicable in Phase 0. No screening classification was made.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[3.281101, 3.254634, 3.258671]`
- candidate_raw_samples_ms: `[3.286430, 3.276976, 3.113909]`
- reference_median_ms: `3.258671`
- candidate_median_ms: `3.276976`
- improvement_pct: `not-applicable: Phase 0` (baseline establishment, not an adoption decision)

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `3.281101` | `3.286430` | `0` |
| 2 | `3.254634` | `3.276976` | `0` |
| 3 | `3.258671` | `3.113909` | `0` |

This descriptive mechanical-adapter comparison is not an optimization-adoption
decision. Round 000 establishes `baseline_adapter.py` as the baseline; the result
is neither `accepted` nor `no-improvement`.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive` (no Phase 0 optimization hypothesis exists)

No decision or `mechanism_observables[]` exists for Phase 0, so there are no missing required observables.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `baseline_base`, `candidate_baseline_adapter`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `log/round_000_forward_50iter.pt.trace.json`, SHA256 `4a19895f74c2c1e4c3c3b867e782c47e00e71915282557c6dd866ca3a5d9540d`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_base` | `48408.123046875` | `968.1624609375` | `6195` | `123.9` | `3.258671` | `0.2971034697695778` |
| `candidate_baseline_adapter` | `48487.466796875` | `969.7493359375` | `6200` | `124.0` | `3.258671` | `0.29759043976440086` |

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000) = 968.162 / 3258.671 ≈ 0.2971
```

Both scopes share an identical kernel sequence and near-identical per-call
totals. The tiny numeric difference between the two scopes (968.162 vs 969.749
us/call, 123.9 vs 124.0 kernels/call) is a measurement artifact of interleaved
scope-boundary sampling, not a semantic difference.

### Baseline Top Kernels (baseline_base scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `index_elementwise_kernel<...index_kernel_impl...>` (gather `x_e = x_rep[mask]`) | `400` | `8.0` | `6357.308` | `127.146` |
| `index_elementwise_kernel<...index_put_kernel_impl...>` (scatter `expert_out[mask]=...`) | `399` | `7.98` | `6356.743` | `127.135` |
| `cub::DeviceSelectSweepKernel<...DispatchSelectIf...>` (mask select) | `799` | `15.98` | `6305.2` | `126.10` |
| `reduce_kernel<1024,1,ReduceOp<bool,or_kernel...>>` (mask.any) | `400` | `8.0` | `4339.0` | `86.78` |
| `cub::DeviceReduceSingleTileKernel<...>` (mask.any reduce) | `800` | `16.0` | `4082.7` | `81.65` |
| `Gemm_tcu_mr_kernel::gemm_tcu_h<128u,128u,32u,32u,32u,1u,...>` (gate/up GEMM) | `400` | `8.0` | `3057.5` | `61.15` |
| `Gemm_tcu_mr_kernel::gemm_tcu_h<32u,64u,32u,16u,32u,2u,...>` (down GEMM) | `400` | `8.0` | `2898.1` | `57.96` |
| `cub::DeviceCompactInitKernel<ScanTileState<int,true>...>` | `799` | `15.98` | `2839.2` | `56.78` |
| `elementwise_kernel_v3<BinaryFunctor<Half,...>>` (chunk) | `400` | `8.0` | `2443.5` | `48.87` |
| `elementwise_kernel_v3<silu_kernel>` (SiLU activation) | `400` | `8.0` | `2006.5` | `40.13` |
| `vectorized_elementwise_kernel<AUnaryFunctor<long,...>>` | `400` | `8.0` | `1546.8` | `30.94` |
| `sbtopk::gatherTopK<float,unsigned int,2,false>` (topk gather) | `50` | `1.0` | `1133.2` | `22.66` |
| `bitonicSortKVInPlace<2,-1,16,16,float,long,GTOp<float,true>>` (topk sort) | `50` | `1.0` | `935.5` | `18.71` |
| `reduce_kernel<512,2,ReduceOp<Half,...>>` (weighted sum) | `49` | `0.98` | `825.4` | `16.51` |
| `reduce_kernel<1024,1,ReduceOp<float,sum_functor>>` (renormalize sum) | `50` | `1.0` | `727.2` | `14.54` |
| `softmax_warp_forward<float,float,float,3,false,false>` | `50` | `1.0` | `253.7` | `5.07` |

### Candidate Top Kernels (candidate_baseline_adapter scope)

Identical kernel sequence to `baseline_base`, with the only ordering difference
being that the two `index_elementwise_kernel` (gather vs index_put) swap places
in the ranking (both at `8.0`/call, `~127 us/call`). Per-call totals:
`index_put` `6386.718 us` / `127.734`, `index` `6373.294 us` / `127.466`,
`DeviceSelectSweepKernel` `16.0`/call. All other kernels match `baseline_base`
within sampling noise. The scope is semantically identical to `baseline_base`.

### Per-Expert Loop Kernel Observation

The per-expert Python loop (`for e in range(8)`) is the dominant structural
feature of this kernel profile. Each of the 8 experts launches, per forward call:

- `1×` gather `index_elementwise_kernel` (`x_e = x_rep[mask]` → `[n_e,128]`)
- `1×` CUB `DeviceSelectSweepKernel` + `DeviceCompactInitKernel` + `DeviceReduceSingleTileKernel` (boolean mask `flat_ids == e` selection and `mask.any()` reduction)
- `1×` `reduce_kernel<or_kernel>` (`mask.any()`)
- `1×` gate/up GEMM `gemm_tcu_h<128u,128u,...>` (`x_e @ w1[e].T`, contraction 128)
- `1×` `elementwise_kernel_v3` (chunk `gate_up.chunk(2)` split)
- `1×` `silu_kernel` + `1×` mul elementwise (`F.silu(gate) * up`)
- `1×` down GEMM `gemm_tcu_h<32u,64u,...>` (`act @ w2[e].T`, contraction 64)
- `1×` scatter `index_elementwise_kernel<index_put_kernel_impl>` (`expert_out[mask] = ...`)

The dispatch/selection overhead (CUB `DeviceSelectSweepKernel` 15.98/call,
`DeviceReduceSingleTileKernel` 16/call, `DeviceCompactInitKernel` 15.98/call,
`index_elementwise` gather+scatter 16/call, `reduce/or` 8/call) is executed once
per expert inside the loop. Together these host/launch-bound dispatch kernels
consume ~263 us/call — more than either GEMM kernel (~61 or ~58 us/call). This
is the primary optimization target: the per-expert Python loop launches ~124
tiny kernels per forward call, and the boolean-mask selection / scatter
(`flat_ids == e`, `x_rep[mask]`, `expert_out[mask]=...`) is the largest single
overhead source.

### GEMM / topk Backend Observation

1. **GEMM backend — Iluvatar TCU (`cublasLt`-style)**: both GEMMs lower to
   `Gemm_tcu_mr_kernel::gemm_tcu_h`, with `ixblasTCUGEMMCategory_t=0`,
   `cublasLtEpilogue_t=1`, fp16 inputs (`__half`), fp32 accumulate (`float`),
   matrix scale `1002`. The gate/up GEMM tiles at `128x128x32x32` (M=128, N=128,
   K=128) and the down GEMM tiles at `32x64x32x16` (M=32, N=64, K=16). Both run
   exactly `8/call` (one per expert). The GEMMs are already on the TCU, so the
   compute itself is efficient; the problem is the surrounding dispatch.
2. **topk backend — standard PyTorch bitonic sort**: `torch.topk` lowers to
   `sbtopk::gatherTopK<float,unsigned int,2,false>` (1/call) plus
   `bitonicSortKVInPlace<2,-1,16,16,float,long,GTOp<float,true>>` (1/call). This
   is the default sort-based topk path, NOT the grouped-topk custom lowering.
   The `softmax_warp_forward` (1/call) handles the routing softmax. Tie semantics
   (descending value, ascending index on equal scores) are the default PyTorch
   `torch.topk` behavior and remain correctness-critical for any Triton
   reimplementation.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `8e5c70232e541a02d83343216376ece9127a1c3e6ea6af77dc77a2723783facf` | same | correctness and wall timing passed; profiler summarized without filtering |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Canonical Phase 0 baseline: `baseline_adapter.py`, wall median `3.258671 ms` from three independent 50/100 samples under measurement fingerprint `5c2a51ab3f3ebaab1123b9fa534d4e4b940f3334f80fac00252df780d3900150`.
- `baseline_base` scope measured `968.1624609375 us/device-call` and `123.9 kernels/call`. Device ratio ≈ `0.297`, so ~30% of wall time is device kernel time and ~70% is host/launch overhead. Unlike flexattention (~0.61), this operator carries real GEMM compute but remains heavily launch-bound due to the ~124 tiny kernels per call.
- The per-expert Python loop (`for e in range(8)`) launches ~124 kernels per forward call. The largest overhead source is dispatch: CUB `DeviceSelectSweepKernel` (15.98/call), `DeviceReduceSingleTileKernel` (16/call), `DeviceCompactInitKernel` (15.98/call), `index_elementwise` gather+scatter (16/call), and `reduce/or` (8/call) — together ~263 us/call, exceeding either GEMM kernel.
- GEMM is already on the Iluvatar TCU (`gemm_tcu_h`, cublasLt-style, fp16 in / fp32 accumulate). The down GEMM tile is only `32x64x16` (M=32), quite small. GEMM is not the bottleneck at baseline.
- `torch.topk` uses the standard PyTorch bitonic sort (`gatherTopK` + `bitonicSortKVInPlace`), one each per call. The grouped-topk lesson (reproduce `torch.topk` descending-value / ascending-index tie order in Triton) applies.
- Base and adapter are semantically equivalent (adapter is a top-level class rename); the small wall/device differences are measurement observations, not an optimization mechanism.
- Tie-rule invariant is correctness-critical: `torch.topk(scores, 2, dim=-1)` selects descending values with ties broken by ascending index.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline is valid (correctness PASS, three 50/100 wall samples, Level 1 profiler summary collected). No optional target is configured, and no terminal-round limit applies to baseline establishment.

Orchestrator owns canonical pointer updates and workflow transition.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (run before and after measurement; all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/fused_moe/base.py kernels/track1-triton/fused_moe/bi150/baseline_adapter.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/baseline_adapter.py --warmup 50 --repeat 100
```

Forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/fused_moe/bi150/log/round_000_forward_50iter.pt.trace.json
```

Separately scoped unmodified repository summaries (both returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/fused_moe/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 3.258671
```

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/fused_moe/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 3.258671
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity |
| runtime fingerprint check | `0` | torch 2.7.1, triton 3.1.0, BI-V150 (7,1) |
| correctness 50/100 | `0` | round_status_000.md; report Correctness table |
| wall sample 1, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 2, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 3, 50/100 | `0` | report Interleaved Wall Timing |
| forward profiler 20/50 | `0` | `log/round_000_forward_50iter.pt.trace.json` |
| summarize `baseline_base` | `0` | report Profiler Evidence |
| summarize `candidate_baseline_adapter` | `0` | report Profiler Evidence |
