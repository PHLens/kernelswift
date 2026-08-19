# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"fuse the 8-expert loop, per-expert mask/gather/scatter, double GEMM, SiLU, and weighted reduction into a single per-token Triton kernel (grid=(T,)); routing (softmax/topk/renorm/cast) stays eager this round","allowed_changes":["ModelNew.forward body","new Triton jit kernel","weight layout helpers"],"invariants":["ModelNew public contract","output dtype and shape","numerical semantics","state_dict keys exactly w1/w2","benchmark semantics"],"expected_wall_improvement_pct":50.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor hidden shape=[T,H] dtype=fp16 layout=row_major memory=global
tensor topk_ids shape=[T,K] dtype=int32 layout=row_major memory=global
tensor topk_weights shape=[T,K] dtype=fp16 layout=row_major memory=global
tensor w1 shape=[E,2I,H] dtype=fp32 layout=row_major memory=global
tensor w2 shape=[E,H,I] dtype=fp32 layout=row_major memory=global
tensor out shape=[T,H] dtype=fp16 layout=row_major memory=global
tile x shape=[H] dtype=fp32 memory=register
tile out_acc shape=[H] dtype=fp32 memory=register
tile gate_up shape=[2I] dtype=fp32 memory=register
tile act shape=[I] dtype=fp32 memory=register

# O Operations
alloc out_acc = zeros(H)
load x <- hidden[token,0:H]
load expert_id <- topk_ids[token,0:K]
load weight <- topk_weights[token,0:K]
compute gate_up[j] = sum_{h in 0..H} x[h] * w1[expert_id,j,h]
compute act[i] = silu(gate_up[i]) * gate_up[i+I]
compute out_acc[h] = sum_{k in 0..K} weight[k] * sum_{i in 0..I} act[i] * w2[expert_id,h,i]
store out[token,0:H] <- out_acc

# C Control
parallel token over T
guard token < T

# H Target Hints
target=triton_gcu
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; routing remains eager"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the 8-expert loop, per-expert mask/gather/scatter, double GEMM, SiLU, and weighted reduction into a single per-token Triton kernel (grid=(T,)); routing (softmax/topk/renorm/cast) stays eager this round","expected_causal_chain":["per-expert Python loop and scatter/gather eager ops disappear","runtime launch count per call drops from 147 to a few routing kernels plus one fused kernel","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"runtime_launch_count_per_call","expectation":"decrease"},{"name":"runtime_launch_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","state_dict keys unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/bottleneck-judgment.md`: 147 GCU runtime launches/call with tiny per-expert GEMMs at T=83 indicates launch/host overhead dominates; device time is unavailable on this GCU exporter, so the normalized observables are `runtime_launch_count_per_call` and `runtime_launch_us_per_call`.
- Consulted `references/anti-patterns.md` and MLU `outcome.md`: the MLU v1 per-token kernel (elementwise outer-product GEMM via `tl.sum`, routing left eager) yielded 12.3x and is the minimum-risk first step; `tl.dot` is Unknown on the GCU profile, so this round uses elementwise `tl.sum` reduction rather than `tl.dot`.
- Consulted `prompts/coder_targets/triton_gcu.md`: `num_warps=1` is the only confirmed warp config; direct launch (no `fast_libentry`) is the only launcher; int64 is unsupported (torch `topk` int64 downgraded to int32 with UserWarning), so index with int32 (expert_id 0..7, max offset fits int32).
- State-dict contract: `ModelNew` must expose exactly `w1 [8,128,128]` and `w2 [8,128,64]` fp32 parameters so `load_state_dict` synchronizes weights; pre-cast fp16 weight copies must be non-persistent buffers, not parameters.

## Rationale and Evidence

The accepted report (`rounds/report_000.md`) records 147 GCU runtime launches per forward call (74 `topsLaunchKernel` + 73 `topsLaunchCooperativeKernel`) with wall median 5.11 ms. The eager implementation loops over 8 experts in Python with per-expert boolean masking and boolean-indexed gather/scatter, plus softmax/topk/renorm/cast and per-expert double GEMM + SiLU as separate eager ops. At T=83 the per-expert GEMMs are tiny (`[n_e,128]x[128,128]`), so device work is minimal and wall is dominated by eager launch/dispatch overhead — the same structural premise that gave MLU its 50.4x path.

The first-round intervention replaces the expert loop and per-expert scatter/gather with a single per-token Triton kernel (`grid=(T,)`) computing the two GEMMs, SiLU, and top-k weighted reduction inline, keeping routing eager for minimal risk (matching MLU v1's 12.3x first step). This drops the launch count from 147 to a few routing kernels plus one fused kernel, a clear, falsifiable mechanism with expected wall improvement well above the 5% adoption threshold.
