# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse the per-expert Python loop's non-GEMM dispatch (boolean mask selection flat_ids==e, x_rep[mask] gather, expert_out[mask]= scatter, mask.any reduce, chunk split, and the weighted sum reduction) into 1-2 fused Triton kernels that eliminate the ~263 us/call of CUB DeviceSelect/Reduce/Compact + index_elementwise gather/scatter + reduce/or overhead without touching torch.topk or the two per-expert TCU GEMMs","allowed_changes":["ModelNew.forward dataflow","fused Triton kernel over the per-expert mask/gather/scatter/chunk/weighted-reduce stages","dispatch from flattened (token,k) rows to experts"],"invariants":["ModelNew public contract (num_experts=8, top_k=2, hidden_size=128, intermediate_size=64, renormalize=True)","forward signature (hidden_states,router_logits)->out[83,128] fp16","torch.topk(scores,2,dim=-1) descending-value / ascending-index tie order preserved bit-exactly","exact routing (fp32 softmax, fp16 cast of weights), GEMM contraction dims (gate/up 128, down 64), SiLU activation, and weighted-sum reduction semantics","input tensors not mutated","caller-selected device and current stream preserved"],"expected_wall_improvement_pct":20.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor hidden_states shape=[83,128] dtype=fp16 layout=contiguous memory=global
tensor router_logits shape=[83,8] dtype=fp32 layout=contiguous memory=global
tensor w1 shape=[8,128,128] dtype=fp16 layout=contiguous memory=global
tensor w2 shape=[8,128,64] dtype=fp16 layout=contiguous memory=global
tensor out shape=[83,128] dtype=fp16 layout=contiguous memory=global
scalar T shape=[1] dtype=int
scalar K shape=[1] dtype=int

# O Operations
load scores <- softmax(router_logits.float(), dim=-1)
compute topk_weights <- topk(scores, K=2, dim=-1) values
compute topk_ids <- topk(scores, K=2, dim=-1) indices
compute topk_weights <- topk_weights / sum(topk_weights, dim=-1, keepdim)
compute flat_w <- topk_weights.view(-1).to(fp16)
compute flat_ids <- topk_ids.view(-1)
compute x_rep <- hidden_states.unsqueeze(1).expand(-1,2,-1).reshape(-1,128)
compute expert_out <- fused_expert_dispatch(x_rep, flat_ids, w1, w2)   # mask/gather/GEMM/SiLU/scatter fused
compute expert_out <- expert_out * flat_w.unsqueeze(-1)
store out <- expert_out.view(83,2,128).sum(dim=1)

# C Control
parallel token over T
guard token < T

# H Target Hints
target=triton_cuda
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; the fused Triton kernel is launched inside forward with no new host-side state, buffer reuse, or allocation caching; hidden_states, router_logits, w1, and w2 are read-only inputs and the output tensor is written once"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the per-expert Python loop's non-GEMM dispatch (boolean mask selection flat_ids==e, x_rep[mask] gather, expert_out[mask]= scatter, mask.any reduce, chunk split, and the weighted sum reduction) into 1-2 fused Triton kernels that eliminate the ~263 us/call of CUB DeviceSelect/Reduce/Compact + index_elementwise gather/scatter + reduce/or overhead without touching torch.topk or the two per-expert TCU GEMMs","expected_causal_chain":["per-call kernel count drops from 123.9 toward a much smaller number by removing the 8x-per-expert CUB mask selection (DeviceSelectSweepKernel 15.98/call, DeviceReduceSingleTileKernel 16/call, DeviceCompactInitKernel 15.98/call), index_elementwise gather+scatter (16/call), and reduce/or (8/call)","the ~263 us/call dispatch/selection overhead that exceeds either GEMM kernel (~61 or ~58 us/call) largely disappears","device_us_per_call decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","torch.topk tie order preserved","routing and GEMM contraction semantics preserved","input not mutated"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No entry's preconditions match this operator: the catalog records grouped-topk selection-network failures — winner-tree for repeated selection (Entry 011), full sort-32/sort-64 network expansion (Entry 012), dynamic `tl.gather` compaction (Entry 013), and cumsum compaction (Entry 016) — all on an MLU590-H8 / Triton 3.2.0 runtime with a hierarchical/partial-selection dataflow over 256 experts. This operator's first-round intervention deliberately keeps `torch.topk` (bitonic sort) and both TCU GEMMs untouched, fusing only the elementwise/mask/scatter/reduce dispatch. The catalog's `tl.gather` compaction failure (Entry 013) is relevant only as a warning against generic dynamic on-chip gather; Coder must avoid reimplementing the mask selection via a dynamic `tl.gather` and instead favor a per-token static-`top_k` layout that lets each program compute its two experts' masked GEMM contributions directly.
- The correctness-critical trap is the **topk tie order** (`torch.topk(scores, 2, dim=-1)` descending value, ties broken by ascending index). The first-round intervention does NOT reimplement `torch.topk` (it is preserved as a host `torch.topk` call), so the tie semantics are inherited unchanged and the grouped-topk lesson is honored by not touching topk at all.
- Target-profile consultation (`prompts/coder_targets/triton_cuda.md`): `tl.load`, `tl.store`, `tl.arange`, `tl.reshape`, `tl.dot`, `tl.sum`, `tl.where`, and `tl.static_range` are Supported on the recorded BI150 / CoreX runtime. `tl.dot` supports `(32,32)@(32,32)` fp32/bf16 matmul with exact/near-exact results, but the per-expert GEMMs here have contraction dims 128 (gate/up) and 64 (down) with fp16 inputs — outside the proven probe. **Therefore round 001 keeps the TCU GEMMs (`gemm_tcu_h`) untouched** and fuses only the non-GEMM dispatch; `tl.dot` remains a round-002+ candidate after a matched local probe. `num_warps`/`num_stages` are Unknown and must stay non-normative.
- The harness AST loader (`auto_bench.py` `_filter_module_ast`) retains `Import`/`ImportFrom`/`ClassDef`/`FunctionDef` and literal assignments, so a `@triton.jit`-decorated top-level function is preserved; the candidate module must still expose `ModelNew`, `get_init_inputs`, and `get_inputs`.

## Rationale and Evidence

Phase 0 profiler evidence (`rounds/report_000.md`) establishes a **mixed** baseline with a hard launch-count wall: `baseline_base` scope measured `968.162 us/device-call` and `123.9 kernels/call` against `3.258671 ms` wall time, giving `device_ratio ≈ 0.297` — about 30% of wall time is device kernel time and ~70% is host/launch overhead, driven by the ~124 tiny kernels the per-expert Python loop (`for e in range(8)`) launches per forward call.

The single largest overhead source is dispatch/selection, not compute. The per-expert loop launches, per expert, a boolean mask selection (`flat_ids == e` → `x_rep[mask]`) and a scatter (`expert_out[mask] = ...`), each lowering to multiple CUB/ATen kernels:

- `cub::DeviceSelectSweepKernel` (mask select): `15.98/call`, `126.10 us/call`
- `index_elementwise_kernel` gather (`x_e = x_rep[mask]`): `8.0/call`, `127.146 us/call`
- `index_elementwise_kernel` scatter (`expert_out[mask]=...`): `7.98/call`, `127.135 us/call`
- `cub::DeviceReduceSingleTileKernel` + `reduce_kernel<or_kernel>` (`mask.any()`): `24/call`, `~168 us/call`
- `cub::DeviceCompactInitKernel`: `15.98/call`, `56.78 us/call`

These dispatch kernels sum to ~263 us/call, **exceeding either GEMM kernel** (`gemm_tcu_h` gate/up `61.15 us/call`, down `57.96 us/call`, both `8/call`). The GEMMs are already on the Iluvatar TCU (cublasLt-style, fp16 in / fp32 accumulate) and are not the bottleneck; `torch.topk` uses the standard bitonic sort (`gatherTopK` + `bitonicSortKVInPlace`, `1/call` each) and is also not the bottleneck.

Fusing the non-GEMM dispatch is the textbook remedy, mirroring the task-8 Sinkhorn win (132.88 → 1 kernel, +87.17%): it removes the 8x-per-expert CUB mask-selection / gather / scatter / reduce / chunk kernels, replacing them with a fused Triton kernel (or two) that computes each token's two selected experts' contributions without the per-expert Python loop's scatter/gather round-trips to global memory. Because this operator carries real GEMM compute plus a large launch-bound dispatch, removing ~263 us/call of pure launch/selection overhead should translate directly into wall-time reduction. The device_ratio (0.297) being below 0.8 means the gain is bounded by the host/launch floor, but the dispatch kernels themselves are a compressible device-and-launch cost (not harness-fixed seed/synchronization), so a substantial fraction of the ~263 us/call should disappear.

The expected gain is bounded but substantial. The ~263 us/call dispatch overhead is almost entirely launch + tiny-kernel inefficiency; a conservative estimate of removing most of it (while the fused kernel still pays its own launches and the TCU GEMMs remain) yields a wall improvement around 15-25%. I record `expected_wall_improvement_pct = 20.0` as a central expectation; the adoption threshold remains the harness `5%` unrounded median. The primary capability risk is whether the fused per-token expert dispatch lowers correctly without reintroducing a dynamic gather (the grouped-topk Entry 013 lesson); the Level 1 device-time and kernel-count observables will directly verify this, and a failed lowering completes the round as `capability-miss`, not a silent numerical regression.

The canonical comparison source is `baseline_adapter.py` (established in Phase 0), and the reference report is `rounds/report_000.md`. The intervention changes only the kernel dataflow inside `ModelNew.forward`; `torch.topk` (and its tie semantics), both TCU GEMMs, the public contract, output structure/shape/dtype, and the exact routing/GEMM/SiLU/reduction semantics are preserved as invariants.
