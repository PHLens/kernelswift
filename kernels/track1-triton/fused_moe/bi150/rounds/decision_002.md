# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_fused_moe_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"gemm-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse the per-expert GEMM loop (gate/up GEMM + chunk + SiLU + mul + down GEMM) and the sort-based bucketing into a single fused Triton kernel using tl.dot, collapsing the ~40 per-expert loop kernels plus the argsort (107 us/call), argsort gather (20 us/call), bincount/cumsum (12.5 us/call), and chunk/SiLU elementwise (88 us/call) into 1-2 kernels, reducing kernel count from 54.1/call toward ~10 and eliminating the now-largest single-kernel dispatch cost without changing the routing (torch.topk) or the output","allowed_changes":["ModelNew.forward dataflow","fused Triton kernel over the per-expert GEMM + chunk + SiLU + mul + down GEMM + weighted-reduce stages using tl.dot","replacement of torch.argsort-based bucketing with an in-kernel expert dispatch","elimination of the host-side inverse-permutation buffer"],"invariants":["ModelNew public contract (num_experts=8, top_k=2, hidden_size=128, intermediate_size=64, renormalize=True)","forward signature (hidden_states,router_logits)->out[83,128] fp16","torch.topk(scores,2,dim=-1) descending-value / ascending-index tie order preserved bit-exactly (topk is not reimplemented)","exact routing (fp32 softmax, fp16 weight cast), GEMM contraction dims (gate/up 128, down 64), SiLU activation, and weighted-sum reduction semantics","input tensors not mutated","caller-selected device and current stream preserved"],"expected_wall_improvement_pct":12.0}
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
compute expert_out <- fused_moe_kernel(hidden_states, flat_ids, flat_w, w1, w2)
store out <- expert_out.view(83,2,128).sum(dim=1)

# C Control
parallel token over T
guard token < T

# H Target Hints
target=triton_cuda
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; the fused Triton kernel replaces the per-expert torch GEMM loop and the torch.argsort/bincount/cumsum dispatch with a single kernel launch, introducing no new host-side state, buffer reuse, or allocation caching; hidden_states, router_logits, w1, and w2 are read-only inputs and the output tensor is written once"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"fuse the per-expert GEMM loop (gate/up GEMM + chunk + SiLU + mul + down GEMM) and the sort-based bucketing into a single fused Triton kernel using tl.dot, collapsing the ~40 per-expert loop kernels plus the argsort (107 us/call), argsort gather (20 us/call), bincount/cumsum (12.5 us/call), and chunk/SiLU elementwise (88 us/call) into 1-2 kernels, reducing kernel count from 54.1/call toward ~10 and eliminating the now-largest single-kernel dispatch cost without changing the routing (torch.topk) or the output","expected_causal_chain":["the argsort (radixSortKVInPlace 107.36 us/call), argsort gather (19.92 us/call), and bincount/cumsum (12.51 us/call) disappear as the expert dispatch moves in-kernel","the ~40 per-expert loop kernels (8x gate/up GEMM + 8x chunk + 8x SiLU + 8x mul + 8x down GEMM) collapse into 1-2 fused Triton kernels, removing their per-launch host overhead and the chunk/SiLU elementwise device time (48.05 + 40.17 us/call)","per-call kernel count drops from 54.1 toward ~10, reducing the dominant host/launch overhead (device_ratio 0.20)","device_us_per_call decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","torch.topk tie order preserved","routing and GEMM contraction semantics preserved","input not mutated"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The relevant entries are the grouped-topk selection-network failures (winner-tree Entry 011, full sort-32/64 Entry 012, dynamic `tl.gather` Entry 013, cumsum compaction Entry 016), all on MLU590-H8 / Triton 3.2.0. None matches this intervention's preconditions: this round does NOT reimplement topk (it stays `torch.topk`), does NOT use on-chip `tl.gather` for compaction, and does NOT build a hierarchical selection network. The per-token dispatch uses a fixed `top_k=2` layout and static `tl.arange`/`tl.load` addressing, avoiding the Entry 013 dynamic-gather trap. The only unproven primitive is `tl.dot`, which is a matmul (not a selection network) and is outside the catalog's scope.
- Target-profile consultation (`prompts/coder_targets/triton_cuda.md`): `tl.dot` is recorded Supported for `(32,32) @ (32,32)` with fp32→exact and bf16→fp32-accumulate results. **This round's GEMMs have contraction dims 128 (gate/up) and 64 (down) with fp16 inputs and a dynamic/small M (`n_e ≈ 20` average, or `M=1` per-token) — entirely outside the proven probe.** This is the primary capability risk. The decision is conditional on a matched local probe: Coder must first validate `tl.dot` lowering for the actual `(M,128)@(128,128)` and `(M,64)@(64,128)` fp16 shapes on BI150. If `tl.dot` cannot lower or produces incorrect results, the round completes as `capability-miss` (a recorded failure), NOT a silent numerical regression, and the canonical kernel remains `triton_fused_moe_001.py`.
- The GEMM `M` dimension is small (`n_e ≈ 20` per expert, or per-token `M=1`). The Iluvatar TCU already executes these GEMMs efficiently (~61/~58 us/call for all 8 experts combined). The round's expected gain is therefore driven primarily by **launch-count reduction and sort elimination**, not by GEMM compute speedup; if the fused `tl.dot` GEMM runs slower than the TCU for these skinny shapes, the device-time observable may not decrease even though kernel count does. The wall-time adoption still requires the 5% unrounded median; Coder must keep the TCU GEMM path as the comparison baseline in mind and report any `tl.dot` regression honestly.
- `num_warps`/`num_stages` are Unknown on this profile and must remain non-normative; `tl.make_block_ptr`, `vectorize`, and `async_copy` are Unknown and must not be required.
- The harness AST loader (`auto_bench.py` `_filter_module_ast`) retains `Import`/`ImportFrom`/`ClassDef`/`FunctionDef` and literal assignments, so `@triton.jit` top-level functions are preserved; the candidate must still expose `ModelNew`, `get_init_inputs`, and `get_inputs`.

## Rationale and Evidence

Round 001 (`triton_fused_moe_001.py`) was accepted at +21.44% wall (3.167858 → 2.488731 ms) by fusing the non-GEMM dispatch: it replaced the per-expert CUB `DeviceSelect` mask selection with a single `torch.argsort` bucketing plus a Triton `_weighted_reduce_kernel`, collapsing kernel count 123.9 → 54.1/call and device time 968.16 → 504.31 us/call.

The round-001 report's candidate profile (`report_001.md`) now shows a **mixed, launch-bound** remainder (device_ratio 0.2026): the single largest kernel is the self-introduced `radixSortKVInPlace` argsort at 107.36 us/call (21% of device time), and the per-expert Python loop still launches ~40 kernels (8x gate/up GEMM + 8x chunk + 8x SiLU + 8x mul + 8x down GEMM), with the chunk/SiLU elementwise contributing 48.05 + 40.17 us/call. Device time (504 us) is now only ~20% of wall (2488 us); the remaining ~80% is host/launch overhead from the still-large 54-kernel launch count.

The argsort is the central obstacle and the key observation for this round: **it cannot be removed in isolation.** Round 001 introduced the sort to make each expert's rows contiguous so the torch GEMMs could use plain slices instead of the baseline's `x_rep[mask]` gather (which triggered the ~263 us/call CUB dispatch). Eliminating the sort therefore requires eliminating the torch GEMM's need for contiguous buckets — which means fusing the GEMMs themselves into a Triton kernel with `tl.dot`. The argsort elimination and the GEMM fusion are the same intervention.

Fusing the per-expert loop into a single Triton kernel with `tl.dot` targets all three remaining overhead sources at once: (1) the argsort chain (107 + 20 + 12.5 ≈ 140 us/call device, 3 kernels), (2) the chunk/SiLU/mul elementwise (88 us/call device, 24 kernels), and (3) the per-expert GEMM launch overhead (16 kernels). Kernel count should fall from 54.1 toward ~10, which is the dominant lever because the operator is now launch-bound (device_ratio 0.20): each eliminated kernel launch removes host enqueue overhead that currently dominates wall time. Even a conservative estimate of removing ~40 kernel launches plus ~230 us of sort/elementwise device time from a 2488 us wall yields a double-digit percentage wall improvement.

I record `expected_wall_improvement_pct = 12.0` as a conservative central expectation. The estimate is deliberately below the sum of removable device time because (a) the skinny `tl.dot` GEMM (`M≈20` or `M=1`) may not match the TCU's throughput, so part of the ~119 us GEMM device time may persist or regress, and (b) the harness-fixed seed/synchronization floor caps the launch-overhead recovery. The adoption threshold remains the harness 5% unrounded median.

The primary risk is capability, not correctness: `tl.dot` for fp16 inputs and contraction dims 128/64 is unproven on the BI150 profile (the recorded probe covers only `(32,32)@(32,32)`). The Evaluation Contract makes this falsifiable — the kernel-count and device-time observables will directly reveal whether `tl.dot` lowers correctly and whether the fusion actually reduces device time. If `tl.dot` fails to lower or produces incorrect results, the round completes as `capability-miss` with `triton_fused_moe_001.py` remaining canonical; there is no silent numerical regression path because `torch.topk` (tie semantics) and the routing remain untouched and correctness is a mandatory Level-0 guardrail.

The canonical comparison source is `triton_fused_moe_001.py` (accepted in round 001), and the reference report is `rounds/report_001.md`. The intervention changes only the kernel dataflow inside `ModelNew.forward`; `torch.topk` (and its tie semantics), the routing, the public contract, output structure/shape/dtype, and the exact GEMM/SiLU/reduction semantics are preserved as invariants.
