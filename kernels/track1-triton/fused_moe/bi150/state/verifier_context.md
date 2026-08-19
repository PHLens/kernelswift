# Verifier Context — fused_moe / BI150

Round 002 complete. Canonical pointer (advanced by Orchestrator) → `triton_fused_moe_002.py`.

## Operator

- `fused_moe` (MoE: softmax router + top-2 gating + per-expert GEMM + SiLU + weighted reduction)
- inputs: `hidden_states[83,128]` fp16, `router_logits[83,8]` fp32; output `[83,128]` fp16
- params: `w1[8,128,128]`, `w2[8,128,64]`; num_experts=8, top_k=2, hidden=128, intermediate=64, renormalize=True

## Canonical history

| Round | Kernel | Wall median | Device us/call | Kernel count/call | device_ratio |
|---|---:|---|---:|---:|---:|
| 000 | baseline_adapter.py | 3.258671 ms | 968.16 | 123.9 | 0.297 |
| 001 | triton_fused_moe_001.py | 2.488731 ms | 504.31 | 54.1 | 0.2026 |
| 002 | triton_fused_moe_002.py | 0.493474 ms | 140.84 | 9.82 | 0.2854 |

## Current canonical (after round 002)

- kernel: `triton_fused_moe_002.py`, SHA `6ac1f44b111285f5bf746110c51f6486868b12beb2deae3390663d74233f8ae5`
- wall median `0.493474 ms`, device `140.84 us/call`, kernel_count `9.82/call`
- single fused Triton kernel `_fused_moe_expert_kernel` (grid=8, one program per expert) using `tl.dot` for both GEMMs + `tl.atomic_add` weighted reduction

## Capability learned (updates target-profile understanding)

`tl.dot` with fp16 inputs and contraction dims 128 (gate/up) and 64 (down) WORKS on this
BI150 profile. Max abs output diff ~1.5e-05 (far below 1e-2 tolerance). This extends the
recorded probe beyond the `(32,32)@(32,32)` case in `coder_targets/triton_cuda.md`.

## Remaining bottleneck (for round 003+, if continued)

1. `_fused_moe_expert_kernel` is now the single largest kernel (55.80 us/call, 1 launch, grid=8).
2. `torch.topk` (`gatherTopK` 21.59 + `bitonicSortKVInPlace` 17.85 us/call) and routing softmax (5.0 us/call) remain torch kernels — correctness-critical (tie semantics), preserved by design.
3. Small remaining overheads: `w1`/`w2` fp16 cast (`float16_copy_kernel` 2.94/call, 15.56 us/call), `out` zero-init (`FillFunctor` 4.60 us/call), renormalize sum/div (13.74 + 6.71 us/call).
4. device_ratio 0.2854 — operator is strongly host/launch-bound; wall 0.493 ms vs device 140.84 us (~71% host/launch/other). Further wall-time gains are bounded by the harness-fixed seed/synchronization floor and the remaining ~9.82 kernels/call.

## Profiler note (recurring, important)

On the CoreX profiler, ANY forward containing a Triton kernel emits overlapping CPU-side
`record_function` interval events (two per scope). `summarize_trace.py` raises
`overlapping scope events` for such scopes. Reliable attribution uses time-interval
separation over `cat=kernel` events, because the reference and candidate intervals are
strictly sequential (a ~95 us gap). Use the `_weighted_reduce_kernel` / `_fused_moe_expert_kernel` kernel-event timestamps as the scope anchors.

## Invariants (correctness-critical, must be preserved in any future candidate)

- tie-rule: `torch.topk(scores, 2, dim=-1)` descending value, ascending index on ties — NOT reimplemented (all rounds keep `torch.topk`).
- GEMM contraction dims: gate/up 128, down 64; SiLU activation; fp32 routing softmax; fp16 weight cast; per-token weighted-sum reduction of top-2 expert outputs.
- input tensors read-only; caller device/stream preserved.
