# Coder Result 001

Result: candidate-ready

## Identity

- Round: `001`
- Decision: `kernels/track1-triton/fused_moe/bi150/rounds/decision_001.md`
- Decision SHA256: `0745c37ddc4a5e27811d9ad20845d8b168017033b3b61f59f253d2129d9f7681`
- Candidate: `kernels/track1-triton/fused_moe/bi150/triton_fused_moe_001.py`
- Candidate SHA256: `8424c7a01bc1d293c2b0ef509dd895950112cfb71dedd145053b4ac3f7eb9ad6`
- Canonical reference (last_accepted_kernel): `baseline_adapter.py`
- Canonical reference SHA256: `8e5c70232e541a02d83343216376ece9127a1c3e6ea6af77dc77a2723783facf`
- Base SHA256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Selected target profile: `triton_cuda`
- Runtime fingerprint: `project.md#runtime-fingerprint` (triton 3.1.0 / torch 2.7.1 / Iluvatar BI-V150, capability (7,1), 16 SM, 16 GiB)
- Measurement fingerprint: `5c2a51ab3f3ebaab1123b9fa534d4e4b940f3334f80fac00252df780d3900150`
- validate_decision: `valid=true` (with `--expected-profile triton_cuda`)

## Implementation

### Fusion strategy

The intervention is **kernel-fusion** over the per-expert Python loop's
non-GEMM dispatch. Two structural changes collapse the dominant launch-bound
overhead identified in `report_000.md` (123.9 kernels/call, ~263 us/call of
CUB/ATen dispatch, device_ratio 0.297):

1. **Sort-based bucketing replaces the per-expert CUB `DeviceSelect`**.
   Instead of running `mask = (flat_ids == e)` plus `x_rep[mask]` gather and
   `mask.any()` reduce once per expert (8x), the flattened `flat_ids` is sorted
   once (`torch.argsort`), so each expert's rows become a contiguous block.
   `torch.bincount` + `torch.cumsum` give the per-expert `[start, end)` slice
   bounds, and a single `x_rep[sort_idx]` gather reorders the rows once. Each
   expert then operates on a plain contiguous slice `x_sorted[start:end]`, with
   no boolean-mask selection, no `mask.any()`, and no CUB compaction.

2. **A single Triton kernel replaces scatter + weighted-sum reduction.** The
   per-expert GEMM results are written into the sorted buffer
   `y_sorted[start:end]` (contiguous store, no `expert_out[mask] =` scatter).
   The final `expert_out * flat_w` + `view(83,2,128).sum(1)` reduction is done
   by one `_weighted_reduce_kernel` with `grid=(num_tokens,)`, one program per
   token. Each program loads its two (k=0, k=1) sorted-slot positions via the
   inverse permutation `inv` (built host-side by `inv[sort_idx] = arange(166)`),
   loads the two rows and two weights, and accumulates the weighted sum into
   `out[token, :]`. No `atomic_add` is needed because each token owns exactly
   `top_k=2` rows.

### Preserved (bit-exact, untouched)

- `torch.softmax(router_logits.float(), dim=-1)` — fp32 routing softmax.
- `torch.topk(scores, 2, dim=-1)` — descending-value / ascending-index tie
  order inherited **bit-exactly**; no Triton reimplementation of topk.
- renormalize `topk_weights / sum(-1, keepdim=True)` + `.to(fp16)`.
- `x_rep = hidden_states.unsqueeze(1).expand(-1,2,-1).reshape(-1,128)`.
- Both per-expert TCU GEMMs (`x_e @ w1[e].T`, `act @ w2[e].T`) as ordinary
  `torch` matmuls with fp16 inputs / fp32 accumulate.
- `gate_up.chunk(2, dim=-1)` split and `F.silu(gate) * up` activation.

### Avoiding dynamic `tl.gather` (anti-pattern Entry 013)

The kernel never uses on-chip `tl.gather` or a dynamic compaction network.
The only dynamic addressing is a scalar `tl.load` of `inv[2t]` / `inv[2t+1]`
(the token's two sorted-slot positions), used purely as global-memory row
offsets into `y_sorted`; the hidden-dimension access is a static
`tl.arange(0, BLOCK_H)` contiguous vector load. This is global-memory
addressing, not the on-chip generic gather that regressed on the MLU590 runtime.

### Kernel structure

```
_weighted_reduce_kernel[(num_tokens,)](
    y_sorted,   # [166,128] fp16 expert outputs in sorted order
    inv,        # [166] int64 inverse permutation (flat -> sorted slot)
    flat_w,     # [166] fp16 top-k weights
    out,        # [83,128] fp16
    H=128, BLOCK_H=128,
)
```

## Gate Evidence

| Gate | Command | Result | Evidence |
|---|---|---|---|
| Decision validation | `python3 skills/kernel-opt-loop/scripts/validate_decision.py .../decision_001.md --expected-profile triton_cuda` | pass | `valid=true` |
| AST parse | `python3 -m py_compile .../triton_fused_moe_001.py` | pass | exit `0` |
| Harness loader | `auto_bench.py` AST loader loaded `ModelNew/get_init_inputs/get_inputs` and the `@triton.jit` top-level function | pass | smoke run completed without load/constructor error |
| Accuracy smoke (run 1) | `auto_bench.py --v0_file base.py --v1_file candidate --warmup 50 --repeat 100 --full-traceback` | pass | `PASS accuracy; v0=3.195127 ms, v1=2.475827 ms, speedup=1.291x` |
| Accuracy smoke (run 2) | same | pass | `speedup=1.307x` |
| Accuracy smoke (run 3) | same | pass | `speedup=1.296x` |
| Accuracy smoke (run 4) | same | pass | `speedup=1.300x` |

### Primitive conformance

| Primitive | Profile status | Used? | Note |
|---|---|---|---|
| `tl.load` | Supported (contiguous fp32; fp16 unproven) | yes | `[128]` fp16 vector rows + scalar `inv`/`flat_w` loads |
| `tl.store` | Supported (contiguous) | yes | `[128]` fp16 row stores |
| `tl.arange` | Supported | yes | `tl.arange(0, BLOCK_H)`, `BLOCK_H=128` |
| `tl.program_id` | Supported (axis 0, 1-D launch) | yes | `grid=(83,)` |
| `tl.sum` | not used | no | reduction done via explicit two-row accumulate |
| `tl.dot` / `tl.gather` | not used | no | GEMMs left on torch TCU; no on-chip gather |

No `num_warps`, `num_stages`, block pointers, `tl.dot`, or `tl.gather` are used,
so no Unknown/Unsupported primitive is required. Scalar dynamic global-memory
loads (`tl.load(inv_ptr + 2*t)`) are exercised and pass; they are not on-chip
gather and fall outside the Entry 013 preconditions.

## Conformance

- Public contract preserved: `ModelNew(num_experts=8, top_k=2, hidden_size=128,
  intermediate_size=64, renormalize=True)`, `forward(hidden_states, router_logits)
  -> out[83,128] fp16`.
- `get_init_inputs()` returns `[8, 2, 128, 64]`; `get_inputs()` returns
  `[hidden_states[83,128] fp16, router_logits[83,8] fp32]`.
- `torch.topk` tie order (descending value, ties ascending index) preserved
  bit-exactly — the operator does not reimplement topk.
- Routing (fp32 softmax, fp16 weight cast), GEMM contraction dims (gate/up 128,
  down 64), SiLU activation, and weighted-sum reduction semantics preserved.
- Output dtype/shape/device unchanged; `forward` does not mutate inputs and
  preserves the caller-selected device/stream.
- No new host-side state, cache, buffer reuse, or lifecycle semantics
  introduced (Host Plan: `not-applicable`).

## Attempt Ledger

| Attempt | Command | Exit | Defect | Candidate before | Candidate after |
|---|---|---|---|---|---|
| 1 | `py_compile` | 0 | - | - | `8424c7a0...` |
| 2 | accuracy smoke 50/100 | 0 | - | `8424c7a0...` | `8424c7a0...` (unchanged) |

No repair was required; the candidate compiled and passed accuracy on the first
attempt.

## Handoff

- Candidate is `candidate-ready`: accuracy PASS on four independent smoke runs
  (speedup ~1.29–1.31x on smoke timing; authoritative wall timing and kernel
  count are Verifier's), no semantic deviation from the immutable decision.
- The candidate must be benchmarked/verified by Verifier; Coder does not return
  `accepted`.

## Exact Reproduction Commands

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
cd /root/CodeBuddy/20260818191200/kernelswift
python3 skills/kernel-opt-loop/scripts/validate_decision.py kernels/track1-triton/fused_moe/bi150/rounds/decision_001.md --expected-profile triton_cuda
python3 -m py_compile kernels/track1-triton/fused_moe/bi150/triton_fused_moe_001.py
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/triton_fused_moe_001.py --warmup 50 --repeat 100 --full-traceback
```
