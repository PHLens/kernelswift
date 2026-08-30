# Coder Result 001

Result: `candidate-ready`

- round: `001`
- source_canonical: `baseline_adapter.py`
- source_canonical_sha256: `b939d91f0f85e299a1102bfceb00da0e38c484a81c8d23ec78777fce68a3ee6f`
- decision: `rounds/decision_001.md`
- decision_sha256: `f03ecaf64e86d3ae01303e5c5ae5390dde32ee2f60ebf3244fd34f9f6aa01c7c`
- candidate: `triton_fused_moe_001.py`
- candidate_sha256: `444eb2fb3e14c48359b27137d11b7f57da22211ad4034e6e56f05af5b4561124`
- selected_profile: `triton_gcu`
- runtime_fingerprint: `project.md#runtime-fingerprint`

## Implementation of the Unified Sketch

The candidate realizes decision_001's sketch verbatim: one Triton program per
token (`grid=(T,)`), `num_warps=1`, direct launch. Each program:

1. loads the token row `x` (`[H]`), and initializes a float32 accumulator
   `out_acc = zeros(H)`.
2. loops `K` times with `tl.static_range(0, K)` (compile-time unrolled):
   - loads `expert_id` (int32 scalar) and `weight` (fp16 scalar) from
     `topk_ids[token,k]` / `topk_weights[token,k]`;
   - loads `w1[expert_id]` as a `[2I, H]` block and computes
     `gate_up = sum_h x[h] * w1[j,h]` via elementwise `tl.sum(axis=1)`;
   - splits `gate_up` into `gate` / `up`, computes
     `act = silu(gate) * up = (gate * (1/(1+exp(-gate)))) * up`;
   - loads `w2[expert_id]` as a `[H, I]` block and computes
     `out_k = sum_i act[i] * w2[h,i]` via `tl.sum(axis=1)`;
   - accumulates `out_acc += weight * out_k`.
3. stores `out_acc` cast back to fp16 into `out[token, :]`.

This collapses the eager per-expert Python loop, boolean mask/gather/scatter,
double GEMM, SiLU, and top-k weighted reduction into a single fused kernel,
matching the sketch's O(Operations) and O(Control) sections exactly.

## Key Design Decisions

- **Elementwise GEMM via `tl.sum`, not `tl.dot`**: `tl.dot` is Unknown on the
  `triton_gcu` profile (no qualifying probe), so both GEMMs are computed as
  outer-product elementwise multiplies followed by a `tl.sum` reduction along
  the contraction axis — the same strategy that gave MLU v1 its 12.3x result
  and avoids an unprovable normative construct.
- **int32 indexing only**: `expert_id` is loaded as int32, and all offset
  arithmetic (`expert_id * TWO_I * H`, `expert_id * H * I`) is int32. No
  `tl.int64` appears anywhere (the MLU reference used `expert_id.to(tl.int64)`;
  this is intentionally removed because GCU has no proven int64 support).
- **gate/up split without tensor slicing**: Triton-GCU rejects Python slice
  indexing on tensors (`ValueError: unsupported tensor index: slice`). Instead
  of slicing a single `[2I]` `gate_up`, the two halves are computed directly as
  two `[I]` GEMMs over `w1[expert_id, 0:I, :]` (gate) and
  `w1[expert_id, I:2I, :]` (up), which is semantically identical to
  `gate, up = gate_up.chunk(2)` but uses only proven `tl.load` / `tl.sum`.
  `tl.split` / `tl.trans` were avoided since neither is in the GCU Supported
  set.
- **Routing stays eager**: `softmax`/`topk`/`renormalize`/`cast` remain in torch
  inside `fused_moe_triton`, matching the Host Plan (`not-applicable`, kernel-only
  change) and the Evaluation Contract.
- **SiLU numerically**: implemented as `gate * (1.0 / (1.0 + tl.exp(-gate)))`,
  equivalent to `F.silu(gate)` (i.e. `x * sigmoid(x)`), within the
  `atol=1e-2, rtol=1e-2` tolerance.

## GCU Adaptation Points (vs MLU v1 reference)

| Concern | MLU v1 reference | GCU candidate |
|---|---|---|
| device import | `import torch_mlu` | `import torch_gcu` |
| launcher | `with torch.mlu.device(...)` + `num_stages=1` | direct launch `kernel[(T,)](...)`, no `num_stages` (Unknown on GCU) |
| index width | `expert_id.to(tl.int64)` | int32 offsets, never `tl.int64` |
| `topk_ids` dtype | `.to(torch.int32)` | `.to(torch.int32)` explicit (torch.topk returns int64; GCU downgrades with UserWarning) |
| weight dtype | `w1/w2 .to(dtype)` in caller | `w1/w2 .to(dtype)` (fp16) before kernel launch |

## State-Dict Contract

`ModelNew` exposes exactly two parameters: `w1` (`[E, 2I, H] = [8,128,128]`,
fp32) and `w2` (`[E, H, I] = [8,128,64]`, fp32), both `nn.Parameter` with
`nn.init.normal_(std=0.02)`, and no extra parameters or buffers. This preserves
the `load_state_dict(model.state_dict())` synchronization used by `compare_case`.
The fp16 cast copies are local temporaries, not persistent state.

## Primitive and Hint Conformance

- `tl.load`, `tl.store`, `tl.arange`, `tl.program_id`, `tl.zeros`,
  `tl.reshape`, `tl.sum`, `tl.exp`, `tl.static_range`, and `tl.broadcast_to`
  (via `[:, None]` / `[None, :]` expand in multiply) are all in the GCU
  Supported set (the first seven from the profile; `tl.sum`/`tl.exp`/
  `tl.static_range` were additionally established by the grouped-topk Round 001
  smoke under the same S60 regime).
- `num_warps=1` is the only confirmed warp config and is used exclusively.
- `tl.dot`, `tl.make_block_ptr`, `num_stages`, `fast_libentry` are all Unknown
  and are intentionally absent from the candidate.

## Deviations from decision_001

None. The candidate follows the Unified Sketch, Optimization Intent, and
Evaluation Contract without modification. The only MLU-derived adjustments
(device import, direct launch, int32 indexing) are GCU target-profile
accommodations that preserve all normative semantics and are recorded as
conformance notes rather than design changes.

## Attempt Ledger

| Attempt | Command | Exit status | Defect | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | `python3 -m py_compile s60/triton_fused_moe_001.py` | 0 | none (py_compile-level) | not-applicable | `9cb247591e84113625e3a63e34d6c40a8e03241620576c3612058addeb9fe45b` |
| 2 | `python3 -m py_compile s60/triton_fused_moe_001.py` | 0 | Verifier compile failure: `unsupported tensor index: slice` at `gate_up[:I]` / `gate_up[I:]`; fixed by splitting into two `[I]` GEMMs over `w1` gate/up halves (no tensor slicing) | `9cb247591e84113625e3a63e34d6c40a8e03241620576c3612058addeb9fe45b` | `444eb2fb3e14c48359b27137d11b7f57da22211ad4034e6e56f05af5b4561124` |

One same-round Verifier repair was applied: the only change was replacing the
tensor slice split with two independent `[I]` GEMMs (gate/up halves of `w1`),
preserving all semantics, the algorithm, the dataflow, and the Evaluation
Contract. No `tl.dot`, no routing change, no other design change was
introduced. `py_compile` passes on the repaired candidate. The real-harness
GCU compile/execution smoke remains the Verifier's responsibility per the
measurement-exclusivity contract.
