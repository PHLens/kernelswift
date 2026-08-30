# Coder Result 002

Result: `candidate-ready`

- round: `002`
- source_canonical: `triton_fused_moe_001.py`
- source_canonical_sha256: `444eb2fb3e14c48359b27137d11b7f57da22211ad4034e6e56f05af5b4561124`
- decision: `rounds/decision_002.md`
- decision_sha256: `31e2767c6b1a65aaa49b0c3a9711dd2f0a3c06a741ef042e1c60bf31a911df6d`
- candidate: `triton_fused_moe_002.py`
- candidate_sha256: `e5d0058d6fb6f23f94e2623ae374d2776c4b2c6d4eb235b2c6c75524fb44eb73`
- selected_profile: `triton_gcu`
- runtime_fingerprint: `project.md#runtime-fingerprint`

## Implementation of the Unified Sketch

The candidate realizes decision_002's sketch: the per-token kernel now reads raw
`router_logits` and computes routing in-place, eliminating all eager routing
launches. Each program:

1. loads `token_logits` (`[E]` fp32) and computes
   `scores = softmax(logits)` via the numerically-stable
   `exp(logits - max_logit) / sum(exp(...))`.
2. computes top-2 via repeated argmax (see below), yielding `topk_vals` (fp32)
   and `topk_ids` (int32), both `[K]`.
3. renormalizes `topk_weights = topk_vals / sum(topk_vals)` and casts to fp16,
   matching base's `topk_weights.to(dtype)`.
4. loads `x` (`[H]`) and runs the same fused FFN as v1: two independent
   `[I]` gate/up GEMMs, SiLU, down GEMM, and top-k weighted accumulation.
5. stores `out_acc` cast back to fp16.

## Top-2 Repeated-Argmax (no `tl.argmax`)

Following the MLU v2 reference exactly, `tl.argmax` is avoided entirely (its
axis-1 reduction is unproven on GCU). The argmax is obtained via the masked-sum
trick, using only Supported primitives:

```python
best_val  = tl.max(remaining, axis=0)                       # max value
is_best   = remaining == best_val                            # bool mask
best_id   = tl.sum(tl.where(is_best, e_idx, 0), axis=0)      # argmax index (int32)
remaining = tl.where(is_best, -1.0, remaining)               # mask out for next pass
```

Because `scores` are softmax values in `(0, 1)`, `-1.0` is a safe sentinel, and
fp32 random logits produce no ties (per the decision's tie assumption). The loop
runs `K=2` times via `tl.static_range`.

## Key Design Decisions

- **Routing fully in-kernel**: softmax + top-2 + renormalize + fp16 cast all
  happen inside `_fused_moe_v2_kernel`; `fused_moe_v2` no longer calls
  `torch.softmax`/`torch.topk`. The only host-side work left is the fp16 weight
  cast (`w1/w2.to(dtype)`) and the single kernel launch.
- **fp16 weight cast preserved**: after renormalization, `topk_weights` is cast
  to `tl.float16`, then `.to(tl.float32)` at the accumulation point — identical
  to v1 and matching base's `topk_weights.to(dtype)` semantic. This keeps the
  `fp16 cast` invariant explicit rather than relying on tolerance.
- **elementwise `tl.sum` GEMM** (not `tl.dot`), same as v1.

## GCU Adaptation Points (vs MLU v2 reference)

| Concern | MLU v2 reference | GCU candidate |
|---|---|---|
| device import | `import torch_mlu` | `import torch_gcu` |
| launcher | `with torch.mlu.device(...)` + `num_stages=1` | direct launch, no `num_stages` (Unknown on GCU) |
| expert offset | `expert_id_scalar.to(tl.int64)` | int32 (`expert_id_scalar` stays int32) |
| gate/up split | `gate_up[:I]` / `gate_up[I:]` slice | two independent `[I]` GEMMs (GCU rejects tensor slice) |
| weight cast | fp32 weight used directly | renorm cast to fp16 then `.to(float32)` at accumulate |

## State-Dict Contract

Unchanged from v1: `ModelNew` exposes exactly `w1 [8,128,128]` and
`w2 [8,128,64]` fp32 `nn.Parameter` (init `normal_(std=0.02)`), no extra
parameters or buffers. `forward` casts `w1/w2.to(dtype)` (fp16) before launch.

## Primitive and Hint Conformance

- `tl.load`, `tl.store`, `tl.arange`, `tl.program_id`, `tl.zeros`, `tl.sum`,
  `tl.exp`, `tl.where`, `tl.max`, `tl.static_range`, and `tl.broadcast_to`
  (via scalar broadcast in `tl.where`) are all used. `tl.max`/`tl.sum` axis-0
  scalar reductions over the `[E]` vector are the routing-stage reductions the
  decision explicitly authorizes (Supported per decision_002 Pitfalls).
- `num_warps=1`, direct launch, no `tl.dot`, no `tl.argmax`, no `tl.int64`.

## Deviations from decision_002

None. The candidate follows the Unified Sketch and Evaluation Contract without
modification. The MLU-derived adjustments (int32 offsets, two independent
gate/up GEMMs, direct launch) are GCU target-profile accommodations that
preserve all normative semantics and are recorded as conformance notes.

## Attempt Ledger

| Attempt | Command | Exit status | Defect | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | `python3 -m py_compile s60/triton_fused_moe_002.py` | 0 | none | not-applicable | `e5d0058d6fb6f23f94e2623ae374d2776c4b2c6d4eb235b2c6c75524fb44eb73` |

`ast.parse` (via `py_compile`) passed on the first attempt. No semantic repair
was required. The real-harness GCU compile/execution smoke remains the
Verifier's responsibility per the measurement-exclusivity contract.
