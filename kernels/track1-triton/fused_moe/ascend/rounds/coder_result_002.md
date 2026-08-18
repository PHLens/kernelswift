# Coder Result 002

Result: `candidate-ready`

- round: `002`
- source_canonical: `triton_fused_moe_001.py`
- source_canonical_sha256: `e42d811a7aee32f3fb34b19c00f88bf7922129faccc3d670dd07abc2df443287`
- decision: `rounds/decision_002.md`
- decision_sha256: `aa051ac4ff036222154badc10ae2051560396720cd9fedbb4f3a8d4f755c9ec2`
- candidate: `triton_fused_moe_002.py`
- candidate_sha256: `1b5c8ecded2008991f0f7cc039f0e06fa072bf5b8a7c6d5630574f64a43f4074`
- selected_profile: `triton_ascend`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `pending` (Verifier-owned)

## Primitive and Hint Conformance

| Primitive / Hint | Status | Note |
|---|---|---|
| `tl.load` | Supported | logits `[E]` fp32, x `[H]` fp16, gate/up `[I,H]`, w2 `[H,I]`. |
| `tl.store` | Supported | `out` `[H]` fp16. |
| `tl.arange` | Supported | Extents 128 (`H`), 64 (`I`), 8 (`E`), 2 (`K`); `up_idx = arange(0,I)+I`. |
| `tl.program_id` | Supported | Axis 0 only (1D grid `(T,)`). |
| `tl.zeros` | Supported | `(H,)` fp32 accumulator; `(K,)` fp32/fp32 topk arrays. |
| `tl.max` | Supported | Axis-0 scalar reduction over `[E]` (softmax max-subtract and per-round best). |
| `tl.exp` | Supported | Elementwise fp32 (softmax + SiLU). |
| `tl.sum` | Supported | Axis-0 (softmax denom, renormalize denom, scalar extract) and rank-1 `axis=1` (gate/up/down GEMM). |
| `tl.argmax` | Supported | Axis-0 over `[E]`; lowest index on tie (matches torch.topk). |
| `tl.where` | Supported | Tie mask for top-k selection and per-k scalar extraction. |
| `tl.static_range` | Supported | Compile-time loops `K=2` (selection + FFN). |
| `num_warps=1` | Constrained (proven) | Only launch hint selected; proven on Ascend. |

No Unsupported or Unknown primitive is made normative. `tl.dot`,
`tl.make_block_ptr`, `async_copy`, `num_stages`, `fast_libentry`, `vectorize`,
`tl.trans`, `tl.split` are not used. Direct Triton launch (the proven Ascend
launcher path); no `triton_ascend` import.

## Conformance Notes (candidate-ready)

Small target-language accommodations that preserve all normative semantics:

- **Routing reproduced exactly from base.py.** Ordering: softmax in fp32
  (`scores = exp(logits - max)/sum(exp)`), top-2 selection over the fp32 scores,
  renormalize in fp32 (`topk_weights /= sum(topk_weights)`), then cast to fp16.
  base.py's exact order is: softmax (fp32) -> topk -> renormalize (fp32) ->
  `.to(dtype)` (fp16) -> weighted reduce with fp16 weights. The kernel casts
  `topk_weights` to `tl.float16` before the reduce (`weight.to(tl.float32)` at the
  multiply, but the stored weight is the fp16-rounded value), matching base.py's
  fp16 weight in `expert_out * flat_w`.
- **Top-k via repeated argmax over E=8** (the decision's prescribed simple loop),
  not any sort network / winner tree / dynamic gather (all forbidden anti-patterns
  for this shape). `tl.argmax` returns the lowest index on ties, matching
  `torch.topk`'s index-order tie-break; for fp32 `randn` logits ties are
  ~impossible, and the decision documents this equivalence.
- **Gate/up split without constexpr slicing** (Ascend rejects `gate_up[:I]`, per
  round 001 finding): gate and up are computed as two independent rank-1
  `tl.sum` reductions over `w1[e,0:I,:]` and `w1[e,I:2I,:]` rows, numerically
  identical to slicing a fused `[2I]` result.
- **Scalar extraction via masked sum**: `expert_id`/`weight` are held in
  `[K]` register arrays (`topk_ids`/`topk_weights`) and extracted per-k with
  `tl.sum(tl.where(k_idx == k, topk_vals, 0))`, the same approach as the proven
  MLU round-2 kernel. No vector indexing by a loop variable.
- **SiLU via `tl.exp`** (`gate * (1/(1+exp(-gate)))`), not `F.silu`, inside the
  kernel.
- **Rank-1 matmuls via `tl.sum`**, no `tl.dot` (per decision and the flexattention
  round-3 host-penalty evidence).
- **`get_inputs` uses `device="npu"`**; `w1`/`w2` cast to fp16 in `forward` via
  `.to(dtype)` exactly as base.py.

## Local Gate

| Gate | Command | Exit | Result |
|---|---|---|---|
| `ast.parse` | harness `_filter_module_ast` AST load | 0 | pass |
| `py_compile` | `python3 -m py_compile triton_fused_moe_002.py` | 0 | pass |
| Harness loader | real `auto_bench.py` AST loader | 0 | pass |
| Correctness smoke | `auto_bench.py --v0_file .../base.py --v1_file .../triton_fused_moe_002.py --warmup 1 --repeat 3 --full-traceback` | 0 | `PASS accuracy; v0=7.990900 ms, v1=0.407550 ms, speedup=19.607x` |

The real harness loader retains the kernel's `@triton.jit` `FunctionDef` and all
top-level imports; correctness passes against the reference under
`atol=1e-2, rtol=1e-2`.

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA256 | After SHA256 |
|---:|---|---:|---|---|---|
| 1 | `python3 -m py_compile triton_fused_moe_002.py` | 0 | none | - | `1b5c8ecded2008991f0f7cc039f0e06fa072bf5b8a7c6d5630574f64a43f4074` |
| 2 | `auto_bench.py ... --warmup 1 --repeat 3 --full-traceback` | 0 | none; correctness PASS, compile smoke PASS | `1b5c8ecd...` | `1b5c8ecd...` |

No semantic repair was required. `tl.argmax`, `tl.max`, `tl.exp`, `tl.sum`, and
`tl.where` all compiled and executed correctly on the Ascend runtime; the routing
fusion produced a correct result on the first compile-smoke attempt.

## Reason Code

`candidate-ready`: the candidate conforms to the immutable design (routing fused
into the single per-token kernel; FFN path and weighted reduce unchanged; no
`tl.dot`); correctness and the local compile-smoke gate pass against the real
harness.
