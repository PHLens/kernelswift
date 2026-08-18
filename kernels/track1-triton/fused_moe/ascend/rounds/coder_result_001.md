# Coder Result 001

Result: `candidate-ready`

- round: `001`
- source_canonical: `baseline_adapter.py`
- source_canonical_sha256: `a7fc0001db3ee9e636241954d2c071b62acee518b23f4c59c19efee886203a02`
- decision: `rounds/decision_001.md`
- decision_sha256: `821c40436ba4af5ed82029405060a4a55b5c3165c5da7d7c26ce4976136218f1`
- candidate: `triton_fused_moe_001.py`
- candidate_sha256: `e42d811a7aee32f3fb34b19c00f88bf7922129faccc3d670dd07abc2df443287`
- selected_profile: `triton_ascend`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `pending` (Verifier-owned)

## Primitive and Hint Conformance

| Primitive / Hint | Status | Note |
|---|---|---|
| `tl.load` | Supported | Contiguous loads: x `[H]`, gate/up `[I,H]`, w2 `[H,I]`, scalar expert_id/weight. |
| `tl.store` | Supported | `out` `[H]` fp16 store. |
| `tl.arange` | Supported | Extents 128 (`H`), 64 (`I`). `up_idx = arange(0,I)+I` offset arange. |
| `tl.program_id` | Supported | Axis 0 only (1D grid `(T,)`). |
| `tl.zeros` | Supported | `(H,)` fp32 accumulator. |
| `tl.sum` | Supported | Rank-1 reduction (`axis=1`) for gate/up GEMM and down GEMM. |
| `tl.exp` | Supported | Elementwise fp32 for SiLU. |
| `tl.static_range` | Supported | Compile-time loop `K=2`. |
| `num_warps=1` | Constrained (proven) | Only launch hint selected; proven on Ascend. |

No Unsupported or Unknown primitive is made normative. `tl.dot`,
`tl.make_block_ptr`, `async_copy`, `num_stages`, `fast_libentry`, `vectorize`,
and `tl.trans` are not used. Direct Triton launch (the proven Ascend launcher
path) is used; no `triton_ascend` import (metadata-only, non-importable).

## Conformance Notes (candidate-ready)

Small target-language accommodations that preserve all normative semantics:

- **No constexpr slicing of reduction output.** The Unified Sketch splits
  `gate_up [2I]` via `gate_up[0:I]` / `gate_up[I:2I]`. The Ascend compiler
  rejects constexpr slice bounds (`ValueError: unsupported tensor index:
  slice(None, constexpr[64], None)`). Instead of slicing the fused `[2I]`
  reduction, gate and up are computed as two independent rank-1 `tl.sum`
  reductions over the `[0:I, :]` and `[I:2I, :]` rows of `w1[e]`, using
  `gate_idx = arange(0,I)` and `up_idx = arange(0,I)+I`. This is numerically
  identical to slicing the fused `[2I]` result (same outer-product sums, same
  `axis=1` reduction) and changes no algorithm, dataflow, or semantics.
- **SiLU via `tl.exp`** (`gate * (1/(1+exp(-gate)))`), not `F.silu`, since the
  gating is inside the kernel (per decision pitfall guidance).
- **Rank-1 matmuls via `tl.sum`** rather than `tl.dot`: the profile's `tl.dot`
  evidence covers only `(16,16)@(16,16)` fp32; rank-1 dot on Ascend is unproven,
  and the decision explicitly prescribes `tl.sum` outer-products. Numerically
  identical for these rank-1 products.
- **`get_inputs` uses `device="npu"`** (not the `"cuda"` placeholder), per the
  target profile's explicit guidance.
- Routing (softmax + top-2 + renormalize + fp16 cast) reproduces `base.py`
  exactly inside `forward`; `topk_ids` cast to int32 for the kernel. `w1`/`w2`
  cast to `dtype` (fp16) before launch, matching base.py's `.to(dtype)`.

## Local Gate

| Gate | Command | Exit | Result |
|---|---|---|---|
| `ast.parse` | harness `_filter_module_ast` AST load | 0 | pass |
| `py_compile` | `python3 -m py_compile triton_fused_moe_001.py` | 0 | pass |
| Harness loader | real `auto_bench.py` AST loader | 0 | pass |
| Correctness smoke | `auto_bench.py --v0_file .../base.py --v1_file .../triton_fused_moe_001.py --warmup 1 --repeat 3 --full-traceback` | 0 | `PASS accuracy; v0=8.029000 ms, v1=0.624290 ms, speedup=12.861x` |

The real harness loader retains the kernel's `@triton.jit` `FunctionDef` and all
top-level imports; correctness passes against the reference under
`atol=1e-2, rtol=1e-2`.

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA256 | After SHA256 |
|---:|---|---:|---|---|---|
| 1 | `python3 -m py_compile triton_fused_moe_001.py` | 0 | none | - | `fused gate_up [2I]` then `gate_up[:I]` slice variant |
| 2 | `auto_bench.py ... --warmup 1 --repeat 3 --full-traceback` | 1 | `ValueError: unsupported tensor index: slice(None, constexpr[64], None)` on `gate_up[:I]` (Ascend rejects constexpr slice) | (slice variant) | (slice variant) |
| 3 | rework gate/up as two independent `tl.sum` reductions over `[0:I]` / `[I:2I]` rows (no slicing) | 0 | none | (slice variant) | `e42d811a7aee32f3fb34b19c00f88bf7922129faccc3d670dd07abc2df443287` |
| 4 | `auto_bench.py ... --warmup 1 --repeat 3 --full-traceback` | 0 | none; correctness PASS, compile smoke PASS | `e42d811a...` | `e42d811a...` |

The defect at attempt 2 was a local target-language accommodation (avoiding an
unsupported constexpr slice), not a semantic change: the gate/up split produces
bit-identical outer-product sums to the fused-then-sliced form. No `tl.dot`
substitution was made; `tl.sum` rank-1 outer-products compile and run correctly
on the Ascend runtime.

## Reason Code

`candidate-ready`: the candidate conforms to the immutable design (single
per-token kernel fusing FFN + weighted reduce, routing stays in PyTorch);
correctness and the local compile-smoke gate pass against the real harness.
