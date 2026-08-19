# Coder Result 001

## Identity

- Round: `001`
- Result: `candidate-ready`
- Decision: `rounds/decision_001.md`
- Decision SHA256: `fa6ffd3d2a08dd78d2f3ad958890d0419a0115b898c68b6bbf4ef88105d43eca`
- Source canonical: `baseline_adapter.py`
- Source canonical SHA256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- Selected target profile: `triton_ascend`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1+cpu, torch_npu 2.7.1.post4, triton 3.2.0, Ascend910B4)
- Candidate path: `triton_attn_001.py`
- Candidate SHA256: `61eeb3367619684e6f61ea3a908c1fc78a575834b4a84c032748277d0e76be74`

## Conformance Notes (candidate-ready)

The candidate realizes the immutable decision exactly: replaces
`F.scaled_dot_product_attention` and its surrounding `view/transpose/reshape`
with a single Triton attention kernel that reads the native `[bsz, seq, hidden]`
contiguous layout via strided loads and writes the same layout directly,
eliminating the three `aclnnFlashAttentionScore_TransposeAiCore_Transpose`
input-transpose kernels and the one `aclnnInplaceCopy_TransposeAiCore_Transpose`
output-transpose kernel. Small target-language accommodations that preserve all
normative semantics:

- **Rank-1 products realized via `tl.sum`** (a Supported primitive) rather than
  `tl.dot`. The Sketch declares per-program register tiles `q[64]`, `scores[83]`,
  `v[64]`, `acc[64]` as rank-1, and the profile's `tl.dot` evidence covers only
  `(16,16)@(16,16)` fp32; a rank-1 `tl.dot` is unproven on Ascend. `tl.sum`
  reduction is numerically identical here (`scores[j] = sum_d q[d]*k[j,d]`,
  `acc[d] = sum_j probs[j]*v[j,d]`). No algorithm, dataflow, or semantics change.
- **Single KV block** (`BLOCK_K=128 >= SEQ=83`): the Sketch's `for k over 83`
  degenerates to one iteration, so a materialized (non-flash) softmax equals the
  global softmax (no online-softmax correction). Correct by construction.
- **1D grid `bsz*heads*seq` with `//`/`%` decomposition** instead of a 2D grid:
  the profile proves `tl.program_id` axis 0 only, so a single-axis launch avoids
  an unproven `program_id(axis=1)`.
- **`tl.where(k_off < SEQ, scores, -inf)` softmax mask**: `BLOCK_K=128` is
  zero-padded beyond `SEQ=83`; the padded scores must be excluded from softmax
  (initially omitted, causing a correctness failure — see Attempt Ledger).
- **`get_inputs` uses `device="npu"`** (not the `"cuda"` placeholder), per the
  target profile's explicit guidance; the harness also rewrites `"cuda"` if needed.

## Primitive and Hint Conformance

| Primitive / Hint | Status | Note |
|---|---|---|
| `tl.load` | Supported | Masked strided loads (`other=0.0`) from native `[bsz,seq,hidden]` layout. |
| `tl.store` | Supported | Contiguous stores back to native layout, fp16. |
| `tl.arange` | Supported | Extents 64 and 128 (powers of two). |
| `tl.program_id` | Supported | Axis 0 only (1D grid). |
| `tl.sum` | Supported | Axis-1 rank-1 reduction (QK^T / AV) and axis-0 scalar (softmax). |
| `tl.max` | Supported | Axis-0 scalar reduction (softmax numerator stabilization). |
| `tl.exp` | Supported | Elementwise fp32. |
| `tl.where` | Supported | Padding mask (`k_off < SEQ`). |
| `num_warps=1` | Constrained (proven) | Direct launch with `num_warps=1`. |

No Unsupported or Unknown primitive is made normative. `tl.make_block_ptr`,
`async_copy`, `num_stages`, `fast_libentry`, `vectorize`, and `tl.dot` are not used.

## Local Gate

| Gate | Command | Exit | Result |
|---|---|---|---|
| `ast.parse` | (harness `ast.parse` during load) | 0 | pass |
| `py_compile` | `python3 -m py_compile triton_attn_001.py` | 0 | pass |
| Harness loader | real `auto_bench.py` AST loader (`_filter_module_ast`) | 0 | pass |
| Correctness smoke | `auto_bench.py --v0_file .../base.py --v1_file .../triton_attn_001.py --warmup 1 --repeat 3 --full-traceback` | 0 | `PASS accuracy` |
| Diagnostic timing | `auto_bench.py ... --warmup 50 --repeat 100` | 0 | `PASS accuracy; v0=0.380115 ms, v1=0.365940 ms, speedup=1.039x` |
| Standalone `__main__` | `python3 triton_attn_001.py` | 0 | prints `torch.Size([2, 83, 512])` |

The real harness loader retains the kernel's `@triton.jit` `FunctionDef` and all
top-level imports; correctness passes against the reference under
`atol=1e-2, rtol=1e-2`.

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA256 | After SHA256 |
|---|---:|---|---|---|---|
| 1 | `python3 -m py_compile triton_attn_001.py` | 0 | - | - | `(pre-fix)` |
| 2 | `auto_bench.py ... --warmup 1 --repeat 3 --full-traceback` | 1 | softmax padded positions (j>=83) leaked 0.0 scores | `(pre-fix)` | `(pre-fix)` |
| 3 | (add `tl.where(k_off < SEQ, scores, -inf)`) + `auto_bench.py --warmup 1 --repeat 3` | 0 | - | `(pre-fix)` | `61eeb336...` |
| 4 | `auto_bench.py --warmup 50 --repeat 100` | 0 | - | `61eeb336...` | `61eeb336...` |

The single defect was a local softmax-masking bug (zero-padded KV positions
contributing 0.0 scores), repaired within the bounded same-round budget. No
semantic change to the algorithm, dataflow, or public contract was required.

## Reason Code

`candidate-ready`: the candidate conforms to the immutable design; correctness
and the local compile-smoke gate pass.
