# Coder Result 001

## Identity

- Round: `001`
- Result: `candidate-ready`
- Decision: `rounds/decision_001.md`
- Decision SHA256: `91cae0bcb4eb0792e59be2c359b21dde2cc038a2d11e25f01e36bb20784bf379`
- Source canonical: `baseline_adapter.py`
- Source canonical SHA256: `31c4e9acea7d94ddd97740dbd3d33e6b505cbc3a118ed891b28f9e1ac5c0696f`
- Selected target profile: `triton_ascend`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1+cpu, torch_npu 2.7.1.post4, triton 3.2.0, Ascend910B4)
- Candidate path: `triton_flexattention_001.py`
- Candidate SHA256: `53e87eff27457f6268040c64979f99dcf30a809effc562caec3db951b141d4a3`

## Conformance Notes (candidate-ready)

The candidate realizes the immutable decision exactly: fuses causal SDPA into one
Triton kernel (QK^T, causal mask, softmax, AV) in a single direct launch with
`num_warps=1`. Small target-language accommodations that preserve all normative
semantics:

- **Rank-1 matmuls realized via `tl.sum`** (a Supported primitive) rather than
  `tl.dot`. The Sketch's `q_tile`/`scores`/`acc` are declared `[1, ...]`
  (M=1), and the profile's `tl.dot` evidence covers only `(16,16)@(16,16)`
  fp32; M=1 dot is unproven on Ascend. `tl.sum` reduction is numerically
  identical for these rank-1 products (`scores = sum_d q[d] k[j,d]`,
  `acc = sum_j probs[j] v[j,d]`). No algorithm, dataflow, or semantics change.
- **Single KV block** (`BLOCK_K=128 >= T=83`): the Sketch's `for block_k over
  KV_BLOCKS` degenerates to one iteration, so the per-block softmax equals the
  global softmax (no online-softmax correction needed). Correct by construction.
- **1D grid `T*H` with `pid // H` / `pid % H`** instead of a 2D grid: the
  profile proves `tl.program_id` axis 0 only, so a single-axis launch avoids an
  unproven `program_id(axis=1)`.
- `get_inputs` uses `device="npu"` (not the `"cuda"` placeholder), per the
  target profile's explicit guidance.

## Primitive and Hint Conformance

| Primitive / Hint | Status | Note |
|---|---|---|
| `tl.load` / `tl.store` | Supported | Masked contiguous loads/stores; verified locally. |
| `tl.arange` | Supported | Extents 64 and 128 (power of two). |
| `tl.program_id` | Supported | Axis 0 only (1D grid). |
| `tl.sum` | Supported | Axis-0 scalar (softmax) and rank-1 reduction for QK^T / AV. |
| `tl.max` | Supported | Axis-0 scalar reduction. |
| `tl.exp` | Supported | Elementwise fp32. |
| `tl.where` | Supported | Causal mask (lower-triangular). |
| `num_warps=1` | Constrained (proven) | Direct launch with `num_warps=1`. |

No Unsupported or Unknown primitive is made normative. `tl.make_block_ptr`,
`async_copy`, `num_stages`, `fast_libentry`, and `vectorize` are not used.

## Local Gate

| Gate | Command | Exit | Result |
|---|---|---|---|
| `ast.parse` | (harness `ast.parse` during load) | 0 | pass |
| `py_compile` | `python3 -m py_compile triton_flexattention_001.py` | 0 | pass |
| Harness loader | real `auto_bench.py` AST loader (`_filter_module_ast`) | 0 | pass |
| Correctness smoke | `auto_bench.py --v0_file .../base.py --v1_file .../triton_flexattention_001.py --warmup 1 --repeat 3 --full-traceback` | 0 | `PASS accuracy; v0=0.423540 ms, v1=0.373550 ms, speedup=1.134x` |
| Standalone `__main__` | `python3 triton_flexattention_001.py` | 0 | prints `torch.Size([83, 512])` |

The real harness loader retains the kernel's `@triton.jit` `FunctionDef` and all
top-level imports; correctness passes against the reference under
`atol=1e-2, rtol=1e-2`.

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA256 | After SHA256 |
|---|---:|---|---|---|---|
| 1 | `python3 -m py_compile` | 0 | - | - | `53e87eff...` |
| 2 | `auto_bench.py ... --warmup 1 --repeat 3 --full-traceback` | 0 | - | `53e87eff...` | `53e87eff...` |

No semantic repair was required.

## Reason Code

`candidate-ready`: the candidate conforms to the immutable design; correctness
and the local compile-smoke gate pass.
