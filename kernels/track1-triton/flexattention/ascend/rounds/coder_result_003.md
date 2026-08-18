# Coder Result 003

## Identity

- Round: `003`
- Result: `candidate-ready`
- Decision: `rounds/decision_003.md`
- Decision SHA256: `c2d0d068f7595bed4aec4e2497b9b390ae875f67dcbcf9de551b448383991b37`
- Source canonical: `triton_flexattention_002.py`
- Source canonical SHA256: `b0fe058c5b5336978e89933e8f0fed0d5a0449aede33c6b1b7a97c9c319c100f`
- Selected target profile: `triton_ascend`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1+cpu, torch_npu 2.7.1.post4, triton 3.2.0, Ascend910B4)
- Candidate path: `triton_flexattention_003.py`
- Candidate SHA256: `4faadac6cd0e3bb5d1faeaddafd899f0fd64c275632d2635f1612bf182686546`

## Conformance Notes (candidate-ready)

The candidate realizes the immutable decision exactly. The two GEMMs (QK^T and
AV) are routed through `tl.dot` on a `BLOCK_M=16` multi-token-per-program layout,
matching the Unified Sketch declaration-for-declaration:

- `BLOCK_M=16`, `BLOCK_D=64`, `BLOCK_KV=128` (>= T=83, single KV block).
- grid `= ceil(T/BLOCK_M) * H = 6 * 8 = 48`; `pid -> (token_block = pid // H,
  head = pid % H)`.
- `q_tile [BLOCK_M, BLOCK_D]`, `k_tile`/`v_tile [BLOCK_KV, BLOCK_D]`, all fp32.
- `scores = tl.dot(q_tile, tl.trans(k_tile)) * scale` -> `[BLOCK_M, BLOCK_KV]`.
- 2D causal mask `tl.where(k_off[None,:] <= m_off[:,None], scores, -inf)`.
- Softmax over axis=1: `scores - max(axis=1)[:,None]`, `exp`, `probs / sum(axis=1)[:,None]`.
- `acc = tl.dot(probs, v_tile)` -> `[BLOCK_M, BLOCK_D]`.
- Store guarded with a 2D mask (`m_off[:,None] < T`) for the partial token block
  (last block: 83-80 = 3 valid tokens).

`tl.dot` fp32 with M=16, K=64, N=128 **compiled and ran** on this runtime (the
decision's explicit capability check passed); no substitution back to `tl.sum`
was made.

The round-002 output buffer cache is retained unchanged (Host Plan
`not-applicable` for this kernel-only change): `self._out_cache` /
`self._out_cache_key` and `_get_output_buffer` are byte-identical to round 002.

## Primitive and Hint Conformance

| Primitive / Hint | Status | Note |
|---|---|---|
| `tl.dot` fp32 | Supported (M=16 proven) | `(16,64)@(64,128)` compiled and ran; correctness pass. |
| `tl.trans` | Standard | 2D transpose of `k_tile`. |
| `tl.load` / `tl.store` | Supported | 2D masked loads/stores. |
| `tl.arange` | Supported | Extents 16/64/128 (powers of two). |
| `tl.program_id` | Supported | Axis 0 only (1D grid of 48). |
| `tl.max` / `tl.sum` | Supported | Axis-1 reductions (softmax), `[:,None]` broadcast. |
| `tl.exp` | Supported | Elementwise fp32. |
| `tl.where` | Supported | 2D causal mask. |
| `num_warps=1` | Constrained (proven) | Direct launch. |

## Local Gate

| Gate | Command | Exit | Result |
|---|---|---|---|
| `ast.parse` | (harness `ast.parse` during load) | 0 | pass |
| `py_compile` | `python3 -m py_compile triton_flexattention_003.py` | 0 | pass |
| Harness loader | real `auto_bench.py` AST loader | 0 | pass |
| Correctness smoke | `auto_bench.py --v0_file .../base.py --v1_file .../triton_flexattention_003.py --warmup 1 --repeat 3 --full-traceback` | 0 | `PASS accuracy; v0=0.419280 ms, v1=0.385430 ms, speedup=1.088x` |

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA256 | After SHA256 |
|---|---:|---|---|---|---|
| 1 | `python3 -m py_compile` | 0 | - | - | `4faadac6...` |
| 2 | `auto_bench.py ... --warmup 1 --repeat 3 --full-traceback` | 0 | - | `4faadac6...` | `4faadac6...` |

No semantic repair was required.

## Reason Code

`candidate-ready`: the candidate conforms to the immutable design (`tl.dot`
multi-token layout); `tl.dot` `(16,64)@(64,128)` is proven to compile and run on
this runtime (not a capability-miss); correctness and the local compile-smoke
gate pass.
