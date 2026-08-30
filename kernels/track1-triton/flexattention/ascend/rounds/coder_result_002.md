# Coder Result 002

## Identity

- Round: `002`
- Result: `candidate-ready`
- Decision: `rounds/decision_002.md`
- Decision SHA256: `1be71c8d099e870321bbcdde02fc6bc078d929fc7ca0b1dc7bce89cb19ee2f06`
- Source canonical: `triton_flexattention_001.py`
- Source canonical SHA256: `53e87eff27457f6268040c64979f99dcf30a809effc562caec3db951b141d4a3`
- Selected target profile: `triton_ascend`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1+cpu, torch_npu 2.7.1.post4, triton 3.2.0, Ascend910B4)
- Candidate path: `triton_flexattention_002.py`
- Candidate SHA256: `b0fe058c5b5336978e89933e8f0fed0d5a0449aede33c6b1b7a97c9c319c100f`

## Conformance Notes (candidate-ready)

The candidate realizes the immutable decision exactly. It is a host-only change:
the Triton kernel, grid, `num_warps`, and numerical semantics are byte-for-byte
unchanged from `triton_flexattention_001.py` (verified via `diff` over the
`@triton.jit` kernel body through `class ModelNew`).

Host Plan conformance:

- `state_owner`: `ModelNew` instance — the cache lives on
  `self._out_cache` / `self._out_cache_key`.
- `lifetime`: allocated lazily on first forward, retained until model
  destruction (no explicit free).
- `cache_key`: `(num_tokens, num_heads, head_size, dtype, device)` =
  `(T, H, D, query.dtype, query.device)`.
- `allocation_reuse` / `invalidation`: reuse when the full key matches;
  otherwise allocate a fresh buffer and replace both buffer and key.
- `concurrency`: per-instance, single-stream; no global or class-level cache.
- `device_stream_behavior`: the cached buffer is allocated on `query.device`
  (caller-selected device); the kernel still launches on the caller's device and
  current stream. Only the output allocation is cached.
- `unchanged_behavior`: returned shape `[83,512]`, fp16 dtype, and numerical
  semantics unchanged (identical kernel, identical inputs).

No Triton primitive changes; no new Unsupported/Unknown primitive introduced.

## Local Gate

| Gate | Command | Exit | Result |
|---|---|---|---|
| `ast.parse` | (harness `ast.parse` during load) | 0 | pass |
| `py_compile` | `python3 -m py_compile triton_flexattention_002.py` | 0 | pass |
| Harness loader | real `auto_bench.py` AST loader | 0 | pass |
| Correctness smoke | `auto_bench.py --v0_file .../base.py --v1_file .../triton_flexattention_002.py --warmup 1 --repeat 3 --full-traceback` | 0 | `PASS accuracy; v0=0.417260 ms, v1=0.354930 ms, speedup=1.176x` |
| Kernel-identity diff | `diff` kernel body vs `triton_flexattention_001.py` | 0 | identical |

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA256 | After SHA256 |
|---|---:|---|---|---|---|
| 1 | `python3 -m py_compile` | 0 | - | - | `b0fe058c...` |
| 2 | `auto_bench.py ... --warmup 1 --repeat 3 --full-traceback` | 0 | - | `b0fe058c...` | `b0fe058c...` |

No semantic repair was required.

## Reason Code

`candidate-ready`: the candidate conforms to the immutable design (host-only
output buffer cache with full cache-key and lifecycle conformance); correctness
and the local compile-smoke gate pass.
