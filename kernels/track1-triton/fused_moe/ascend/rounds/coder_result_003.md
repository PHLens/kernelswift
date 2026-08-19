# Coder Result 003

Result: `candidate-ready`

- round: `003`
- source_canonical: `triton_fused_moe_002.py`
- source_canonical_sha256: `1b5c8ecded2008991f0f7cc039f0e06fa072bf5b8a7c6d5630574f64a43f4074`
- decision: `rounds/decision_003.md`
- decision_sha256: `29d8c079c7ccc67c89bb363ff2f4b905346c47b8d122755448478119ea8a2737`
- candidate: `triton_fused_moe_003.py`
- candidate_sha256: `eb065f9a4371686b7ad028bb003501047b512265190b42438a559df05e85fb0d`
- selected_profile: `triton_ascend`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `pending` (Verifier-owned)

## Primitive and Hint Conformance

No new Triton primitive or launch hint is introduced. The kernel body, grid
`(T,)`, `num_warps=1`, routing, and FFN are byte-for-byte identical to the
canonical `triton_fused_moe_002.py` (only the `ModelNew.__init__` and
`forward` output allocation path changed). See `coder_result_002.md` for the
primitive conformance table, which remains fully applicable.

## Conformance Notes (candidate-ready)

The candidate realizes the immutable host-only decision exactly:

- **Per-instance output buffer cache.** `state_owner` = `ModelNew` instance
  (`self._out_cache` / `self._out_cache_key`); no global or class-level cache.
- **Cache key** `(num_tokens, hidden_size, dtype, device)` matches the Host Plan.
  On first forward the buffer is lazily allocated; on key match it is reused; on
  mismatch a fresh `torch.empty((num_tokens, hidden_size), dtype=dtype,
  device=device)` is allocated on the caller device and replaces the cache.
- **Device/stream preservation.** The buffer is allocated with
  `device=hidden_states.device` (the caller-selected device); no explicit stream
  or device context is created, preserving the current stream and caller device.
- **Concurrency.** The cache is per-instance, single-stream; no cross-instance or
  concurrent sharing is introduced.
- **Numerical semantics unchanged.** The kernel and inputs are byte-for-byte
  identical to round 002; the reused buffer is overwritten each forward and only
  the allocation path differs.
- **Out of scope (deferred, unchanged).** `w1/w2 .to(dtype)` casts remain inside
  `forward` (not moved to `__init__`), per the decision.

## Local Gate

| Gate | Command | Exit | Result |
|---|---|---|---|
| `ast.parse` | harness `_filter_module_ast` AST load | 0 | pass |
| `py_compile` | `python3 -m py_compile triton_fused_moe_003.py` | 0 | pass |
| Harness loader | real `auto_bench.py` AST loader | 0 | pass |
| Correctness smoke | `auto_bench.py --v0_file .../base.py --v1_file .../triton_fused_moe_003.py --warmup 1 --repeat 3 --full-traceback` | 0 | `PASS accuracy; v0=7.876730 ms, v1=0.404820 ms, speedup=19.457x` |

The real harness loader retains the kernel's `@triton.jit` `FunctionDef` and all
top-level imports; correctness passes against the reference under
`atol=1e-2, rtol=1e-2`.

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA256 | After SHA256 |
|---:|---|---:|---|---|---|
| 1 | `python3 -m py_compile triton_fused_moe_003.py` | 0 | none | - | `eb065f9a4371686b7ad028bb003501047b512265190b42438a559df05e85fb0d` |
| 2 | `auto_bench.py ... --warmup 1 --repeat 3 --full-traceback` | 0 | none; correctness PASS, compile smoke PASS | `eb065f9a...` | `eb065f9a...` |

No semantic repair was required. The host-only allocation-reuse change compiled
and passed correctness on the first smoke attempt.

## Reason Code

`candidate-ready`: the candidate conforms to the immutable host-only design
(per-instance output buffer reuse keyed on shape/dtype/device); the kernel is
byte-for-byte unchanged from round 002 and correctness and the local
compile-smoke gate pass against the real harness.
