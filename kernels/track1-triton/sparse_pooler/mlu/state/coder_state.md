# Coder State

Concise implementation facts and resume context for the current candidate.
No canonical-state claims.

## Current round

- Round: 004
- Phase: coding (complete from Coder's side; awaiting Orchestrator handoff to Verifier)
- Decision: `rounds/decision_004.md` (SHA-256 `dc33e45ee2c95319608bc08f9ed8a5a3e3ae0882305f52eb07f0a449ea33f111`, `proceed`, change_scope=host, change_family=host-allocation-reuse)
- Source canonical: `triton_sparse_pooler_001.py` (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)
- Candidate: `triton_sparse_pooler_004.py` (SHA-256 `81cdea2b958c288e1382aef0b30cfc6dffb544c55a0e44825fab51b53cac7842`)
- Result file: `rounds/coder_result_004.md`
- Classification: `candidate-ready`

## Candidate facts

- Host-only change. The fused `_sparse_pooler_max_kernel` body, `BLOCK_V=1024`, `num_warps=1`, grid `(num_seq, cdiv(vocab_size, BLOCK_V))`, on-device `seq_offset` prefix scan, and the library MLM head pipeline (`self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))`) are byte-identical to the accepted Round 001 reference. Verified by SHA-256 of the extracted kernel body: `f3ebee376d9e7732b622c41acd5ff175932943166068c3b70adf22e9ae1c4bb6` for both.
- Two host-side changes:
  1. `from triton.runtime import fast_libentry` added (first-choice import form; probed to succeed on this runtime). The class-body `globals()` trick populates `globals()["_sparse_pooler_max_fast"] = fast_libentry()(_sparse_pooler_max_kernel)` inside the `ModelNew` class body. This survives the harness AST loader (`_filter_module_ast` retains `ClassDef` nodes; the class body executes at import time and injects the wrapped kernel into module globals). Proven in `flexattention/triton_flexattention_003.py`. In `forward`, the wrapped kernel is retrieved via `globals()["_sparse_pooler_max_fast"]` and launched with the same grid, args, and `num_warps=1`.
  2. `ModelNew.__init__` initializes `self._out_cache: torch.Tensor | None = None` (plain Python attribute, NOT `register_buffer`/`register_parameter`, so it is not in `state_dict()` and does not affect `load_state_dict`). On every forward, `_out_cache` is checked against `(num_seq, vocab_size, dtype, device)`: if `None` or any component mismatches, a fresh `[num_seq, vocab_size]` fp32 tensor is allocated with `torch.empty` and stored; otherwise the existing buffer is reused. The cached/fresh buffer is passed to the fused kernel; the returned list is `[out[i] for i in range(num_seq)]`.
- Cache key: `(num_seq, vocab_size, dtype, device)`. All four components must match for reuse; any mismatch triggers reallocation and replaces the cache.
- `kernel_count_per_call` remains 5 by construction: no kernel added, removed, or modified; the library MLM head (dense matmul, GELU, LayerNorm, decoder matmul) and the fused reduction kernel are unchanged. The `fast_libentry` wrapper changes the launcher path, not the kernel count.
- `load_state_dict` compatibility maintained: `state_dict` keys and shapes match `base.py Model` exactly; `_out_cache` is not in the state_dict.
- No `torch.mlu.device()` context introduced or removed; caller-selected device and current stream preserved.
- `num_warps=1` passed through to the wrapped kernel unchanged. `num_warps=2` not used.
- `pooling == "sum"` fallback, `get_inputs`, `get_init_inputs`, and `__main__` block unchanged from the accepted reference.

## Probe outcomes (resume reference)

- `fast_libentry` import probe: `python3 -c "from triton.runtime import fast_libentry"` succeeds on this runtime (first-choice form). No fallback to `from triton.runtime.fast_libentry import fast_libentry` or to the default launcher was needed.
- Harness loader probe (`load_ks_module` with `pathlib.Path`): after load, `_sparse_pooler_max_fast` is in `mod.__dict__` (class-body `globals()` trick succeeded). `ModelNew`, `get_inputs`, `get_init_inputs` all present.
- Correctness smoke vs `base.py`: `load_state_dict` ok; `state_dict` keys/shapes match; `_out_cache` not in state_dict; 4/4 outputs `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`, `max_abs_diff=1.788139e-07`.
- Cache reuse probe: first forward allocates (`_out_cache=None` -> `[4,30522]` fp32 mlu:0); second forward reuses same buffer (same `data_ptr`); forward with `num_seq=3` reallocates (`[3,30522]`); return to `num_seq=4` reallocates again (cache was replaced). Returned slices share storage with the cached buffer.
- Harness end-to-end smoke (warmup=5, repeat=5): `PASS accuracy; v0=0.899668 ms, v1=0.569278 ms, speedup=1.580x` (smoke only; Verifier owns the 50/100 measurement).
- Kernel body identity: extracted kernel body SHA-256 matches between `triton_sparse_pooler_001.py` and `triton_sparse_pooler_004.py` (`f3ebee37...`).
- All throwaway probe and smoke scripts were run inline via `python3 -c "..."`; no files written to the project root; no `__pycache__` left.

## Repair budget

- Repairs used this round: 0 of 2. No non-semantic syntax, import, or loader defect was encountered. No semantic change was required.

## Ownership

- Coder owns: `triton_sparse_pooler_004.py`, `rounds/coder_result_004.md`, `state/coder_state.md`.
- Coder must not edit: decision, team-state, project.md, base.py, harness, target profile, Verifier-owned files.
