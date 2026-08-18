# Coder Result 004

## Metadata

```json
{
  "schema_version": 1,
  "round": "004",
  "result": "candidate-ready",
  "result_reason": "candidate conforms to the immutable host-only decision; the fused _sparse_pooler_max_kernel body, BLOCK_V=1024, num_warps=1, grid (num_seq, cdiv(vocab_size, BLOCK_V)), on-device seq_offset prefix scan, and the library MLM head pipeline are byte-identical to the accepted reference; two host-side changes are implemented: (a) the existing kernel is wrapped with fast_libentry via the class-body globals() trick (proven on this runtime and retained by the harness AST loader), and (b) a per-instance _out_cache attribute (NOT registered via register_buffer/register_parameter) caches the [num_seq, vocab_size] fp32 output buffer keyed on (num_seq, vocab_size, dtype, device) and reuses it on steady-state cache hits; load_state_dict compatibility maintained; correctness smoke 4/4 within tolerance; harness end-to-end smoke passes",
  "source_canonical_path": "triton_sparse_pooler_001.py",
  "source_canonical_sha256": "182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd",
  "decision_path": "rounds/decision_004.md",
  "decision_sha256": "dc33e45ee2c95319608bc08f9ed8a5a3e3ae0882305f52eb07f0a449ea33f111",
  "selected_profile": "triton_mlu",
  "runtime_fingerprint": {
    "triton_version": "3.2.0",
    "backend_target": "BangDriver (mlu)",
    "backend_version": "torch_mlu 1.32.0+torch2.11.0; MLU driver 6.5.49",
    "device_arch": "MLU590-H8 (capability 5.0)"
  },
  "candidate_path": "triton_sparse_pooler_004.py",
  "candidate_sha256": "81cdea2b958c288e1382aef0b30cfc6dffb544c55a0e44825fab51b53cac7842"
}
```

## Decision validation

- Command: `python3 /home/lipenghui/.claude/skills/kernel-opt-loop/scripts/validate_decision.py /projs/framework/lipenghui/projects/kernelswift/sparse_pooler/rounds/decision_004.md --expected-profile triton_mlu`
- Exit code: 0
- Output: `{"valid": true, ...}` (full normalized contract returned)
- Language (`triton`), backend (`mlu`), and target profile (`triton_mlu`) match the manifest's Identity and Match rules. `change_scope=host`, `change_family=host-allocation-reuse`. No environment-blocked condition.

## Runtime fingerprint verification

- `triton_version=3.2.0` (matches `project.md#runtime-fingerprint`)
- `torch_mlu_version=1.32.0+torch2.11.0` (matches)
- `driver_version=6.5.49` (matches)
- `device_name=MLU590-H8` (matches)
- `device_capability=(5, 0)` (matches cap 5.0)

No fingerprint mismatch. The runtime is the one the decision was authored against.

## Primitive and hint conformance

Round 004 is a host-only change. The fused `_sparse_pooler_max_kernel` body is byte-identical to the accepted Round 001 reference (verified by SHA-256 of the kernel body extracted from both files: `f3ebee376d9e7732b622c41acd5ff175932943166068c3b70adf22e9ae1c4bb6` for both). No device-side primitive is added, removed, or modified. All primitives proven in Rounds 001-003 (`tl.load` with `other=-inf`, `tl.where`, `tl.log`, `tl.maximum`, `tl.full`, `tl.arange`, `tl.program_id`, `tl.zeros`, `tl.store`) are reused unchanged.

| Primitive / hint | Decision status | Profile status | Probe outcome | Conformance |
|---|---|---|---|---|
| `fast_libentry` import (`from triton.runtime import fast_libentry`) | Required (first-choice import form per decision) | Runtime and Launcher Conventions: both `from triton.runtime import fast_libentry` and `from triton.runtime.fast_libentry import fast_libentry` are observed forms; the former is the first choice | Local probe: `python3 -c "from triton.runtime import fast_libentry"` succeeds on this runtime (returns `<function fast_libentry at 0x...>`). The first-choice import form is used; no fallback to the alternate form or to the default launcher was needed | Supported; first-choice import form compiles on this runtime |
| `fast_libentry()(_kernel)` class-body `globals()` trick | Required (the harness AST loader strips module-level non-literal assignments) | Runtime and Launcher Conventions: "The observed compatible pattern initializes `fast_libentry()(_kernel)` from a retained class body when the loader requires it." Evidence: `flexattention/triton_flexattention_003.py` | Local probe via the actual harness loader (`auto_bench.load_ks_module` with a `pathlib.Path`): after load, `_sparse_pooler_max_fast` is present in the module `__dict__` (the class body executed at import time and injected the wrapped kernel into module globals). Proven on this runtime | Supported; the class-body `globals()` trick survives the harness AST loader exactly as the flexattention v3 evidence records |
| `_out_cache` per-instance attribute | Required (Host Plan: per-instance output buffer cache) | Not a primitive; a Host Plan buffer-lifecycle pattern | Local probe: `self._out_cache` initialized to `None` in `__init__`; first forward allocates and stores; second forward with matching cache key reuses the same buffer (same `data_ptr`); forward with a different cache key (different `num_seq`) reallocates; returning to the original key reallocates again (the cache was replaced). Verified via the harness loader | Conforms; cache key, invalidation, and reuse semantics match the Host Plan |
| `_out_cache` NOT registered via `register_buffer`/`register_parameter` | Required (load_state_dict compatibility) | Not a primitive; a Host Plan ownership rule | Local probe: `model_new.state_dict()` keys match `model.state_dict()` keys exactly; `_out_cache` is not in the state_dict; `model_new.load_state_dict(model.state_dict())` succeeds. No `self.register_buffer` or `self.register_parameter` call present in the code (only comment text mentioning the prohibited APIs) | Conforms; load_state_dict compatibility maintained |
| `num_warps=1` | Required (Constrained, normative; passed through to the wrapped kernel) | Constrained: `num_warps=1` proven; `num_warps=2` failed | Used `num_warps=1` exactly as the decision and profile require. The `fast_libentry` wrapper changes the launcher path, not the kernel's `num_warps`; the `num_warps=1` argument is passed through to the wrapped kernel exactly as in the accepted reference | Proven value, no fallback needed; `num_warps=2` not used |
| `BLOCK_V=1024` | Required (normative target hint; unchanged from accepted reference) | Not a primitive; a `tl.constexpr` tiling parameter | Unchanged from accepted reference; correctness smoke confirms | Conforms |
| Grid `(num_seq, cdiv(vocab_size, BLOCK_V))` = `(4, 30)` | Required (normative; unchanged from accepted reference) | Not a primitive; a grid expression | Unchanged from accepted reference; correctness smoke confirms | Conforms |
| On-device `seq_offset` prefix scan | Required (normative; unchanged from accepted reference) | Not a primitive; a kernel control-flow pattern | Unchanged from accepted reference (kernel body byte-identical) | Conforms |
| Library MLM head `self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))` | Required (normative; unchanged from accepted reference) | Not a primitive; a PyTorch library op pipeline | Unchanged from accepted reference; the four library ops (dense matmul, GELU, LayerNorm, decoder matmul) remain | Conforms; `kernel_count_per_call` remains 5 by construction |
| `torch.mlu.device()` context | NOT introduced (the accepted reference has none) | Target-specific Pitfall: "Removing `torch.mlu.device()` is valid only when the caller already owns device selection" | No `torch.mlu.device()` context introduced or removed; the caller-selected device and current stream are preserved | Conforms; the pitfall is not applicable because the accepted reference has no device context to remove |
| Return `[out[i] for i in range(num_seq)]` | Required (normative; unchanged from accepted reference) | Not a primitive; a Python return expression | Unchanged from accepted reference; the returned list of slices shares storage with the cached buffer, matching the flexattention v3 pattern | Conforms |

No Unsupported or unprovable Unknown primitive is required. No silent substitution of a normative construct occurred. The host-side changes (launcher wrapper + output cache) are exactly the two interventions the decision specifies; the kernel body and the library MLM head pipeline are unchanged.

## Implementation summary

The candidate is a copy of `triton_sparse_pooler_001.py` with exactly two host-side changes:

1. **`fast_libentry` launcher wrapper.** A new import `from triton.runtime import fast_libentry` is added (first-choice form per the decision; probed to succeed on this runtime). Inside the `ModelNew` class body, the class-body `globals()` trick populates `_sparse_pooler_max_fast = fast_libentry()(_sparse_pooler_max_kernel)` into module globals. This pattern is required because the harness AST loader (`auto_bench._filter_module_ast`) strips module-level non-literal assignments; a module-level `_sparse_pooler_max_fast = fast_libentry()(_sparse_pooler_max_kernel)` would be dropped and cause a `NameError` at runtime. The loader retains `ClassDef` nodes, so the class body executes at import time and injects the wrapped kernel into module globals. This is the exact pattern proven in `flexattention/triton_flexattention_003.py`. In `ModelNew.forward`, the wrapped kernel is retrieved via `globals()["_sparse_pooler_max_fast"]` and launched with the same grid, arguments, and `num_warps=1` as the accepted reference. The kernel body, `BLOCK_V=1024`, grid `(num_seq, cdiv(vocab_size, BLOCK_V))`, and on-device `seq_offset` prefix scan are byte-identical to the accepted reference.

2. **Per-instance `_out_cache` output buffer.** `ModelNew.__init__` initializes `self._out_cache: torch.Tensor | None = None` as a plain Python attribute (NOT registered via `register_buffer` or `register_parameter`, so it does not appear in `state_dict()` and does not affect `load_state_dict`). On every forward, `ModelNew.forward` checks `self._out_cache`: if it is `None` or its `(num_seq, vocab_size, dtype, device)` does not match the current forward, a fresh `[num_seq, vocab_size]` fp32 tensor is allocated with `torch.empty` and stored in `self._out_cache`; otherwise the existing buffer is reused. The cached or freshly allocated buffer is passed to the fused kernel as `out_ptr`. The returned list is `[out[i] for i in range(num_seq)]` as before; the returned slices share storage with the cached buffer (matching the flexattention v3 pattern, which returns `out.reshape(T, H * D)` from a cached buffer). The cache key includes `num_seq`, `vocab_size`, `dtype`, and `device` — all mandatory compatibility components per the Host Plan. The cache is invalidated (replaced) whenever any cache-key component changes between forwards; the buffer is never shared across `ModelNew` instances; `load_state_dict` does not touch `_out_cache` because the cache stores only the output tensor, not model parameters.

The library MLM head pipeline (`self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))`), the `pooling == "sum"` Python fallback, `get_inputs`, `get_init_inputs`, and the `if __name__ == "__main__"` block are unchanged from the accepted reference.

## Verification

### ast.parse

- Command: `python3 -c "import ast; ast.parse(open('triton_sparse_pooler_004.py').read())"`
- Exit code: 0
- Top-level nodes: 8 imports (7 `Import` + 1 `ImportFrom` for `fast_libentry`), `FunctionDef _sparse_pooler_max_kernel`, `ClassDef ModelNew`, `FunctionDef get_inputs`, `FunctionDef get_init_inputs`, `If` (the `__main__` guard). No module-level non-literal assignment is present (the `fast_libentry()(_sparse_pooler_max_kernel)` call lives inside the class body, which the loader retains).

### Harness loader (`auto_bench.load_ks_module` with `pathlib.Path`)

- Command: `python3 -c "from auto_bench import load_ks_module; mod = load_ks_module(Path('triton_sparse_pooler_004.py')); ..."`
- Exit code: 0
- `hasattr(mod, 'ModelNew')` = True
- `hasattr(mod, 'get_inputs')` = True
- `hasattr(mod, 'get_init_inputs')` = True
- `'_sparse_pooler_max_fast' in mod.__dict__` = True (the class-body `globals()` trick successfully injected the wrapped kernel into module globals; the harness AST loader retained the `ClassDef` and the class body executed at import time)

### Correctness smoke (vs `base.py`)

- Procedure: instantiate `Model` from `base.py` and `ModelNew` from the candidate via the harness loader; move both to `mlu:0`; run `model_new.load_state_dict(model.state_dict())`; run both forwards on `get_inputs()` moved to `mlu:0`; compare 4 outputs with `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`.
- `load_state_dict` succeeded.
- `state_dict` keys match exactly; all shapes match; `_out_cache` is not in the state_dict.
- 4/4 outputs `allclose=True`, `max_abs_diff=1.788139e-07` for each output.
- Result: `CORRECTNESS: PASS`

### Cache reuse semantics

- Before first forward: `self._out_cache = None`.
- After first forward: `_out_cache` shape `[4, 30522]`, dtype `fp32`, device `mlu:0`, `data_ptr = P1`.
- After second forward (same cache key): `_out_cache.data_ptr == P1` (same buffer reused).
- `out1[0].data_ptr() == cache1[0].data_ptr()` = True (returned slices share storage with the cached buffer).
- After forward with different cache key (`num_seq=3`): `_out_cache` shape `[3, 30522]` (reallocation happened).
- After returning to the original key (`num_seq=4`): `_out_cache.data_ptr != P1` (the cache was replaced when the key changed, so a fresh buffer was allocated).
- Result: `CACHE_BEHAVIOR: PASS`

### Harness end-to-end smoke (`auto_bench.py --warmup 5 --repeat 5`)

- Command: `python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_004.py --warmup 5 --repeat 5`
- Exit code: 0
- Output: `PASS accuracy; v0=0.899668 ms, v1=0.569278 ms, speedup=1.580x`
- Result: accuracy PASS; smoke-only timing (Verifier owns the authoritative 50/100 measurement).

### Kernel body identity

- The fused `_sparse_pooler_max_kernel` body (lines from `@triton.jit` to `class ModelNew`) was extracted from both `triton_sparse_pooler_001.py` and `triton_sparse_pooler_004.py` and SHA-256 hashed.
- `ref kernel body sha256 = f3ebee376d9e7732b622c41acd5ff175932943166068c3b70adf22e9ae1c4bb6`
- `cand kernel body sha256 = f3ebee376d9e7732b622c41acd5ff175932943166068c3b70adf22e9ae1c4bb6`
- The kernel body is byte-identical. The device-side change boundary is empty; only the two host-side interventions were applied.

## Attempt ledger

| Attempt | Command | Exit code | Defect | Before SHA | After SHA | Outcome |
|---|---|---:|---|---|---|---|
| 1 | `python3 -c "import ast; ast.parse(open('triton_sparse_pooler_004.py').read())"` | 0 | none | n/a | `81cdea2b...` | ast.parse ok |
| 2 | `python3 -c "from auto_bench import load_ks_module; load_ks_module(Path('triton_sparse_pooler_004.py'))"` | 0 | none | `81cdea2b...` | `81cdea2b...` | harness loader ok; `_sparse_pooler_max_fast` in module globals |
| 3 | correctness smoke (instantiate `ModelNew`, `load_state_dict` from `base.py Model`, forward, compare 4 outputs) | 0 | none | `81cdea2b...` | `81cdea2b...` | 4/4 allclose, max_abs_diff 1.8e-7 |
| 4 | cache reuse semantics probe | 0 | none | `81cdea2b...` | `81cdea2b...` | first alloc, second reuse, diff-key realloc, return-to-original-key realloc |
| 5 | `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_004.py --warmup 5 --repeat 5` | 0 | none | `81cdea2b...` | `81cdea2b...` | PASS accuracy; speedup 1.580x (smoke only) |

Repairs used: 0 of 2. No non-semantic syntax, import, or loader defect was encountered. No semantic change was required.

## Ownership

- Coder owns: `triton_sparse_pooler_004.py`, `rounds/coder_result_004.md`, `state/coder_state.md`.
- Coder must not edit (and did not edit): `decision_004.md`, `team-state.md`, `project.md`, `base.py`, `auto_bench.py`, the triton_mlu target profile, or any other role's state file.
- All throwaway probe and smoke scripts were run inline via `python3 -c "..."` and were not written to the project root. No `__pycache__` was left in the project root.
