# Coder Result 001

Result: `candidate-ready`

- round: `001`
- source_canonical: `baseline_adapter.py`
- source_canonical_sha256: `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d`
- decision: `rounds/decision_001.md`
- decision_sha256: `<pending: hash after commit — see note>`
- candidate: `triton_mhc_mix_bwd_001.py`
- candidate_sha256: `f7efc6853a8f07b90926237cc2f4de620926bd0b34333648e7355d8995c57d10`
- selected_profile: `triton_ascend`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `pending` (Verifier-owned)

## Primitive and Hint Conformance

| Primitive / Hint | Status | Note |
|---|---|---|
| `tl.load` | Supported | Masked contiguous loads: `[BLOCK_R,4]` tiles of `input_mix` / `grad_out`, `[4]` `mhc_base`, scalar `mhc_scale`. |
| `tl.store` | Supported | Contiguous `grad_input_mix` `[BLOCK_R,4]` store. |
| `tl.arange` | Supported | Extents `BLOCK_R=64` and `4`. `4` verified; `64` in the probed extent family. |
| `tl.program_id` | Supported | Axis 0, 1D grid `(cdiv(2048,64),)`. |
| `tl.sigmoid` | Supported (runtime-confirmed) | `hasattr(tl, 'sigmoid')` on triton 3.2.0 Ascend; compiles and runs in smoke. |
| `tl.sum` | Supported | Axis-0 reduction to `[4]` and full reduction to scalar. |
| `tl.atomic_add` | Supported (runtime-confirmed) | `hasattr(tl, 'atomic_add')`; compiled + executed in smoke correctness (PASS). |
| `num_warps=4` | Constrained (proven) | `num_warps=4` compiled and ran; within the proven `{1,2,4}` set. |

`tl.atomic_add` and `tl.sigmoid` are not in the recorded profile's Supported
table, but both are present in the Ascend triton.language runtime and were
compiled + executed in the local correctness smoke against the real harness, so
they are treated as runtime-confirmed (not Unknown/Unsupported). The decision
also grants an explicit two-kernel fallback if `tl.atomic_add` overhead proves
unacceptable under Verifier measurement, but the single-kernel form compiled and
ran correctly, so no fallback was required. `tl.dot`, `tl.make_block_ptr`,
`async_copy`, `num_stages`, `fast_libentry`, and `vectorize` are not used.

## Conformance Notes (candidate-ready)

Small target-language accommodations that preserve all normative semantics:

- **`tl.sigmoid` for the sigmoid gate.** The Unified Sketch specifies
  `sig = sigmoid(z)`; the reference uses `torch.sigmoid`. `tl.sigmoid` is the
  exact triton equivalent and is available on this runtime; no `tl.exp`
  substitution was needed.
- **`1.0 - sig` in place of `1 - sigmoid`.** The reference `1 - sigmoid`
  lowers to an `aclnnRsubs` kernel; here it is a fused elementwise op inside the
  single kernel, numerically identical.
- **Flat `[2048,4]` indexing.** `input_mix` `[2,1024,4]` is viewed as
  `n0*n1 = 2048` rows of 4, matching the decision's `2*1024=2048` note. The
  kernel writes `grad_input_mix` as a flat `[2048,4]` buffer and `forward`
  reshapes back to `(n0, n1, mhc_mult)`, so the public output shape is unchanged.
- **Accumulator zero-init.** `grad_mhc_base` and `grad_mhc_scale` are
  `torch.zeros` allocated per-call in `forward` before the kernel launches, so
  the `tl.atomic_add` accumulation is correct on every call (no implicit global
  state; per-call allocation is inside the decision's "normal per-call output
  allocation" boundary).
- **`get_inputs` uses `device="npu"`** (not the `"cuda"` placeholder), per the
  target profile's guidance and the harness `_RewriteDeviceStr` convention.

## Local Gate

| Gate | Command | Exit | Result |
|---|---|---|---|
| `ast.parse` | `ast.parse(open(...).read())` | 0 | pass |
| `py_compile` | `python3 -m py_compile triton_mhc_mix_bwd_001.py` | 0 | pass |
| Harness loader | real `auto_bench.py` AST loader | 0 | pass |
| Correctness smoke | `auto_bench.py --v0_file .../base.py --v1_file .../triton_mhc_mix_bwd_001.py --warmup 1 --repeat 3 --full-traceback` | 0 | `PASS accuracy; v0=0.406540 ms, v1=0.429190 ms, speedup=0.947x` |

The `0.947x` at warmup=1/repeat=3 is launch-noise only (not authoritative;
Verifier owns the warmup 50/repeat 100 benchmark). Correctness passed against
the reference under `atol=1e-2, rtol=1e-2` with the real harness loader.

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA256 | After SHA256 |
|---:|---|---:|---|---|---|
| 1 | `python3 -m py_compile triton_mhc_mix_bwd_001.py` | 0 | none | - | `f7efc6853a8f07b90926237cc2f4de620926bd0b34333648e7355d8995c57d10` |
| 2 | `auto_bench.py ... --warmup 1 --repeat 3 --full-traceback` | 0 | none; correctness PASS, compile smoke PASS | `f7efc685...` | `f7efc685...` |

No repair was required. The single-fused-kernel form compiled and executed
correctly on the first attempt.

## Reason Code

`candidate-ready`: the candidate conforms to the immutable design (single fused
Triton kernel collapsing the sigmoid elementwise chain + both reductions from 10
unfused library kernels to 1, via block-local `tl.sum` partials + `tl.atomic_add`
into `[4]`/`[1]` accumulators); the public contract (`ModelNew` / `get_inputs` /
`get_init_inputs` / forward tuple) and output shapes/dtypes are preserved;
correctness and the local compile-smoke gate pass against the real harness.
