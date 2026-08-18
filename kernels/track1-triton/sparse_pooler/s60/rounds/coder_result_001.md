# Coder Result 001

## Metadata

```json
{"round":"001","result":"candidate-ready","source_canonical":"mlu/triton_sparse_pooler_001.py","decision":"rounds/decision_001.md","selected_profile":"triton_gcu","language":"triton","backend":"gcu","candidate":"triton_sparse_pooler_001.py"}
```

## Candidate

- path: `kernels/track1-triton/sparse_pooler/s60/triton_sparse_pooler_001.py`
- sha256: `60aadce88e02776b71960092bf9df59c0adcdc8ace75319e845f7c9122a3f80e`

## Source canonical

- path: `kernels/track1-triton/sparse_pooler/mlu/triton_sparse_pooler_001.py`
- The MLU v1 accepted canonical (the same fusion, validated on MLU at +33.39%) was used as the structural template. GCU adaptions were applied on top (see below).

## How decision_001 is realized

The candidate fuses `relu + log1p + per-sequence max pooling` into a single
`@triton.jit` kernel `_sparse_pooler_max_kernel`, launched once per forward over
a 2D grid `(num_seq, num_vocab_tiles)` with `num_warps=1`, matching the Unified
Sketch exactly:

1. **Fused kernel grid**: `pid_s = tl.program_id(0)` over sequences and
   `pid_v = tl.program_id(1)` over vocab tiles. `num_vocab_tiles =
   (vocab_size + BLOCK_V - 1) // BLOCK_V` with `BLOCK_V = 256`.
2. **Device-side prefix scan**: `seq_len = tl.load(seq_lens_ptr + pid_s)` and
   `seq_offset = sum(seq_lens[0:pid_s])` computed on-device via a bounded
   `for i in range(pid_s)` loop (at most 3 extra `tl.load` calls for `num_seq=4`).
   This eliminates the `seq_lens.tolist()` D2H synchronization and the Python
   host for-loop.
3. **Per-segment max reduction**: accumulator `acc` starts at `-inf`; for each
   `row in range(seq_len)` the vocab tile is loaded (masked), `relu` via
   `tl.where(x > 0.0, x, 0.0)`, `log1p` via `tl.log(1.0 + x)` (stable since relu
   output >= 0), then `acc = tl.where(acc < x, x, acc)`.
4. **MLM head unchanged**: `dense` (Linear 768→768), `act` (nn.GELU),
   `layer_norm` (LayerNorm eps=1e-12), `decoder` (Linear 768→30522, bias=True)
   remain PyTorch library ops. Submodule names/params identical to
   `baseline_adapter.py` ModelNew: `dense.weight/bias`, `layer_norm.weight/bias`,
   `decoder.weight/bias`. Only `forward` dispatch changed.
5. **Output is a Python list**: kernel writes a contiguous `[num_seq, vocab_size]`
   fp32 buffer, then `forward` returns `[out[i] for i in range(num_seq)]` — 4
   independent `[30522]` fp32 tensors, not a stacked `[4, 30522]`.
6. **get_inputs** returns `[hidden_states [83,768] fp32, seq_lens [4] int32]`;
   **get_init_inputs** returns `[768, 30522, "max"]`. Device literal `'cuda'`
   (rewritten to `'gcu'` by the harness `_rewrite_device_for_backend`).

## GCU adaption points

| Item | MLU canonical | GCU candidate | Reason |
|---|---|---|---|
| imports | `import torch_mlu` + `torch_mlu.utils.gpu_migration` | `import torch_gcu` + `import triton_gcu` | GCU target; profile requires both imports before tensor allocation / launch |
| device | `'cuda'` (harness rewrites to mlu) | `'cuda'` (harness rewrites to gcu) | invariant: literal `'cuda'` remapped to `'gcu'` when target == gcu |
| max update | `acc = tl.maximum(acc, x)` | `acc = tl.where(acc < x, x, acc)` | `tl.where` is proven Supported on GCU; `tl.maximum` is unproven — this is the documented fallback in decision_001 (both preserve semantics) |
| BLOCK_V | 1024 | 256 | decision asks for a conservative value (256/512); extent-256 `tl.arange` is proven by `groupedtopk/s60` (BLOCK_E=256); 256 also bounds last-tile masking |
| `total_seq` arg | present but unused | removed | not in the Unified Sketch D-declarations; removal is non-semantic |

## Design decisions

- **BLOCK_V = 256** (not 512): `vocab_size = 30522` gives 120 vocab tiles and a
  last tile of 58 live lanes (`v_offs < vocab_size` mask). Extent-256 `tl.arange`
  is the largest arange extent already proven on this GCU runtime (groupedtopk
  `BLOCK_E = 256`); 512 would be a larger unproven extent. The larger tile count
  (120 vs 60) is acceptable given `num_warps=1` and the small per-program work.
- **`tl.where(acc < x, x, acc)` for max** instead of `tl.maximum`: `tl.where` and
  scalar/tensor comparisons are proven on GCU; `tl.maximum` is not listed in the
  Supported table. This is the exact fallback decision_001 specifies and preserves
  the Evaluation Contract observables and guardrails.
- **`for row in range(seq_len)`** (dynamic bound): carried over from the MLU
  accepted canonical. `seq_len` is a runtime int32 scalar tensor; `range()` over a
  scalar tensor is standard Triton semantics (dynamic loop), distinct from the
  compile-time `tl.static_range` proven in groupedtopk. This is the one construct
  not explicitly covered by the GCU probe — flagged below.
- **`seq_offset = tl.zeros([], dtype=tl.int32)`** then accumulated with
  `tl.load(...).to(tl.int32)`; all offset arithmetic stays int32 (never `tl.int64`),
  per the GCU constraint.

## Primitive and hint conformance

| Primitive / hint | Decision status | Profile status | Handling |
|---|---|---|---|
| `tl.load` (masked) | Required | Supported | used for seq_lens and vocab tiles, `other=-inf` |
| `tl.store` | Required | Supported | contiguous per-row store with `mask=v_mask` |
| `tl.arange(0, BLOCK_V)` | Required | extent 256 proven (groupedtopk) | BLOCK_V=256 chosen conservatively |
| `tl.program_id` axis 0 | Required | Supported (axis 0) | `pid_s = program_id(0)` |
| `tl.program_id` axis 1 | Required | not individually proven (only axis 0 recorded) | `pid_v = program_id(1)`; standard Triton 2D-grid usage, mandated by the sketch — conformance note, not a substitution |
| `tl.zeros([], int32)` | Required | scalar init; not individually probed | standard Triton 0-d zero |
| `tl.full((BLOCK_V,), -inf, fp32)` | Required | proven in groupedtopk (`tl.full((K,), v, tl.float32)`) | used for accumulator init |
| `tl.where` | Required | Supported | relu + max update |
| `tl.log` | Required | not listed (tl.exp proven) | `tl.log(1.0 + x)` standard libdevice mapping; no `tl.log1p` needed |
| `tl.maximum` | Optional | not listed | **not used**; `tl.where` fallback chosen |
| `tl.dot` | Not required | Unknown | not used (decoder matmul stays a library op) |
| `num_warps=1` | Required | Constrained (proven) | pinned |
| `fast_libentry` | Not required | Unknown (import failed) | not used; direct `kernel[(grid,)]` launch |
| dynamic `range(scalar_tensor)` | Required | not in probe (only `tl.static_range` proven) | conformance note — standard Triton semantics, carried from MLU accepted canonical |

## Deviations from decision

None. The implementation matches the Unified Sketch and all Host Plan / invariant
requirements. Conformance notes (not deviations):
- `program_id(axis=1)` and dynamic `range(seq_len)` are standard Triton constructs
  not explicitly covered by the recorded GCU primitive probe; both are mandated by
  the approved sketch and match the MLU accepted canonical.
- `tl.maximum` was intentionally replaced by the decision-documented `tl.where`
  fallback rather than probed locally, because the task scope limited Coder to
  `py_compile` (no runtime probe).

## Local gate

- `python3 -m py_compile triton_sparse_pooler_001.py` → exit 0 (`COMPILE_OK`).
- Runtime warm-up / compile smoke is left to Verifier (per task scope: no
  benchmark/measurement by Coder this round).

## Attempt ledger

| # | command | exit | defect | before hash | after hash |
|---|---|---|---|---|---|
| 0 | `python3 -m py_compile triton_sparse_pooler_001.py` | 0 | none | n/a | `60aadce88e02776b71960092bf9df59c0adcdc8ace75319e845f7c9122a3f80e` |
