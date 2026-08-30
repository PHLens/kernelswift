# Coder Result 002

## Metadata

- round: 002
- result: `candidate-ready`
- language: triton
- backend: ascend
- target_profile: triton_ascend
- source_canonical_path: `ascend/triton_sparse_pooler_001.py`
- source_canonical_sha256: `dc2a8b6582cf9d6fef3e044081426762b88833e056ce8d7f04086e0d92f429e0`
- decision_path: `rounds/decision_002.md`
- decision_sha256: `8beb3c34e9eb88aac1722ee9a99117a7c05453e2f08ed5487ccd49a5004b003f`
- candidate_path: `ascend/triton_sparse_pooler_002.py`
- candidate_sha256: `a7338d89a1f5a30843e84d3f533ac151245d6547453ddc5a2dcff66f93cb7957`
- runtime_fingerprint_ref: `project.md#runtime-fingerprint`

## Result

`candidate-ready` — the candidate conforms to the immutable design in decision_002.

## Conformance Notes

- Host-only change: a per-instance output buffer cache was added to `ModelNew`.
- `state_owner` = `ModelNew` instance; cache fields `self._out_cache` (tensor or
  None) and `self._out_cache_key` (tuple or None) are plain Python attributes —
  NOT `register_buffer`/`register_parameter`, so `load_state_dict` is untouched.
- `cache_key` = `(num_seq, vocab_size, dtype, device)`; lazily allocated on first
  forward, reused on exact key match, replaced on mismatch.
- The fused `_sparse_pooler_max_kernel` body, `BLOCK_V=1024`, `num_warps=1`,
  grid `(num_seq, cdiv(vocab_size, BLOCK_V))`, and on-device prefix scan are
  byte-for-byte unchanged from round 001 (no device-side change).
- MLM head (dense → GELU → LayerNorm → decoder) library ops unchanged.
- `pooling == "sum"` fallback unchanged.
- Output is still a Python list of `num_seq` tensors; the kernel writes into the
  cached buffer in place, then `return [out[i] for i in range(num_seq)]`.
- No `tl.dot`, no `fast_libentry`, no tile-tuning.

## Gate Evidence

- `py_compile` (real interpreter `/usr/local/python3.11.15/bin/python3`): PASS.
- Real harness loader + accuracy + warm-up/compile smoke:
  `auto_bench.py --v0_file .../base.py --v1_file .../triton_sparse_pooler_002.py --warmup 1 --repeat 3`
  -> `PASS accuracy; v0=0.925630 ms, v1=0.644010 ms, speedup=1.437x` (smoke timing,
  not authoritative).

## Attempt Ledger

| Command | Exit | Defect | Before SHA | After SHA |
|---|---|---|---|---|
| `python3 -m py_compile .../triton_sparse_pooler_002.py` | 0 | none | - | `a7338d89...` |
| `python3 auto_bench.py --v0_file .../base.py --v1_file .../triton_sparse_pooler_002.py --warmup 1 --repeat 3` | 0 | none | `a7338d89...` | `a7338d89...` |
