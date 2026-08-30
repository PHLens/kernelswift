# Coder Result 001

## Metadata

- round: 001
- result: `candidate-ready`
- language: triton
- backend: ascend
- target_profile: triton_ascend
- source_canonical_path: `../baseline_adapter.py`
- source_canonical_sha256: `94d00f1a5d26f453fd5078fd9d50dfcddbb0c11d20a145d223544e59234add0f`
- decision_path: `rounds/decision_001.md`
- decision_sha256: `9b4f5333dd5145d8f8075047a89882f81f2c77fce888b3013e83c19e169256bf`
- candidate_path: `ascend/triton_sparse_pooler_001.py`
- candidate_sha256: `dc2a8b6582cf9d6fef3e044081426762b88833e056ce8d7f04086e0d92f429e0`
- runtime_fingerprint_ref: `project.md#runtime-fingerprint`

## Result

`candidate-ready` — the candidate conforms to the immutable design in decision_001.

## Conformance Notes

- The Triton kernel `_sparse_pooler_max_kernel` fuses relu + log1p + per-sequence
  max pooling, launched once per forward with `grid = (num_seq, cdiv(vocab_size, BLOCK_V))`,
  `BLOCK_V = 1024`, `num_warps = 1`, direct launch `kernel[(grid,)](...)`.
- MLM head (dense 768->768 -> GELU -> LayerNorm eps=1e-12 -> decoder 768->30522
  with bias) stays as PyTorch library ops, UNCHANGED from baseline_adapter.py.
  Only the post-MLM-head relu + log1p + per-seq max pool goes into the kernel.
- relu + log1p are FOLDED into the kernel (`tl.where(x > 0.0, x, 0.0)` then
  `tl.log(1.0 + x)`); no separate `torch.relu`/`torch.log1p` calls in the max path.
- Per-sequence offset via on-device prefix scan `seq_offset = sum(seq_lens[0:pid])`;
  at most 3 extra `tl.load`s for num_seq=4. No `seq_lens.tolist()` in the max path
  (`num_seq = seq_lens.shape[0]` only, no D2H sync).
- Output is a Python `list` of num_seq x `[vocab_size]` fp32 tensors
  (`return [out[i] for i in range(num_seq)]`), preserving the public contract.
- `tl.log` and `tl.maximum` both compiled and executed correctly on the Ascend
  runtime (accuracy passed); no `tl.where`-based fallback was needed.
- `pooling == "sum"` branch preserves the original reference fallback path
  (`torch.log1p(F.relu(logits))` + Python loop + `chunk.sum(dim=0)`).
- `import triton` + `import torch_npu`; no `import triton_ascend` (metadata-only).
  `get_inputs` uses `device="npu"`; `get_init_inputs` returns `[768, 30522, "max"]`.
- `fast_libentry` NOT used (Unknown on Ascend); direct Triton launch is the proven path.
- No module-level non-literal assignments: the `@triton.jit`-decorated kernel is a
  top-level `FunctionDef` and `ModelNew` is a `ClassDef`, both retained by the
  harness AST loader (`_filter_module_ast`).

## Gate Evidence

- `py_compile` (real interpreter `/usr/local/python3.11.15/bin/python3`): PASS.
- Real harness loader + accuracy + warm-up/compile smoke:
  `auto_bench.py --v0_file .../base.py --v1_file .../triton_sparse_pooler_001.py --warmup 1 --repeat 3`
  -> `PASS accuracy; v0=1.042550 ms, v1=0.726550 ms, speedup=1.435x` (smoke timing,
  not authoritative).

## Attempt Ledger

| Command | Exit | Defect | Before SHA | After SHA |
|---|---|---|---|---|
| `python3 -m py_compile .../triton_sparse_pooler_001.py` | 0 | none | - | `dc2a8b65...` |
| `python3 auto_bench.py --v0_file .../base.py --v1_file .../triton_sparse_pooler_001.py --warmup 1 --repeat 3` | 0 | none | `dc2a8b65...` | `dc2a8b65...` |
