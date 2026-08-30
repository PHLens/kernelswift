# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: 1
- last_completed_round: null
- accepted_kernel: null
- accepted_report: null
- recent_three_round_evidence: `<none: Phase 0>`
- open_hypotheses: `<none: Phase 0>`
- artifact_read_hashes: `<see ledger below>`

## Current Bottleneck

- `<none: baseline not yet established (Phase 0; Verifier has not measured)`

## Recent Three-round Evidence

- `<none: Phase 0>`

## Open Hypotheses or Checks

- MLU sibling campaign reached 1.60x by fusing ONLY pooling+relu+log1p into a
  Triton kernel (MLM head stays as library `nn.Linear`/`GELU`/`LayerNorm` ops),
  with on-device prefix scan to avoid `seq_lens.tolist()` D2H sync. Reference:
  `mlu/triton_sparse_pooler_004.py`. NOT ported verbatim: `fast_libentry` is
  Unknown on Ascend (target profile) and `torch_mlu` is MLU-specific — both must
  NOT be assumed for Ascend. Direct Triton launch is the proven Ascend launcher
  path.

## Baseline Adapter Confirmation (Phase 0)

- `baseline_adapter.py` is `base.py` with `Model` renamed to `ModelNew`.
- Exactly one `ModelNew` class: confirmed (line 5).
- `get_inputs` present: confirmed (returns `[hidden_states, seq_lens]`).
- `get_init_inputs` present: confirmed (returns `[768, 30522, "max"]`).
- No stray `class Model`: confirmed (no `class Model` definition).
- Output is a Python `list` of tensors: confirmed (per-sequence `result.append`).
- Conformance note: `__main__` block (line 38) still references `Model` instead
  of `ModelNew`. Harmless — never executed by the harness AST loader; public
  contract (`ModelNew`/`get_inputs`/`get_init_inputs`) is unaffected.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `2b740bba37a87a7bcb022af36537486179538feed5dada3f3c1d5e32cd3f6c36` | 000 |
| `ascend/baseline_adapter.py` | `94d00f1a5d26f453fd5078fd9d50dfcddbb0c11d20a145d223544e59234add0f` | 000 |
| `skills/kernel-opt-loop/prompts/coder.md` | `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196` | 000 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_ascend.md` | `db54aa6269174f7f7d8707c6a084a36c4451b6826e8d5153003ce7e1b0523cc8` | 000 |
| `mlu/triton_sparse_pooler_004.py` | `81cdea2b958c288e1382aef0b30cfc6dffb544c55a0e44825fab51b53cac7842` | 000 |
