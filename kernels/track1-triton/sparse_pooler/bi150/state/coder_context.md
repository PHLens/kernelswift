# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `1`
- last_completed_round: `001`
- accepted_kernel: `null`
- accepted_report: `null`
- recent_three_round_evidence: `round 001 activation-pooling-fusion candidate-ready`
- open_hypotheses: `fused Triton kernel for log1p(relu) + per-sequence max-pool (candidate produced)`
- artifact_read_hashes: see table below

## Current Bottleneck

- device-bound (device_ratio 0.694); dominant cost is the two GEMMs (dense + decoder) on the TCU (~581 us/call ≈ 78% of device time), plus the post-decoder tail (~144 us/call).

## Recent Three-round Evidence

- round `000`: baseline (`baseline_adapter.py`), wall median `1.070492 ms`.
- round `001`: activation-pooling-fusion candidate (`triton_sparse_pooler_001.py`), smoke `1.201x`.

## Open Hypotheses or Checks

- Verifier to measure wall time / device time / kernel count for the fused kernel candidate.
- A future (unprobed) hypothesis: fp32 large-N `tl.dot` rewrite of the decoder GEMM (N=30522); currently capability-miss risk (only `(32,32)@(32,32)` `tl.dot` evidence).

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `kernels/track1-triton/sparse_pooler/base.py` | `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58` | `001` |
| `kernels/track1-triton/sparse_pooler/bi150/baseline_adapter.py` | `359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8` | `001` |
| `kernels/track1-triton/sparse_pooler/bi150/rounds/decision_001.md` | `0fbbdb6929e1b75f939fc2d513c28878b7a53587f33e8fcaf66401f1269256f1` | `001` |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | `001` |
