# Final Summary — sparse_pooler (S60 Epoch 2)

## Delivery Result (measurement-bound, deliverable shipped)

- status: **complete — measurement-bound (terminal)**
- **deliverable**: `triton_sparse_pooler_e2_001.py` (correctness-PASS, first Triton for sparse_pooler)
- **speedup vs base**: ~0.249x (no-improvement, -302%)
- **epoch-1 stays optimal**: 0.79x (epoch-1 fused candidate remains the best submission)

## What was tried

| metric | base | candidate |
|---|---:|---:|
| launches/call | 11.0 | 8.0 |
| aten cpu_ops/call | 83 | 59 |
| D2H sync (tolist) | present | eliminated (device-side cumsum/sub offsets) |
| wall | 0.857ms | 3.448ms |

## Root cause (conclusive)

1. **GEMM 481us (61%) vendor-bound and untouchable**: dense [83,768]@[768,768] (~165us)
   + decoder [83,768]@[768,30522] (~316us). Both 768 and 30522 are NOT power-of-2,
   so tl.dot is capability-blocked (power-of-2 constraint) — hand-written GEMM is
   not even legal, let alone faster.
2. **Hand-written segment-max is ~4x slower**: the fused-tail Triton kernel's
   segment reduction costs ~2.59ms device, vs base's PyTorch `log1p(relu)`+4×`chunk.max`
   tail. This far exceeds the preflight ~150us estimate.
3. **D2H sync (125us, 16%) elimination requires the slower hand-written reduction**,
   net negative — confirming the preflight second clause.

## Falsified directions (full chain)

1. epoch-1 fused relu/log1p/max + prefix-scan: -26.79%
2. scatter_reduce segment max: 7x slower (unoptimized on GCU)
3. post-decoder tail fusion (this round): -302% (hand-written segment-max ~4x slower)

## Terminal Classification

- terminal_result: no-improvement
- stop_reason: measurement-bound (GEMM vendor-bound + segment reduction hand-write-slow)
- DELIVERABLE RULE met: correctness-PASS Triton candidate shipped (0 tl.dot, stateless,
  envelope-legal); canonical stays baseline_adapter.py (epoch-1 0.79x remains best).

## Note

- `log1p(relu(x))` implemented as `tl.log(1.0 + tl.maximum(x, 0.0))` (GCU has no tl.log1p).
- offsets via `torch.cumsum(seq_lens,0)-seq_lens` cast back to int32 (GCU rejects int64).
