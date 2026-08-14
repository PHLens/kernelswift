# Verifier Context State

- role_contract_sha256: `1f55c297cc01b16fbbb4c083487e0f835d6194c524bbaafda10994807f9400b8`
- context_epoch: 2
- last_completed_round: `002`
- accepted_kernel: `triton_sparse_pooler_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `000 baseline (0.909974 ms, 180.05 us/call, ratio 0.198); 001 accepted (0.606758 ms, +33.39%, 210.12 us/call, ratio 0.346, partially-confirmed); 002 no-improvement (0.621848 ms, +0.65%, 213.02 us/call, ratio 0.343, falsified)`
- open_hypotheses: `await Designer decision_003; Verifier will use triton_sparse_pooler_001.py as accepted reference and run v2 screening (two short interleaved pairs) before authoritative timing`
- artifact_read_hashes: `see ledger below`

## Current Bottleneck

- (Verifier-backed) Accepted wall time 0.606758 ms with device_ratio 0.346 (mixed class). The fused `_sparse_pooler_max_kernel` at 98.73 us/call is the dominant device kernel and is 30.86 us/call slower than the 6 library kernels it replaced (67.87 us/call combined). Round 002 BLOCK_V=2048 was falsified (fused kernel 99.71->102.42 us/call). BLOCK_V=1024 remains the best-known tiling for the current reduction strategy.

## Recent Three-round Evidence

- Round 000 (baseline): `baseline_adapter.py` vs `base.py`. Wall 0.909974 ms, device 180.05 us/call, device_ratio 0.198, 10 kernels/call. Evidence: `rounds/report_000.md`.
- Round 001 (accepted): `triton_sparse_pooler_001.py` vs `baseline_adapter.py`. Wall 0.606758 ms (+33.39%), device 210.12 us/call (up 30.79), 5 kernels/call (down from 10). Hypothesis partially-confirmed: `kernel_count_per_call` and `host_sync_count_per_call` confirmed; `device_us_per_call` falsified (fused kernel 98.73 us/call vs 67.87 for 6 library kernels). Evidence: `rounds/report_001.md`.
- Round 002 (no-improvement): `triton_sparse_pooler_002.py` vs `triton_sparse_pooler_001.py`. Wall 0.621848 ms (+0.65%), device 213.02 us/call (up 2.92), fused kernel 99.71->102.42 us/call. Hypothesis falsified: BLOCK_V=2048 larger per-program work outweighed launch-dispatch savings. Evidence: `rounds/report_002.md`.

## Open Hypotheses or Checks

- Awaiting Designer's decision_003. Verifier's accepted reference is `triton_sparse_pooler_001.py` (the Round 001 canonical); Round 002's rejected candidate is never the comparison source.
- v2 screening protocol: after correctness passes, run exactly two short interleaved accepted-reference/candidate pairs. A correct candidate is `screened-out` only when both pairs are at least 10% slower than the accepted reference. Otherwise proceed to authoritative 3-pair timing.
- `summarize_trace.py` hits "overlapping scope events" on this harness's traces (each scope has 2 overlapping events: `user_annotation` on CPU + `gpu_user_annotation` on GPU). Use the gpu_user_annotation-only fallback scoping documented in `rounds/report_001.md` and `rounds/report_002.md`.
- Verifier does not update `last_accepted_kernel`; Orchestrator applies the accepted transition.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `triton_sparse_pooler_001.py` | `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd` | 002 |
| `triton_sparse_pooler_002.py` | `62dc853db5423cb5d99ad53433f3fb35919abe901a64d6e3acb3d815ac678248` | 002 |
| `baseline_adapter.py` | `d7e69ed4b66a4193a4475cc307fc0929cef807f875785652df6cc36fb2c487e5` | 000 |
| `base.py` | `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de` | 000 |
| `rounds/report_001.md` | (see report Identity block) | 002 |
| `rounds/report_002.md` | (see report Identity block) | 002 |
