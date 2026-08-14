# Designer Context State

- role_contract_sha256: `afadcef690ff4d28d47a7958ce91415bb9a99b6096950c3742cfbfd0b02b6733`
- context_epoch: 2
- last_completed_round: `002`
- accepted_kernel: `triton_sparse_pooler_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `000 baseline (0.909974 ms); 001 accepted fused relu+log1p+per-segment max (0.606758 ms, +33.39%, partially-confirmed: device time rose 179.33->210.12 us/call, host loop+D2H sync elimination drove the wall gain); 002 no-improvement BLOCK_V 1024->2048 (0.621848 ms, +0.65%, falsified: fused kernel regressed 99.71->102.42 us/call)`
- open_hypotheses: `fused kernel is 47% of candidate device time and 34.55 us/call slower than the 6 library kernels it replaced; decoder matmul MLUFusedMatMulGepm 90.36 us/call is the second-largest device kernel and largest non-fused kernel; remaining host time ~396 us/call (~65% of wall) is launcher+wrapper+alloc+harness fixed cost`
- artifact_read_hashes: `see ledger below`

## Current Bottleneck

- Candidate wall time 0.606758 ms with device_ratio 0.346 (mixed class). The fused `_sparse_pooler_max_kernel` at 98.73 us/call is the dominant device kernel and is 30.86 us/call slower than the 6 library kernels it replaced (67.87 us/call combined). The decoder matmul (MLUFusedMatMulGepm) at 90.36 us/call is the second-largest device kernel. Host-side Python loop and D2H sync are already eliminated; remaining host time (~396 us/call) is launcher, wrapper, allocation, and harness-fixed cost.

## Recent Three-round Evidence

- Round 000 (baseline): `baseline_adapter.py` vs `base.py`, wall 0.909974 ms, device 180.05 us/call, device_ratio 0.198. Evidence: `rounds/report_000.md`. Change family: baseline.
- Round 001 (accepted): fused relu+log1p+per-segment max into one Triton kernel, grid (4,30), BLOCK_V=1024, num_warps=1, on-device seq_offset prefix scan. Wall 0.606758 ms (+33.39%), device 210.12 us/call (up 30.79), kernel_count 10->5. Hypothesis partially-confirmed: host loop+D2H sync elimination drove the gain; fused kernel is slower on device than the 6 library kernels it replaced. Evidence: `rounds/report_001.md`. Change family: kernel-fusion (mixed scope).
- Round 002 (no-improvement): BLOCK_V 1024->2048 tuning. Wall 0.621848 ms (+0.65%), device 213.02 us/call (up 2.92), fused kernel 99.71->102.42 us/call. Falsified: larger per-program work outweighed launch-dispatch savings. Evidence: `rounds/report_002.md`. Change family: kernel-tile-tuning.

## Open Hypotheses or Checks

- Device-side headroom in the fused kernel: a better-tuned reduction (e.g., load each logits row once and accumulate across vocab tiles, or a different num_warps/BLOCK_V combination) could recover the 30.86 us/call device regression. Round 002 narrowed the parameter space: BLOCK_V=2048 is worse, BLOCK_V=1024 remains the best-known tiling for the current reduction strategy.
- Decoder matmul fusion via `tl.dot` with shape [83,768]x[768,30522]: larger change boundary, would replace the 90.36 us/call MLUFusedMatMulGepm library op. Flagged as a future-round candidate in decision_001.
- Host-side launcher reduction / allocation reuse: targets the remaining ~396 us/call host time but requires Host Plan lifecycle changes (output buffer caching, fast_libentry). Independent of device-side work.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `triton_sparse_pooler_001.py` | `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd` | 002 |
| `rounds/report_001.md` | (see report Identity block) | 002 |
| `rounds/report_002.md` | (see report Identity block) | 002 |
| `rounds/decision_001.md` | `0816c943dfcfd157c9c4268196f4779b9804b9107de5fff0ba135d66f4f5bc75` | 001 |
| `rounds/decision_002.md` | `0d39de9e280f6ffa2cc3d1d3322d393fa400eb8f405b7e7ee3ceb3ef845b3dd4` | 002 |
| `baseline_adapter.py` | `d7e69ed4b66a4193a4475cc307fc0929cef807f875785652df6cc36fb2c487e5` | 000 |
| `base.py` | `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de` | 000 |
