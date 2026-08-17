# Designer Context State

- role_contract_sha256: `afadcef690ff4d28d47a7958ce91415bb9a99b6096950c3742cfbfd0b02b6733`
- context_epoch: 2
- last_completed_round: `002`
- accepted_kernel: `triton_sparse_pooler_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `000 baseline (0.909974 ms); 001 accepted fused relu+log1p+per-segment max (0.606758 ms, +33.39%, partially-confirmed: device time rose 179.33->210.12 us/call, host loop+D2H sync elimination drove the wall gain); 002 no-improvement BLOCK_V 1024->2048 (0.621848 ms, +0.65%, falsified: fused kernel regressed 99.71->102.42 us/call)`
- open_hypotheses: `Round 003 targets decoder matmul fusion via tl.dot (kernel-matmul-fusion): fuse decoder matmul + bias + relu + log1p + per-segment max into one kernel, eliminating MLUFusedMatMulGepm (90.36 us/call) and the existing fused reduction kernel (98.73 us/call) and avoiding intermediate logits tensor materialization; remaining host time ~396 us/call (~65% of wall) is launcher+wrapper+alloc+harness fixed cost, deferred to a future host-allocation-reuse round`
- artifact_read_hashes: `see ledger below`

## Current Bottleneck

- Candidate wall time 0.606758 ms with device_ratio 0.346 (mixed class). The decoder matmul (MLUFusedMatMulGepm) at 90.36 us/call (43.0% of device time) and the existing fused _sparse_pooler_max_kernel at 98.73 us/call (47.0%) together account for 189.09 us/call — 90.0% of device time. The decoder matmul materializes the intermediate logits tensor [83, 30522] fp32 (10.16 MB) in global memory, which the fused reduction kernel re-reads. Host-side Python loop and D2H sync are already eliminated; remaining host time (~396 us/call) is launcher, wrapper, allocation, and harness-fixed cost.

## Recent Three-round Evidence

- Round 000 (baseline): `baseline_adapter.py` vs `base.py`, wall 0.909974 ms, device 180.05 us/call, device_ratio 0.198. Evidence: `rounds/report_000.md`. Change family: baseline.
- Round 001 (accepted): fused relu+log1p+per-segment max into one Triton kernel, grid (4,30), BLOCK_V=1024, num_warps=1, on-device seq_offset prefix scan. Wall 0.606758 ms (+33.39%), device 210.12 us/call (up 30.79), kernel_count 10->5. Hypothesis partially-confirmed: host loop+D2H sync elimination drove the gain; fused kernel is slower on device than the 6 library kernels it replaced. Evidence: `rounds/report_001.md`. Change family: kernel-fusion (mixed scope). NONCANONICAL for future rounds: the fused reduction kernel's tiling and per-program work are proven; BLOCK_V=1024 is the best-known tiling for the current reduction strategy.
- Round 002 (no-improvement): BLOCK_V 1024->2048 tuning. Wall 0.621848 ms (+0.65%), device 213.02 us/call (up 2.92), fused kernel 99.71->102.42 us/call. Falsified: larger per-program work outweighed launch-dispatch savings. Evidence: `rounds/report_002.md`. Change family: kernel-tile-tuning. NONCANONICAL for future rounds: kernel-tile-tuning is exhausted for this kernel on this runtime; the bottleneck in the fused kernel is per-program elementwise compute, not launch-dispatch overhead.

## Open Hypotheses or Checks

- Round 003 (in progress): decoder matmul fusion via tl.dot (change_family=kernel-matmul-fusion). The new fused kernel replaces the decoder matmul (90.36 us/call) and the existing fused reduction kernel (98.73 us/call) with one matmul+bias+relu+log1p+max kernel, avoiding the intermediate logits tensor materialization. Expected wall improvement 8.0%. Primary risk: tl.dot with small M (BLOCK_M=32, actual seq_len=18-25) may be inefficient on MLU590-H8. Falsifiable via decoder_matmul_kernel_count_per_call, total_kernel_count_per_call, device_us_per_call, fused_kernel_us_per_call observables. Evidence: `rounds/decision_003.md`.
- Host-side launcher reduction / allocation reuse: targets the remaining ~396 us/call host time but requires Host Plan lifecycle changes (output buffer caching, fast_libentry). Independent of device-side work. Deferred to a future round (change_family=host-allocation-reuse).
- Dense matmul + GELU + LayerNorm fusion: the three remaining library ops (8.42 + 7.21 + 5.40 = 21.03 us/call) could be fused into the same or a separate Triton kernel in a future round if the decoder matmul fusion succeeds and these become the new bottleneck.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `triton_sparse_pooler_001.py` | `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd` | 003 |
| `rounds/report_001.md` | (see report Identity block) | 003 |
| `rounds/report_002.md` | (see report Identity block) | 003 |
| `rounds/decision_001.md` | `0816c943dfcfd157c9c4268196f4779b9804b9107de5fff0ba135d66f4f5bc75` | 003 |
| `rounds/decision_002.md` | `0d39de9e280f6ffa2cc3d1d3322d393fa400eb8f405b7e7ee3ceb3ef845b3dd4` | 003 |
| `baseline_adapter.py` | `d7e69ed4b66a4193a4475cc307fc0929cef807f875785652df6cc36fb2c487e5` | 003 |
| `base.py` | `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de` | 003 |
| `rounds/decision_003.md` | (pending validation) | 003 |
