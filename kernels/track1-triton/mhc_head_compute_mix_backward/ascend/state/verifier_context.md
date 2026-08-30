# Verifier Context State

- role_contract_sha256: `f9d06fdf3ddbb18944568412f7d86d88266245f8dfa974a2ab3cf282f37bbd27`
- context_epoch: 2
- last_completed_round: 001
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `round 000 baseline (host-bound 9.5%, 10 kernels); round 001 kernel-fusion rejected (device 41->17us but wall only +3.26%, host-bound persists)`
- open_hypotheses: `host-bound wall not moved by device fusion; next lever is host-side (zero-init kernels, allocation reuse) or measurement-bound`
- artifact_read_hashes: `<see table below>`

## Current Bottleneck

- Host-bound, and device fusion did not move wall time. After round 001 fusion, candidate device_ratio fell to ~3.9% but wall stayed ~430 us. Remaining wall is host-side: triton kernel launch, 2 zero-init kernels (from per-call `torch.zeros`), per-call `torch.empty_like`/`torch.zeros` allocation, and harness seed+sync (fixed for regime).

## Recent Three-round Evidence

- round 000 (baseline): host-bound, device_ratio ~9.5%, 10 kernels/call (2 ReduceSum + 5 Mul + 1 Add + 1 Rsubs + 1 Sigmoid).
- round 001 (kernel-fusion, H-001): falsified. device_us_per_call 41.06->16.85 us (2.4x), but kernel_count only 10->3 (not ->1; 2 zero-init kernels added). wall +3.26% < 5%. tl.atomic_add is device-cheap (fused kernel 15.41us < two ReduceSum 22.44us). Host-bound wall unchanged.

## Open Hypotheses or Checks

- Next round: whether eliminating the 2 zero-init kernels + per-call allocation, or a host-side / allocation-reuse change, can move wall time; or whether the remaining host cost is harness-fixed (measurement-bound stop). Any change must stay inside the decision's allowed change boundary and preserve per-call output-allocation semantics.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc` | 000 |
| `baseline_adapter.py` | `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d` | 001 |
| `triton_mhc_mix_bwd_001.py` | `f7efc6853a8f07b90926237cc2f4de620926bd0b34333648e7355d8995c57d10` | 001 |
| `decision_001.md` | `5ee9ea5d9b74de678482dd801066bf9883d0d0bf76af231f0325689665d5f88d` | 001 |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 001 |
| `project.md` | `<orchestrator-owned>` | 001 |
