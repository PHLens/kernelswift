# Verifier Context State

- role_contract_sha256: `f9d06fdf3ddbb18944568412f7d86d88266245f8dfa974a2ab3cf282f37bbd27`
- context_epoch: 1
- last_completed_round: `000` (Phase 0 baseline)
- accepted_kernel: `baseline_adapter.py` (canonical pointer to be set by Orchestrator)
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `round 000 = baseline (Phase 0); flexattention causal SDPA -> FlashAttnFwdF16Ixmma CausalM_t=2; wall median 0.150070 ms; device 12.88 us/call; device_ratio 0.086`
- open_hypotheses: `none yet (Phase 0)`
- artifact_read_hashes: see table below

## Current Bottleneck

- Device time is a single fused `FlashAttnFwdF16Ixmma` flash-attention kernel
  (Ixmma tensor-core, fp16, `CausalM_t=2`, `AlibiMode_t=0`); ~91% of wall time
  is host/launch overhead (`device_ratio ≈ 0.086`), so there is no internal
  kernel-count reduction opportunity below one kernel per call.

## Recent Three-round Evidence

- round 000 (baseline): flexattention causal SDPA dispatches to
  `FlashAttnFwdF16Ixmma<128,128,16,64,64, CausalM_t=2, AlibiMode_t=0>` — the
  same Ixmma FlashAttention backend as task 6 but with `CausalM_t=2` (causal)
  vs task 6's `CausalM_t=0` (non-causal). wall median `0.150070 ms` (v0) /
  `0.149600 ms` (v1).

## Open Hypotheses or Checks

- None (Phase 0). Designer will formulate the first optimization decision.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `kernels/track1-triton/flexattention/base.py` | `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` | 0 |
| `kernels/track1-triton/flexattention/bi150/baseline_adapter.py` | `b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1` | 0 |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 0 |
| `skills/kernel-opt-loop/scripts/summarize_trace.py` | `f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c` | 0 |

## Key Phase 0 Facts (persisted for future rounds)

- Harness drift: `auto_bench.py` actual hash `71fb3ad0...` ≠ `project.md`-recorded
  `3d4fa4ee...`. base/adapter hashes match project.md. Orchestrator must refresh
  the measurement fingerprint before optimization rounds.
- Causal SDPA backend: `FlashAttnFwdF16Ixmma` with `CausalM_t=2` (the actual
  observed causal enum value; task brief's "Causal=1" was not what the kernel
  name reports).
