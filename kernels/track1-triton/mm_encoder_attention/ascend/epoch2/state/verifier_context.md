# Verifier Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f`
- context_epoch: `2`
- last_completed_round: `000`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Phase 0 baseline measured; no candidate verification has occurred in this epoch.`
- open_hypotheses: `awaiting a candidate from Coder for round 001.`
- artifact_read_hashes: `see the table below`

## Current Bottleneck

Facts this role established in `rounds/report_000.md`:

- Device time `116.1696 us/call` (reference) and `104.1264 us/call` (candidate)
  at `iterations=50`; `device_ratio` 0.2885 / 0.2684 against profiler-run wall
  values, about 0.33 against benchmark medians.
- `6.98` / `6.96` kernels per call.
- Device composition: 3x `aclnnFlashAttentionScore_TransposeAiCore_Transpose`
  (`48.0884 us/call`), `EVENT_WAIT_SQE` (`29.7752 us/call`), 1x
  FlashAttentionScore (`23.0232 us/call`), 1x
  `aclnnInplaceCopy_TransposeAiCore_Transpose` (`15.2628 us/call`).

## Recent Three-round Evidence

- `000` / `baseline` / `rounds/report_000.md`: correctness pass; reference median
  `0.349625` ms, candidate median `0.347800` ms, improvement `0.5220%`.
  Baseline drift versus epoch 1 is `+9.04%` (`0.320635` -> `0.349625`) under an
  identical measurement fingerprint.

## Open Hypotheses or Checks

- Device evidence on this target requires `torch_npu.profiler` plus the CANN
  sqlite at `device_0/sqlite/ai_core_op_summary.db`, summarized by
  `skills/kernel-opt-loop/scripts/summarize_cann_trace.py`. A raw
  `torch.profiler` chrome trace alone is host-side `cpu_op` events only and is
  not device evidence.
- Report device and synchronized wall measurements separately; never present a
  device win as a wall win.
- The profiler run inflates wall time relative to the un-profiled benchmark, so
  `device_ratio` must state which wall value it uses.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 000 |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 000 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
