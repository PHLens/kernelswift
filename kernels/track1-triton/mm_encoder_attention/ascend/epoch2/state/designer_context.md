# Designer Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5`
- context_epoch: `2`
- last_completed_round: `000`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Phase 0 baseline only; no optimization round has completed in this epoch. Epoch-1 history under ../ is labeled noncanonical and is not a source baseline.`
- open_hypotheses: `Collapse the materialized transpose/copy chain and the 6.98 launches/call into a single fused Triton attention kernel that reads and writes the native [bsz, seq, heads*head_size] layout, using the now-qualified fp16 tl.dot path with seq_len 83 unpadded.`
- artifact_read_hashes: `see the table below`

## Current Bottleneck

Verifier-backed facts from `rounds/report_000.md` only:

- Wall time is dominated by host-side launch and synchronization: device time is
  `116.1696 us/call` (reference) and `104.1264 us/call` (candidate) against a
  benchmark median of `0.349625 / 0.347800 ms`, i.e. `device_ratio` ~0.29 under
  the profiler and ~0.33 against benchmark medians. About two thirds of wall time
  is not device compute.
- Launch profile is `6.96-6.98` kernels per call.
- Inside device time, layout conversion dominates the actual attention math:
  `aclnnFlashAttentionScore_TransposeAiCore_Transpose` 3/call at `48.0884 us/call`
  and `aclnnInplaceCopy_TransposeAiCore_Transpose` 1/call at `15.2628 us/call`
  together cost `63.35 us/call` (54.5% of device), while
  `aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore` costs only
  `23.0232 us/call` (19.8%).
- `EVENT_WAIT_SQE` / `EVENT_RECORD_SQE` synchronization events cost `29.7752 / 0.02 us/call`.

Implication: a device-only improvement cannot clear the 5% wall threshold. Any
hypothesis must reduce launch count or remove materialized layout conversions, or
both. This is the `device-win-wall-loss` pattern recorded in KernelWiki
(`skills/kernelwiki/wiki/patterns/device-win-wall-loss.md`).

## Recent Three-round Evidence

- `000` / `baseline` / `rounds/report_000.md` / change family `not-applicable`:
  baseline_adapter.py is a faithful reproduction; improvement 0.52% (within noise).
  Baseline drifted +9.04% versus epoch 1 (`0.320635` -> `0.349625`) under an
  identical measurement fingerprint.

## Open Hypotheses or Checks

1. Fused single-kernel attention with fp16 `tl.dot`, reading q/k/v directly in
   `[bsz, 83, 512]` and writing `[bsz, 83, 512]`, eliminating the three
   transposes and the inplace copy, and collapsing `6.98` launches to one.
   Expected mechanism: removes `63.35 us/call` of device layout work and most of
   the host launch cost.
2. If the fused kernel needs a stable softmax, prefer a single-pass or
   two-pass-in-kernel formulation that keeps the launch count at one rather than
   reintroducing a separate reduction kernel.
3. Do not pad `seq_len 83` to 128: the frozen profile establishes fp16 `tl.dot`
   as numerically correct on non-multiple-of-16 tiles including `(83,64,64)`.
4. `num_warps` 1/2/4/8 and `num_stages` 1/2/3/4 are profile-legal and may be
   declared only as finite `preferred|exploratory` configuration fields.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 000 |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 000 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `rounds/report_000.md` | `64fe68820ac2b5b45211477dca5de66ac53b9bdbadbbc96297b0b6ae925dfb55` | 000 |
| `state/implementation_profile_snapshot/profile.yaml` | `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321` | 000 |
