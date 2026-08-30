# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: 0
- last_completed_round: `002`
- accepted_kernel: `null`
- accepted_report: `null`
- recent_three_round_evidence: `[000: baseline, wall 0.151139 ms, fused FlashAttnFwdF16Ixmma 14.949 us/call, device_ratio 0.099 | 001: aborted (tl.dot Unknown capability-miss) | 002: aborted (tl.dot correct but perf unproven; host-bound harness-fixed)]`
- open_hypotheses: `<none> — two aborts; device is single vendor Ixmma kernel, host is harness-fixed`
- artifact_read_hashes: see Artifact Read Hashes table below

## Current Bottleneck

- `mm_encoder_attention` is `host-bound` with harness-fixed overhead: 100% of
  device time is one vendor-tuned fused `FlashAttnFwdF16Ixmma` flash-attention
  kernel (`14.949 us/call`, `0.86 kernels/call`), `device_ratio ≈ 0.099`, so
  ~90% of wall (`151.139 us` total) is `set_seed` + `torch.cuda.synchronize()`
  harness overhead outside `ModelNew.forward` and outside any candidate
  boundary.

## Recent Three-round Evidence

- `000`: Result `baseline`. Wall median `0.151139 ms`. Device
  `14.9492578125 us/call`, `0.86 kernels/call`. SDPA → fused flash backend
  (`FlashAttnFwdF16Ixmma`, Causal=0, Alibi=0, `__half`); no bmm/softmax.
  `device_ratio ≈ 0.099`. See `rounds/report_000.md`.
- `001`: Result `aborted`. Justification: `tl.dot` Unknown in profile (no BI150
  probe), so Triton flash attention was a capability-miss. See
  `rounds/decision_001.md`.
- `002`: Result `aborted`. Justification: `tl.dot` is now Supported (correctness
  only, `(32,32)@(32,32)` exact fp32 / near-exact bf16), but its BI150
  tensor-core performance is unverified; device is a single vendor Ixmma kernel
  and ~90% of wall is harness-fixed host overhead, so no falsifiable ≥5% wall
  improvement exists. See `rounds/decision_002.md`.

## Open Hypotheses or Checks

- `<none>` — two consecutive aborts. The `tl.dot` capability-miss is resolved but
  the bottleneck is unchanged: host-bound with harness-fixed seed/synchronize
  overhead, and device time is a single vendor tensor-core kernel with no
  compressible structure. A proceeding decision would require either (a) a
  matched BI150 probe demonstrating `tl.dot` tensor-core device time that can
  plausibly beat the vendor Ixmma kernel, or (b) a change to the benchmark
  regime/harness that is outside the Designer's authority.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `kernels/track1-triton/mm_encoder_attention/base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 000 |
| `kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py` | `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f` | 000 |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | 002 |
| `kernels/track1-triton/mm_encoder_attention/bi150/rounds/report_000.md` | `<recorded at read>` | 002 |
| `kernels/track1-triton/mm_encoder_attention/bi150/rounds/decision_001.md` | `<recorded at read>` | 002 |
| `prompts/coder_targets/triton_cuda.md` | `<recorded at read>` | 002 |
