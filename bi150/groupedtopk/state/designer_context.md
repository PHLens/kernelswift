# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `2`
- last_completed_round: `001`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Round 000 baseline established on BI150; reference median 0.474612 ms, baseline adapter median 0.474995 ms; correctness passed; device profiler available. Round 001 was capability-miss under the prior profile. Round 002 proposes a direct fixed-shape Triton grouped-top-k fusion under the updated matched profile, with exact PyTorch tie/order correctness as a hard guardrail.`
- open_hypotheses: `H-002: Fuse the fixed [83,256] fp32 grouped-top-k path into one direct Triton kernel. Expected mechanism is fewer intermediate materializations and library launches, with at least 5% wall improvement required. Reject on any tie/order mismatch or insufficient paired wall improvement.`
- artifact_read_hashes: `designer.md, project.md, team-state.md, designer_context.md, report_000.md, triton_cuda.md, decision-template.md, invariants.md, bottleneck-judgment.md, and anti-patterns.md read for Round 002.`

## Current Bottleneck

- Verifier-backed: baseline profiler reports top-k gather at 48.7290625 us/call and bitonic sort at 36.879697265625 us/call; device time is available on BI150.
- Classification: mixed, with device_ratio 0.3769941822 for the canonical baseline adapter; the Round 002 intervention targets device launches and intermediate materialization only.

## Recent Three-round Evidence

- Round 000, baseline, `rounds/report_000.md`: reference device time 177.181318359375 us/call; baseline adapter device time 179.0703515625 us/call; 14.8 versus 14.96 kernels/call.
- Round 001, `rounds/decision_001.md`: capability-miss under the earlier profile; no candidate was created and canonical baseline remained unchanged.
- Round 002, `rounds/decision_002.md`: proceeding kernel-only fused grouped-top-k hypothesis; validation pending until Orchestrator runs the deterministic decision validator.

## Open Hypotheses or Checks

- H-002, ranked first, `grouped-topk-fusion`: fuse softmax, grouped filtering, stable top-k selection, renormalization, and scaling for the fixed `[83,256]` fp32 regime. Expected wall gain >=5%; risk is constrained repeated argmax tie behavior and resource growth. Evidence: `rounds/report_000.md` top-k gather and bitonic sort dominance. Validation cost: correctness with exact IDs/ties plus Level 1 targeted profiling. Change family: `grouped-topk-fusion`.
- A future host or allocation hypothesis requires a complete Host Plan with cache key, invalidation, device/stream, and concurrency semantics; it is not selected in Round 002.
- Do not require or introduce `tl.dot`, `num_warps`, `num_stages`, `fast_libentry`, block pointers, mixed precision, arbitrary layouts, or non-contiguous inputs.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `base.py` | `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5` | 000 |
| `baseline_adapter.py` | `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016` | 002 |
| `rounds/report_000.md` | `39a512eed23f1f0889e7845cde5f854cf0c2ca9d377ff23588148f239139f1e5` | 002 |
| `rounds/decision_001.md` | `7e899d6cee2ad8fe6ab586b902ff0d18226e77d7d3cfb3ecf791b572e2371365` | 002 |
| `rounds/decision_002.md` | `d3c0f316945706acaad5c6f68ae0d93e9bbf3c848ca735b20c2356b304107d37` | 002 |
