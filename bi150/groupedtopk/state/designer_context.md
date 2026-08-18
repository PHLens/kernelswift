# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `3`
- last_completed_round: `002`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Round 000 baseline established on BI150; reference median 0.474612 ms, baseline adapter median 0.474995 ms; correctness passed and device profiler is available. Round 001 was capability-miss under the prior profile. Round 002 direct fused selection passed seeded smoke and tie checks but failed structured group-tie exact IDs after bounded repair: reference [32,0,64,96,4,3,1,2] versus candidate [0,32,64,96,7,6,4,5].`
- open_hypotheses: `H-003: Partial routing fusion for the fixed [83,256] fp32 regime. Fuse softmax, group-score reduction, and group masking, but retain exact torch.topk for group and final expert selection. Expected wall gain >=5%; exact adversarial tie ordering is mandatory. Reject any custom final top-k replacement or any semantic mismatch.`
- artifact_read_hashes: `project.md, team-state.md, designer_context.md, report_000.md, decision_002.md, coder_result_002.md, triton_grouped_topk_002.py, triton_cuda.md, decision-template.md, invariants.md, bottleneck-judgment.md, and anti-patterns.md read for Round 003.`

## Current Bottleneck

- Verifier-backed: baseline profiler reports top-k gather at 48.7290625 us/call and bitonic sort at 36.879697265625 us/call; device time is available on BI150.
- Classification: mixed, with device_ratio 0.3769941822 for the canonical baseline adapter.
- Round 002 evidence identifies tie-order semantics, not primitive availability, as the blocking design risk for custom final selection. Round 003 is explicitly kernel-only; torch.topk remains the exact library boundary and no host lifecycle behavior is changed.

## Recent Three-round Evidence

- Round 000, baseline, `rounds/report_000.md`: reference device time 177.181318359375 us/call; baseline adapter device time 179.0703515625 us/call; 14.8 versus 14.96 kernels/call.
- Round 001, `rounds/decision_001.md`: capability-miss under the earlier profile; no candidate was created and canonical baseline remained unchanged.
- Round 002, `rounds/coder_result_002.md`: implementation-failed after two bounded repairs. Seeded smoke passed, but exact tie ordering failed for structured groups; canonical baseline remained unchanged.

## Open Hypotheses or Checks

- H-003, ranked first, `partial-routing-fusion`: fuse only softmax, group-score reduction, and group masking, retaining exact `torch.topk` for both group and final expert selection. Expected wall gain >=5%; risk is feeding the library selections without changing active sets or values. Evidence: baseline top-k/bitonic device dominance and Round 002 active-set-dependent tie mismatch. Validation cost: exact seeded and adversarial tie correctness plus Level 1 targeted profiling. Change family: `partial-routing-fusion`.
- Do not repeat Round 002's custom final top-k or static priority replacement; its tie semantics are disproven by the structured case.
- Any host or allocation hypothesis requires a complete Host Plan with cache key, invalidation, device/stream, and concurrency semantics; it is not selected in Round 003.
- Do not require or introduce `tl.dot`, `num_warps`, `num_stages`, `fast_libentry`, block pointers, mixed precision, arbitrary layouts, non-contiguous inputs, or custom repeated argmax ordering.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `base.py` | `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5` | 000 |
| `baseline_adapter.py` | `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016` | 003 |
| `rounds/report_000.md` | `39a512eed23f1f0889e7845cde5f854cf0c2ca9d377ff23588148f239139f1e5` | 003 |
| `rounds/decision_001.md` | `7e899d6cee2ad8fe6ab586b902ff0d18226e77d7d3cfb3ecf791b572e2371365` | 003 |
| `rounds/decision_002.md` | `d3c0f316945706acaad5c6f68ae0d93e9bbf3c848ca735b20c2356b304107d37` | 003 |
| `rounds/coder_result_002.md` | `pending ledger hash` | 003 |
| `triton_grouped_topk_002.py` | `pending ledger hash` | 003 |
| `rounds/decision_003.md` | `dfe241e2b7b6f2609a3d59185d2d067072b986e9316f0b1e857a4023d0ac5030` | 003 |
