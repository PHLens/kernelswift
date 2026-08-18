# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `5`
- last_completed_round: `004`
- accepted_kernel: `triton_grouped_topk_004.py`
- accepted_report: `rounds/report_004.md`
- recent_three_round_evidence: `Round 002 direct fused selection passed seeded smoke but failed structured group-tie exact IDs. Round 003 one-kernel partial fusion was design-rejected because post-selection masking needs group_idx from library topk. Round 004 two-stage fusion retained exact library topk boundaries and was accepted at 7.455430192% paired median wall improvement with device time 127.260771484375 us/call.`
- open_hypotheses: `The accepted two-stage routing fusion is canonical. Future design must start from triton_grouped_topk_004.py, preserve exact torch.topk tie behavior and Host Plan, and target remaining gather/sort bottlenecks only with a complete falsifiable mechanism.`
- artifact_read_hashes: `project.md, team-state.md, designer_context.md, report_000.md, decision_003.md, coder_result_003.md, triton_cuda.md, decision-template.md, invariants.md, bottleneck-judgment.md, and anti-patterns.md read for Round 004.`

## Current Bottleneck

- Verifier-backed accepted profile: `at::native::sbtopk::gatherTopK` 48.852978515625 us/call and `at::native::bitonicSortKVInPlace` 36.45123046875 us/call; accepted candidate device ratio 0.2945183072.
- Classification: mixed; the accepted two-stage fusion reduced preprocessing/mask work while retaining exact library selection.
- Round 004 evidence: device time 127.260771484375 us/call and 9.9 kernels/call versus reference 178.991259765625 us/call and 14.86 kernels/call.

## Recent Three-round Evidence

- Round 002, `rounds/coder_result_002.md`: custom selector failed structured group-tie IDs; canonical baseline retained.
- Round 003, `rounds/coder_result_003.md`: one-kernel post-selection mask dependency required a major design revision; canonical baseline retained.
- Round 004, `rounds/report_004.md`: two-stage partial fusion accepted; wall median `0.432098 ms`, device `127.260771484375 us/call`, `9.9 kernels/call`.

## Open Hypotheses or Checks

- Accepted H-004, `two-stage-routing-fusion`: stage-one Triton softmax/group-score kernel, exact `torch.topk(group_scores,4)`, stage-two Triton group_idx-dependent mask kernel, exact `torch.topk(masked_scores,8)`. Verifier evidence supports the >=5% wall hypothesis and exact four-case tie correctness.
- Start any future round from accepted `triton_grouped_topk_004.py`; do not repeat Round 002 custom final selection or Round 003's one-kernel post-selection mask dependency.
- Preserve the accepted Host Plan: per-forward scores/group_scores/masked_scores allocations, no cross-call cache, input-device allocation, caller-current-stream launches, same-stream torch.topk consumption, distinct buffers for concurrency, and baseline fallback for non-target shapes/scoring modes.
- Do not require or introduce `tl.dot`, `num_warps`, `num_stages`, `fast_libentry`, block pointers, mixed precision, arbitrary layouts, non-contiguous inputs, or custom repeated argmax ordering.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `base.py` | `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5` | 000 |
| `baseline_adapter.py` | `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016` | 004 |
| `rounds/report_000.md` | `39a512eed23f1f0889e7845cde5f854cf0c2ca9d377ff23588148f239139f1e5` | 004 |
| `rounds/decision_001.md` | `7e899d6cee2ad8fe6ab586b902ff0d18226e77d7d3cfb3ecf791b572e2371365` | 004 |
| `rounds/decision_002.md` | `d3c0f316945706acaad5c6f68ae0d93e9bbf3c848ca735b20c2356b304107d37` | 004 |
| `rounds/coder_result_002.md` | `pending ledger hash` | 004 |
| `triton_grouped_topk_002.py` | `pending ledger hash` | 004 |
| `rounds/decision_003.md` | `dfe241e2b7b6f2609a3d59185d2d067072b986e9316f0b1e857a4023d0ac5030` | 004 |
| `rounds/coder_result_003.md` | `pending ledger hash` | 004 |
| `rounds/decision_004.md` | `307f4a03c15b08daca8bb571f0391418997a07d864ff357b9f2d113cf2fb8f65` | 004 |
| `triton_grouped_topk_004.py` | `881a549cf95746dda93ee4c898e7ab0e67e3133a526088553091f8b8d7431d83` | 004 |
| `rounds/coder_result_004.md` | `aba12a62645a7d789a9a172fb55efe0eac86784c5c81ab86fa29c8d069c79e1c` | 004 |
| `rounds/report_004.md` | `40400c3764ebcbf3825ba0530a59e3f7d081e5728dfd5068a64c26476874cd23` | 004 |
