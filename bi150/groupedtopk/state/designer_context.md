# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `8`
- last_completed_round: `008`
- accepted_kernel: `triton_grouped_topk_008.py`
- accepted_report: `rounds/report_008.md`
- recent_three_round_evidence: `Rounds 005 to 007 established that direct modifications cannot clear a five-percent wall gain while exact library top-k selection remains mandatory. Round 008 found a new host mechanism: constrained torch.compile dispatch preserved the accepted two-stage Triton algorithm and exact tie behavior, then achieved 19.987917795% paired median wall improvement with device time 111.120595703125 us/call.`
- open_hypotheses: `The compiled two-stage candidate is canonical. H-009 (compile-mode-reduce-overhead) is validated and in flight: compile the accepted fixed-shape forward with torch.compile mode reduce-overhead, preserving exact torch.topk ties, compiled-callable lifecycle, current stream/device behavior, per-forward buffer ownership, and eager fallback. Resume Round 009 by gating triton_grouped_topk_009.py on BI150, then verify.`
- artifact_read_hashes: `project.md, team-state.md, designer_context.md, triton_grouped_topk_004.py, report_004.md, decisions_001_to_004.md, coder_results_002_to_004.md, triton_cuda.md, decision-template.md, invariants.md, bottleneck-judgment.md, and anti-patterns.md read for Round 005.`

## Current Bottleneck

- Verifier-backed compiled profile: `at::native::sbtopk::gatherTopK` 49.368779296875 us/call and `at::native::bitonicSortKVInPlace` 37.02447265625 us/call; accepted candidate device ratio 0.3226872915.
- Classification: mixed; exact library selection is still required for BI150/PyTorch active-set-dependent ties, while compiled dispatch fuses surrounding eager framework work.
- Round 008 removed 16.338291015625 us/call of attributed device time and 0.94 kernels/call without modifying selector semantics.

## Recent Three-round Evidence

- Round 003, `rounds/coder_result_003.md`: design-revision-required because a one-kernel design cannot consume library-produced group_idx for masking; canonical baseline retained.
- Round 004, `rounds/report_004.md`: two-stage partial fusion accepted; wall median `0.432098 ms`, device `127.260771484375 us/call`, `9.9 kernels/call`, exact tie suite passed.
- Round 005, `rounds/decision_005.md`: aborted. No supported direct-kernel intervention could credibly clear the five-percent wall threshold from the Round 004 candidate.
- Round 008, `rounds/report_008.md`: compiled dispatch candidate accepted; `0.344360 ms` wall median, `111.120595703125 us/call`, `8.96 kernels/call`, exact tie suite passed.

## Open Hypotheses or Checks

- Accepted H-004, `two-stage-routing-fusion`: stage-one Triton softmax/group-score kernel, exact `torch.topk(group_scores,4)`, stage-two Triton group_idx-dependent mask kernel, exact `torch.topk(masked_scores,8)`. Verifier evidence supports the >=5% wall hypothesis and exact four-case tie correctness.
- Rejected families: custom final selection and static-priority repair (`topk-tie-ordering-active-set-mismatch`); one-stage post-selection masking (`post-selection-mask-requires-second-stage`).
- Round 005 abort: do not target the 8.76669921875 us/call int32 copy without matched proof of a new semantics-preserving mechanism and a credible >=5% wall causal chain.
- Preserve accepted Host Plan: per-forward scores/group_scores/masked_scores allocations, no cross-call cache, input-device allocation, caller-current-stream launches, same-stream torch.topk consumption, distinct buffers for concurrency, and baseline fallback for non-target shapes/scoring modes.
- Do not require or introduce `tl.dot`, `num_warps`, `num_stages`, `fast_libentry`, block pointers, mixed precision, arbitrary layouts, non-contiguous inputs, custom repeated argmax ordering, or unproven integer-transfer semantics.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `base.py` | `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5` | 000 |
| `baseline_adapter.py` | `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016` | 004 |
| `rounds/report_000.md` | `39a512eed23f1f0889e7845cde5f854cf0c2ca9d377ff23588148f239139f1e5` | 004 |
| `rounds/decision_001.md` | `7e899d6cee2ad8fe6ab586b902ff0d18226e77d7d3cfb3ecf791b572e2371365` | 005 |
| `rounds/decision_002.md` | `d3c0f316945706acaad5c6f68ae0d93e9bbf3c848ca735b20c2356b304107d37` | 005 |
| `rounds/coder_result_002.md` | `pending ledger hash` | 005 |
| `triton_grouped_topk_002.py` | `pending ledger hash` | 005 |
| `rounds/decision_003.md` | `dfe241e2b7b6f2609a3d59185d2d067072b986e9316f0b1e857a4023d0ac5030` | 005 |
| `rounds/coder_result_003.md` | `pending ledger hash` | 005 |
| `rounds/decision_004.md` | `307f4a03c15b08daca8bb571f0391418997a07d864ff357b9f2d113cf2fb8f65` | 005 |
| `triton_grouped_topk_004.py` | `881a549cf95746dda93ee4c898e7ab0e67e3133a526088553091f8b8d7431d83` | 005 |
| `rounds/coder_result_004.md` | `aba12a62645a7d789a9a172fb55efe0eac86784c5c81ab86fa29c8d069c79e1c` | 005 |
| `rounds/report_004.md` | `40400c3764ebcbf3825ba0530a59e3f7d081e5728dfd5068a64c26476874cd23` | 005 |
| `rounds/decision_005.md` | `ce1e0f7808982c90273959eb6b07783925085bd63f5d959bd5e810a740d6160e` | 005 |
| `rounds/decision_006.md` | `4da137fb5d59463663e71b08fefa5421f8377560396538c8759b33cec53045bf` | 006 |
| `rounds/decision_007.md` | `e51cb64103bc8b5de6f16bd9fe7bf7a3cd3a502986f43a9b27d22280c5d107c2` | 007 |
| `rounds/decision_008.md` | `bec59b81693001fd27302a610ab48123e38a4a81c44b65cedfff9530b059e5d1` | 008 |
| `triton_grouped_topk_008.py` | `d1fb6b03d3be92cdd6423f1f44f33ea81d13f0e4df18227fe2d5f7dceb582535` | 008 |
| `rounds/coder_result_008.md` | `255ac5a2a36a162e17701f91233d258895790b144d711fd8c9540ed3bd4dae94` | 008 |
| `rounds/report_008.md` | `f1fa38cef46804c96be6c4eb3f5eddeaa7ec509830dae6f2bea58f8be0e2b3b7` | 008 |
| `rounds/decision_009.md` | `066045e737fa1aedcc283c4058d2eceb28b8630013c7b93342abdb516af908b8` | 009 |
