# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `6`
- last_completed_round: `008`
- accepted_kernel: `triton_grouped_topk_008.py`
- accepted_report: `rounds/report_008.md`
- recent_three_round_evidence: `Round 004 two-stage candidate was accepted at 7.455430192% paired wall improvement. Rounds 005 to 007 rejected the exhausted direct-kernel path without a falsifiable five-percent mechanism. Round 008 retained the two-stage Triton algorithm and exact library top-k calls, added constrained torch.compile dispatch, and was accepted at 19.987917795% paired median improvement.`
- open_hypotheses: `The compiled two-stage candidate is canonical. Future changes must begin from triton_grouped_topk_008.py, preserve exact library top-k tie behavior, compiled callable lifecycle, per-forward buffer ownership, caller device/current stream, and fallback behavior.`
- artifact_read_hashes: `baseline_adapter.py, project.md, team-state.md, decision_004.md, triton_cuda.md, invariants.md, coder.md, coder_result_004.md, report_004.md, and triton_grouped_topk_004.py read for Round 004.`

## Current Bottleneck

- `Retained library top-k gather and bitonic sort remain dominant at 49.37 and 37.02 us/call in the accepted compiled candidate; future changes must preserve exact torch.topk ordering and compiled graph lifecycle.`

## Recent Three-round Evidence

- Round 002, `rounds/coder_result_002.md`: custom selector failed structured group-tie IDs.
- Round 003, `rounds/coder_result_003.md`: one-kernel post-selection mask was a major-deviation dependency.
- Round 004, `rounds/report_004.md`: two-stage candidate accepted; device time `127.260771484375 us/call`, kernel count `9.9/call`, wall improvement `7.455430192%`.
- Round 008, `rounds/report_008.md`: compiled two-stage candidate accepted; device time `111.120595703125 us/call`, kernel count `8.96/call`, wall improvement `19.987917795%`.

## Open Hypotheses or Checks

- Preserve exact library torch.topk group/final ordering, per-forward buffer ownership, current device/stream, and fallback behavior.
- Any future candidate starts from `triton_grouped_topk_004.py`, not the failed Round 002 source.
- If later timing misses the 5% threshold, retain the accepted candidate and classify no-improvement.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `baseline_adapter.py` | `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016` | 004 |
| `rounds/decision_004.md` | `307f4a03c15b08daca8bb571f0391418997a07d864ff357b9f2d113cf2fb8f65` | 004 |
| `triton_grouped_topk_004.py` | `881a549cf95746dda93ee4c898e7ab0e67e3133a526088553091f8b8d7431d83` | 004 |
| `rounds/coder_result_004.md` | `aba12a62645a7d789a9a172fb55efe0eac86784c5c81ab86fa29c8d069c79e1c` | 004 |
| `rounds/report_004.md` | `40400c3764ebcbf3825ba0530a59e3f7d081e5728dfd5068a64c26476874cd23` | 004 |
| `rounds/decision_008.md` | `bec59b81693001fd27302a610ab48123e38a4a81c44b65cedfff9530b059e5d1` | 008 |
| `triton_grouped_topk_008.py` | `d1fb6b03d3be92cdd6423f1f44f33ea81d13f0e4df18227fe2d5f7dceb582535` | 008 |
| `rounds/coder_result_008.md` | `255ac5a2a36a162e17701f91233d258895790b144d711fd8c9540ed3bd4dae94` | 008 |
| `rounds/report_008.md` | `f1fa38cef46804c96be6c4eb3f5eddeaa7ec509830dae6f2bea58f8be0e2b3b7` | 008 |
