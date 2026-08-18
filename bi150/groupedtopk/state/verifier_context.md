# Verifier Context State

- role_contract_sha256: `6bc3277527523950cab2c9295b07e6a5dc66030618c3b7e648c69a095a48f0a2`
- context_epoch: `2`
- last_completed_round: `008`
- accepted_kernel: `triton_grouped_topk_008.py`
- accepted_report: `rounds/report_008.md`
- recent_three_round_evidence: `Round 004 two-stage routing fusion was accepted at 0.432098 ms and 127.260771484375 us/call. Rounds 005 to 007 aborted without a defensible direct-kernel intervention. Round 008 compiled the accepted two-stage forward, passed exact tie suites, and was accepted at 0.344360 ms and 111.120595703125 us/call.`
- open_hypotheses: `The compiled two-stage candidate is canonical. Preserve exact torch.topk boundaries, torch.compile lifecycle/fallback, current device/stream behavior, and per-forward ownership. Keep raw trace authoritative when CPU/GPU scope annotations overlap.`
- artifact_read_hashes: `base.py, baseline_adapter.py, project.md, team-state.md, report_000.md, round_status_000.md, and baseline trace recorded.`

## Current Bottleneck

- The BI150 profiler provides attributable CUDA `cat=kernel` events. The largest compiled-candidate contributors are exact top-k gather at 49.368779296875 us/call and bitonic sort at 37.02447265625 us/call.

## Recent Three-round Evidence

- Round 000, baseline, `rounds/report_000.md`: 177.181318359375 device us/call for base and 179.0703515625 device us/call for the generated adapter; 14.8 and 14.96 kernels/call respectively.
- Round 008, `rounds/report_008.md`: compiled candidate accepted with exact tie behavior; paired wall median `0.344360 ms` versus `0.430385 ms`, device `111.120595703125 us/call`, `8.96 kernels/call`.

## Open Hypotheses or Checks

- Correctness must precede all candidate timing.
- Future reports must retain separate `baseline_base` and candidate scopes and normalize profiler totals by 50 forward calls.
- Adoption remains controlled by unrounded wall-time median and the 5% threshold.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `log/groupedtopk_baseline_forward_50iter.pt.trace.json` | `0ed6dfa64748d1226baac93d0cd32ec4f16c0b64555b3f16022ef103efc77af` | 000 |
| `rounds/report_000.md` | `39a512eed23f1f0889e7845cde5f854cf0c2ca9d377ff23588148f239139f1e5` | 000 |
| `rounds/decision_008.md` | `bec59b81693001fd27302a610ab48123e38a4a81c44b65cedfff9530b059e5d1` | 008 |
| `triton_grouped_topk_008.py` | `d1fb6b03d3be92cdd6423f1f44f33ea81d13f0e4df18227fe2d5f7dceb582535` | 008 |
| `rounds/coder_result_008.md` | `255ac5a2a36a162e17701f91233d258895790b144d711fd8c9540ed3bd4dae94` | 008 |
| `rounds/report_008.md` | `f1fa38cef46804c96be6c4eb3f5eddeaa7ec509830dae6f2bea58f8be0e2b3b7` | 008 |
| `log/groupedtopk_round008_forward_50iter.pt.trace.json` | `bfd01278b7487ec053467d677ac0e912a089db1d521be8b7614505e65af2910f` | 008 |
