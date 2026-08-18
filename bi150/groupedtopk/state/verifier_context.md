# Verifier Context State

- role_contract_sha256: `6bc3277527523950cab2c9295b07e6a5dc66030618c3b7e648c69a095a48f0a2`
- context_epoch: `3`
- last_completed_round: `009`
- accepted_kernel: `triton_grouped_topk_009.py`
- accepted_report: `rounds/report_009.md`
- recent_three_round_evidence: `Round 007 aborted (persistent-no-falsifiable-intervention). Round 008 compiled the accepted two-stage forward with torch.compile default mode, passed exact tie suites, and was accepted at 0.344360 ms and 111.120595703125 us/call. Round 009 switched the constructor-owned compiled callable to mode="reduce-overhead", passed exact tie suites, and was accepted at 0.277234 ms (22.510768061134083% improvement vs 008) with a CUDA-Graph-replay attribution caveat in the profiler.`
- open_hypotheses: `The reduce-overhead compiled two-stage candidate is canonical. Preserve exact torch.topk boundaries, torch.compile lifecycle/fallback, current device/stream behavior, and per-forward ownership. Keep raw trace authoritative when CPU/GPU scope annotations overlap.`
- artifact_read_hashes: `base.py, baseline_adapter.py, project.md, team-state.md, report_000.md, report_004.md, report_008.md, round_status_000.md, round_status_004.md, round_status_008.md, decision_009.md, coder_result_009.md, baseline and round 008/009 traces recorded.`

## Current Bottleneck

- The BI150 profiler provides attributable CUDA `cat=kernel` events for eager and default-mode-compiled candidates, but under reduce-overhead CUDA Graph replay the graph-internal kernels collapse to a single `multi_tensor_apply_kernel ... Copy<float,float>` replay event, so `device_us_per_call` / `kernel_count_per_call` are under-counted and cannot be compared 1:1 against a non-graph reference. Wall time is now dominated by the graph-replay device event plus host launch/dispatch of the CUDA Graph.

## Recent Three-round Evidence

- Round 007, aborted: persistent no-falsifiable-intervention.
- Round 008, `rounds/report_008.md`: default-mode compiled candidate accepted with exact tie behavior; paired wall median `0.344360 ms` versus `0.430385 ms`, device `111.120595703125 us/call`, `8.96 kernels/call`.
- Round 009, `rounds/report_009.md`: reduce-overhead compiled candidate accepted; paired wall median `0.277234 ms` versus `0.357771 ms` (improvement `22.510768061134083%`), reference device `109.198896484375 us/call` / `8.78 kernels/call`, candidate attributed device `14.898740234375 us/call` / `1.22 kernels/call` (CUDA Graph replay attribution caveat).

## Open Hypotheses or Checks

- Correctness must precede all candidate timing.
- Future reports must retain separate reference and candidate scopes and normalize profiler totals by 50 forward calls.
- Adoption remains controlled by unrounded wall-time median and the 5% threshold.
- For reduce-overhead (CUDA Graph) candidates, treat `device_us_per_call` / `kernel_count_per_call` as under-attributed; do not compare them 1:1 against an eager or default-mode reference.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `log/groupedtopk_baseline_forward_50iter.pt.trace.json` | `0ed6dfa64748d1226baac93d0cd32ec4f16c0b64555b3f16022ef103efc77af` | 000 |
| `rounds/report_000.md` | `39a512eed23f1f0889e7845cde5f854cf0c2ca9d377ff23588148f239139f1e5` | 000 |
| `rounds/decision_008.md` | `bec59b81693001fd27302a610ab48123e38a4a81c44b65cedfff9530b059e5d1` | 008 |
| `triton_grouped_topk_008.py` | `d1fb6b03d3be92cdd6423f1f44f33ea81d13f0e4df18227fe2d5f7dceb582535` | 009 |
| `rounds/coder_result_008.md` | `255ac5a2a36a162e17701f91233d258895790b144d711fd8c9540ed3bd4dae94` | 008 |
| `rounds/report_008.md` | `f1fa38cef46804c96be6c4eb3f5eddeaa7ec509830dae6f2bea58f8be0e2b3b7` | 009 |
| `log/groupedtopk_round008_forward_50iter.pt.trace.json` | `bfd01278b7487ec053467d677ac0e912a089db1d521be8b7614505e65af2910f` | 008 |
| `rounds/decision_009.md` | `066045e737fa1aedcc283c4058d2eceb28b8630013c7b93342abdb516af908b8` | 009 |
| `triton_grouped_topk_009.py` | `9b58f861ef6c3de86577dfe819327895311298cc4edf4b3f514f7fe9f4bff194` | 009 |
| `rounds/coder_result_009.md` | `ad9705c4cb2c7120f593cf0a6e5bd4b4b71438d4066360812755f3dd17e4dd8b` | 009 |
| `log/groupedtopk_round009_forward_50iter.pt.trace.json` | `56dd0b9e1e3a6f772274b2efe0dab087f412f248a749271b8e50ee5f8a2a3036` | 009 |
