# Designer Context State

- role_contract_sha256: `<computed-at-first-use>`
- context_epoch: 1
- last_completed_round: "002"
- accepted_kernel: `triton_grouped_topk_002.py`
- accepted_report: `rounds/report_002.md`
- recent_three_round_evidence: `R002 accepted; allocation-reuse removed 2 per-call torch.empty allocations -> host 291.743->232.086 us/call (-20.45%), wall 0.326705->0.267220 ms (+18.21%); device flat 35.134 us/call; device_ratio 0.131 => host-bound`
- open_hypotheses: `none — R003 halted (abort)`
- artifact_read_hashes: `decision_002 a3b8aebf; report_002 9315412c(candidate)/b7b47d1f(ref); base 12f3324; decision_003 (written this round)`

## Current Bottleneck

- `host-bound`: device_ratio 0.131 (device 35.134 us/call vs wall 0.267220 ms). Remaining ~232 us/call host cost is the Triton launch/dispatch path, of which ~107 us is measured fixed launch overhead independent of kernel size. Allocation reuse exhausted (0 steady-state allocations). Launcher primitives (`fast_libentry`, stream/context) are `Unknown` on this runtime; direct launch already proven. No falsifiable >=5% intervention remains.

## Recent Three-round Evidence

- `R001 accepted | report_001.md | kernel-fusion | candidate triton_grouped_topk_001.py; 1.0 kernel/call, device 34.634 us/call, wall 0.321620 ms, +54.88%, device_ratio 0.108`
- `R002 accepted | report_002.md | allocation-reuse | candidate triton_grouped_topk_002.py; device 35.134 us/call, wall 0.267220 ms, +18.21%, device_ratio 0.131; host 232.086 us/call`
- `R003 aborted | decision_003.md | no-change | halt: remaining host cost is fixed backend launch/dispatch overhead`

## Open Hypotheses or Checks

- `H-002 allocation-reuse: CONFIRMED and exhausted (0 steady-state allocations)`
- `H-003 (rejected) launcher-path reduction: fast_libentry/stream/context Unknown on Ascend; direct launch already proven; no implementable lever`
- `H-004 (rejected) device-side work: 13.1% of wall; bounded below 5% after fixed launch overhead; anti-patterns show device selection regressions`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58` | 002 |
| `triton_grouped_topk_002.py` | `9315412c0dd7e2c56a6ce16924c74d7fbf0c4872edab454dd51ce2f62e91413f` | 003 |
| `rounds/report_002.md` | (candidate ref `b7b47d1f...`) | 003 |
| `rounds/decision_002.md` | `a3b8aebf92a887ec07def2f9a3f804726db620b37f9b6e9f7bb7bbaba6aebf78` | 002 |
