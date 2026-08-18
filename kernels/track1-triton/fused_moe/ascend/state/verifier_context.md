# Verifier Context State

- role_contract_sha256: `<computed-at-first-use>`
- context_epoch: 1
- last_completed_round: 000
- accepted_kernel: baseline_adapter.py
- accepted_report: rounds/report_000.md
- recent_three_round_evidence: `<Phase 0 baseline established; no optimization rounds yet>`
- open_hypotheses: `<next: establish baseline bottleneck hypothesis from report_000 profiler evidence>`
- artifact_read_hashes: `<base.py, baseline_adapter.py, auto_bench.py hashes verified>`

## Current Bottleneck

- Baseline wall time ~7.16 ms (median, 100 repeats), with 126 kernels per forward
  call and device_ratio ~0.104 (~10% of wall time is AI Core device time). The
  dominant device-time families are the per-expert mask/gather/scatter
  (`aclnnNonzeroV2` ~337 us/call, `aclnnIndex` ~82 us/call,
  `aclnnIndexPutImpl` ~57 us/call) plus 16 MatMul launches, driven by the Python
  for-loop over 8 experts. Remaining ~90% of wall time is host-side launch and
  Python overhead.

## Recent Three-round Evidence

- Round 000 (baseline): wall median 7.159420 ms (candidate) / 7.158795 ms
  (reference), device 743.517 us/call, 126 kernels/call. No optimization rounds
  yet.

## Open Hypotheses or Checks

- Baseline bottleneck: host-side launch overhead + small-kernel scatter/gather
  dominates; candidate optimizations should target kernel fusion / reduced
  launch count.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b` | 000 |
| `baseline_adapter.py` | `a7fc0001db3ee9e636241954d2c071b62acee518b23f4c59c19efee886203a02` | 000 |
| `../../../../auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 000 |
