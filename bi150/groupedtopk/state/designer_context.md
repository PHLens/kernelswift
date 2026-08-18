# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `1`
- last_completed_round: `000`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Round 000 baseline established on BI150; reference median 0.474612 ms, baseline adapter median 0.474995 ms; correctness passed; device profiler available.`
- open_hypotheses: `Use the baseline profiler's top-k gather and bitonic-sort evidence to select one falsifiable >=5% optimization hypothesis; do not infer a kernel change before reading the exact CUDA profile.`
- artifact_read_hashes: `base.py, baseline_adapter.py, project.md, team-state.md, report_000.md, and triton_cuda.md recorded in the Phase 0 ledger.`

## Current Bottleneck

- Verifier-backed: baseline profiler reports top-k gather at 48.7290625 us/call and bitonic sort at 36.879697265625 us/call; device time is available on BI150.

## Recent Three-round Evidence

- Round 000, baseline, `rounds/report_000.md`: reference device time 177.181318359375 us/call; baseline adapter device time 179.0703515625 us/call; 14.8 versus 14.96 kernels/call.

## Open Hypotheses or Checks

- Any next kernel hypothesis must preserve grouped top-k semantics, output dtypes, top-k tie behavior, and the CUDA/BI150 target profile.
- Any host or allocation hypothesis requires a complete Host Plan with cache key, invalidation, device/stream, and concurrency semantics.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `base.py` | `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5` | 000 |
| `baseline_adapter.py` | `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016` | 000 |
| `rounds/report_000.md` | `pending-local-commit` | 000 |
