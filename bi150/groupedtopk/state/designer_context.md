# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `1`
- last_completed_round: `000`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Round 000 baseline established on BI150; reference median 0.474612 ms, baseline adapter median 0.474995 ms; correctness passed; device profiler available. Round 001 is capability-miss: the profile does not establish reductions, masked/indexed selection, or the row/layout semantics required for a Triton grouped-top-k candidate.`
- open_hypotheses: `Do not write a candidate under triton_cuda until matched BI150/CoreX evidence establishes the exact primitive and semantic envelope required for grouped top-k. A future round may reconsider only after local probes cover group reductions, indexed/masked selection, fp32 row layout, and top-k tie behavior.`
- artifact_read_hashes: `base.py, baseline_adapter.py, project.md, team-state.md, report_000.md, triton_cuda.md, invariants.md, bottleneck-judgment.md, anti-patterns.md, designer.md, and codex.md read for Round 001.`

## Current Bottleneck

- Verifier-backed: baseline profiler reports top-k gather at 48.7290625 us/call and bitonic sort at 36.879697265625 us/call; device time is available on BI150.
- Classification: mixed, with device_ratio 0.3769941822 for the canonical baseline adapter; no Verifier-backed host decomposition supports a host intervention.

## Recent Three-round Evidence

- Round 000, baseline, `rounds/report_000.md`: reference device time 177.181318359375 us/call; baseline adapter device time 179.0703515625 us/call; 14.8 versus 14.96 kernels/call.
- Round 001, decision only, `rounds/decision_001.md`: capability-miss. The selected profile proves only one-dimensional contiguous fp32 load/store/arange/program-id operations and cannot support a normative grouped-top-k fusion specification.

## Open Hypotheses or Checks

- Do not turn the rejected Round 001 Triton fusion into a source baseline or candidate.
- Any next kernel hypothesis must preserve grouped top-k semantics, output dtypes, top-k tie behavior, and the CUDA/BI150 target profile.
- Any host or allocation hypothesis requires a complete Host Plan with cache key, invalidation, device/stream, and concurrency semantics.
- Reconsider a Triton selection kernel only with matched target evidence for every required reduction, masking/indexing, shape/layout, dtype, and tie-ordering requirement. Unknown primitives remain capability-miss when normative.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `base.py` | `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5` | 000 |
| `baseline_adapter.py` | `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016` | 001 |
| `rounds/report_000.md` | `39a512eed23f1f0889e7845cde5f854cf0c2ca9d377ff23588148f239139f1e5` | 001 |
| `rounds/decision_001.md` | `7e899d6cee2ad8fe6ab586b902ff0d18226e77d7d3cfb3ecf791b572e2371365` | 001 |
