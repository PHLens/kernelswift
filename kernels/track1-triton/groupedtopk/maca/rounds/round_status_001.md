# Round Status 001

- phase: `verification-complete`
- result: `accepted`
- decision: `rounds/decision_001.md`
- candidate: `triton_grouped_topk_001.py`
- accepted_reference: `baseline_adapter.py`
- harness_v0_for_timing: `base.py` (normalized-AST-equivalent to the accepted adapter except the required `Model`/`ModelNew` class name)
- measurement_fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- timing_order: `sequential complete base-reference block, then complete candidate block`
- required_pre_timing_targeted_gate: `group-cutoff and expert-cutoff tie-ID parity`

## Artifact Hashes

| Artifact | Local SHA256 | Remote SHA256 | Verdict |
|---|---|---|---|
| `base.py` | `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb` | `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb` | match |
| `baseline_adapter.py` | `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827` | `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827` | match |
| `triton_grouped_topk_001.py` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | match |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | match |
| `rounds/decision_001.md` | `c71c970e3bcf6d7984272611627d711ce64b6f3c18d1a057b2aab440c50c173f` | not-applicable | match Coder result |
| `rounds/coder_result_001.md` | `c8d583749759c718429a0ed118d695908de714f89edd12c8121ed44f67d03f65` | not-applicable | read |
| `rounds/report_000.md` | `9b8374ee96d72fa8eed02415440eb778867d9ee0f3d0e8914608695a0c299f00` | not-applicable | accepted evidence |
| `rounds/report_001.md` | `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a` | not-applicable | written |
| `state/verifier_context.md` | `af563d7b591556ddc1b5e6f6c810311d509d4914720f5a1386d71fd3dfd0e32c` | not-applicable | written |

## Completed Commands

1. Read latest Round 001 durable inputs: exit `0`.
2. Local SHA256 verification: exit `0`.
3. Remote executable-file SHA256 verification over the existing SSH control connection: exit `0`.
4. Frozen standard correctness (`warmup=5`, `repeat=10`, `--full-traceback`): exit `0`; `PASS accuracy`; summary `1 passed, 0 failed, 1 total`; evidence `log/correctness_001.log`.
5. Intended SCP of `log/tie_id_parity_001.py`: not started; execution policy rejected the transfer before process creation; no remote side effect; evidence `log/tie_script_transfer_001.log`.
6. Orchestrator-assisted approved script upload followed by remote SHA256 check: exit `0`; local and remote `8808ce0dee5004a62008cedb158f92f5f4a6d0ac06641901424536fed24726bf`.
7. Targeted tie-ID parity, two exact fast-path cases: exit `0`; both `PASS`; `overall=PASS`; evidence `log/tie_id_parity_001.log` and script `log/tie_id_parity_001.py`.
8. Formal wall sample 1 (`warmup=200`, `repeat=500`): exit `0`; accepted-reference proxy `0.223698 ms`; candidate `0.068280 ms`; SSH elapsed `15.1700544 s`; evidence `log/wall_001_sample_1.log`.
9. Formal wall sample 2 (`warmup=200`, `repeat=500`): exit `0`; accepted-reference proxy `0.224533 ms`; candidate `0.068671 ms`; SSH elapsed `15.4756862 s`; evidence `log/wall_001_sample_2.log`.
10. Formal wall sample 3 (`warmup=200`, `repeat=500`): exit `0`; accepted-reference proxy `0.225974 ms`; candidate `0.067233 ms`; SSH elapsed `15.0783929 s`; evidence `log/wall_001_sample_3.log`.
11. Separately scoped forward profiler (`--profile-reference-file baseline_adapter.py`, `20/100`): exit `0`; evidence `log/profile_001.log` and raw trace.
12. Raw summaries for `reference_baseline_adapter` and `candidate_triton_grouped_topk_001`: exit `1/1`; known duplicate nested GPU scope markers; raw trace retained.
13. Exact Round 001 duplicate-marker filter: exit `0`; removed exactly 2 GPU scope markers; kernel events remained `1600`, cuda_runtime remained `3002`.
14. Derived summaries with unmodified repository script: exit `0/0`; all six Evaluation Contract observables present and passing.

## Raw Samples

- standard correctness: `pass`; smoke-only timing `base=0.230905 ms`, `candidate=0.070167 ms`; evidence `log/correctness_001.log`
- targeted tie parity: `pass`; group-cutoff IDs ref/candidate `[0,32,64,96,1,2,3,4]`; expert-cutoff IDs ref/candidate `[0,32,64,96,1,33,65,2]`; both fast-path probes passed
- reference wall samples ms: `[0.223698, 0.224533, 0.225974]`; median `0.224533`
- candidate wall samples ms: `[0.068280, 0.068671, 0.067233]`; median `0.068280`
- unrounded improvement_pct: `69.59021613749428`
- speedup from medians: `3.288415348564734x`
- raw trace SHA256: `c3db406b8bd6213a56cc4fe92977e6ed24c378702f2dc2a139d69536dbbea1e8`
- derived trace SHA256: `67a675bdd50280c165d46bbec5bb06af9e8b693f19c807cbdfc5efdf3d744b36`
- accepted reference profile: `147.591396484375 us/call`, `15.0 kernels/call`, device ratio `0.6573260789477493`
- candidate profile: `10.7442822265625 us/call`, `1.0 kernel/call`, device ratio `0.15735621304280173`
- candidate kernel: `_grouped_topk_fixed_kernel`, `1.0/call`
- candidate gatherTopK plus bitonicSort: `0 us/call`
- device-time reduction vs concurrent accepted reference: `92.72025166609224%`
- hypothesis verdict: `confirmed`

## Next Safe Action

Orchestrator validates `rounds/report_001.md`, advances canonical pointers to the accepted candidate/report, updates counters/state, and releases measurement exclusivity. Verifier must not mutate canonical state.
