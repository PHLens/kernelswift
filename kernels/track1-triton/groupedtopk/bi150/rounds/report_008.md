# Report 008

Result: accepted

## Identity

- Round: `008`
- Decision: `rounds/decision_008.md`
- Decision SHA-256: `bec59b81693001fd27302a610ab48123e38a4a81c44b65cedfff9530b059e5d1`
- Candidate: `triton_grouped_topk_008.py`
- Candidate SHA-256: `d1fb6b03d3be92cdd6423f1f44f33ea81d13f0e4df18227fe2d5f7dceb582535`
- Accepted reference: `triton_grouped_topk_004.py`
- Accepted reference SHA-256: `881a549cf95746dda93ee4c898e7ab0e67e3133a526088553091f8b8d7431d83`
- Decision source report: `rounds/report_004.md`
- Base SHA-256: `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5`
- Harness SHA-256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Measurement fingerprint: `57bf01d317ee03ca2b09730e648f0f93d2bf4f226639ca3af2b1ff57b2865575`
- verification_tier: `full`

## Correctness and Guardrails

| Check | Observation | Verdict |
|---|---|---|
| Unchanged harness | Accepted wrapper versus candidate passed `auto_bench.py --warmup 5 --repeat 10 --full-traceback`; v0 `0.444320 ms`, v1 `0.355490 ms`. | pass |
| All-equal ties | Exact IDs `[7,6,4,5,1,0,2,3]`; weights allclose. | pass |
| Two-expert tie | Exact IDs `[1,0,2,3,4,5,6,7]`; weights allclose. | pass |
| Structured group tie | Exact IDs `[32,0,64,96,4,3,1,2]`; weights allclose. | pass |
| Selection ownership | Candidate retains both exact library `torch.topk` calls and the accepted two direct Triton stages. | pass |
| Compile lifecycle | Constructor-owned `torch.compile` callable only dispatches for the accepted fixed regime; exceptions/non-target inputs fall back to accepted eager behavior. | pass |
| Device/stream/aliasing | Per-forward buffers remain distinct on `gating_output.device`; no stream/context mutation or output cache is introduced. | pass |

## Interleaved Wall Timing

The accepted reference was exposed to the v0-only harness through a temporary
remote wrapper that renames only `ModelNew` to `Model`; wrapper SHA-256:
`7b2c36de7ae3d1a8e789bd520de557401bc5513eb690668aca9d45922a90bb45`.
The wrapper was removed after collection.

- warmup: `50`
- repeat: `100`
- reference_raw_samples_ms: `[0.430385, 0.438276, 0.411169]`
- candidate_raw_samples_ms: `[0.344360, 0.347371, 0.327867]`
- reference_median_ms: `0.430385`
- candidate_median_ms: `0.344360`
- improvement_pct: `19.987917795`
- adoption_threshold_pct: `5`
- verdict: `pass`

The unrounded paired median improvement exceeds the frozen threshold.

## Evaluation Contract Mirror

- hypothesis_id: `H-008`
- intervention: `torch.compile` dispatch for the accepted fixed-shape two-stage forward
- hypothesis verdict: `supported`
- primary metric: `unrounded paired median wall_time_ms`
- mechanism outcome: compiled dispatch reduces wall time without changing the accepted direct-kernel/top-k dataflow.

## Profiler Evidence

- profiler_level: `targeted`
- iterations: `50`
- trace: `log/groupedtopk_round008_forward_50iter.pt.trace.json`
- trace_sha256: `bfd01278b7487ec053467d677ac0e912a089db1d521be8b7614505e65af2910f`
- device time: `available`

| Scope | Device us/call | Kernels/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|
| reference_triton_grouped_topk_004 | 127.45888671875 | 9.9 | 0.430385 | 0.2961508573 |
| candidate_triton_grouped_topk_008 | 111.120595703125 | 8.96 | 0.344360 | 0.3226872915 |

The raw trace contained overlapping CPU/GPU reference annotations. Its
reference summary removed only the duplicate CPU `user_annotation`, retaining
the GPU annotation. The compiled candidate has only its CPU harness annotation,
which unambiguously encloses compiled graph kernels; its candidate summary
removed only the reference duplicate. The raw trace remains authoritative.

### Candidate Top Kernels

| Kernel | Count/call | Us/call |
|---|---:|---:|
| `at::native::sbtopk::gatherTopK` | 2.0 | 49.368779296875 |
| `at::native::bitonicSortKVInPlace` | 1.98 | 37.02447265625 |
| `_softmax_group_scores_kernel` | 1.0 | 7.250966796875 |
| `_group_mask_kernel` | 0.98 | 5.36640625 |
| `triton_poi_fused__to_copy_1` | 0.98 | 3.925068359375 |
| `triton_per_fused_div_sum_2` | 0.98 | 3.816435546875 |
| `triton_poi_fused_0` | 0.98 | 3.807412109375 |

The compiled graph replaces accepted eager-framework elementwise/reduction
launches with fused Triton kernels, reducing device time by
`16.338291015625 us/call` and kernel count by `0.94/call` while retaining the
exact library top-k selection path.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 008 resets failure counters and establishes a new canonical baseline; the campaign remains below its configured round budget.

## Exact Reproduction Commands

```bash
cd /root/kernelswift-bi150
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
sed 's/^class ModelNew/class Model/' kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_004.py > /tmp/bi150_accepted_model_008.py
python3 auto_bench.py --v0_file /tmp/bi150_accepted_model_008.py --v1_file kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_008.py --warmup 50 --repeat 100
rm -f /tmp/bi150_accepted_model_008.py
```

```bash
cd /root/kernelswift-bi150
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
sed 's/^class ModelNew/class Model/' kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_004.py > /tmp/bi150_accepted_model_008.py
python3 auto_bench.py --v0_file /tmp/bi150_accepted_model_008.py --v1_file kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_008.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_004.py --profile-output kernels/track1-triton/groupedtopk/bi150/log/groupedtopk_round008_forward_50iter.pt.trace.json
rm -f /tmp/bi150_accepted_model_008.py
```
