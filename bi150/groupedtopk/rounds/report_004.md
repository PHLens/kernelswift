# Report 004

Result: accepted

## Identity

- Round: `004`
- Decision: `rounds/decision_004.md`
- Decision SHA-256: `307f4a03c15b08daca8bb571f0391418997a07d864ff357b9f2d113cf2fb8f65`
- Candidate: `triton_grouped_topk_004.py`
- Candidate SHA-256: `881a549cf95746dda93ee4c898e7ab0e67e3133a526088553091f8b8d7431d83`
- Accepted reference: `baseline_adapter.py`
- Accepted reference SHA-256: `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016`
- Base SHA-256: `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5`
- Harness SHA-256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `57bf01d317ee03ca2b09730e648f0f93d2bf4f226639ca3af2b1ff57b2865575`
- verification_tier: `full`
- screening_pairs: `base.py` versus candidate passed three corrected pairs before canonical-adapter timing.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| Harness correctness | unchanged harness accuracy pass | `PASS accuracy`; v0 `0.475544 ms`, v1 `0.440146 ms` with warmup 5/repeat 10. | pass | BI150 `auto_bench.py --full-traceback` |
| All-equal ties | exact IDs and floating tolerance | Reference and candidate IDs `[7,6,4,5,1,0,2,3]`; weights allclose. | pass | Independent verifier tie suite |
| Two-expert tie | exact IDs and floating tolerance | Reference and candidate IDs `[1,0,2,3,4,5,6,7]`; weights allclose. | pass | Independent verifier tie suite |
| Structured group tie | exact IDs and floating tolerance | Reference and candidate IDs `[32,0,64,96,4,3,1,2]`; weights allclose. | pass | Independent verifier tie suite |
| Selection ownership | exact library group and final selection | Candidate uses `torch.topk(group_scores,4)` and `torch.topk(masked_scores,8)`; neither direct kernel performs selection. | pass | Candidate source and verifier ties |
| Device/stream/lifecycle | current device/stream and no aliasing | Per-forward scores/group_scores/masked_scores are allocated on `gating_output.device`; no persistent cache or global state. | pass | Candidate source, Coder Host Plan conformance |

## Screening Evidence

Corrected base-vs-candidate screening pairs (`warmup=50`, `repeat=100`) were:

| Pair | Base ms | Candidate ms | Speedup |
|---:|---:|---:|---:|
| 1 | 0.462277 | 0.428671 | 1.078x |
| 2 | 0.465368 | 0.429210 | 1.084x |
| 3 | 0.465255 | 0.431601 | 1.078x |

The initial attempted baseline-adapter-as-v0 invocation was rejected by the
unchanged harness before any forward because v0 requires `Model`; it produced
no timing sample and was discarded. Canonical timing below uses a temporary
remote v0 wrapper created by renaming only `ModelNew` to `Model` in the
canonical adapter, with no behavior or body changes. Its SHA-256 was
`265dc639a9de1427d7f26181c4d5a487a23c457e59b2de1a2a654afb6750b60d`.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `three ordered canonical-adapter/candidate harness pairs`
- canonical adapter wrapper: remote temporary `sed 's/^class ModelNew/class Model/' baseline_adapter.py`; removed after collection
- reference_raw_samples_ms: `[0.470285, 0.466908, 0.466055]`
- candidate_raw_samples_ms: `[0.432791, 0.430251, 0.432098]`
- reference_median_ms: `0.466908`
- candidate_median_ms: `0.432098`
- improvement_pct: `7.455430192`
- adoption_threshold_pct: `5`
- verdict: `pass`

The unrounded paired median improvement exceeds the frozen adoption threshold.

## Evaluation Contract Mirror

- hypothesis_id: `H-004`
- intervention: `two direct Triton stages around unchanged library group/final torch.topk selection`
- expected_causal_chain: `fused softmax/group reduction and post-group mask reduce preprocessing/masking work while retained library top-k preserves active-set-dependent tie behavior`
- primary_metric: `unrounded interleaved median wall_time_ms`
- mechanism_observables: `kernel_count_per_call`, `device_us_per_call`, direct stage-one/stage-two kernels
- Hypothesis verdict: `supported`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available: BI150 trace contains cat=kernel CUDA device-duration events`
- iterations: `50`
- trace: `log/groupedtopk_round004_forward_50iter.pt.trace.json`
- trace_sha256: `4fa1253d085e95f68c0e2029ab1b1a7946ce354c7a8174e7d6622a09db60bd5e`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_baseline_adapter | 8949.56298828125 | 178.991259765625 | 743 | 14.86 | 0.466908 | 0.3833544505 |
| candidate_triton_grouped_topk_004 | 6363.03857421875 | 127.260771484375 | 495 | 9.9 | 0.432098 | 0.2945183072 |

The raw trace contains overlapping CPU and GPU candidate annotations with the
same label. The candidate summary removed only the overlapping CPU
`user_annotation` record in a temporary derived JSON and scoped to the
non-overlapping GPU annotation; the raw trace above remains authoritative.

### Reference Top Kernels

| Kernel | Count/call | Us/call |
|---|---:|---:|
| `at::native::sbtopk::gatherTopK` | 1.98 | 48.91419921875 |
| `at::native::bitonicSortKVInPlace` | 1.98 | 37.42125 |
| `at::native::reduce_kernel MaxOps` | 1.0 | 18.417490234375 |
| `at::native::reduce_kernel sum` | 0.98 | 14.0286328125 |
| `at::native::elementwise_kernel direct_copy` | 1.0 | 10.024609375 |

### Candidate Top Kernels

| Kernel | Count/call | Us/call |
|---|---:|---:|
| `at::native::sbtopk::gatherTopK` | 1.98 | 48.852978515625 |
| `at::native::bitonicSortKVInPlace` | 1.98 | 36.45123046875 |
| `at::native::reduce_kernel sum` | 0.98 | 13.679365234375 |
| `at::native::direct_copy int32` | 1.98 | 8.76669921875 |
| `_softmax_group_scores_kernel` | 1.0 | 7.1426171875 |
| `at::native::elementwise_kernel div` | 0.98 | 7.0394140625 |
| `_group_mask_kernel` | 1.0 | 5.328466796875 |

The retained library top-k kernels remain the primary device contributors, as
required for exact tie semantics. The two direct kernels replace the baseline
softmax, group reduction, mask expansion, boolean inversion, and masked-fill
path sufficiently to reduce total device time by `51.73048828125 us/call` and
kernel count by `4.96` per forward call.

## Retry History

No Verifier repair was needed. The Coder handoff was independently verified on
BI150 before timing and profiling.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 004 is accepted and resets both failure streaks; the campaign remains below its round budget.

## Exact Reproduction Commands

```bash
cd /root/kernelswift-bi150
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
python3 auto_bench.py --v0_file bi150/groupedtopk/base.py --v1_file bi150/groupedtopk/triton_grouped_topk_004.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift-bi150
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
sed 's/^class ModelNew/class Model/' bi150/groupedtopk/baseline_adapter.py > /tmp/bi150_baseline_model_004.py
python3 auto_bench.py --v0_file /tmp/bi150_baseline_model_004.py --v1_file bi150/groupedtopk/triton_grouped_topk_004.py --warmup 50 --repeat 100
rm -f /tmp/bi150_baseline_model_004.py
```

```bash
cd /root/kernelswift-bi150
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
python3 auto_bench.py --v0_file bi150/groupedtopk/base.py --v1_file bi150/groupedtopk/triton_grouped_topk_004.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file bi150/groupedtopk/baseline_adapter.py --profile-output bi150/groupedtopk/log/groupedtopk_round004_forward_50iter.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_trace.py bi150/groupedtopk/log/groupedtopk_round004_forward_50iter.pt.trace.json --iterations 50 --scope reference_baseline_adapter --wall-ms 0.466908
```
