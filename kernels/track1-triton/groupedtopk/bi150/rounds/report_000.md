# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016`
- Accepted reference SHA256: `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5`
- Base SHA256: `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `57bf01d317ee03ca2b09730e648f0f93d2bf4f226639ca3af2b1ff57b2865575`
- verification_tier: `baseline`
- screening_pairs: `not-applicable: Phase 0`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `base.py` and `baseline_adapter.py` outputs matched; floating outputs passed `atol=1e-2, rtol=1e-2` and integer IDs matched exactly. | pass | Remote auto_bench smoke: v0 `0.485843` ms, v1 `0.479700` ms, exit 0 |
| immutable base | base bytes unchanged by adapter generation | Base SHA-256 was `d57ace7d...` before and after adapter generation. | pass | `make_baseline_adapter.py` and local/remote SHA-256 ledger |
| CUDA/BI150 runtime | selected profile matches runtime | CoreX bootstrap succeeded; `torch.cuda` reports one `Iluvatar BI-V150`, capability `[7,1]`, 16 processors, 16 GiB; Triton vector-add smoke passed with max error `0.0`. | pass | Runtime discovery and `scripts/bi150_triton_smoke.py` |
| harness AST loader | reference and generated adapter load through the unchanged harness | Correctness command completed with `PASS accuracy` and no traceback. | pass | Remote `auto_bench.py --full-traceback` command |

## Screening Evidence

Not applicable to Phase 0.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `ordered reference/candidate pairs; each pair uses the unchanged harness`
- reference_raw_samples_ms: `[0.473436, 0.479897, 0.474612]`
- candidate_raw_samples_ms: `[0.472600, 0.479476, 0.474995]`
- reference_median_ms: `0.474612`
- candidate_median_ms: `0.474995`
- improvement_pct: `not-applicable: baseline adapter is the executable canonical baseline`

The baseline adapter is retained as canonical because it is generated from the
immutable base, not because its small timing difference is an optimization claim.
The raw pairs were collected in three remote BI150 invocations with identical
CoreX environment and harness flags.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `not-applicable: Phase 0`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available: BI150 trace contains cat=kernel CUDA device-duration events`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/groupedtopk_baseline_forward_50iter.pt.trace.json`
- trace_sha256: `0ed6dfa64748d1226baac93d0cd32ec4f16c0b64555b3f16022ef103efc77af`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| baseline_base | 8859.06591796875 | 177.181318359375 | 740 | 14.8 | 0.474612 | 0.3733182439 |
| candidate_baseline_adapter | 8953.517578125 | 179.0703515625 | 748 | 14.96 | 0.474995 | 0.3769941822 |

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `at::native::sbtopk::gatherTopK` | 99 | 1.98 | 2436.453125 | 48.7290625 |
| `at::native::bitonicSortKVInPlace` | 99 | 1.98 | 1843.98486328125 | 36.879697265625 |
| `at::native::reduce_kernel MaxOps` | 50 | 1.0 | 917.17138671875 | 18.343427734375 |
| `at::native::reduce_kernel sum` | 49 | 0.98 | 695.56298828125 | 13.911259765625 |
| `at::native::elementwise_kernel direct_copy` | 49 | 0.98 | 483.3857421875 | 9.66771484375 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `at::native::sbtopk::gatherTopK` | 100 | 2.0 | 2469.0205078125 | 49.38041015625 |
| `at::native::bitonicSortKVInPlace` | 99 | 1.98 | 1848.04052734375 | 36.960810546875 |
| `at::native::reduce_kernel MaxOps` | 50 | 1.0 | 919.9677734375 | 18.39935546875 |
| `at::native::reduce_kernel sum` | 50 | 1.0 | 708.4453125 | 14.16890625 |
| `at::native::elementwise_kernel direct_copy` | 50 | 1.0 | 496.93603515625 | 9.938720703125 |

The baseline and generated adapter have equivalent execution structure. The
profiler shows the top-k gather and bitonic sort paths dominate device time;
these are observations for the next Designer round, not a prescribed change.

## Retry History

No retries; Phase 0 baseline gate passed.

## evidence_for_next_round

- The BI150 CUDA profiler exposes attributable `cat=kernel` events, so device time and kernel-count evidence are available for future candidate rounds.
- Top-k gather and bitonic sort are the largest observed device-time contributors in the grouped-top-k baseline.
- Baseline correctness, harness loading, runtime fingerprint, and measurement fingerprint are established on `cuda:0`.

## Stop Recommendation

- recommendation: `continue`
- evidence: baseline is established and no candidate round has been evaluated yet

## Exact Reproduction Commands

```bash
cd /root/kernelswift-bi150
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/bi150/base.py --v1_file kernels/track1-triton/groupedtopk/bi150/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift-bi150
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/bi150/base.py --v1_file kernels/track1-triton/groupedtopk/bi150/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-bi150
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/bi150/base.py --v1_file kernels/track1-triton/groupedtopk/bi150/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/groupedtopk/bi150/log/groupedtopk_baseline_forward_50iter.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150/log/groupedtopk_baseline_forward_50iter.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 0.474612
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150/log/groupedtopk_baseline_forward_50iter.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 0.474995
```
