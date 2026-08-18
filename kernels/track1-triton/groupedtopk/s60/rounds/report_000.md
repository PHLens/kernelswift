# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `6713aa567c945e98628f5b3c58d2bf5d71c3df85af8ad19438c00a447890fdd1`
- Accepted reference SHA256: `a5b37db46753a7458802c87bd7996ca9fd073795c914178d3e1298ccfb6aea0f`
- Base SHA256: `a5b37db46753a7458802c87bd7996ca9fd073795c914178d3e1298ccfb6aea0f`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`
- verification_tier: `baseline`
- screening_pairs: `not-applicable: Phase 0`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `base.py` and `baseline_adapter.py` outputs matched at `atol=1e-2, rtol=1e-2`; shapes and dtypes matched. | pass | S60 auto_bench smoke and formal profile command |
| immutable base | base bytes unchanged by adapter generation | base hash recorded as `a5b37db4...` before and after adapter generation | pass | `make_baseline_adapter.py` and SHA-256 ledger |
| GCU runtime | selected profile matches runtime | `torch_gcu` and `triton_gcu` imported; `gcu:0` available; architecture `major=3, minor=0` | pass | S60 runtime discovery commands and baseline harness run |

## Screening Evidence

Not applicable to Phase 0.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `base.py`, then `baseline_adapter.py` in the unchanged harness
- reference_raw_samples_ms: `[0.482833]`
- candidate_raw_samples_ms: `[0.459285]`
- reference_median_ms: `0.482833`
- candidate_median_ms: `0.459285`
- improvement_pct: `not-applicable: baseline adapter is the executable canonical baseline`

The baseline adapter is retained as canonical because it is generated from the
immutable base, not because its small timing difference is an optimization claim.

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
- profiler_device_time: `unavailable: recorded GCU PrivateUse1 trace has no cat=kernel device-duration events`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels` are `null/unavailable`; `runtime_launch_*` fields are available
- trace: `log/groupedtopk_baseline_forward_50iter.pt.trace.json`
- trace_sha256: `cfea5cd92a62d2eee78db6a2f801212f1920723121f21793dd9724c0194952a2`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio | Runtime launches/call | Runtime launch us/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_base | unavailable | unavailable | unavailable | unavailable | 0.482833 | unavailable | 12.0 | 131.759590 |
| candidate_baseline_adapter | unavailable | unavailable | unavailable | unavailable | 0.459285 | unavailable | 12.0 | 132.298057 |

GCU trace evidence is diagnostic: the baseline and generated adapter each issue
12 runtime launches per forward call. Runtime launch duration is not device
kernel duration and is not converted to `device_ratio`.

## Retry History

No retries; Phase 0 baseline gate passed.

## evidence_for_next_round

- The unchanged eager reference issues 12 GCU runtime launches per forward call in the recorded scope.
- GCU profiler export does not provide `cat=kernel` device durations on this runtime; runtime-launch evidence is the available normalized backend diagnostic.
- Baseline correctness, harness loading, and measurement fingerprint are established for `gcu:0`.

## Stop Recommendation

- recommendation: `continue`
- evidence: baseline is established and no candidate round has been evaluated yet

## Exact Reproduction Commands

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file base.py --v1_file baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file base.py --v1_file baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file base.py --v1_file baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_baseline_forward_50iter.pt.trace.json
```
