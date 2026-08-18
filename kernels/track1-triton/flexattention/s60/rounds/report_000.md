# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `1532b55e399da3a8404f75d31ee7f2453a32f7baef41d10425f556931400ac0c`
- Accepted reference SHA256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0`
- Base SHA256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `115c2e1f54ec7c9973ce8cfa822498e737bd022793ef3b6a7db93ef760479668`
- verification_tier: `baseline`
- screening_pairs: `not-applicable: Phase 0`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `base.py` and `baseline_adapter.py` outputs matched at `atol=1e-2, rtol=1e-2`; shapes and dtypes matched (`[83,512] fp16`) | pass | S60 auto_bench smoke and formal benchmark command |
| immutable base | base bytes unchanged by adapter generation | base hash recorded as `dd1359ad...` before and after adapter generation | pass | `make_baseline_adapter.py` and SHA-256 ledger |
| GCU runtime | selected profile matches runtime | `torch_gcu` and `triton_gcu` imported; `gcu:0` available; architecture `major=3, minor=0` | pass | S60 runtime discovery commands and baseline harness run |

## Screening Evidence

Not applicable to Phase 0.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `base.py`, then `baseline_adapter.py` in the unchanged harness
- reference_raw_samples_ms: `[0.268253]`
- candidate_raw_samples_ms: `[0.269216]`
- reference_median_ms: `0.268253`
- candidate_median_ms: `0.269216`
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
- trace: `log/flexattention_baseline_forward_50iter.pt.trace.json`
- trace_sha256: `c0b45f247dd40db7e043b53bd0a1881ab09b87f20fc177b7944b65affb1a2e28`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio | Runtime launches/call | Runtime launch us/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_base | unavailable | unavailable | unavailable | unavailable | 0.252247 | unavailable | 1.0 | 10.514160 |
| candidate_baseline_adapter | unavailable | unavailable | unavailable | unavailable | 0.252474 | unavailable | 1.0 | 10.514160 |

GCU trace evidence is diagnostic: the baseline and generated adapter each issue
1 runtime launch (`topsLaunchKernel`) per forward call, indicating the GCU
eager `F.scaled_dot_product_attention` path is already fused to a single kernel.
Runtime launch duration is not device kernel duration and is not converted to
`device_ratio`.

## Retry History

No retries; Phase 0 baseline gate passed.

## evidence_for_next_round

- The unchanged eager reference issues exactly 1 GCU runtime launch
  (`topsLaunchKernel`) per forward call in the recorded scope.
- GCU profiler export does not provide `cat=kernel` device durations on this
  runtime; runtime-launch evidence is the available normalized backend diagnostic.
- Baseline correctness, harness loading, and measurement fingerprint are
  established for `gcu:0`.
- Baseline wall median is ~0.269 ms (benchmark) / ~0.252 ms (profiler run);
  the eager SDPA path is already a single fused kernel on GCU, so the
  optimization headroom is host/launcher-bound rather than device kernel count.

## Stop Recommendation

- recommendation: `continue`
- evidence: baseline is established and no candidate round has been evaluated yet

## Exact Reproduction Commands

```bash
cd /root/kernelswift/.worktrees/flexattention-s60
python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/s60/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift/.worktrees/flexattention-s60
python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/s60/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/flexattention-s60
python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/s60/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/flexattention/s60/log/flexattention_baseline_forward_50iter.pt.trace.json
```
