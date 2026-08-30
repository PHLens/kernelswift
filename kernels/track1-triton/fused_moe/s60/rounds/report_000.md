# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `b939d91f0f85e299a1102bfceb00da0e38c484a81c8d23ec78777fce68a3ee6f`
- Accepted reference SHA256: `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d`
- Base SHA256: `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `d8f8f6bf8965ab279eb59215a7cc0c6f24f7dd0ad5ea7d8436162336955af6c3`
- verification_tier: `baseline`
- screening_pairs: `not-applicable: Phase 0`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `base.py` and `baseline_adapter.py` outputs matched at `atol=1e-2, rtol=1e-2`; shapes and dtypes matched (`[83,128] fp16`) | pass | S60 auto_bench smoke and formal benchmark command |
| immutable base | base bytes unchanged by adapter generation | base hash recorded as `21e75853...` before and after adapter generation | pass | `make_baseline_adapter.py` and SHA-256 ledger |
| GCU runtime | selected profile matches runtime | `torch_gcu` and `triton_gcu` imported; `gcu:0` available; architecture `major=3, minor=0` | pass | S60 runtime discovery commands and baseline harness run |

## Screening Evidence

Not applicable to Phase 0.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `base.py`, then `baseline_adapter.py` in the unchanged harness
- reference_raw_samples_ms: `[5.112406]`
- candidate_raw_samples_ms: `[5.232766]`
- reference_median_ms: `5.112406`
- candidate_median_ms: `5.232766`
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
- trace: `log/fused_moe_baseline_forward_50iter.pt.trace.json`
- trace_sha256: `ad37ecb991e1d4fd5521aba0362084941263f8b1e68591aae6e2fc4486e68614`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio | Runtime launches/call | Runtime launch us/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_base | unavailable | unavailable | unavailable | unavailable | 5.112406 | unavailable | 147.0 | 1788.388701 |
| candidate_baseline_adapter | unavailable | unavailable | unavailable | unavailable | 5.232766 | unavailable | 147.0 | 1788.388701 |

GCU trace evidence is diagnostic: the eager reference issues 147 runtime
launches per forward call (74 `topsLaunchKernel` + 73 `topsLaunchCooperativeKernel`),
driven by the Python for-loop over 8 experts plus per-expert mask/gather/scatter
and eager FFN ops (softmax/topk/renorm/cast + per-expert double GEMM + SiLU).
Runtime launch duration is not device kernel duration and is not converted to
`device_ratio`. This is the primary fusion headroom (contrast flexattention's
already-fused single launch).

## Retry History

No retries; Phase 0 baseline gate passed.

## evidence_for_next_round

- The eager reference issues 147 GCU runtime launches per forward call; the
  per-expert Python loop and per-expert mask/gather/scatter are the structural
  source, giving a large kernel-fusion headroom (unlike flexattention).
- GCU profiler export does not provide `cat=kernel` device durations on this
  runtime; runtime-launch evidence is the available normalized backend diagnostic.
- `torch.topk` returns int64 `topk_ids` but GCU downgrades to int32
  (UserWarning); any Triton candidate must use int32 indices only.
- State-dict contract: candidate `ModelNew` must expose exactly `w1 [8,128,128]`
  and `w2 [8,128,64]` fp32 parameters for `load_state_dict` to synchronize weights.
- Baseline wall median is ~5.11 ms (benchmark); eager launch overhead dominates
  at T=83 (tiny per-expert GEMMs), matching the MLU 50.4x fusion premise.

## Stop Recommendation

- recommendation: `continue`
- evidence: baseline is established; 147-launch fusion headroom is clear and no
  candidate round has been evaluated yet

## Exact Reproduction Commands

```bash
cd /root/kernelswift/.worktrees/fused-moe-s60
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift/.worktrees/fused-moe-s60
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/fused-moe-s60
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/fused_moe/s60/log/fused_moe_baseline_forward_50iter.pt.trace.json
```
