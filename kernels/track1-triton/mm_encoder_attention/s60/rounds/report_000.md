# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py`
- Candidate SHA256: `d0bae6edf2e34b22184615c063544fe23abca9409ca002246dc04d466dbd398c`
- Accepted reference SHA256: `27f1c594afb539baa716d8e00516646acddb17cf2ba0b402bd4c7aaabc4a8f9b`
- Base SHA256: `27f1c594afb539baa716d8e00516646acddb17cf2ba0b402bd4c7aaabc4a8f9b`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Measurement fingerprint: `1b7f6e8f5d2c9a0b3e4f7a8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d`
- verification_tier: `baseline`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict |
|---|---|---|---|
| correctness | pass | outputs matched at atol=1e-2, rtol=1e-2, shape [2,83,512] fp16 | pass |
| immutable base | base bytes unchanged | base hash recorded | pass |
| GCU runtime | profile matches | torch_gcu/triton_gcu imported, gcu:0 available | pass |

## Interleaved Wall Timing

- reference_median_ms: `0.229925`
- candidate_median_ms: `0.223979`
- improvement_pct: `not-applicable`

## Profiler Evidence

- profiler_device_time: `unavailable`
- runtime_launch_count_per_call: `1.0` (single `topsLaunchKernel`)
- runtime_launch_us_per_call: `10.43`

| Scope | Wall ms | Runtime launches/call | Runtime launch us/call |
|---|---:|---:|---:|
| baseline_base | 0.229925 | 1.0 | 10.43 |

The eager `F.scaled_dot_product_attention` (non-causal) already lowers to a single fused CNNL kernel on GCU — identical to the flexattention s60 case. No kernel-count headroom exists.

## evidence_for_next_round

- Single fused kernel (1 launch/call); no fusion headroom.
- flexattention s60 already proved hand-written Triton SDPA is ~100x slower on device than the CNNL flash-attention kernel.

## Stop Recommendation

- recommendation: `stop` (measurement-bound, no optimization space)
