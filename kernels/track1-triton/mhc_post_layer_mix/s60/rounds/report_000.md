# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py`
- Candidate SHA256: `2c0c121e2e4504e791fee3675ac1cd54d1322059fc537555578a91cabd2a24e6`
- Base SHA256: `58c67cedac8aac3fe1e35a32833616a80f2c3af74f184698a6338a59497695f5`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- verification_tier: `baseline`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict |
|---|---|---|---|
| correctness | pass | outputs matched at atol=1e-2, rtol=1e-2, shape [1,12,128,1024] fp16 | pass |
| immutable base | base bytes unchanged | super() fix recorded (semantically identical) | pass |
| GCU runtime | profile matches | gcu:0 available | pass |

## Interleaved Wall Timing

- reference_median_ms: `4.270324`
- candidate_median_ms: `4.251942`
- improvement_pct: `not-applicable`

## Profiler Evidence

- profiler_device_time: `unavailable`
- runtime_launch_count_per_call: `6.0`
- runtime_launch_us_per_call: `67.21`

| Scope | Wall ms | Runtime launches/call | Runtime launch us/call |
|---|---:|---:|---:|
| baseline_base | 4.270324 | 6.0 | 67.21 |

Launch overhead is only ~67 us/call ≈ 1.6% of the 4.27 ms wall; the einsum matmul
`[1,12,128,1024] x [1,12,128,1024]` dominates wall time on device.

## evidence_for_next_round

- Device-bound (matmul), launch overhead negligible (1.6%); no kernel-fusion headroom.
- Hand-written Triton `tl.dot` is Unknown on GCU and would not beat the CNNL matmul library kernel.

## Stop Recommendation

- recommendation: `stop` (measurement-bound, device matmul bound)
