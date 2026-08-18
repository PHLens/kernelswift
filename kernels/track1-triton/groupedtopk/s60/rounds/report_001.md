# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_grouped_topk_001.py`
- Accepted reference: `baseline_adapter.py` (benchmark v0 uses byte-identical `base.py` semantics because the unchanged harness requires a top-level `Model`)
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `f49d72923a1e274a5ae00725947db509665c9ef899f0113c2db07a4d7336f6af`
- Candidate SHA256: `f42ff6b47b28996199bbe9b8df0a181db2834be99473453f3eea35df51df693e`
- Accepted reference SHA256: `6713aa567c945e98628f5b3c58d2bf5d71c3df85af8ad19438c00a447890fdd1`
- Base SHA256: `a5b37db46753a7458802c87bd7996ca9fd073795c914178d3e1298ccfb6aea0f`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`
- verification_tier: `authoritative`
- screening_pairs: `not-run; formal ordered pairs completed after correctness`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | Candidate matched reference outputs with `atol=1e-2, rtol=1e-2`; all three formal pairs and the compile smoke passed. | pass | S60 auto_bench output, three ordered benchmark pairs |
| output shape | `[83,8]` weights and ids | Candidate returned `[83,8]` for both outputs. | pass | auto_bench compare_values |
| output dtype | weights fp32, ids int32 | Candidate returned fp32 weights and int32 ids. | pass | auto_bench compare_values |
| target/device | preserve GCU execution | Candidate ran on `gcu:0` with direct Triton-GCU launch. | pass | S60 compile smoke and formal benchmark |
| launcher conformance | direct launch is the selected GCU path | Both observed fast-libentry imports failed; direct module launch compiled and ran. | pass | S60 import probe and candidate smoke |

## Screening Evidence

Screening was not run as a separate two-pair stage; three formal ordered
reference/candidate pairs were run after correctness and used for the adoption
decision. No screened-out classification was used.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `reference, candidate` for each ordered pair; unchanged harness command
- reference_raw_samples_ms: `[0.449626, 0.459056, 0.447581]`
- candidate_raw_samples_ms: `[0.274446, 0.273881, 0.270696]`
- reference_median_ms: `0.449626`
- candidate_median_ms: `0.273881`
- improvement_pct: `39.08693002628853`

```text
improvement_pct = (0.449626 - 0.273881) / 0.449626 * 100
                = 39.08693002628853
```

The unrounded median clears the 5% adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| runtime_launch_count_per_call | decrease from the baseline scoped runtime trace | `12.0` reference to `1.0` candidate | pass | `scripts/summarize_trace.py` on `log/groupedtopk_round_001_forward_50iter.pt.trace.json` |
| runtime_launch_us_per_call | decrease in the GCU runtime-launch diagnostic | `131.004023` us reference to `10.409482` us candidate | pass | same scoped trace summary |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse grouped softmax routing and masked top-k selection into one direct Triton-GCU kernel`
- expected_causal_chain: `separate GCU runtime launches disappear -> runtime launch count decreases -> wall time decreases while correctness passes`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `unavailable: recorded GCU PrivateUse1 trace has no cat=kernel device-duration events`
- iterations: `50`
- normalized_fields: device totals, device us/call, kernel counts, device ratio, and top kernels are unavailable; backend `runtime_launch_*` fields are normalized per forward call
- trace: `log/groupedtopk_round_001_forward_50iter.pt.trace.json`
- trace_sha256: `d2eb35974a9617b3e114397b54548883a822cbbabd6886a58f6c7955469e9ce6`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio | Runtime launches/call | Runtime launch us/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| accepted_reference (`baseline_base`) | unavailable | unavailable | unavailable | unavailable | 0.449626 | unavailable | 12.0 | 131.004023 |
| candidate | unavailable | unavailable | unavailable | unavailable | 0.273881 | unavailable | 1.0 | 10.409482 |

### Accepted Reference Runtime Launches

| Event | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `topsLaunchKernel` | 450 | 9.0 | 4851.998047 | 97.039960 |
| `topsLaunchCooperativeKernel` | 150 | 3.0 | 1698.203125 | 33.964063 |

### Candidate Runtime Launches

| Event | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `topsModuleLaunchKernel` | 50 | 1.0 | 520.474121 | 10.409482 |

The runtime-launch values are diagnostic and are not device kernel duration.
No device ratio is calculated.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | `f42ff6b47b28996199bbe9b8df0a181db2834be99473453f3eea35df51df693e` | `f42ff6b47b28996199bbe9b8df0a181db2834be99473453f3eea35df51df693e` | correctness, smoke, timing, and profile completed |

No Verifier-to-Coder repair was required.

## evidence_for_next_round

- Direct kernel fusion reduces the observed GCU runtime launches from 12.0 to 1.0 per forward call on the exact `T=83,E=256` regime.
- Wall time improves from `0.449626 ms` to `0.273881 ms` at the unrounded median, clearing the 5% threshold.
- GCU PrivateUse1 traces still expose no `cat=kernel` device durations; future device-bound claims require a different matched profiler exporter or remain unmeasured.
- The accepted candidate still allocates two output tensors per forward call; allocator lifecycle was intentionally outside this decision.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 001 is accepted, but the current run has one completed round and no user stop or policy stop condition.

## Exact Reproduction Commands

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file base.py --v1_file triton_grouped_topk_001.py --warmup 1 --repeat 3 --full-traceback
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file base.py --v1_file triton_grouped_topk_001.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file base.py --v1_file triton_grouped_topk_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_round_001_forward_50iter.pt.trace.json
```
