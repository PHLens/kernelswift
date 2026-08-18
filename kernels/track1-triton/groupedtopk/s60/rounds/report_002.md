# Report 002

Result: accepted

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_grouped_topk_002.py`
- Accepted reference: `reference_triton_grouped_topk_001.py` (adapter of the accepted `triton_grouped_topk_001.py`; only `ModelNew -> Model` was changed for the unchanged harness v0 contract)
- Canonical source before round: `triton_grouped_topk_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `8d56aaf1e9ca91f59a439e3ace0bba74d0234b7f02f4a3712f592100884f0805`
- Canonical source SHA256: `f42ff6b47b28996199bbe9b8df0a181db2834be99473453f3eea35df51df693e`
- Reference adapter SHA256: `800ec0080e66589f6dfcf3a71ee79f08e01be68f145b4cb3c6c6b50dd7c03027`
- Candidate SHA256: `90d7b09569d1d155c8e44e1626f2c0f3b3f41e0919a8a9e5b76719e874b17ce3`
- Base SHA256: `a5b37db46753a7458802c87bd7996ca9fd073795c914178d3e1298ccfb6aea0f`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`
- verification_tier: `authoritative`
- screening_pairs: `not-run as a separate stage; three formal paired runs completed after correctness and guardrails`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass against accepted Round 001 reference | Smoke and all three formal paired runs passed `atol=1e-2, rtol=1e-2`. | pass | S60 auto_bench output |
| output shape | `[83,8]` weights and ids | Both outputs retained the expected shapes. | pass | auto_bench compare_values |
| output dtype/layout | weights fp32, ids int32, contiguous | Candidate preserves output dtypes and contiguous layout. | pass | auto_bench and candidate source |
| target/device | preserve GCU execution | Candidate ran on `gcu:0`; device and current stream are used for allocation and launch. | pass | S60 smoke and lifecycle command |
| retained output lifetime | prior output is not overwritten | Retained output received distinct storage and remained equal to its snapshot after a later call. | pass | S60 lifecycle command |
| retained alias lifetime | view alias is not overwritten | Retained view received distinct storage and remained equal to its snapshot. | pass | S60 lifecycle command |
| concurrent forward safety | distinct storage for overlapping calls | Two same-instance concurrent calls received distinct output storage. | pass | S60 lifecycle command |
| separate instance ownership | no shared pool state | Pool is initialized in `ModelNew.__init__`; no module/global cache exists. | pass | candidate source |
| launch conformance | keep accepted launch path | One unchanged direct Triton-GCU module launch per forward. | pass | candidate source and profile |

## Screening Evidence

Screening was not run as a separate two-pair stage. Three formal ordered
reference/candidate pairs were run after correctness and lifecycle guardrails.
No screened-out classification was used.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `reference, candidate` for each ordered pair; unchanged harness command
- reference_raw_samples_ms: `[0.301983, 0.280062, 0.322022]`
- candidate_raw_samples_ms: `[0.271376, 0.274740, 0.290498]`
- reference_median_ms: `0.301983`
- candidate_median_ms: `0.274740`
- improvement_pct: `9.02136875254568`
- speedup: `1.0991592050666086`

```text
improvement_pct = (0.301983 - 0.274740) / 0.301983 * 100
                = 9.02136875254568
```

The unrounded median clears the 5% adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| output_allocations_per_call | decrease from two after warmup for compatible sequential calls | Sequential lifecycle check reused the same output pair after warmup; pool size remained 1. | pass | S60 lifecycle command |
| live_output_storage_conflicts | remain zero for retained outputs, aliases, and overlapping calls | Retained output, alias, and concurrent checks all received distinct storage and preserved snapshots. | pass | S60 lifecycle command |
| runtime_launch_count_per_call | remain `1.0` | Reference and candidate both emitted `1.0` runtime launch/call. | pass | Round 002 trace summary |
| wall_time | improve by at least 5% | `9.02136875254568%` median improvement. | pass | three formal ordered pairs |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: `model-instance-owned, device-and-stream-keyed output buffer lease pool`
- expected_causal_chain: `compatible sequential calls reuse safe output storage -> allocation work decreases without extra launch -> wall time decreases while lifetime guardrails pass`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `unavailable: recorded GCU PrivateUse1 trace has no cat=kernel device-duration events`
- iterations: `50`
- normalized_fields: device totals, device us/call, kernel counts, device ratio, and top kernels are unavailable; backend `runtime_launch_*` fields are normalized per forward call
- trace: `log/groupedtopk_round_002_forward_50iter.pt.trace.json`
- trace_sha256: `c01d6f966b94e5476d77ad008df9c6d2e5072702d7ffbcce17a9f9c348f68a62`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio | Runtime launches/call | Runtime launch us/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| accepted_reference (`baseline_reference_triton_grouped_topk_001`) | unavailable | unavailable | unavailable | unavailable | 0.301983 | unavailable | 1.0 | 11.261884765625 |
| candidate (`candidate_triton_grouped_topk_002`) | unavailable | unavailable | unavailable | unavailable | 0.274740 | unavailable | 1.0 | 10.782412109375 |

The runtime-launch values are diagnostic and are not device kernel duration.
No device ratio is calculated. Launch count is unchanged from the accepted
reference, consistent with the host-only change boundary.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | lifecycle guardrail failure in initial lease threshold | `36abc6656221a61414b85a6eb838efbd9519205aba2814f1a8eff9138229ff58` | `36abc6656221a61414b85a6eb838efbd9519205aba2814f1a8eff9138229ff58` | retained-output and concurrent pointer checks exposed unsafe reuse |
| 2 | calibrated S60 reference count baseline | `36abc6656221a61414b85a6eb838efbd9519205aba2814f1a8eff9138229ff58` | `90d7b09569d1d155c8e44e1626f2c0f3b3f41e0919a8a9e5b76719e874b17ce3` | all lifecycle guardrails passed; correctness, timing, and profile completed |

## evidence_for_next_round

- Round 002 accepted a safe output-buffer lifecycle optimization and improved wall
  median by `9.02136875254568%` against the accepted Round 001 candidate.
- The accepted candidate still performs host shape-derived metadata work each
  forward (`triton.next_power_of_2`, expert/group arithmetic, and repeated launch
  argument construction); this is only a source-backed hypothesis, not a measured
  host-time attribution.
- Runtime launch count is already one per call and GCU device duration remains
  unavailable. Future kernel dataflow changes require new attributable device
  evidence or a matched microbenchmark.
- The GCU-specific storage use-count API and stream id were observed on the exact
  fingerprint; portability outside this GCU runtime is not claimed.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 002 is accepted, `valid_no_improvement_limit=3` has not been
  reached, `max_rounds=20` has not been reached, and the user explicitly asked
  to continue until a stop threshold.

## Exact Reproduction Commands

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_001.py --v1_file triton_grouped_topk_002.py --warmup 5 --repeat 5 --full-traceback
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_001.py --v1_file triton_grouped_topk_002.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_001.py --v1_file triton_grouped_topk_002.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_round_002_forward_50iter.pt.trace.json
```
