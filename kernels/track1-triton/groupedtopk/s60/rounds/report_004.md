# Report 004

Result: no-improvement

## Identity

- Round: `004`
- Decision: `rounds/decision_004.md`
- Candidate: `triton_grouped_topk_004.py`
- Accepted reference: `reference_triton_grouped_topk_003.py` (adapter of accepted `triton_grouped_topk_003.py`; only `ModelNew -> Model` changed for the unchanged harness v0 contract)
- Canonical source before round: `triton_grouped_topk_003.py`
- Decision SHA256: `a126c9abc86da11734be828bc6c5900e0b1107ba07ecbfa079fc4f74d1416713`
- Canonical source SHA256: `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37`
- Reference adapter SHA256: `9977aaf9ec96c851be33f2582e6284451fd41686a1acc4607deb4e104dca5ea7`
- Candidate SHA256: `ea9be7896731f7f371f9ba087c8d01daca6556c66a3e50b2c6146fe6de118bb0`
- Base SHA256: `a5b37db46753a7458802c87bd7996ca9fd073795c914178d3e1298ccfb6aea0f`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`
- verification_tier: `authoritative`
- screening_pairs: `not-run as a separate stage; three formal paired runs completed after correctness and guardrails`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass against accepted Round 003 reference | Smoke, all three formal pairs, and profile run passed `atol=1e-2, rtol=1e-2`. | pass | S60 auto_bench output |
| explicit stream snapshot | candidate samples current stream once | Stack trace showed one `_stream_snapshot` call in candidate per forward. | pass | S60 stack trace |
| stream backend behavior | preserve direct launch semantics | A second current-stream call came from `triton_gcu` backend internals during launch, outside candidate code. | pass | S60 stack trace |
| metadata/output cache | preserve prior exact-key and lifecycle behavior | Hit/miss/invalidation, retained-output, and concurrency checks passed. | pass | S60 guardrail command |
| target/device | preserve GCU execution | Candidate ran on `gcu:0` with unchanged direct Triton-GCU launch. | pass | S60 smoke and profile |
| adoption threshold | at least 5% wall improvement | Formal median improvement was `2.058982586436897%`. | fail | three formal paired runs |

## Screening Evidence

Screening was not run as a separate two-pair stage. Three formal ordered
reference/candidate pairs were run after correctness and guardrails. The
candidate was valid but not adopted because its unrounded median improvement was
below the 5% threshold.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `reference, candidate` for each ordered pair; unchanged harness command
- reference_raw_samples_ms: `[0.350635, 0.275174, 0.277370]`
- candidate_raw_samples_ms: `[0.342246, 0.271659, 0.269279]`
- reference_median_ms: `0.277370`
- candidate_median_ms: `0.271659`
- improvement_pct: `2.058982586436897`
- speedup: `1.0210226791676331`

```text
improvement_pct = (0.277370 - 0.271659) / 0.277370 * 100
                = 2.058982586436897
```

The valid candidate missed the 5% adoption threshold and was not promoted.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| explicit_stream_snapshot_calls | one candidate call per forward | One candidate `_stream_snapshot` call; one additional call originates in Triton-GCU backend launch internals. | pass | S60 stack trace |
| metadata_cache_hit_rate | preserve compatible exact-key hits | Cache and invalidation checks passed. | pass | S60 guardrail command |
| runtime_launch_count_per_call | remain `1.0` | Both scopes emitted one `topsModuleLaunchKernel` per call. | pass | Round 004 trace summary |
| wall_time | improve by at least 5% | `2.058982586436897%`, below threshold. | fail | three formal ordered pairs |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-004`
- intervention: `single candidate-owned device/current-stream snapshot shared by metadata and output-pool lookup`
- expected_causal_chain: `duplicate candidate stream lookup disappears -> host setup decreases -> wall time improves while launch and lifecycle guardrails remain unchanged`
- primary_metric: `wall_time`
- Hypothesis verdict: `not-confirmed: valid no-improvement`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `unavailable: recorded GCU PrivateUse1 trace has no cat=kernel device-duration events`
- iterations: `50`
- normalized_fields: device totals, device us/call, kernel counts, device ratio, and top kernels are unavailable; backend `runtime_launch_*` fields are normalized per forward call
- trace: `log/groupedtopk_round_004_forward_50iter.pt.trace.json`
- trace_sha256: `ba3ddc328cd2cb36b06cd4401c2db423994f5be5dba29ee245766dff3bc609db`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio | Runtime launches/call | Runtime launch us/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| accepted_reference (`baseline_reference_triton_grouped_topk_003`) | unavailable | unavailable | unavailable | unavailable | 0.277370 | unavailable | 1.0 | 11.01330078125 |
| candidate (`candidate_triton_grouped_topk_004`) | unavailable | unavailable | unavailable | unavailable | 0.271659 | unavailable | 1.0 | 10.595556640625 |

Runtime-launch values are diagnostic and are not device kernel duration. No
device ratio is calculated.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial Round 004 implementation | `ea9be7896731f7f371f9ba087c8d01daca6556c66a3e50b2c6146fe6de118bb0` | `ea9be7896731f7f371f9ba087c8d01daca6556c66a3e50b2c6146fe6de118bb0` | guardrails, correctness, timing, and profile completed |

No repair was required. Candidate remains a valid non-canonical experiment.

## evidence_for_next_round

- Candidate 004 removed one candidate-owned stream lookup but did not clear the
  5% threshold; the additional backend-internal stream lookup is outside the
  candidate boundary.
- Canonical remains `triton_grouped_topk_003.py`, not candidate 004.
- Runtime launch count remains one per call and GCU device duration remains
  unavailable. Further host-only work should be conservative and must not claim
  device-time attribution.

## Stop Recommendation

- recommendation: `continue`
- evidence: This is the first valid no-improvement round; `valid_no_improvement_limit=3` and `max_rounds=20` are not reached, and the user explicitly requested continuation.

## Exact Reproduction Commands

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_003.py --v1_file triton_grouped_topk_004.py --warmup 5 --repeat 5 --full-traceback
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_003.py --v1_file triton_grouped_topk_004.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_003.py --v1_file triton_grouped_topk_004.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_round_004_forward_50iter.pt.trace.json
```
