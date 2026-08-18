# Report 003

Result: accepted

## Identity

- Round: `003`
- Decision: `rounds/decision_003.md`
- Candidate: `triton_grouped_topk_003.py`
- Accepted reference: `reference_triton_grouped_topk_002.py` (adapter of accepted `triton_grouped_topk_002.py`; only `ModelNew -> Model` changed for the unchanged harness v0 contract)
- Canonical source before round: `triton_grouped_topk_002.py`
- Decision SHA256: `2f90569b0cbf786f217cd45fac38c51990d7b5c041dc1f9a5ac6e5ac38129594`
- Canonical source SHA256: `90d7b09569d1d155c8e44e1626f2c0f3b3f41e0919a8a9e5b76719e874b17ce3`
- Reference adapter SHA256: `9d3a368e93afc557d18eba6241df83757ec4c7478686809e90c7b8f1945fa8cd`
- Candidate SHA256: `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37`
- Base SHA256: `a5b37db46753a7458802c87bd7996ca9fd073795c914178d3e1298ccfb6aea0f`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`
- verification_tier: `authoritative`
- screening_pairs: `not-run as a separate stage; three formal paired runs completed after correctness and guardrails`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass against accepted Round 002 reference | Smoke and all three formal paired runs passed `atol=1e-2, rtol=1e-2`. | pass | S60 auto_bench output |
| output shape/dtype/layout | unchanged | Weights and ids retained expected shape, dtype, and contiguous layout. | pass | auto_bench and candidate source |
| metadata exact-key hit | repeated compatible calls hit one entry | 256-expert repeated calls left metadata cache size at 1. | pass | S60 metadata command |
| metadata invalidation | incompatible key creates separate entry | 128-expert call created a second entry; 256-expert returned to the original entry. | pass | S60 metadata command |
| instance ownership | no cross-instance cache | Separate model instance had an independent metadata dictionary. | pass | S60 metadata command |
| output lifetime/concurrency | preserve Round 002 pool guarantees | Retained output and concurrent checks passed. | pass | S60 lifecycle command |
| target/device/stream | preserve GCU path | Candidate used selected `gcu:0` and current stream; no explicit synchronization or context switch. | pass | S60 smoke and candidate source |
| launch conformance | unchanged direct launch | One direct Triton-GCU launch per forward; `num_warps=1`. | pass | candidate source and profile |

## Screening Evidence

Screening was not run as a separate two-pair stage. Three formal ordered
reference/candidate pairs were run after correctness and guardrails. No
screened-out classification was used.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `reference, candidate` for each ordered pair; unchanged harness command
- reference_raw_samples_ms: `[0.292588, 0.294687, 0.267353]`
- candidate_raw_samples_ms: `[0.271637, 0.273673, 0.273697]`
- reference_median_ms: `0.292588`
- candidate_median_ms: `0.273673`
- improvement_pct: `6.464721724746064`
- speedup: `1.0691153310702919`

```text
improvement_pct = (0.292588 - 0.273673) / 0.292588 * 100
                = 6.464721724746064
```

The unrounded median clears the 5% adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| metadata_cache_hit_rate | compatible repeated calls hit after one miss | Cache size remained 1 for repeated exact-key calls. | pass | S60 metadata command |
| metadata_derivation_count_per_call | derive once per new exact key | Cache size increased only for the 128-expert key and remained stable on return to 256 experts. | pass | S60 metadata command |
| runtime_launch_count_per_call | remain `1.0` | Both scopes emitted one `topsModuleLaunchKernel` per call. | pass | Round 003 trace summary |
| wall_time | improve by at least 5% | `6.464721724746064%` median improvement. | pass | three formal ordered pairs |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-003`
- intervention: `exact-key instance-private host metadata specialization`
- expected_causal_chain: `metadata derivation moves from every forward to every new key -> unchanged one-launch path remains -> wall time decreases while output pool and correctness guardrails pass`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `unavailable: recorded GCU PrivateUse1 trace has no cat=kernel device-duration events`
- iterations: `50`
- normalized_fields: device totals, device us/call, kernel counts, device ratio, and top kernels are unavailable; backend `runtime_launch_*` fields are normalized per forward call
- trace: `log/groupedtopk_round_003_forward_50iter.pt.trace.json`
- trace_sha256: `dc1b1ac9b8dbf8d21b52804dc540c116dd48a3560c347886ca3780ccd8c4af34`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio | Runtime launches/call | Runtime launch us/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| accepted_reference (`baseline_reference_triton_grouped_topk_002`) | unavailable | unavailable | unavailable | unavailable | 0.292588 | unavailable | 1.0 | 11.284130859375 |
| candidate (`candidate_triton_grouped_topk_003`) | unavailable | unavailable | unavailable | unavailable | 0.273673 | unavailable | 1.0 | 11.1054296875 |

Runtime-launch values are diagnostic and are not device kernel duration. No
device ratio is calculated.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial Round 003 implementation | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` | metadata guardrails, correctness, timing, and profile completed |

No Verifier-to-Coder repair was required.

## evidence_for_next_round

- Round 003 accepted exact-key host metadata specialization and improved wall
  median by `6.464721724746064%` against the accepted Round 002 candidate.
- The accepted candidate now has safe output lifecycle reuse and exact-key host
  metadata cache; runtime launches remain one per call.
- GCU device duration remains unavailable. Kernel dataflow changes require a
  matched device profiler exporter or a same-runtime microbenchmark.
- Further host changes must preserve both caches and all lifetime/concurrency
  guardrails; likely gains are increasingly small.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 003 is accepted; `valid_no_improvement_limit=3` and `max_rounds=20` are not reached, and the user explicitly requested continuation until stop.

## Exact Reproduction Commands

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_002.py --v1_file triton_grouped_topk_003.py --warmup 5 --repeat 5 --full-traceback
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_002.py --v1_file triton_grouped_topk_003.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_002.py --v1_file triton_grouped_topk_003.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_round_003_forward_50iter.pt.trace.json
```
