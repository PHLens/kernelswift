# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_mha_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `ab2f5bb98a8f491ed67e2a05850fc28e9bf0958a09ef89ec3f32c8f24a0a949d`
- Candidate SHA256: `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b`
- Accepted reference SHA256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `29ecde127206fc1808c2d7f28951e44ee55a257aadfda78517e64d3493ce1862`
- verification_tier: authoritative
- screening_pairs: `not-run (correct candidate proceeded directly to authoritative timing per epoch-2 deliverable policy)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | v0 `Model` and v1 `ModelNew` outputs allclose (atol=1e-2, rtol=1e-2, equal_nan=True) | `PASS accuracy; v0=0.139143 ms, v1=0.186143 ms, speedup=0.748x` | pass | correctness command (warmup 5 repeat 10 full-traceback) `RETURN_CODE=0` |
| output dtype/shape/device | `(bsz, seq_len, hidden)` fp16 on input device | compare passed (no shape/dtype/device mismatch raised) | pass | correctness command output `PASS accuracy` |
| no input mutation | forward must not mutate query/key/value | kernel reads q/k/v only; output written to fresh `torch.empty_like(q)` | pass | candidate `triton_mha_001.py` kernel body + `torch.empty_like(q)` |
| caller-selected device/current stream preserved | candidate preserves device and stream | candidate uses input tensors' device; no explicit device/stream override | pass | candidate `forward` operates on input tensors in place (no `.cuda()`/stream switch) |
| SDPA fallback preserved | non-benchmark shapes keep `F.scaled_dot_product_attention` | `is_benchmark` gate guards the Triton path; else verbatim SDPA branch | pass | candidate `forward` `is_benchmark` conditional |

Conformance, correctness, and every declared guardrail pass before adoption.

## Screening Evidence

Not run: this is a deliverable round (epoch-2 policy). Correctness passed, and the candidate proceeded directly to authoritative timing. The candidate is expected to be slower than the flash-attention baseline (smoke showed 0.898x), so screening (which would only `screened-out` a >=10% slower candidate) is intentionally bypassed in favor of the deliverable policy that accepts a correct Triton kernel.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `sequential complete accepted-reference block, then complete candidate block`
- reference_raw_samples_ms (v0): `[0.108863, 0.111197, 0.110801]`
- candidate_raw_samples_ms (v1): `[0.164166, 0.166405, 0.163559]`
- reference_median_ms: `0.110801`
- candidate_median_ms: `0.164166`
- improvement_pct: `-48.162922717304006`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
```

The candidate is ~48% slower in median wall time than the accepted reference (speedup ~0.675x). This regression is expected and acceptable under the epoch-2 deliverable policy (recorded in `team-state.md` Policy Revisions): the round's deliverable requirement — ship a correct Triton kernel — is satisfied by the correctness pass, independent of wall-time improvement. The 5% adoption threshold is deliberately overridden for this deliverable round.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| candidate_kernel_count_per_call | single fused attention kernel name replaces the two mcFlashAttn kernels | The fused Triton kernel `_mha_fwd_kernel` runs 1.0 per call (replacing the 2 mcFlashAttn kernels), BUT the candidate also emits 4.0 `transpose12_copy_64` copy kernels per call from the `contiguous()` calls on q/k/v, so total device kernel count is 5.0 per call. The fused MHA kernel itself is exactly 1.0/call. | pass (with note) | summarize_trace candidate scope: `_mha_fwd_kernel` count/call 1.0; `transpose12_copy_64` count/call 4.0 |
| candidate_device_us_per_call | recorded for the fused Triton kernel; may be comparable or higher than the ~15 us baseline | `_mha_fwd_kernel` = 67.13333984375 us/call; total candidate device = 79.697666015625 us/call (incl. 12.564326171875 us/call of transpose copies). Higher than baseline ~15 us/call as anticipated. | pass (recorded) | summarize_trace candidate scope |
| correctness_parity | candidate output allclose reference (atol=1e-2, rtol=1e-2, equal_nan=True) = pass | `PASS accuracy` (allclose 1e-2) across all runs | pass | correctness command `RETURN_CODE=0`, `PASS accuracy` |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `implement a hand-written fused Triton multi-head attention kernel (one program per (batch, head, query_row))`
- expected_causal_chain: `Triton kernel accumulates in fp32 with online (two-pass max-subtracted) softmax, matching fp32-accumulation reference within 1e-2; emits a single fused device kernel replacing two mcFlashAttn kernels; correctness parity achieved`
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed`

The hypothesis is partially-confirmed: correctness parity is confirmed (the acceptance gate), and the single fused `_mha_fwd_kernel` replaced the two mcFlashAttn kernels as predicted. However, the expected "single device kernel per forward" is only partly realized — the `contiguous()` calls in the candidate's benchmark path introduce 4 additional `transpose12_copy_64` copy kernels per call, so total device kernels are 5.0/call, not 1.0. Device time (~79.7 us/call, dominated by `_mha_fwd_kernel` at ~67.1 us/call) is higher than the baseline ~15 us/call, as the decision anticipated.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not applicable (device kernel time available on C500)`

### C500 trace filtering note (documented)

The raw trace `round_001_forward_50iter.pt.trace.json` contains duplicate nested scope markers (the same known C500 issue as report_000): each scope (`baseline_base`, `candidate_triton_mha_001`) has a CPU-side `cat=user_annotation` X event that fully contains a nested GPU-side `cat=gpu_user_annotation` X event. `summarize_trace.py` matches any `ph=X, cat!=kernel, name==scope` event, so both events would be reported as `overlapping scope events`.

Fix applied (identical to report_000): the raw trace was preserved unchanged; a filtered trace `round_001_forward_50iter.filtered.pt.trace.json` was produced by dropping the two `cat=user_annotation` X events (outer CPU-side markers), keeping the `cat=gpu_user_annotation` device-side markers. No kernel events were touched; totals are unchanged by the filter. Summaries below are from the filtered trace.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| baseline_base (v0) | 754.1796875 | 15.08359375 | 100 | 2.0 | 0.110801 | 0.13613228896851112 |
| candidate_triton_mha_001 (v1) | 3984.88330078125 | 79.697666015625 | 250 | 5.0 | 0.164166 | 0.4854699877905596 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### Accepted Reference (v0, baseline_base) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void flash_fwd_splitkv_kernel<...>(mcFlashAttn::Flash_fwd_params, int, int)` | 50 | 1.0 | 438.0205078125 | 8.76041015625 |
| `void flash_fwd_splitkv_combine_kernel<...>(mcFlashAttn::Flash_fwd_params)` | 50 | 1.0 | 316.1591796875 | 6.32318359375 |

### Candidate (v1, candidate_triton_mha_001) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `_mha_fwd_kernel` | 50 | 1.0 | 3356.6669921875 | 67.13333984375 |
| `void at::native::elementwise_kernel_4_1_transpose12_copy_64<512, ...>` | 200 | 4.0 | 628.21630859375 | 12.564326171875 |

The candidate emits exactly one fused `_mha_fwd_kernel` per forward call (replacing the two mcFlashAttn kernels), plus four `transpose12_copy_64` copy kernels per call from the `.contiguous()` calls on the reshaped/transposed q, k, v tensors in the benchmark path.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b` | `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b` | correctness pass; authoritative timing + profiler complete |

No Verifier-to-Coder repair was required in this round.

## evidence_for_next_round

- Correctness parity is achieved: the hand-written Triton MHA kernel matches the fp16 SDPA reference within atol/rtol 1e-2 (all three authoritative runs and the correctness run all report `PASS accuracy`).
- The fused `_mha_fwd_kernel` is 1.0 kernel/call, replacing the two mcFlashAttn kernels (2.0/call) in the baseline. The fused-kernel replacement mechanism is confirmed.
- Candidate device time is ~79.7 us/call (vs baseline ~15.1 us/call), dominated by `_mha_fwd_kernel` at ~67.1 us/call. This is ~4.5x the baseline flash-attention device time, confirming the decision's prediction that a hand-written Triton MHA would regress against hardware-optimized `mcFlashAttn`.
- The candidate benchmark path emits 4 additional `transpose12_copy_64` copy kernels/call from `.contiguous()` on q/k/v (total 5.0 kernels/call). This is an observable inefficiency: the transpose+contiguous materialization adds ~12.6 us/call of device copy work.
- Wall time regresses ~48% (median 0.110801 ms -> 0.164166 ms). This is expected and acceptable under the epoch-2 deliverable policy; it is not a `no-improvement` failure.

## Stop Recommendation

- recommendation: `continue`
- evidence: `Deliverable Triton kernel accepted (correctness parity pass). Wall regression ~48% is expected under epoch-2 deliverable policy and does not burn the no-improvement streak.`

## Exact Reproduction Commands

```bash
cd /root/kernelswift-mma && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/triton_mha_001.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift-mma && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/triton_mha_001.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-mma && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/triton_mha_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mm_encoder_attention/maca/log/round_001_forward_50iter.pt.trace.json
```

```bash
cd /root/kernelswift-mma && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/maca/log/round_001_forward_50iter.filtered.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 0.110801
cd /root/kernelswift-mma && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/maca/log/round_001_forward_50iter.filtered.pt.trace.json --iterations 50 --scope candidate_triton_mha_001 --wall-ms 0.164166
```
