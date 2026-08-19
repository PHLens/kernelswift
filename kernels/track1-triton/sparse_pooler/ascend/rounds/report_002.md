# Report 002

Result: no-improvement

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_sparse_pooler_002.py`
- Accepted reference: `triton_sparse_pooler_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `8beb3c34e9eb88aac1722ee9a99117a7c05453e2f08ed5487ccd49a5004b003f`
- Candidate SHA256: `a7338d89a1f5a30843e84d3f533ac151245d6547453ddc5a2dcff66f93cb7957`
- Accepted reference SHA256: `dc2a8b6582cf9d6fef3e044081426762b88833e056ce8d7f04086e0d92f429e0`
- Base SHA256: `2b740bba37a87a7bcb022af36537486179538feed5dada3f3c1d5e32cd3f6c36`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `f4305d20c3f39dba64e252050fcc6cb437a1ba7a24fb0480530287bcd4e7a6e1`
- verification_tier: `authoritative`
- screening_pairs: `not-run` (authoritative timing; correctness passed, candidate not ≥10% slower so no screening gate)

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | PASS accuracy | `PASS accuracy; v0=0.950040 ms, v1=0.641920 ms` (warmup 5 / repeat 10) | pass | `auto_bench.py --v0_file base.py --v1_file triton_sparse_pooler_002.py --warmup 5 --repeat 10 --full-traceback` |
| list-output comparison | output is Python list of num_seq tensors, element-wise equal | harness `compare_values` recurses list→element→Tensor(allclose); PASS despite cached buffer being overwritten each forward | pass | correctness run PASS |
| numerical semantics | log(1+relu(logits)) max-pooled within tolerance | PASS accuracy (allclose) | pass | correctness run |
| device/stream preserved | caller-selected device/stream | `logits.device` used; no `torch.npu.device()` context | pass | source read |
| fused kernel unchanged | `_sparse_pooler_max_kernel` body, BLOCK_V=1024, num_warps=1, grid, prefix scan unchanged | kernel byte-identical to round 001 | pass | source read |
| MLM head unchanged | dense/GELU/LayerNorm/decoder library ops | unchanged | pass | source read |
| signature unchanged | `ModelNew(hidden_size, vocab_size, pooling)`, `forward(hidden_states, seq_lens)` | unchanged | pass | source read |
| load_state_dict compatibility | cache is plain attribute, not buffer/parameter | `self._out_cache`/`self._out_cache_key` plain attrs | pass | source read |
| kernel_count_per_call remains 5 | no kernel added/removed | 5 → 5 (profiler) | pass | CANN scope summary |
| device_us_per_call must not increase | no device-side change | 194.38 → 183.03 us/call (within noise, not increased) | pass | CANN scope summary |

## List-output aliasing safety (harness behavior confirmed)

The candidate reuses a cached `out` buffer and returns `[out[i] for i in range(num_seq)]` — the returned list shares storage with the cached buffer, which the fused kernel overwrites in place on the next forward. The harness reads each forward's output immediately: `compare_case` calls `run_forward` (which returns the list) then `compare_values(v0_output, v1_output, ...)` before the next `time_forward` iteration overwrites the buffer. There is no cross-forward retention of output references in the harness, so the aliasing is safe. Correctness passed, confirming the list-output comparison still holds under buffer reuse.

## Screening Evidence

- screening not run: correctness passed and the candidate is not ≥10% slower than the accepted reference, so the `screened-out` gate does not apply; the candidate proceeded to authoritative timing.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (reference = triton_sparse_pooler_001.py; candidate = triton_sparse_pooler_002.py; base.py held constant as v0 anchor)
- reference_raw_samples_ms: `[0.637610, 0.636715, 0.592040]`
- candidate_raw_samples_ms: `[0.640425, 0.619190, 0.616975]`
- reference_median_ms: `0.636715`
- candidate_median_ms: `0.619190`
- improvement_pct: `2.752`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (0.636715 - 0.619190) / 0.636715 * 100 = 2.752 %
```

The unrounded improvement (2.752%) is below the 5% adoption threshold. The host-only allocation-reuse change delivered a small (but sub-threshold) wall-time reduction; the NPU caching allocator already made the per-call `torch.empty` largely free, as the decision's own falsification clause anticipated.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: `reuse a single preallocated [num_seq, vocab_size] fp32 output buffer across compatible forward calls`
- expected_causal_chain: `per-call output allocation amortized; host work decreases; device time and kernel count unchanged; wall time decreases ≥5%`
- primary_metric: `wall_time`
- Hypothesis verdict: `falsified`

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| output_allocations_per_call | decrease from 1 to 0 on cache hit | cache present (`self._out_cache` keyed by (num_seq, vocab_size, dtype, device)); first forward allocates, subsequent reuse. Confirmed by source. | pass | source read (forward) |
| device_us_per_call | unchanged within noise (~202.86 us/call) | reference 194.38, candidate 183.03 us/call (both within per-capture noise of the round-001 baseline ~202.86); candidate NOT increased | pass | CANN scope summary |
| kernel_count_per_call | remains 5 exactly | 5 → 5 (unchanged) | pass | CANN scope summary |

The three mechanism observables are all satisfied (the host change is real and correct), but the **primary_metric** (wall_time ≥5% improvement) is NOT met (2.75%). The hypothesis is therefore falsified: the output allocation was not a meaningful contributor to wall time on this runtime (the NPU caching allocator already amortizes `torch.empty`).

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable`

Reference and candidate profiled in separate CANN msprof captures via `--profile-reference-file triton_sparse_pooler_001.py` (labels `reference_triton_sparse_pooler_001`, `candidate_triton_sparse_pooler_002`).

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference | 9718.82 | 194.3764 | 250 | 5.0 | 0.636715 | 0.3053 |
| candidate | 9151.36 | 183.0272 | 250 | 5.0 | 0.619190 | 0.2956 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnAddmm_MatMulCommon_MatMulV2 | 100 | 2.0 | 7355.56 | 147.1112 |
| _sparse_pooler_max_kernel | 50 | 1.0 | 1902.88 | 38.0576 |
| aclnnLayerNormWithImplMode_LayerNormV3WithImplMode_LayerNormV3 | 50 | 1.0 | 315.08 | 6.3016 |
| aclnnGelu_Gelu_Gelu | 50 | 1.0 | 145.30 | 2.9060 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnAddmm_MatMulCommon_MatMulV2 | 100 | 2.0 | 6774.54 | 135.4908 |
| _sparse_pooler_max_kernel | 50 | 1.0 | 1897.54 | 37.9508 |
| aclnnLayerNormWithImplMode_LayerNormV3WithImplMode_LayerNormV3 | 50 | 1.0 | 327.40 | 6.5480 |
| aclnnGelu_Gelu_Gelu | 50 | 1.0 | 151.88 | 3.0376 |

Kernel set is identical between reference and candidate (4 kernel types, 5 launches/call), confirming the change is host-only (byte-identical kernel, no device-side modification). Device times differ slightly between captures (per-capture variation artifact, as observed in prior rounds).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `a7338d89a1f5a30843e84d3f533ac151245d6547453ddc5a2dcff66f93cb7957` | correctness pass, timing +2.75% (sub-threshold), profiler complete |

## evidence_for_next_round

- H-002 falsified: output-buffer reuse is correct and eliminates the per-call `torch.empty`, but wall improvement was only +2.75% (< 5% threshold). The NPU caching allocator already makes the `[4,30522]` fp32 (~0.5 MB) output allocation near-free on this runtime — unlike the flexattention-ascend case (+14.71%) where the pattern paid off.
- kernel_count_per_call and kernel set are unchanged (5/call, byte-identical kernel), confirming the host-only change did not touch device work.
- The dominant remaining bottleneck is unchanged: `aclnnAddmm_MatMulCommon_MatMulV2` x2 (~135-147 us/call, ~74-76% of device time), the MLM-head dense + decoder matmuls. device_ratio ≈ 0.30, so ~70% of wall time remains host-side (launch/dispatch/harness-fixed, which allocation-reuse did not meaningfully reduce).
- The host-side time that remains is NOT output allocation — it is launch/dispatch and harness-fixed overhead (sync_devices, seed setup). These are "Fixed for the regime" per bottleneck-judgment and were not targeted by this round.

## Stop Recommendation

- recommendation: `continue`
- evidence: H-002 falsified (no-improvement, +2.75%); no target mode set; valid_no_improvement_limit has one miss remaining in the streak. Matmul (device) and launch-dispatch (host) remain unaddressed bottlenecks.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/triton_sparse_pooler_002.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/triton_sparse_pooler_001.py --warmup 50 --repeat 100
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/triton_sparse_pooler_002.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/triton_sparse_pooler_002.py --profile --profile-reference-file kernels/track1-triton/sparse_pooler/ascend/triton_sparse_pooler_001.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/sparse_pooler/ascend/log/sparse_pooler_round002_forward_50iter.pt.trace.json
```

```bash
# Summarize per-scope device time (separate CANN captures):
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/sparse_pooler/ascend/log/profiling_data/reference_triton_sparse_pooler_001/profiling_data --iterations 50 --wall-ms 0.636715
/usr/local/python3.11.15/bin/python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/sparse_pooler/ascend/log/profiling_data/candidate_triton_sparse_pooler_002/profiling_data --iterations 50 --wall-ms 0.619190
```
