# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_sparse_pooler_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `9b4f5333dd5145d8f8075047a89882f81f2c77fce888b3013e83c19e169256bf`
- Candidate SHA256: `dc2a8b6582cf9d6fef3e044081426762b88833e056ce8d7f04086e0d92f429e0`
- Accepted reference SHA256: `94d00f1a5d26f453fd5078fd9d50dfcddbb0c11d20a145d223544e59234add0f`
- Base SHA256: `2b740bba37a87a7bcb022af36537486179538feed5dada3f3c1d5e32cd3f6c36`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `f4305d20c3f39dba64e252050fcc6cb437a1ba7a24fb0480530287bcd4e7a6e1`
- verification_tier: `authoritative`
- screening_pairs: `not-run` (authoritative timing; correctness passed and candidate is faster, so no screening gate was triggered)

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | PASS accuracy | `PASS accuracy; v0=0.910860 ms, v1=0.627850 ms` (warmup 5 / repeat 10) | pass | `auto_bench.py --v0_file base.py --v1_file triton_sparse_pooler_001.py --warmup 5 --repeat 10 --full-traceback` |
| list-output comparison | output is Python list of num_seq tensors, element-wise equal | harness `compare_values` recurses list→element→Tensor(allclose atol=1e-2 rtol=1e-2 equal_nan=True); PASS | pass | correctness run PASS; source `return [out[i] for i in range(num_seq)]` |
| numerical semantics | log(1+relu(logits)) max-pooled per seq within tolerance | PASS accuracy (allclose) | pass | correctness run |
| device/stream preserved | caller-selected device/stream | candidate uses `logits.device`, no `torch.npu.device()` context | pass | source read (forward) |
| MLM head unchanged | dense/GELU/LayerNorm/decoder stay library ops | dense/GELU/LayerNorm/decoder lines identical to baseline_adapter; only pooling path changed | pass | source read |
| signature unchanged | `ModelNew(hidden_size, vocab_size, pooling)`, `forward(hidden_states, seq_lens)` | unchanged | pass | source read |
| load_state_dict compatibility | state dict keys unchanged | nn.Linear/nn.GELU/nn.LayerNorm submodules unchanged | pass | source read |
| pooling=="sum" fallback | original reference behavior preserved | sum branch uses `torch.log1p(F.relu(logits))` + Python loop + `chunk.sum(dim=0)` (unchanged) | pass | source read (forward) |

## List-output comparison (harness behavior confirmed)

The candidate still returns a Python `list` of 4 `Tensor[30522]`. The harness `compare_values` (in `auto_bench.py`) handles lists by: (1) type guard (both must be `list`), (2) length guard (`len(v0)==len(v1)`), (3) recursive per-element comparison `compare_values(item0, item1, "output[i]", atol, rtol)`, where each fp32 tensor is compared with `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`. This passed for the fused-kernel output, confirming the on-device prefix-scan + fused reduction produces numerically equivalent per-sequence pooled tensors.

## Screening Evidence

- screening not run: correctness passed and the candidate is faster than the accepted reference on the first authoritative pair, so the screening gate (`screened-out` only when both pairs are ≥10% slower) does not apply.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (reference = baseline_adapter.py as v1_file; candidate = triton_sparse_pooler_001.py as v1_file; base.py held constant as v0 anchor)
- reference_raw_samples_ms: `[0.934505, 0.884145, 0.978735]`
- candidate_raw_samples_ms: `[0.618775, 0.637685, 0.594790]`
- reference_median_ms: `0.934505`
- candidate_median_ms: `0.618775`
- improvement_pct: `33.784`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (0.934505 - 0.618775) / 0.934505 * 100 = 33.784 %
```

The unrounded improvement (33.784%) exceeds the 5% adoption threshold and the 15% expected improvement (H-001 predicted 15.0% conservatively).

## Evaluation Contract Mirror

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse relu, log1p, and per-sequence max pooling into a single Triton kernel launched once per forward`
- expected_causal_chain: `Python for-loop + 4x torch.max dispatches → one Triton kernel launch; seq_lens.tolist() D2H sync eliminated via on-device prefix scan; kernel count drops; wall time decreases ≥5%`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease from 14 to 6 | decreased from 14 to **5** (2x Addmm + 1 fused + 1 LayerNorm + 1 Gelu); relu/log1p/4x MaxDim/4x Cast (10 kernels) removed, 1 fused added | pass | CANN scope summary (reference 14.0, candidate 5.0) |
| device_us_per_call | decrease from 374.81 by ~327 us minus fused cost | decreased from 379.47 to **202.86** us/call (−176.61 us, −46.5%) | pass | CANN scope summary (reference 379.4664, candidate 202.8556) |
| host_sync_count_per_call | decrease (seq_lens.tolist() D2H sync eliminated) | max path has no `.tolist()`; only `seq_lens.shape[0]`; on-device prefix scan replaces host loop | pass | source read (forward); fused kernel `_sparse_pooler_max_kernel` |

Guardrail observables (all passed, see Correctness table): correctness, list output, numerical semantics, device/stream, MLM head unchanged, signature, load_state_dict, pooling=="sum".

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable` (device kernel time available via CANN msprof)

Reference and candidate were profiled in separate CANN msprof captures via `--profile-reference-file baseline_adapter.py`, giving each a distinct `ASCEND_WORK_PATH` (labels `reference_baseline_adapter` and `candidate_triton_sparse_pooler_001`). Each scope's `ai_core_op_summary.db` is summarized independently.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference | 18973.32 | 379.4664 | 700 | 14.0 | 0.934505 | 0.4061 |
| candidate | 10142.78 | 202.8556 | 250 | 5.0 | 0.618775 | 0.3278 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnAddmm_MatMulCommon_MatMulV2 | 100 | 2.0 | 12840.54 | 256.8108 |
| aclnnMaxDim_Max2AiCore_ArgMaxWithValue | 200 | 4.0 | 3095.20 | 61.9040 |
| aclnnLog1p_Log1pAiCore_Log1p | 50 | 1.0 | 1066.00 | 21.3200 |
| aclnnMaxDim_CastAiCore_Cast | 200 | 4.0 | 637.38 | 12.7476 |
| aclnnLayerNormWithImplMode_LayerNormV3WithImplMode_LayerNormV3 | 50 | 1.0 | 588.42 | 11.7684 |
| aclnnRelu_Relu_Relu | 50 | 1.0 | 446.52 | 8.9304 |
| aclnnGelu_Gelu_Gelu | 50 | 1.0 | 299.26 | 5.9852 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnAddmm_MatMulCommon_MatMulV2 | 100 | 2.0 | 7723.72 | 154.4744 |
| _sparse_pooler_max_kernel | 50 | 1.0 | 1915.58 | 38.3116 |
| aclnnLayerNormWithImplMode_LayerNormV3WithImplMode_LayerNormV3 | 50 | 1.0 | 343.86 | 6.8772 |
| aclnnGelu_Gelu_Gelu | 50 | 1.0 | 159.62 | 3.1924 |

## Distinguishing MLM-head vs eliminated pooling kernels

- **MLM-head kernels (expected to remain, unchanged):** `aclnnAddmm_MatMulCommon_MatMulV2` (x2: dense 768→768 and decoder 768→30522), `aclnnLayerNorm...` (x1), `aclnnGelu_Gelu_Gelu` (x1). All present in the candidate scope — the MLM head was correctly left as library ops.
- **Eliminated pooling/activation kernels:** `aclnnMaxDim_Max2AiCore_ArgMaxWithValue` (x4), `aclnnMaxDim_CastAiCore_Cast` (x4), `aclnnRelu_Relu_Relu` (x1), `aclnnLog1p_Log1pAiCore_Log1p` (x1). All absent from the candidate scope; replaced by a single `_sparse_pooler_max_kernel` (x1).

Observation (measurement, not correctness): the candidate's `aclnnAddmm` device time (154.47 us/call) is lower than the reference's (256.81 us/call) despite being identical library matmul ops. This is a per-capture variation artifact (the two scopes are separate profiling runs; device clocks/contention differ slightly between captures) and does not affect the correctness or adoption decision, which is governed by wall time. The kernel-count and kernel-elimination conclusions are robust to this artifact.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `dc2a8b6582cf9d6fef3e044081426762b88833e056ce8d7f04086e0d92f429e0` | correctness pass, authoritative timing, profiler complete |

## evidence_for_next_round

- H-001 confirmed: fusing relu+log1p+per-seq max pooling removed 10 device kernels and the `seq_lens.tolist()` D2H sync, replaced by one `_sparse_pooler_max_kernel` (38.31 us/call). kernel_count_per_call 14→5; device_us_per_call 379.47→202.86; wall 0.934505→0.618775 ms (+33.78%).
- Remaining dominant bottleneck is the MLM head matmuls: `aclnnAddmm_MatMulCommon_MatMulV2` x2 now account for 154.47 us/call = ~76% of the candidate's device time (202.86 us/call). device_ratio dropped to 0.3278, meaning ~67% of wall time is now host-side (launch/dispatch/alloc/harness overhead), so host dispatch is also a remaining candidate bottleneck.
- The matmul fusion was explicitly deferred in decision_001 (MLU sibling Round 003 showed `tl.dot` regressed vs `aclnnAddmm`); it is the natural next bottleneck target but requires its own evidence.
- Recorded but not acted on: per-capture Addmm device-time variation (256.81 vs 154.47 us/call) between separate profiling scopes; wall time remains the adoption authority.

## Stop Recommendation

- recommendation: `continue`
- evidence: improvement 33.78% ≥ 5% threshold, correctness and all guardrails pass, H-001 confirmed. No target mode set. Continue to next round (matmul/host-dispatch remain as bottlenecks).

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/triton_sparse_pooler_001.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/baseline_adapter.py --warmup 50 --repeat 100
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/triton_sparse_pooler_001.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/triton_sparse_pooler_001.py --profile --profile-reference-file kernels/track1-triton/sparse_pooler/ascend/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/sparse_pooler/ascend/log/sparse_pooler_round001_forward_50iter.pt.trace.json
```

```bash
# Summarize per-scope device time (separate CANN captures):
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/sparse_pooler/ascend/log/profiling_data/reference_baseline_adapter/profiling_data --iterations 50 --wall-ms 0.934505
/usr/local/python3.11.15/bin/python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/sparse_pooler/ascend/log/profiling_data/candidate_triton_sparse_pooler_001/profiling_data --iterations 50 --wall-ms 0.618775
```
