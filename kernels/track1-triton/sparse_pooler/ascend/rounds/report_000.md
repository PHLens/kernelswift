# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py for Phase 0`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `94d00f1a5d26f453fd5078fd9d50dfcddbb0c11d20a145d223544e59234add0f`
- Accepted reference SHA256: `2b740bba37a87a7bcb022af36537486179538feed5dada3f3c1d5e32cd3f6c36`
- Base SHA256: `2b740bba37a87a7bcb022af36537486179538feed5dada3f3c1d5e32cd3f6c36`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `f4305d20c3f39dba64e252050fcc6cb437a1ba7a24fb0480530287bcd4e7a6e1`
- verification_tier: `baseline`
- screening_pairs: `not-run`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | PASS accuracy | `PASS accuracy; v0=1.001665 ms, v1=1.013420 ms` (short warmup 5 / repeat 10) | pass | `auto_bench.py --v0_file base.py --v1_file baseline_adapter.py --warmup 5 --repeat 10` |

No Evaluation Contract guardrails exist at Phase 0 (no round decision). The only gate is correctness, which passed.

## Correctness — list-output comparison mechanism

This operator returns a Python `list` of 4 `Tensor[30522]` (NOT a single tensor). The harness's `compare_values` (in `auto_bench.py`) handles list outputs recursively:

1. Type guard: `if isinstance(v0, list) or isinstance(v1, list)` — if one side is a list and the other is not, it raises `output type mismatch`.
2. Length guard: `len(v0) != len(v1)` raises `list length mismatch: 4 vs ...`.
3. Element-wise recursion: for each index `i`, it calls `compare_values(item0, item1, f"output[{i}]", atol, rtol)`, so each of the 4 tensors is compared independently.

Each tensor element comparison (fp32) uses `torch.allclose(lhs, rhs, atol=1e-2, rtol=1e-2, equal_nan=True)`, with `rhs` moved to `lhs.device`. On mismatch it reports `max_abs_diff` / `mean_abs_diff`.

Verdict: the harness compares list outputs correctly, element-wise across all 4 tensors, with shape/type/length pre-checks.

## Host-cost characteristic — `seq_lens.tolist()` D2H sync

Both `base.py` (line 29) and `baseline_adapter.py` (line 21) call `seq_lens.tolist()` inside `forward`. On Ascend NPU, `.tolist()` on a device tensor forces a device-to-host (D2H) copy + synchronization, because the host Python loop `for L in seq_lens.tolist()` iterates over concrete host ints. This is a key baseline host-cost source: every forward call pays one NPU→CPU sync to materialize the 4 sequence lengths, then issues 4 separate `chunk.max(dim=0)` reductions whose slices depend on those host values. This is recorded as an observed fact for future rounds (the candidate must preserve this exact list semantics unless a design decision explicitly changes the dataflow/lifecycle).

Note on `get_inputs`: both files use `device="cuda"` for `hidden_states` and `seq_lens`; the harness `_rewrite_device_for_backend` remaps the `"cuda"` string literal to `"npu"` at AST load time (`_auto_accel_name()` returns `npu` on this host). Confirmed working — the run produced NPU tensors and passed.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `Phase 0 baseline (reference=base.py, candidate=baseline_adapter.py)`
- reference_raw_samples_ms: `[single run median]`
- candidate_raw_samples_ms: `[single run median]`
- reference_median_ms: `0.935560`
- candidate_median_ms: `0.939350`
- improvement_pct: `-0.405`

```text
improvement_pct = (0.935560 - 0.939350) / 0.935560 * 100 = -0.405 %
```

The baseline adapter is the reference renamed `Model`→`ModelNew`; it is expected to match, not improve. `improvement_pct` is trivially near zero (within run-to-run noise), confirming semantic identity.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `wall_time`
- Hypothesis verdict: `inconclusive`

No round decision exists at Phase 0, so there are no `mechanism_observables` to mirror.

## Profiler Evidence

- profiler_applicability: `required` (baseline Level 1 summary)
- profiler_level: `summary`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable` (device kernel time available via CANN msprof)

Reference and candidate scopes are collected in separate CANN msprof captures (distinct `ASCEND_WORK_PATH` per scope). The chrome trace only retains the candidate scope (`candidate_baseline_adapter`) because `_export_profile_npu` calls `export_chrome_trace` inside the per-scope loop (overwrites); however each scope's `ai_core_op_summary.db` is written to its own directory, so each DB is summarized independently without time-range isolation.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (base.py) | 18740.52 | 374.8104 | 700 | 14.0 | 0.935560 | 0.4006 |
| candidate (baseline_adapter) | 18917.58 | 378.3516 | 700 | 14.0 | 0.939350 | 0.4028 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnAddmm_MatMulCommon_MatMulV2 | 100 | 2.0 | 12605.18 | 252.1036 |
| aclnnMaxDim_Max2AiCore_ArgMaxWithValue | 200 | 4.0 | 3088.62 | 61.7724 |
| aclnnLog1p_Log1pAiCore_Log1p | 50 | 1.0 | 1120.56 | 22.4112 |
| aclnnMaxDim_CastAiCore_Cast | 200 | 4.0 | 603.64 | 12.0728 |
| aclnnLayerNormWithImplMode_LayerNormV3WithImplMode_LayerNormV3 | 50 | 1.0 | 569.10 | 11.3820 |
| aclnnRelu_Relu_Relu | 50 | 1.0 | 478.36 | 9.5672 |
| aclnnGelu_Gelu_Gelu | 50 | 1.0 | 275.06 | 5.5012 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnAddmm_MatMulCommon_MatMulV2 | 100 | 2.0 | 12794.44 | 255.8888 |
| aclnnMaxDim_Max2AiCore_ArgMaxWithValue | 200 | 4.0 | 3086.86 | 61.7372 |
| aclnnLog1p_Log1pAiCore_Log1p | 50 | 1.0 | 1122.08 | 22.4416 |
| aclnnMaxDim_CastAiCore_Cast | 200 | 4.0 | 607.20 | 12.1440 |
| aclnnLayerNormWithImplMode_LayerNormV3WithImplMode_LayerNormV3 | 50 | 1.0 | 562.16 | 11.2432 |
| aclnnRelu_Relu_Relu | 50 | 1.0 | 478.08 | 9.5616 |
| aclnnGelu_Gelu_Gelu | 50 | 1.0 | 266.76 | 5.3352 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `94d00f1a5d26f453fd5078fd9d50dfcddbb0c11d20a145d223544e59234add0f` | correctness pass, baseline recorded |

## evidence_for_next_round

- Baseline wall time: reference median `0.935560 ms`, candidate median `0.939350 ms` (warmup 50 / repeat 100).
- Baseline device time: `374.8104 us/call` (reference), `378.3516 us/call` (candidate); 14 kernels/call each; device_ratio ≈ 0.40 (device time is ~40% of wall time, so ~60% is host-side).
- Dominant device kernel is `aclnnAddmm_MatMulCommon_MatMulV2` at `252.1 us/call` (2 calls: MLM head dense + decoder matmul), ~67% of device time. Second is `aclnnMaxDim_Max2AiCore_ArgMaxWithValue` at `61.77 us/call` (4 calls, one per sequence chunk).
- Host-side cost is significant (device_ratio ≈ 0.40); `seq_lens.tolist()` forces a D2H sync per forward call, then 4 sequential `chunk.max(dim=0)` reductions driven by host-loop values. This is a candidate bottleneck for future optimization (but must preserve list-of-4-tensor output semantics).
- Output is a Python list of 4 `Tensor[30522]`; the harness compares it element-wise correctly (see Correctness section). Any future candidate must keep returning a list of 4 tensors.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline established; no optimization rounds run yet.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/sparse_pooler/ascend/log/sparse_pooler_baseline_forward_50iter.pt.trace.json
```

```bash
# Summarize per-scope device time (separate CANN captures):
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
/usr/local/python3.11.15/bin/python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/sparse_pooler/ascend/log/profiling_data/baseline_base/profiling_data --iterations 50 --wall-ms 0.935560
/usr/local/python3.11.15/bin/python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/sparse_pooler/ascend/log/profiling_data/candidate_baseline_adapter/profiling_data --iterations 50 --wall-ms 0.939350
```
