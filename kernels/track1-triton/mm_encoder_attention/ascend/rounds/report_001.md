# Report 001

Result: no-improvement

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_attn_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `fa6ffd3d2a08dd78d2f3ad958890d0419a0115b898c68b6bbf4ef88105d43eca`
- Candidate SHA256: `61eeb3367619684e6f61ea3a908c1fc78a575834b4a84c032748277d0e76be74`
- Accepted reference SHA256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `pending` (owned by Orchestrator)
- verification_tier: authoritative
- screening_pairs: `not-run` (candidate is correct; proceeded to authoritative timing)

Note on reference identity: the harness requires the v0 file to define `Model` and the v1 file to define `ModelNew`. The canonical accepted reference `baseline_adapter.py` defines `ModelNew` (it is the Phase 0 renamed copy of `base.py`). Therefore the authoritative wall-time comparison is `base.py` (defines `Model`, byte-identical logic to `baseline_adapter.py`) as v0 vs `triton_attn_001.py` as v1. The profiler reference scope is `baseline_adapter.py` (via `--profile-reference-file`), which is the canonical accepted reference. `base.py` and `baseline_adapter.py` are semantically identical (differ only by the `Model`→`ModelNew` class rename).

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | candidate output matches reference within atol=1e-2, rtol=1e-2 (equal_nan=True) | `PASS accuracy` on all three interleaved pairs | pass | `auto_bench.py --v0_file .../base.py --v1_file .../triton_attn_001.py --warmup 50 --repeat 100` |
| output dtype and shape | `Tensor[2,83,512]` fp16 unchanged | shape [2,83,512], fp16 (verified via `__main__` coder gate + correctness compare_values) | pass | `python3 triton_attn_001.py` prints `torch.Size([2, 83, 512])` |
| fp16 within tolerance | within atol=1e-2 rtol=1e-2 | `PASS accuracy` | pass | harness allclose |
| non-GQA head arithmetic | num_heads=8, head_size=64 preserved | head h indexed at `h*HEAD_SIZE + d`, 1D grid `bsz*heads*seq`, heads=8, head_size=64 | pass | candidate source lines 24-30, 81-92 |
| ModelNew public contract | `ModelNew(num_heads=8, head_size=64, num_kv_heads=8)` | `get_init_inputs` returns [8,64,8], `ModelNew.__init__` matches | pass | candidate source lines 68-75, 113-115 |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (3 pairs in one Verifier turn)
- reference_raw_samples_ms: `[0.348605, 0.346175, 0.357905]`
- candidate_raw_samples_ms: `[0.339685, 0.336795, 0.356130]`
- reference_median_ms: `0.348605`
- candidate_median_ms: `0.339685`
- improvement_pct: `2.5588`

```text
improvement_pct = (0.348605 - 0.339685) / 0.348605 * 100 = 2.5588
```

The unrounded improvement is 2.56%, below the 5% adoption threshold. Classified `no-improvement`.

Per-pair detail:

| Pair | Reference ms | Candidate ms | speedup |
|---:|---:|---:|---:|
| 1 | 0.348605 | 0.339685 | 1.026x |
| 2 | 0.346175 | 0.336795 | 1.028x |
| 3 | 0.357905 | 0.356130 | 1.005x |

## Evaluation Contract Mirror

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: replace F.scaled_dot_product_attention and its surrounding view/transpose/reshape with a single Triton attention kernel that reads the native [bsz, seq, num_heads*head_size] contiguous layout via strided loads and writes the same layout directly, eliminating the three .contiguous() transpose kernels and the output InplaceCopy transpose kernel
- expected_causal_chain: `[layout kernels disappear, kernel_count_per_call ~6.7→~1, device_us_per_call decreases by ~62us, host launch gaps shrink, wall time decreases]`
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed`

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease | 6.78 → 1.0 (single `_mm_enc_attn_kernel`) | pass | CANN summary: candidate kernel_count_per_call=1.0 |
| device_us_per_call | decrease | 118.94 → 104.15 us (−14.79 us, −12.4%), far short of predicted −62 us | partially-confirmed | CANN summary: candidate device_us_per_call=104.15 |
| transpose_wrapper_kernel_count_per_call | decrease toward zero | 0 (all 3 Transpose + 1 InplaceCopy layout kernels eliminated) | pass | CANN summary: candidate has zero `TransposeAiCore` kernels |

The causal chain is only partially realized. The layout-kernel elimination and kernel-count collapse succeeded (6.78→1.0, transpose wrappers→0), but the expected device-time saving was not achieved: device time fell only ~14.8 us, not ~62 us, because the single Triton kernel costs 104.15 us on device — 4.5× the native flash-attention kernel's 23.35 us. The Triton materialized-attention formulation (rank-1 `tl.sum` reductions, one program per (batch,head,seq), full 83×64 K/V block loads) is compute-inefficient on Ascend910B4, replacing cheap native FA math + layout with expensive Triton compute.

## Profiler Evidence

- profiler_applicability: `required` (targeted Level 2 per Evaluation Contract)
- profiler_level: `targeted`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable`

Reference (baseline_adapter) and candidate (triton_attn_001) scopes collected independently. All totals normalized by `iterations=50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_baseline_adapter | 5947.02 | 118.9404 | 339 | 6.78 | 0.361855 | 0.3287 |
| candidate_triton_attn_001 | 5207.48 | 104.1496 | 50 | 1.0 | 0.347425 | 0.2998 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
```

### Accepted Reference Top Kernels (reference_baseline_adapter)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnFlashAttentionScore_TransposeAiCore_Transpose | 150 | 3.0 | 2406.58 | 48.1316 |
| EVENT_WAIT_SQE | 50 | 1.0 | 1626.44 | 32.5288 |
| aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore | 50 | 1.0 | 1167.58 | 23.3516 |
| aclnnInplaceCopy_TransposeAiCore_Transpose | 50 | 1.0 | 745.64 | 14.9128 |
| EVENT_RECORD_SQE | 39 | 0.78 | 0.78 | 0.0156 |

### Candidate Top Kernels (candidate_triton_attn_001)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _mm_enc_attn_kernel | 50 | 1.0 | 5207.48 | 104.1496 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification (authoritative timing + profiler) | `not-applicable` | `61eeb3367619684e6f61ea3a908c1fc78a575834b4a84c032748277d0e76be74` | correctness pass; 2.56% improvement (< 5%); no-improvement |

No Verifier-to-Coder repair was needed (correctness passed on first verification).

## evidence_for_next_round

- **Mechanism confirmed (layout elimination)**: kernel_count_per_call collapsed from 6.78 → 1.0, and all four transpose/inplace-copy layout kernels were eliminated (transpose_wrapper_kernel_count → 0). The decision's layout-fusion hypothesis is structurally correct.
- **Mechanism falsified (device-time saving)**: device_us_per_call fell only 118.94 → 104.15 us (−14.8 us), not the predicted ~62 us. The single Triton kernel `_mm_enc_attn_kernel` costs 104.15 us on device — 4.5× the native `aclnnFlashAttentionScore` kernel's 23.35 us. The Triton materialized-attention formulation (per-(batch,head,seq) program, rank-1 `tl.sum` QK^T and AV reductions, full 83×64 K/V block loads with `tl.where` masking) is compute-inefficient on Ascend910B4.
- **Wall time is host-bound**: device_ratio is only ~0.30-0.33. The candidate still spends ~0.34 ms wall vs ~0.104 ms device; the dominant cost is host-side launch + synchronization (single-kernel launch overhead + `torch.empty` alloc + `contiguous()` no-op checks + `sync_devices()`). Eliminating device layout kernels does not move wall time much when device time is not the bottleneck.
- **Current bottleneck**: wall time is dominated by host-side launch/synchronization overhead (device_ratio < 0.33). Any device-side kernel rewrite alone cannot exceed ~5% wall improvement while host overhead dominates. The native flash-attention path's device time (~119 us) is already small relative to ~0.35 ms wall; the residual win from a kernel rewrite is bounded by the ~104 us of Triton compute vs ~23 us native FA, which currently does not beat native FA even on device.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 001 candidate is a valid no-improvement (2.56% < 5%). No target set; valid_no_improvement_limit=3 not yet reached; round budget (20) not exhausted.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/mm-encoder-attn-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/triton_attn_001.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/mm-encoder-attn-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/triton_attn_001.py --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/ascend/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mm_encoder_attention/ascend/log/round_001_forward_50iter.pt.trace.json
```

```bash
/usr/local/python3.11.15/bin/python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/log/profiling_data/reference_baseline_adapter/profiling_data/c8843a4fa93a_295492_20260818153648211_ascend_pt" --iterations 50 --scope reference_baseline_adapter --wall-ms 0.361855

/usr/local/python3.11.15/bin/python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/log/profiling_data/candidate_triton_attn_001/profiling_data/c8843a4fa93a_295492_20260818153653054_ascend_pt" --iterations 50 --scope candidate_triton_attn_001 --wall-ms 0.347425
```
