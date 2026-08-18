# Report 002

Result: accepted

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_fused_moe_002.py`
- Accepted reference: `triton_fused_moe_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `aa051ac4ff036222154badc10ae2051560396720cd9fedbb4f3a8d4f755c9ec2`
- Candidate SHA256: `1b5c8ecded2008991f0f7cc039f0e06fa072bf5b8a7c6d5630574f64a43f4074`
- Accepted reference SHA256: `e42d811a7aee32f3fb34b19c00f88bf7922129faccc3d670dd07abc2df443287`
- Base SHA256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `47e60b0db91c4c67e55f92cf79f5dddf591925620ec4db38704dfb42f0f185dd`
- verification_tier: authoritative
- screening_pairs: `not-run (candidate correctness passes and first pair is clearly faster; proceeded to authoritative timing)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass (torch.allclose atol=1e-2, rtol=1e-2) | `PASS accuracy; v0=8.740895 ms, v1=0.430155 ms, speedup=20.320x` | pass | `auto_bench.py --v0_file base.py --v1_file triton_fused_moe_002.py --warmup 5 --repeat 10 --full-traceback` |
| output shape [83,128] fp16 unchanged | preserved | correctness PASS implies shape/dtype match (harness `compare_values`) | pass | harness `compare_values` |
| softmax+topk+renormalize routing over E=8 preserved | preserved | kernel computes fp32 softmax (max-subtract+exp+sum), top-2 by repeated argmax, fp32 renormalize, then fp16 cast; correctness PASS against base.py confirms numerical equivalence including top-2 selection | pass | `triton_fused_moe_002.py` kernel lines 40-62 |
| weighted top-k reduce over exactly 2 experts preserved | preserved | `tl.static_range(0,K)` K=2, accumulates `weight * out_k` | pass | `triton_fused_moe_002.py` kernel lines 69-97 |
| no tl.dot introduced | preserved | FFN uses `tl.sum` rank-1 outer-products only; no `tl.dot` anywhere | pass | source review |

Top-2 selection equivalence confirmed: the kernel's repeated-argmax (`tl.argmax` returns lowest index on ties, matching `torch.topk`) over E=8 fp32 `randn` logits matches base.py's `torch.topk(scores, 2)`; correctness PASS at atol=1e-2/rtol=1e-2 confirms the selected experts and weights agree with base.py on the actual benchmark inputs.

## Screening Evidence

Not run. Candidate correctness passed and the first authoritative pair was clearly faster;
no 10%-slower screen was possible.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `three interleaved reference(001)/candidate(002) pairs (base.py as the constant Model anchor for the harness v0 slot)`
- reference_raw_samples_ms: `not-collected (harness reports median only)`
- candidate_raw_samples_ms: `not-collected (harness reports median only)`

| Pair | Reference wall ms (001) | Candidate wall ms (002) | Evidence |
|---:|---:|---:|---|
| 1 | 0.545465 | 0.406495 | `auto_bench.py --v0_file base.py --v1_file <001/002> --warmup 50 --repeat 100` |
| 2 | 0.598405 | 0.352400 | same |
| 3 | 0.575985 | 0.368980 | same |

- reference_median_ms: `0.575985`
- candidate_median_ms: `0.368980`
- improvement_pct: `35.939304`

```text
improvement_pct = (0.575985 - 0.368980) / 0.575985 * 100 = 35.939304
```

The unrounded improvement (35.94%) far exceeds the 5% adoption threshold.
Correctness passed, so the terminal result is `accepted`.

Note on timing mechanics: the harness requires `--v0_file` to define `Model` and
`--v1_file` to define `ModelNew`. Both the accepted reference (001) and the
candidate (002) define only `ModelNew`, so neither can occupy the `--v0_file`
slot. To obtain interleaved reference/candidate wall samples with byte-for-byte
identical flags, each run used `base.py` as the constant `Model` anchor and
alternated `--v1_file` between 001 (reference) and 002 (candidate), extracting
each `v1_ms` median. The reported medians are the unrounded per-file `v1_ms`
values; the base.py anchor `v0_ms` is ignored (it is the harness's required
Model slot, not the round's accepted reference).

## Evaluation Contract Mirror

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: `fuse routing (softmax over E=8, top-2 argmax selection, renormalize) into the existing per-token Triton kernel so forward passes router_logits directly and the 11 PyTorch routing kernels disappear; keep the elementwise tl.sum FFN path and do NOT introduce tl.dot`
- expected_causal_chain: `11 PyTorch routing kernels disappear → kernel_count 12→1 → device_us ~97→~20 → host dispatch decreases → wall_time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease from 12 toward 1 | 12 → **3** | pass | candidate kernel_count_per_call=3.0 (vs reference 12.0) |
| device_us_per_call | decrease from ~97 us/call | 94.9 → **26.7** | pass | candidate device_us_per_call=26.678 vs reference 94.931 |
| aclnnTopk_presence | absent from candidate scope | absent (0 occurrences) | pass | candidate top-kernels has no `aclnnTopk*` |
| aclnnSoftmax_presence | absent from candidate scope | absent (0 occurrences) | pass | candidate top-kernels has no `aclnnSoftmax*` |

Note: kernel_count reached 3, not exactly 1, because `forward` still performs two
`w1 = self.w1.to(dtype)` / `w2 = self.w2.to(dtype)` fp16 casts (2x
`aclnnInplaceCopy_Cast`) plus the single fused `_fused_moe_per_token_kernel`. All
routing kernels (Topk, Softmax, ReduceSum, Div, GatherElements) are fully
eliminated, exactly as the intervention predicted.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (Level 1)
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`,
  `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable (device time available via CANN msprof)`

Reference (001) and candidate (002) profiled in separate CANN msprof captures
(via `--profile-reference-file triton_fused_moe_001.py`), summarized
independently with `summarize_cann_trace.py`. All totals normalized by
`iterations=50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference (triton_fused_moe_001.py) | 4746.56 | 94.931 | 600 | 12.0 | 0.575985 | 0.164815 |
| candidate (triton_fused_moe_002.py) | 1333.90 | 26.678 | 150 | 3.0 | 0.368980 | 0.072302 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### Reference Top Kernels (triton_fused_moe_001.py)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnTopk_TopkV2AiCore_TopKV2 | 100 | 2.0 | 1930.62 | 38.612 |
| _fused_moe_per_token_kernel | 50 | 1.0 | 1043.58 | 20.872 |
| aclnnSoftmax_SoftmaxAiCore_SoftmaxV2 | 50 | 1.0 | 505.16 | 10.103 |
| aclnnInplaceCopy_CastAiCore_Cast | 200 | 4.0 | 480.76 | 9.615 |
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 50 | 1.0 | 354.56 | 7.091 |
| aclnnTopk_GatherElements_GatherElements | 50 | 1.0 | 299.34 | 5.987 |
| aclnnDiv_RealDivAiCore_RealDiv | 50 | 1.0 | 81.04 | 1.621 |
| aclnnTopk_CastAiCore_Cast | 50 | 1.0 | 51.50 | 1.030 |

### Candidate Top Kernels (triton_fused_moe_002.py)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _fused_moe_per_token_kernel | 50 | 1.0 | 1096.24 | 21.925 |
| aclnnInplaceCopy_CastAiCore_Cast | 100 | 2.0 | 237.66 | 4.753 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial Round 2 verification | `1b5c8ecded2008991f0f7cc039f0e06fa072bf5b8a7c6d5630574f64a43f4074` | `1b5c8ecded2008991f0f7cc039f0e06fa072bf5b8a7c6d5630574f64a43f4074` | correctness PASS, accepted |

No repair was required.

## evidence_for_next_round

- Fusing routing removed all 11 routing kernels: kernel_count 12 → 3,
  device_us 94.9 → 26.7, wall 0.576 → 0.369 ms (35.9% improvement). Confirmed.
- The candidate device time is now ~22 us of the single fused Triton kernel plus
  ~4.8 us of two fp16 `w1/w2.to(dtype)` casts. The fused kernel itself (21.9 us)
  is essentially unchanged from Round 1 (20.9 us) — routing fusion removed the
  surrounding PyTorch kernels without changing the FFN compute.
- Candidate device_ratio dropped to 0.072: device work is now a small fraction of
  wall time, and the remaining ~0.37 ms wall is dominated by host-side launch +
  Python dispatch (single Triton launch + 2 casts + `torch.empty_like`).
- Remaining device kernels: the two `aclnnInplaceCopy_Cast` (from `w1`/`w2`
  `.to(dtype)` casts) are the only non-Triton kernels left. The kernel count is
  already near-minimal (3). Further wall reduction would target host-side
  overhead (e.g. cast removal / launch overhead), not device kernels.

## Stop Recommendation

- recommendation: `continue`
- evidence: `large accepted improvement (35.9%); no stop condition met (not target-reached, streak 0, round 2 of 20).`

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/ascend/triton_fused_moe_002.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/ascend/triton_fused_moe_001.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/ascend/triton_fused_moe_002.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/ascend/triton_fused_moe_002.py --warmup 50 --repeat 100 --profile --profile-reference-file kernels/track1-triton/fused_moe/ascend/triton_fused_moe_001.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/fused_moe/ascend/log/fused_moe_round_002_forward_50iter.pt.trace.json
```

CANN profiler summarization:

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py \
  kernels/track1-triton/fused_moe/ascend/log/profiling_data/reference_triton_fused_moe_001/profiling_data/<ts>_ascend_pt \
  --iterations 50 --scope reference_triton_fused_moe_001 --wall-ms 0.575985
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py \
  kernels/track1-triton/fused_moe/ascend/log/profiling_data/candidate_triton_fused_moe_002/profiling_data/<ts>_ascend_pt \
  --iterations 50 --scope candidate_triton_fused_moe_002 --wall-ms 0.368980
```

Raw profiler trace: `kernels/track1-triton/fused_moe/ascend/log/fused_moe_round_002_forward_50iter.pt.trace.json`
