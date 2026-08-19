# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` (`proceed`)
- Candidate: `candidate_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `cfce60f6110bb21802b878f61a6238d89fed0320835560d2cfbd723107b881ef`
- Candidate SHA256: `3eda8a14dede15a91f1a04c37bc5ff178a83fc87ecb7137b3569756c17f94f10`
- Accepted reference SHA256: `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`
- Base SHA256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `52025b1bb12ac09c6a26db2a94fd681e9ac9b325db572734a4af3689a43c38ed`
- verification_tier: authoritative
- screening_pairs: `not-run` (proceeded directly to authoritative timing; no screening requested)

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | candidate output matches base within atol=1e-2, rtol=1e-2 (tuple pre/post/comb element-wise) | PASS accuracy | pass | `auto_bench.py --v0_file base.py --v1_file candidate_001.py --warmup 50 --repeat 100` exit 0 |
| output dtype and shape | `(pre[2,8,4], post[2,8,4], comb[2,8,4,4])` fp32 tuple | matched by harness `compare_values` (shape + fp32 allclose) | pass | correctness gate |
| numerical semantics | `+eps` on pre only; factor 2 on post (no +eps); row_max-stabilized softmax; 20 row + 20 column normalizations | comb maxdiff 8.9e-8 (coder smoke); full harness allclose passes | pass | correctness gate + coder_result_001.md |
| ModelNew contract | `ModelNew(hc_mult=4, sinkhorn_iters=20, eps=1e-6)`, `get_inputs`/`get_init_inputs` | preserved | pass | harness loader resolved `ModelNew`/`get_inputs`/`get_init_inputs` |

Conformance, correctness, and every declared guardrail passed.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (3 pairs; reference = base.py shared `Model`, candidate = `ModelNew` in candidate_001.py)
- reference_raw_samples_ms: `[3.526815, 3.467300, 3.889960]`
- candidate_raw_samples_ms: `[0.392115, 0.394090, 0.386750]`
- reference_median_ms: `3.526815`
- candidate_median_ms: `0.392115`
- improvement_pct: `88.8819`

```text
improvement_pct = (3.526815 - 0.392115) / 3.526815 * 100 = 88.8819%
```

The unrounded improvement (88.88%) far exceeds the 5% adoption threshold.
Authoritative timing controls adoption; profiler time does not substitute.

Note: the accepted reference `baseline_adapter.py` defines only `ModelNew`, while the
harness requires the v0 file to define `Model`. The shared immutable `base.py`
(`Model`) is numerically identical to `baseline_adapter.py` (same semantics; baseline
wall medians 3.396440 vs 3.400635 ms differ only by noise). The authoritative
comparison therefore runs v0=`base.py` against v1=`candidate_001.py`, matching the
established regime from report_000.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease | 136.0 → 1.0 | pass | CANN: reference 6800 kernels/50 calls; candidate 50 kernels/50 calls |
| device_us_per_call | decrease | 282.354 → 8.784 us | pass | CANN: reference 14117.68 us/50; candidate 439.22 us/50 |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `collapse the entire forward into a single Triton kernel with an internal static_range loop, reducing 136 per-call kernel launches to 1`
- expected_causal_chain:
  1. Python for-loop and per-iteration torch reductions replaced by one kernel launch — CONFIRMED (kernel_count_per_call 136 → 1).
  2. kernel_count_per_call drops from 136 to 1 — CONFIRMED.
  3. per-launch host overhead collapses — CONFIRMED (wall 3.53 ms → 0.39 ms).
  4. wall time decreases despite near-identical device compute — CONFIRMED.
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary` (Level 1)
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`,
  `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable` (device time available via CANN)

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations=50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference | 14117.680 | 282.354 | 6800 | 136.00 | 3.526815 | 0.0801 |
| candidate | 439.220 | 8.784 | 50 | 1.00 | 0.392115 | 0.0224 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
reference: 282.354 / 3526.815 = 0.0801
candidate: 8.784 / 392.115 = 0.0224
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 2000 | 40.0 | 8429.420 | 168.588 |
| aclnnDiv_RealDivAiCore_RealDiv | 2000 | 40.0 | 2425.540 | 48.511 |
| aclnnAdds_AddAiCore_Add | 2050 | 41.0 | 2031.540 | 40.631 |
| aclnnMul_SliceAiCore_Slice | 150 | 3.0 | 399.920 | 7.998 |
| aclnnAmax_ReduceMaxAiCore_ReduceMax | 50 | 1.0 | 236.520 | 4.730 |
| aclnnAdd_AddAiCore_Add | 150 | 3.0 | 209.220 | 4.184 |
| aclnnMul_MulAiCore_Mul | 150 | 3.0 | 145.020 | 2.900 |
| aclnnSigmoid_SigmoidAiCore_Sigmoid | 100 | 2.0 | 88.340 | 1.767 |
| aclnnSub_SubAiCore_Sub | 50 | 1.0 | 61.060 | 1.221 |
| aclnnMuls_MulAiCore_Mul | 50 | 1.0 | 50.260 | 1.005 |
| aclnnExp_ExpAiCore_Exp | 50 | 1.0 | 40.840 | 0.817 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _mhc_head_compute_mix_kernel | 50 | 1.0 | 439.220 | 8.784 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial authoritative verification | not-applicable | `3eda8a14dede15a91f1a04c37bc5ff178a83fc87ecb7137b3569756c17f94f10` | accepted |

## evidence_for_next_round

- The sinkhorn-loop-fusion intervention is confirmed: kernel_count_per_call
  dropped from 136 to 1 and device_us_per_call from 282.4 us to 8.8 us, yielding
  an 88.88% wall-time improvement (3.5268 ms → 0.3921 ms).
- The operator is now launch-bound on a single fused Triton kernel. The candidate
  device_ratio is 0.0224, so the ~390 us wall time is still dominated by host-side
  cost (single kernel launch + forward wrapper: reshape/to/contiguous/empty
  allocations + view). The device kernel itself is only ~8.8 us.
- Remaining compressible host work: the forward allocates 3 output tensors per call
  (`torch.empty`), and does `.to(fp32).contiguous()` on inputs each call. A Host
  Plan (allocation reuse / output caching) may reduce the residual wall time, but
  this is Level 2 evidence and not yet measured.
- Next bottleneck is host launch/allocation overhead of a single kernel, not the
  Sinkhorn loop (already fused).

## Stop Recommendation

- recommendation: `continue`
- evidence: 88.88% improvement accepted; residual wall time (~390 us) still
  host-bound (device_ratio 0.022). Further host-side optimization (allocation
  reuse) may yield additional gains.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/mhc-head-compute-mix-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/ascend/candidate_001.py --warmup 50 --repeat 100
```

```bash
python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/ascend/candidate_001.py --profile --profile-reference-file kernels/track1-triton/mhc_head_compute_mix/ascend/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mhc_head_compute_mix/ascend/log/round_001_forward_50iter.pt.trace.json
```

```bash
python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py <scope_ascend_pt_dir> --iterations 50 --wall-ms <scope_wall_ms>
```
