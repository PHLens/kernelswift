# Report 001

Result: no-improvement

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_mhc_mix_bwd_001.py`
- Accepted reference: `baseline_adapter.py` (last_accepted_kernel)
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `5ee9ea5d9b74de678482dd801066bf9883d0d0bf76af231f0325689665d5f88d`
- Candidate SHA256: `f7efc6853a8f07b90926237cc2f4de620926bd0b34333648e7355d8995c57d10`
- Accepted reference SHA256: `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d`
- Base SHA256: `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `72952842694ec2990df6b4d83a7750193963ade9a98d045828840df282e35270`
- verification_tier: authoritative
- screening_pairs: `not-run` (correctness passed; proceeded to authoritative timing)

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | v1 output matches v0 (atol=1e-2, rtol=1e-2) | `PASS accuracy` on all 6 runs | pass | `auto_bench.py --warmup 50 --repeat 100` -> PASS |
| output tuple structure | `(grad_input_mix, grad_mhc_scale, grad_mhc_base)` | tuple of 3 tensors, harness `compare_values` matched all | pass | harness tuple recursion |
| grad_input_mix shape | `[2,1024,4]` fp32 | matched (reshaped `[2048,4]` -> `[2,1024,4]`) | pass | harness shape check |
| grad_mhc_scale shape | `[1]` fp32 | matched | pass | harness shape check |
| grad_mhc_base shape | `[4]` fp32 | matched | pass | harness shape check |
| reduction accumulator zero-init | `grad_mhc_base`/`grad_mhc_scale` zero-initialized before atomic_add | `torch.zeros` per-call in forward (line 84-85) | pass | candidate source + correctness |
| dtype | fp32 throughout | matched | pass | harness allclose fp32 |
| ModelNew contract | `forward(input_mix, mhc_scale, mhc_base, grad_out) -> tuple` | conforms | pass | harness loader |

All guardrails pass. Correctness holds for every run.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (6 pairs across two Verifier batches)

reference_raw_medians_ms: `[0.379945, 0.441645, 0.474750, 0.444120, 0.447325, 0.452830]`
candidate_raw_medians_ms: `[0.394690, 0.429085, 0.433335, 0.421225, 0.433885, 0.434960]`

- reference_median_ms: `0.445723`
- candidate_median_ms: `0.431210`
- improvement_pct: `3.256`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (0.445723 - 0.431210) / 0.445723 * 100
               = 3.256%
```

The unrounded improvement (3.26%) is below the 5% adoption threshold. The
reference median itself is noisy (0.380–0.475 ms across runs, ~20% spread), so
the candidate's ~3% edge is within run-to-run variance and does not clear the
adoption bar.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease from 10 to 1 | decreased from 10.0 to **3.0** | fail | profiler: reference 10.0/call, candidate 3.0/call |
| device_us_per_call | decrease | decreased from 41.06 us to 16.85 us (2.4x) | pass | profiler scopes |
| primary_metric wall_time | expected_improvement_pct 5.0 | +3.26% (below threshold) | fail | interleaved timing |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the sigmoid elementwise chain and both reductions into one Triton kernel, collapsing 10 unfused library kernels to 1`
- expected_causal_chain: `10 kernels -> 1 kernel -> host launch overhead down -> wall time down`
- primary_metric: `wall_time`
- Hypothesis verdict: `falsified`

The causal chain is falsified at two points:
1. Kernel count dropped to 3, not 1 — the two per-call `torch.zeros` allocations
   for the reduction accumulators each launch their own `aclnnInplaceZero`/`ZerosLike`
   kernel (2 kernels) on top of the single fused triton kernel.
2. Even though device time fell 2.4x (41.06 -> 16.85 us), wall time did not move
   (~3% within noise). The operator is host-bound (device_ratio ~9% -> ~4%), so
   removing device compute does not remove the host dispatch/launch/allocation
   overhead that dominates wall time.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (Level 1 summary)
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (baseline_adapter.py) | 2053.08 | 41.062 | 500 | 10.0 | 0.445723 | 0.0921 |
| candidate (triton_mhc_mix_bwd_001.py) | 842.50 | 16.850 | 150 | 3.0 | 0.431210 | 0.0391 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
reference:  41.062 / (0.445723 * 1000) = 0.0921
candidate:  16.850 / (0.431210 * 1000) = 0.0391
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 100 | 2.0 | 1121.92 | 22.438 |
| aclnnMul_MulAiCore_Mul | 250 | 5.0 | 446.32 | 8.926 |
| aclnnAdd_AddAiCore_Add | 50 | 1.0 | 269.20 | 5.384 |
| aclnnRsubs_SubAiCore_Sub | 50 | 1.0 | 113.80 | 2.276 |
| aclnnSigmoid_SigmoidAiCore_Sigmoid | 50 | 1.0 | 101.84 | 2.037 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _mhc_mix_bwd_fused_kernel | 50 | 1.0 | 770.72 | 15.414 |
| aclnnInplaceZero_ZerosLikeAiCore_ZerosLike | 100 | 2.0 | 71.78 | 1.436 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial authoritative verification | - | `f7efc6853a8f07b90926237cc2f4de620926bd0b34333648e7355d8995c57d10` | no-improvement (correct, but <5% wall gain) |

No repair was requested: correctness passes; the defect is the mechanism
(kernel count 10->3, host-bound wall not moved), not a local implementation bug.

## evidence_for_next_round

- **Kernel fusion DID collapse device compute**: device_us_per_call fell 41.06 -> 16.85 us (2.4x); the two `aclnnReduceSum` (22.4 us) + 8 elementwise kernels became one `_mhc_mix_bwd_fused_kernel` at 15.41 us.
- **Kernel count is 3, not 1**: the two per-call `torch.zeros` accumulator allocations (`grad_mhc_base[4]`, `grad_mhc_scale[1]`) each emit an `aclnnInplaceZero`/`ZerosLike` kernel (2 kernels, 1.44 us/call). The decision's "10 -> 1" mechanism is falsified to "10 -> 3".
- **`tl.atomic_add` is NOT the bottleneck**: the fused kernel (15.41 us) is cheaper than the two baseline ReduceSum kernels alone (22.44 us), so atomic_add on the tiny [4]/[1] accumulators is device-cheap. It does not explain the missing wall-time gain.
- **Root cause: host-bound.** device_ratio fell from 9.2% to 3.9%, but wall time stayed ~430 us. ~96% of candidate wall time is now host-side (triton kernel launch/dispatch, the 2 zero-init kernel launches, per-call `torch.empty_like`/`torch.zeros` allocation, harness seed setup + device sync). Device-side fusion cannot move wall time when the wall is host-bound.
- **Next lever candidates (recorded as evidence, not prescription):** (a) eliminate the 2 zero-init kernels by writing the accumulators via a full tile store in the fused kernel instead of `torch.zeros` + `atomic_add`, or by zeroing inside the kernel; (b) reduce host dispatch — e.g. a 2-kernel split would still leave ~2-3 launches and not address the host-bound wall; (c) the dominant remaining cost is the harness's own seed setup + `sync_devices()` per iteration, which is fixed for the regime (measurement-bound risk). A host-side or allocation-reuse change may be required to move wall time further, but any such change must stay inside the decision's allowed boundary and preserve the per-call output-allocation semantics.

## Stop Recommendation

- recommendation: `continue`
- evidence: `improvement_pct 3.26% < 5% (no-improvement), performance_miss_streak will increment to 1. valid_no_improvement_limit=3 not yet reached; round budget 20 not reached.`

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/mhc-head-backward-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/ascend/triton_mhc_mix_bwd_001.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/mhc-head-backward-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/ascend/triton_mhc_mix_bwd_001.py --profile --profile-reference-file kernels/track1-triton/mhc_head_compute_mix_backward/ascend/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mhc_head_compute_mix_backward/ascend/log/round_001_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift/.worktrees/mhc-head-backward-ascend/kernels/track1-triton/mhc_head_compute_mix_backward/ascend
/usr/local/python3.11.15/bin/python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py log/profiling_data/reference_baseline_adapter/profiling_data/c8843a4fa93a_452164_20260818230505221_ascend_pt --iterations 50 --wall-ms 0.445723
/usr/local/python3.11.15/bin/python3 /workspace/kernelswift/skills/kernel-opt-loop/scripts/summarize_cann_trace.py log/profiling_data/candidate_triton_mhc_mix_bwd_001/profiling_data/c8843a4fa93a_452164_20260818230510071_ascend_pt --iterations 50 --wall-ms 0.431210
```
