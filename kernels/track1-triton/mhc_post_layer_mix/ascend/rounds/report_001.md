# Report 001 — Kernel Fusion Candidate

- round: `001`
- result: `accepted`
- change_family: `kernel-fusion`
- bottleneck_class: `device-bound`

## Decision

- decision: `proceed` (decision_001.md, H-001 kernel-fusion)
- candidate: `candidate_001.py` (single Triton kernel `mhc_fused_kernel`)
- accepted-reference: `baseline_adapter.py`

## Source Hashes

| Artifact | SHA-256 |
|---|---|
| base `../base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` |
| accepted reference `baseline_adapter.py` | `a4f0aa8ac2d59c57059223b1710d20718af1b0f892cd7c373174e531c927133e` |
| candidate `candidate_001.py` | `b74e407348d424c9265ddf831b245cda90297a48bdbaa576fa7e6b57b5d121f9` |
| decision `decision_001.md` | `6c9bf2b10c30b3a1205fe3c94f3ba4b6dc8abe1f2753b41536bdf7e29acb32ad` |

## Correctness and Guardrails

Correctness PASS in all three timing pairs and the profiler run (atol=1e-2 / rtol=1e-2).

| Guardrail | Expectation | Observation | Verdict |
|---|---|---:|---|
| correctness | pass within atol/rtol 1e-2 | PASS (all runs) | pass |
| output dtype/shape | `Tensor[2,4096,4,1280]` bf16 | PASS (return `out.reshape(n0,n1,mhc,h)`, bf16) | pass |
| fp32 accumulation before single bf16 cast | fp32 acc, single cast | explicit fp32 FMA reduction, one `.to(tl.bfloat16)` | pass |
| ModelNew public contract | `ModelNew().forward(...)` | present | pass |
| input tensors unmodified | no in-place on inputs | loads only, `torch.empty` output | pass |
| get_inputs returns 4 tensors | 4 tensor args | returns `[x, residual, post_layer_mix, comb_res_mix]` | pass |

## Authoritative Timing (3 interleaved pairs)

Command (v0=base.py as immutable reference proxy; v1=candidate; warmup 50 / repeat 100, npu:0):

| Pair | v0 (ref) median ms | v1 (candidate) median ms | speedup |
|---|---:|---:|---:|
| 1 | 3.194895 | 0.876125 | 3.647x |
| 2 | 3.227110 | 0.887760 | 3.635x |
| 3 | 3.197965 | 0.879715 | 3.635x |

- Unrounded median of v0 samples ≈ 3.198 ms; candidate ≈ 0.880 ms.
- Speedup ≈ 3.64x, improvement_pct ≈ 264% — far above the 5.0 adoption threshold.

Note: the harness `--v0_file` must define `Model` (not `ModelNew`), so the
immutable `base.py` is the v0 reference. `base.py` and `baseline_adapter.py`
are byte-semantically identical (established in Phase 0), so this is a valid
comparison against the accepted reference.

## Profiler Evidence (CANN per-scope, forward, 20 warmup / 50 iter)

### reference_baseline_adapter
- device_us_per_call: 3093.30
- kernel_count_per_call: 6.0
- device_ratio: 0.9662 (vs wall 3.201345 ms)
- top kernels: BatchMatMul (1111.40 us), Add (903.67 us), Cast ×3 (814.87 us), Mul (263.36 us)

### candidate_candidate_001
- device_us_per_call: 619.76
- kernel_count_per_call: 1.0
- device_ratio: 0.7044 (vs wall 0.879860 ms)
- top kernel: `mhc_fused_kernel` (619.76 us/call, count 1.0)

## Evaluation Contract Mirror (H-001)

| Mechanism observable | Expectation | Observation | Verdict |
|---|---|---:|---|
| kernel_count_per_call | decrease from 6.0 to 1.0 | 6.0 → 1.0 | confirmed |
| device_us_per_call | decrease | 3093.30 → 619.76 us (−80%) | confirmed |
| cast_kernel_eliminated | `aclnnInplaceCopy_CastAiCore_Cast` disappears from top-k | all 6 baseline kernels (incl. Cast) replaced by single `mhc_fused_kernel` | confirmed |
| wall_time | improve ≥5% | 3.198 → 0.880 ms (+264%) | confirmed |

## Hypothesis Verdict

`confirmed` — the six baseline kernels collapse into a single fused Triton
kernel; device_us_per_call falls ~80% (3093→620 us) and wall time falls ~3.64x.
The causal chain (intermediate cast kernels disappear → kernel count 6→1 →
device time falls → wall time falls) is fully observed.

## Retry History

- Round 000 (Phase 0): baseline established after a baseline-adapter defect
  (incident_000) was repaired by Orchestrator. No candidate retries in round 001;
  correctness passed on first authoritative run.

## evidence_for_next_round

- Observed: fused kernel achieves device_us_per_call ≈ 620 us (single kernel),
  but device_ratio drops to 0.70 (host-side launch/overhead now a larger
  fraction of the ~0.88 ms wall time). The remaining ~0.26 ms gap between
  device (620 us) and wall (880 us) is host overhead + launch latency, now more
  exposed because the kernel is so much faster.
- Falsified/remaining: the single kernel is now the only device-side cost; the
  matmul contraction is explicit 4-way fp32 FMA (no tl.dot). Potential further
  gains may lie in reducing host launch overhead or tuning the kernel
  (block/warp config), but this is left to Designer.
- Bottleneck: device-bound fusion succeeded; wall time is now ~0.88 ms with
  device_ratio ~0.70, indicating growing host-overhead share.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/mhc-post-layer-mix-ascend
/usr/local/python3.11.15/bin/python3 auto_bench.py \
  --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py \
  --v1_file kernels/track1-triton/mhc_post_layer_mix/ascend/candidate_001.py \
  --warmup 50 --repeat 100

/usr/local/python3.11.15/bin/python3 auto_bench.py \
  --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py \
  --v1_file kernels/track1-triton/mhc_post_layer_mix/ascend/candidate_001.py \
  --profile --profile-reference-file kernels/track1-triton/mhc_post_layer_mix/ascend/baseline_adapter.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output kernels/track1-triton/mhc_post_layer_mix/ascend/log/round_001_forward_50iter.pt.trace.json
```
