# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 source reference; no pre-existing canonical implementation)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `66a3a2c31863d18c725a52ab57fd1b9f89fe655dd7bab7cb4da158b8130b5d07`
- Accepted reference SHA256: `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3`
- Base SHA256: `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `c17c7f45d44b1bec047c7c3a315275d33b21af3226dd34e884d483899ef039b6`
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`
- completed_at: `2026-08-18T18:30:00Z`

The adapter, base, and harness hashes match the frozen project values exactly.
The adapter SHA-256 (`66a3a2c3...`) reflects the Orchestrator-repaired adapter
(the Phase 0 generation defect `super(Model, ...)` was corrected to
`super(ModelNew, ...)` and the recorded hash updated in `project.md`).

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=8.154829 ms, v1=8.184198 ms, speedup=0.996x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| public loader contract | `base.py` exposes `Model/get_init_inputs/get_inputs`; adapter exposes `ModelNew/get_init_inputs/get_inputs` through the actual AST loader | Frozen harness loaded, constructed, moved, and executed both sides without load or constructor error | pass | correctness return code `0` |
| output structure/dtype | Single tensor `[2,4096,4,1280]` `bfloat16` | Harness recursive comparator accepted structure, shape, dtype | pass | correctness return code `0`; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | No floating mismatch | pass | correctness return code `0` |
| frozen artifact identity | local hashes equal project.md before and after measurement | base `e392799f...`, adapter `66a3a2c3...`, harness `3d4fa4ee...` all match | pass | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, seed/tolerances defaults, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte | pass | round_status_000.md |

The correctness command's `v0=8.154829 ms` and `v1=8.184198 ms` values are smoke
timing only and do not replace the frozen 50/100 baseline samples.

## Screening Evidence

Not applicable in Phase 0. No screening classification was made.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[8.189047, 8.198384, 8.176003]`
- candidate_raw_samples_ms: `[8.185878, 8.193696, 8.104427]`
- reference_median_ms: `8.189047`
- candidate_median_ms: `8.185878`
- improvement_pct: `0.03869899017091595`

```text
improvement_pct = (8.189047 - 8.185878) / 8.189047 * 100
               = 0.03869899017091595
```

This descriptive mechanical-adapter comparison is not an optimization-adoption
decision. Round 000 establishes `baseline_adapter.py` as the baseline; the result
is neither `accepted` nor `no-improvement`.

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `8.189047` | `8.185878` | `0` |
| 2 | `8.198384` | `8.193696` | `0` |
| 3 | `8.176003` | `8.104427` | `0` |

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive` (no Phase 0 optimization hypothesis exists)

No decision or `mechanism_observables[]` exists for Phase 0, so there are no
missing required observables.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `baseline_base`, `candidate_baseline_adapter`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `log/round_000_forward_50iter.pt.trace.json`, SHA256 `8c21c97ddca24e78ebb0f4dd37e4aba65f7e942fbddfdfc76a65b7d406ea9b26`
- unmodified summarizer SHA256: `f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_base` | `366192.359375` | `7323.8471875` | `274` | `5.48` | `8.189047` | `0.8943467032854981` |
| `candidate_baseline_adapter` | `397822.60888671875` | `7956.452177734375` | `298` | `5.96` | `8.189047` | `0.971596838769441` |

The kernel-count difference (274 vs 298) is a scope-boundary sampling artifact of
the 50-iteration forward profile, not a semantic difference: the two scopes share
an identical top-kernel set and the same computation.

### Baseline Top Kernels (baseline_base scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `Gemm_tcu_mr_kernel::gemm_tcu_h<...float...>` (TCU batched GEMM via cublasLt) | `46` | `0.92` | `243327.205` | `4866.544` |
| `modern::elementwise_kernel<CUDAFunctor_add<float>>` | `45` | `0.90` | `38890.063` | `777.801` |
| `elementwise_kernel<...MulFunctor<float>>` | `46` | `0.92` | `31295.656` | `625.913` |
| `elementwise_kernel<...direct_copy_kernel_cuda>` | `92` | `1.84` | `26342.239` | `526.845` |
| `vectorized_elementwise_kernel<4, bfloat16_copy_kernel_cuda>` | `45` | `0.90` | `26337.196` | `526.744` |

### Candidate Top Kernels (candidate_baseline_adapter scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `Gemm_tcu_mr_kernel::gemm_tcu_h<...float...>` (TCU batched GEMM via cublasLt) | `50` | `1.00` | `264487.807` | `5289.756` |
| `modern::elementwise_kernel<CUDAFunctor_add<float>>` | `49` | `0.98` | `42321.976` | `846.440` |
| `elementwise_kernel<...MulFunctor<float>>` | `49` | `0.98` | `33301.760` | `666.035` |
| `vectorized_elementwise_kernel<4, bfloat16_copy_kernel_cuda>` | `50` | `1.00` | `29258.785` | `585.176` |
| `elementwise_kernel<...direct_copy_kernel_cuda>` | `100` | `2.00` | `28452.281` | `569.046` |

### einsum Dispatch Observation (Designer open question resolved)

The `torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())` contraction
dispatches to an Iluvatar **TCU batched GEMM** kernel
(`Gemm_tcu_mr_kernel::gemm_tcu_h<...>`), selected through the `cublasLt`
interface (`cublasLtEpilogue_t` / `cublasLtMatmulMatrixScale_t` template
parameters are present). This is the `[4,4] @ [4,1280]` batched matrix multiply
over the `(a=2, b=4096)` batch dimension (`ixblasGEMMBatchMode_t`). The kernel is
instantiated with `float` accumulators/output and `matrix_scale_t=1002` (bf16
input storage promoted to fp32 for compute).

The full forward kernel sequence is:
1. bf16→fp32 promotion of `residual` and `x` (`direct_copy_kernel_cuda` /
   `bfloat16_copy_kernel_cuda` cast kernels, `~0.92`/`~1.0` per call each);
2. the `comb_res_mix` (already fp32) and the promoted `residual.float()` feed the
   single TCU batched GEMM (`term2`);
3. fp32 elementwise multiply-add (`MulFunctor<float>` then `CUDAFunctor_add<float>`,
   implementing `x.float().unsqueeze(-2) * post_layer_mix + term2`);
4. final fp32→bf16 cast (`bfloat16_copy_kernel_cuda`, the `.bfloat16()` output).

Optimization implication: the GEMM dominates device time (`~4866-5290 us/call`,
≈66% of device time), with the three elementwise/cast kernels contributing the
remainder. A Triton fusion could target the multiply-add + cast elementwise tail,
and potentially the GEMM. Note: `tl.dot` was verified as `Supported` on this
target profile after this report was first written (see
`scripts/bi150_tl_dot_probe_bf16.py`; fp32 exact, bf16 verified), so a Triton
replacement of the GEMM is no longer a capability-miss. However, the
`[4,4]@[4,1280]` contraction is small (contraction dim 4), so the elementwise
tail remains the safer first fusion target while the GEMM replacement is a
higher-risk, higher-reward candidate for a later round.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness (adapter generation defect) | `ceaff44f4c97ee360ce10f69c80e628aaa5fa7bdb2157a6035e8dd8c98d37b0b` | same | correctness failed: `ModelNew(...) failed: name 'Model' is not defined` |
| 2 | Orchestrator-repaired adapter (`super(ModelNew, ...)`) | `66a3a2c31863d18c725a52ab57fd1b9f89fe655dd7bab7cb4da158b8130b5d07` | same | correctness, wall timing, and profiler all passed |

Attempt 1 is a Phase 0 adapter-generation repair reconciled by Orchestrator
(not a Verifier-to-Coder repair); no candidate source was edited by Verifier.

## evidence_for_next_round

- Canonical Phase 0 baseline: `baseline_adapter.py`, wall median `8.189047 ms`
  from three independent 50/100 samples under measurement fingerprint
  `c17c7f45d44b1bec047c7c3a315275d33b21af3226dd34e884d483899ef039b6`.
- `baseline_base` scope measured `7323.8471875 us/device-call` and
  `5.48 kernels/call`. Device ratio ≈ `0.894`, so ~11% of wall time is host /
  launch overhead (much higher device occupancy than the rotary-embedding
  reference, whose ratio was ~0.19).
- einsum dispatches to a single TCU batched GEMM (`gemm_tcu_h`, cublasLt,
  bf16→fp32, fp32 accumulate) that dominates device time at `~4866-5290 us/call`
  (~66%). The remaining device time is split across a fp32 elementwise
  multiply-add (`MulFunctor<float>` + `CUDAFunctor_add<float>`) and two
  bf16↔fp32 cast/copy kernels.
- Base and adapter are semantically equivalent (adapter is a top-level class
  rename); the small wall/device differences are measurement observations, not an
  optimization mechanism.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline is valid; no optional target is configured, and no
  terminal-round limit applies to baseline establishment.

Orchestrator owns canonical pointer updates and workflow transition.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (run before and after measurement; all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/mhc_post_layer_mix/base.py kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py --warmup 50 --repeat 100
```

Forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mhc_post_layer_mix/bi150/log/round_000_forward_50iter.pt.trace.json
```

Separately scoped unmodified repository summaries (both returned code `0`):

```bash
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_post_layer_mix/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 8.189047
```

```bash
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_post_layer_mix/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 8.189047
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity |
| runtime fingerprint check | `0` | torch 2.7.1, triton 3.1.0, BI-V150 (7,1) |
| correctness 50/100 | `0` | round_status_000.md; report Correctness table |
| wall sample 1, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 2, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 3, 50/100 | `0` | report Interleaved Wall Timing |
| forward profiler 20/50 | `0` | `log/round_000_forward_50iter.pt.trace.json` |
| summarize `baseline_base` | `0` | report Profiler Evidence |
| summarize `candidate_baseline_adapter` | `0` | report Profiler Evidence |
| frozen-file SHA256 after measurement | `0` | hashes in Identity |
