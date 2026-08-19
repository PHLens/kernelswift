# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 source reference; no pre-existing canonical implementation)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f`
- Accepted reference SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `b8029499f0964a738f50b09164e419511d0bc89df5e260573e607bb7345afc2e`
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`
- completed_at: `2026-08-18T16:20:00Z`

The adapter, base, and harness hashes all match the frozen project.md values exactly.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=0.148653 ms, v1=0.148315 ms, speedup=1.002x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| public loader contract | `base.py` exposes `Model/get_init_inputs/get_inputs`; adapter exposes `ModelNew/get_init_inputs/get_inputs` through the actual AST loader | Frozen harness loaded, constructed, moved, and executed both sides without load or constructor error | pass | correctness return code `0` |
| output structure/shape/dtype | Single tensor `[2, 83, 512]`, float16 | Harness recursive comparator accepted structure, shape, and dtype | pass | correctness return code `0`; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | No floating mismatch | pass | correctness return code `0` |
| frozen artifact identity | local hashes equal project.md before measurement | base `86ac5703...`, adapter `c3980a2c...`, harness `3d4fa4ee...` all match | pass | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, seed/tolerances defaults, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte | pass | round_status_000.md |

The correctness command's `v0=0.148653 ms` and `v1=0.148315 ms` values are smoke timing only and do not replace the frozen 50/100 baseline samples.

## Screening Evidence

Not applicable in Phase 0. No screening classification was made.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[0.150876, 0.151139, 0.151994]`
- candidate_raw_samples_ms: `[0.149624, 0.150183, 0.149352]`
- reference_median_ms: `0.151139`
- candidate_median_ms: `0.149624`
- improvement_pct: `not-applicable: Phase 0` (baseline establishment, not an adoption decision)

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `0.150876` | `0.149624` | `0` |
| 2 | `0.151139` | `0.150183` | `0` |
| 3 | `0.151994` | `0.149352` | `0` |

This descriptive mechanical-adapter comparison is not an optimization-adoption
decision. Round 000 establishes `baseline_adapter.py` as the baseline; the result
is neither `accepted` nor `no-improvement`.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive` (no Phase 0 optimization hypothesis exists)

No decision or `mechanism_observables[]` exists for Phase 0, so there are no missing required observables.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `baseline_base`, `candidate_baseline_adapter`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `log/round_000_forward_50iter.pt.trace.json`, SHA256 `140ce325b62c0ac03e08f1e8f9f9bbbe586ed382e18407d212c8d02ad985b94c`
- unmodified summarizer SHA256: `f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_base` | `747.462890625` | `14.9492578125` | `43` | `0.86` | `0.151139` | `0.09891065715996535` |
| `candidate_baseline_adapter` | `794.85400390625` | `15.897080078125` | `46` | `0.92` | `0.149624` | `0.10625596439902468` |

Both scopes share a single, identical top-kernel. The kernel-count difference
(43 vs 46) and the sub-1.0 per-call value are scope-boundary sampling artifacts
of the 50-iteration forward profile (kernels straddling the `X` scope markers
are excluded), not a semantic difference.

### Baseline Top Kernels (baseline_base scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)0, (AlibiMode_t)0, false, __half, false>(FlashAttnFwdParams, ...)` | `43` | `0.86` | `747.463` | `14.949` |

### Candidate Top Kernels (candidate_baseline_adapter scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)0, (AlibiMode_t)0, false, __half, false>(FlashAttnFwdParams, ...)` | `46` | `0.92` | `794.854` | `15.897` |

## Designer Open Questions — Runtime Observations

### Q1: SDPA backend dispatch

The CPU operator trace shows the exact dispatch chain:

```text
aten::scaled_dot_product_attention
  -> aten::_scaled_dot_product_flash_attention
    -> aten::_flash_attention_forward
```

The single device kernel is `FlashAttnFwdF16Ixmma<128, 128, 16, 64, 64, Causal=0, Alibi=0, ...>`.

**Conclusion: SDPA dispatches to the fused flash-attention backend (FlashAttention), not the math backend and not the mem-efficient backend.** There are no separated `bmm` / `softmax` / `scaled_dot_product` kernels anywhere in the trace; the entire QK^T + softmax + PV computation is fused into exactly one `FlashAttnFwdF16Ixmma` kernel per forward call. The kernel name confirms `CausalM_t = 0` (no causal mask) and `AlibiMode_t = 0` (no ALiBi), matching the project.md semantics (no mask, no causal).

### Q2: fp16 accumulation path

The kernel name `FlashAttnFwdF16Ixmma` with element type `__half` indicates the attention is computed with fp16 inputs on the Iluvatar Ixmma tensor-core matmul unit. Because the computation is fully fused into a single flash-attention kernel, the QK^T product and softmax are accumulated inside the fused kernel — there is no separate fp32 intermediate tensor materialized between a bmm and a softmax (which would be the signature of the math backend). The exact internal accumulation width (fp16 vs fp32 inside the Ixmma MMA/softmax) is not directly readable from the kernel name or device-time evidence; what is observable is that a single fused fp16 flash kernel owns 100% of device time, so there is no exposed fp32 accumulation stage.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f` | same | correctness and wall timing passed; profiler summarized without filtering |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Canonical Phase 0 baseline: `baseline_adapter.py`, wall median `0.151139 ms` from three independent 50/100 samples under measurement fingerprint `b8029499f0964a738f50b09164e419511d0bc89df5e260573e607bb7345afc2e`.
- `baseline_base` scope measured `14.9492578125 us/device-call` and `0.86 kernels/call`. Device ratio ≈ `0.099`, so ~90% of wall time is host / launch overhead rather than device kernel time.
- The device time is entirely a single fused `FlashAttnFwdF16Ixmma` flash-attention kernel (Ixmma tensor-core path, fp16, Causal=0, Alibi=0). There is no bmm/softmax decomposition and no opportunity to reduce internal kernel count below one per call.
- SDPA dispatches to the flash-attention backend (`_scaled_dot_product_flash_attention` → `_flash_attention_forward`), so any future candidate that replaces SDPA must reproduce flash-attention numerics within `atol=1e-2, rtol=1e-2` or the correctness gate will fail against this fused reference.
- Base and adapter are semantically equivalent (adapter is a top-level class rename); the small wall/device differences are measurement observations, not an optimization mechanism.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline is valid (correctness PASS, three 50/100 wall samples, Level 1 profiler summary collected). No optional target is configured, and no terminal-round limit applies to baseline establishment.

Orchestrator owns canonical pointer updates and workflow transition.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (run before and after measurement; all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/mm_encoder_attention/base.py kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py --warmup 50 --repeat 100
```

Forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mm_encoder_attention/bi150/log/round_000_forward_50iter.pt.trace.json
```

Separately scoped unmodified repository summaries (both returned code `0`):

```bash
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 0.151139
```

```bash
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 0.149624
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
