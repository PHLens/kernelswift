# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 source reference; no pre-existing canonical implementation)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8`
- Accepted reference SHA256: `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58`
- Base SHA256: `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1), 16 SM, 16 GiB)
- Measurement fingerprint: `72be9562432197795bf6a24300483ccb2c3219b804b73258611048014cd804a9`
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`

The base, adapter, and harness hashes all match the frozen `project.md` values exactly.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive list comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=1.070367 ms, v1=1.071914 ms, speedup=0.999x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| public loader contract | `base.py` exposes `Model/get_init_inputs/get_inputs`; adapter exposes `ModelNew/get_init_inputs/get_inputs` through the actual AST loader | Frozen harness loaded, constructed, moved, and executed both sides without load or constructor error | pass | correctness return code `0` |
| output structure/shape/dtype | list of 4 fp32 tensors each `[30522]` | Harness recursive comparator (list branch) accepted structure, shape, and dtype for all 4 elements | pass | correctness return code `0`; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` per list element | No floating mismatch | pass | correctness return code `0` |
| frozen artifact identity | local hashes equal project.md before measurement | base `46106baa...`, adapter `359f4c80...`, harness `71fb3ad0...` all match | pass | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, seed/tolerances defaults, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte | pass | round_status_000.md |

The correctness command's `v0=1.070367 ms` and `v1=1.071914 ms` values are smoke
timing only and do not replace the frozen 50/100 baseline samples.

## Screening Evidence

Not applicable in Phase 0. No screening classification was made.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[1.069584, 1.070644, 1.070492]`
- candidate_raw_samples_ms: `[1.068099, 1.068803, 1.064938]`
- reference_median_ms: `1.070492`
- candidate_median_ms: `1.068099`
- improvement_pct: `not-applicable: Phase 0` (baseline establishment, not an adoption decision)

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `1.069584` | `1.068099` | `0` |
| 2 | `1.070644` | `1.068803` | `0` |
| 3 | `1.070492` | `1.064938` | `0` |

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
- raw trace: `log/round_000_forward_50iter.pt.trace.json`, SHA256 `74a3604cb6fade42a2ecfb4dc6de409f8329c5e363e47457140e952cc81e995a`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_base` | `37153.182` | `743.064` | `596` | `11.92` | `1.070492` | `0.69413` |
| `candidate_baseline_adapter` | `37190.913` | `743.818` | `600` | `12.0` | `1.070492` | `0.69484` |

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000) = 743.064 / 1070.492 ≈ 0.6941
```

Both scopes share an identical kernel sequence and near-identical per-call
totals. The tiny numeric difference between the two scopes (743.064 vs 743.818
us/call, 11.92 vs 12.0 kernels/call) is a measurement artifact of interleaved
scope-boundary sampling (the 4 per-sequence max-pool `reduce_kernel` boundaries
fall differently across the two scope sampling windows), not a semantic
difference.

### Baseline Top Kernels (baseline_base scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `Gemm_tcu_mr_kernel::gemm_tcu_h<64u,64u,64u,16u,16u,2u,...>` (dense + decoder GEMM, TCU) | `100` | `2.0` | `24924.771` | `498.495` |
| `at::native::reduce_kernel<1024,1,ReduceOp<float,MaxOps...>>` (per-sequence max-pool) | `196` | `3.92` | `4440.672` | `88.813` |
| `GEMM_Epilogue<float,...>` (Linear bias-add epilogue) | `100` | `2.0` | `4149.554` | `82.991` |
| `elementwise_kernel<log1p_kernel_cuda...>` (SPLADE log1p) | `50` | `1.0` | `1690.745` | `33.815` |
| `elementwise_kernel<launch_clamp_scalar...>` (ReLU) | `50` | `1.0` | `1062.754` | `21.255` |
| `vectorized_layer_norm_kernel<float,float>` (LayerNorm) | `50` | `1.0` | `509.437` | `10.189` |
| `elementwise_kernel<GeluCUDAKernelImpl...>` (GELU) | `50` | `1.0` | `375.249` | `7.505` |

### Candidate Top Kernels (candidate_baseline_adapter scope)

Identical kernel sequence to `baseline_base`. Per-call totals:
`gemm_tcu_h` `497.957 us` / `2.0` call, `reduce_kernel<MaxOps>` `89.786 us` /
`4.0` call, `GEMM_Epilogue` `83.404 us` / `2.0` call, `log1p` `33.642 us` /
`1.0` call, `clamp_scalar` `21.258 us` / `1.0` call, `layer_norm` `10.297 us` /
`1.0` call, `GELU` `7.474 us` / `1.0` call. The only difference from
`baseline_base` is the max-pool reduce count sampled as exactly `4.0`/call (vs
`3.92`/call) due to scope-boundary sampling. The scope is semantically
identical to `baseline_base`.

### Decoder GEMM Backend Observation

The large decoder GEMM (768×30522) lowers to the Iluvatar **TCU** path
(`Gemm_tcu_mr_kernel::gemm_tcu_h`), running `2.0`/call — one for the `dense`
768×768 projection and one for the `decoder` 768×30522 projection. Template
parameters: tile `64u,64u,64u,16u,16u,2u`, `(ixblasTrans_t)1` (B-transposed),
`(ixblasTCUGEMMCategory_t)0`, `(ixblasGEMMBoundary_t)2`,
`(cublasLtEpilogue_t)1`, `float` in/accumulate/out, matrix scale `1002`. The
GEMM epilogue is a separate `GEMM_Epilogue<float,...>` kernel (bias add),
`2.0`/call. Combined GEMM+epilogue is ~581 us/call, i.e. ~78% of the total
device time — this operator is **compute-bound**, unlike launch-bound operators
like fused_moe.

### Per-Sequence Pooling Loop Observation

The per-sequence Python loop (`for L in seq_lens.tolist()`, 4 iterations) emits
one `reduce_kernel<1024,1,ReduceOp<float,MaxOps>>` per iteration
(`chunk.max(dim=0)`), i.e. **4 max-pool reductions per forward call** (sampled
`3.92`/call in baseline scope and `4.0`/call in candidate scope). Each reduction
is over a `[L, 30522]` chunk (column-wise max along the sequence axis). This is
a small structural loop (4 launches, ~89 us/call total) relative to the GEMM.

### Elementwise Chain Observation

The SPLADE elementwise chain is 4 kernels per call, executed once on the full
`[83, 30522]` tensor (not per-sequence): `GELU` (~7.5 us), `LayerNorm` (~10.2
us), ReLU via `clamp_scalar` (~21.3 us), and `log1p` (~33.8 us). Together ~72
us/call. These are already single vectorized elementwise kernels; the dominant
cost in this operator is the decoder GEMM, not the activation chain.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8` | same | correctness and wall timing passed; profiler summarized without filtering |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Canonical Phase 0 baseline: `baseline_adapter.py`, wall median `1.070492 ms` from three independent 50/100 samples under measurement fingerprint `72be9562432197795bf6a24300483ccb2c3219b804b73258611048014cd804a9`.
- `baseline_base` scope measured `743.064 us/device-call` and `11.92 kernels/call`. Device ratio ≈ `0.694`, so ~69% of wall time is device kernel time. This operator is compute-bound, unlike launch-bound operators (fused_moe ≈ 0.297).
- The decoder GEMM (768×30522) + `dense` (768×768) both run on the Iluvatra TCU (`gemm_tcu_h`, fp32), `2.0`/call, ~498 us/call; combined with the `GEMM_Epilogue` bias-add kernel (~83 us/call) they total ~581 us/call ≈ 78% of device time. GEMM compute is the dominant bottleneck.
- The per-sequence pooling loop launches 4 `reduce_kernel<MaxOps>` (one `chunk.max(dim=0)` per sequence), ~89 us/call total. Small relative to GEMM.
- The elementwise chain (GELU → LayerNorm → ReLU → log1p) is 4 kernels/call, ~72 us/call total, already single vectorized kernels.
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
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/sparse_pooler/base.py kernels/track1-triton/sparse_pooler/bi150/baseline_adapter.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/bi150/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/bi150/baseline_adapter.py --warmup 50 --repeat 100
```

Forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/sparse_pooler/bi150/log/round_000_forward_50iter.pt.trace.json
```

Separately scoped unmodified repository summaries (both returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/sparse_pooler/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 1.070492
```

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/sparse_pooler/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 1.070492
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity |
| correctness 50/100 | `0` | round_status_000.md; report Correctness table |
| wall sample 1, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 2, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 3, 50/100 | `0` | report Interleaved Wall Timing |
| forward profiler 20/50 | `0` | `log/round_000_forward_50iter.pt.trace.json` |
| summarize `baseline_base` | `0` | report Profiler Evidence |
| summarize `candidate_baseline_adapter` | `0` | report Profiler Evidence |
