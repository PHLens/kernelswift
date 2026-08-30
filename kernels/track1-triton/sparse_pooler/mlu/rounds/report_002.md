# Report 002

Result: no-improvement

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_sparse_pooler_002.py`
- Accepted reference: `triton_sparse_pooler_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `0d39de9e280f6ffa2cc3d1d3322d393fa400eb8f405b7e7ee3ceb3ef845b3dd4`
- Candidate SHA256: `62dc853db5423cb5d99ad53433f3fb35919abe901a64d6e3acb3d815ac678248`
- Accepted reference SHA256: `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`
- Base SHA256: `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `a0208c7da7e371d45c88f82ebddd3850d01669aa5d912f31db9234a7a56ebab7`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` between `base.py` Model and `triton_sparse_pooler_002.py` ModelNew outputs (list of 4 x Tensor[30522]) | `PASS accuracy` across the correctness gate, all 3 candidate timing runs, and the profiler run | pass | `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_002.py --warmup 50 --repeat 100` |
| output structure | list of 4 tensors, each shape [30522], dtype fp32, device mlu:0 | candidate returns `[out[i] for i in range(num_seq)]` from a `[num_seq, vocab_size]` fp32 mlu:0 tensor; confirmed by correctness gate and profiler run | pass | `triton_sparse_pooler_002.py` forward; harness `PASS accuracy` |
| numerical semantics | `log(1+relu(decoder_logits))` max-pooled per sequence within atol=1e-2 rtol=1e-2 | fused kernel computes `log(1+relu(x))` per element then per-segment max; correctness gate passed | pass | `_sparse_pooler_max_kernel` body; `PASS accuracy` |
| caller-selected device and current stream preserved | no explicit `torch.mlu.device()` introduced; kernel launches on current stream | candidate uses `device = x.device` for output allocation; no device context; kernel launch inherits stream | pass | `triton_sparse_pooler_002.py` forward |
| dense GELU LayerNorm decoder matmul pipeline unchanged | library ops preserved | `self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))` unchanged from Round 001 | pass | `triton_sparse_pooler_002.py` forward |
| ModelNew public constructor and forward signature unchanged | `ModelNew(hidden_size=768, vocab_size=30522, pooling="max")` and `forward(hidden_states, seq_lens) -> list[Tensor]` | signatures match Round 001 exactly | pass | `triton_sparse_pooler_002.py` |
| load_state_dict compatibility | candidate accepts reference state dict | harness runs `model_new.load_state_dict(model.state_dict())` before timing; `PASS accuracy` confirms | pass | harness behavior; `PASS accuracy` |
| kernel_count_per_call remains 5 | no kernels added or removed; only the fused kernel's tiling parameter changes | profiler: reference 5.0 kernels/call -> candidate 5.0 kernels/call (exactly) | pass | profiler scope summary below |
| num_warps=2 is not used | known to fail on this runtime per triton_mlu target profile | candidate uses `num_warps=1` exactly as the decision requires | pass | `triton_sparse_pooler_002.py` dispatch |

Correctness and every declared guardrail pass. The candidate is a conforming implementation of the decision; it simply does not clear the 5% adoption threshold.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`
- accepted reference: `triton_sparse_pooler_001.py` (the Round 001 accepted canonical, compared as v1 against `base.py` v0)
- candidate: `triton_sparse_pooler_002.py` (compared as v1 against `base.py` v0)
- reference_raw_samples_ms: `[0.690104, 0.625936, 0.604951]`
- candidate_raw_samples_ms: `[0.621848, 0.631596, 0.613131]`
- reference_median_ms: `0.625936`
- candidate_median_ms: `0.621848`
- improvement_pct: `0.6531019145727505`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
                = (0.625936 - 0.621848) / 0.625936 * 100
                = 0.6531019145727505
```

The unrounded improvement controls the 5% adoption threshold. Profiler time does
not replace this benchmark result. The improvement of 0.65% is well below the 5%
adoption threshold and well below the decision's 7.0% expected wall improvement.

Raw per-run harness output (v0 = base.py Model, v1 = accepted reference or candidate):

| Pair | Role | v0 ms | v1 ms | speedup | exit |
|---:|---|---:|---:|---:|---:|
| 1 | reference (triton_sparse_pooler_001) | 0.975735 | 0.690104 | 1.414x | 0 |
| 1 | candidate (triton_sparse_pooler_002) | 0.934121 | 0.621848 | 1.502x | 0 |
| 2 | reference (triton_sparse_pooler_001) | 0.932037 | 0.625936 | 1.489x | 0 |
| 2 | candidate (triton_sparse_pooler_002) | 0.935818 | 0.631596 | 1.482x | 0 |
| 3 | reference (triton_sparse_pooler_001) | 0.890085 | 0.604951 | 1.471x | 0 |
| 3 | candidate (triton_sparse_pooler_002) | 0.906601 | 0.613131 | 1.479x | 0 |

## Evaluation Contract Mirror

- hypothesis_id: `H-002`
- intervention: tune the fused `_sparse_pooler_max_kernel` by increasing BLOCK_V from 1024 to 2048, halving the vocab tile count from 30 to 15 and the total grid programs from 120 to 60, to recover the 30.86 us/call device regression against the six library kernels it replaced and reduce per-program launch overhead
- expected_causal_chain: BLOCK_V increases from 1024 to 2048 so the vocab dimension is covered by 15 tiles instead of 30; the fused kernel grid drops from (4,30)=120 programs to (4,15)=60 programs; per-program overhead (prefix scan, loop setup, program dispatch) is halved because half as many programs execute; the fused kernel device time decreases from 98.73 us/call toward or below the 67.87 us/call combined cost of the six library kernels it replaced; total device_us_per_call decreases from 210.12 us/call; wall time decreases by at least 5%
- primary_metric: `wall_time`, expected_improvement_pct 5.0
- profiling_level: `targeted`

### Mechanism observables

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `fused_kernel_us_per_call` | decrease from 98.73 us/call; target is to recover the 30.86 us/call regression and approach or fall below the 67.87 us/call combined cost of the six replaced library kernels | reference 99.71 us/call -> candidate 102.42 us/call (increased by 2.71 us/call); the fused kernel got SLOWER, not faster, with BLOCK_V=2048 | falsified | profiler scope summary below |
| `device_us_per_call` | decrease from 210.12 us/call as the fused kernel cost drops; the four non-fused kernels (decoder matmul 90.36, dense matmul 8.42, LayerNorm 7.21, GELU 5.40 us/call) are unchanged | reference 209.50 us/call -> candidate 213.02 us/call (increased by 3.52 us/call); the four non-fused kernels are unchanged within noise (decoder matmul 89.61->90.33, dense matmul 8.41->8.53, LayerNorm 7.21->7.20, GELU 4.56->4.54); the entire device increase is attributable to the fused kernel | falsified | profiler scope summary below |
| `fused_kernel_grid_programs` | decrease from 120 (4 sequences x 30 vocab tiles) to 60 (4 sequences x 15 vocab tiles) | reference grid (4, 30)=120 -> candidate grid (4, 15)=60; confirmed by code inspection: `BLOCK_V = 1024` -> `BLOCK_V = 2048` and `num_vocab_tiles = triton.cdiv(30522, BLOCK_V)` recomputed from 30 to 15 | confirmed | `triton_sparse_pooler_001.py` line 85 vs `triton_sparse_pooler_002.py` line 85; `triton.cdiv(30522, 2048) = 15` |

### Hypothesis verdict: `falsified`

The wall improvement of 0.65% is far below the 5% adoption threshold and the 7.0% expected improvement. Two of the three mechanism observables (`fused_kernel_us_per_call` and `device_us_per_call`) moved in the OPPOSITE direction of the hypothesis: the fused kernel got slower (99.71 -> 102.42 us/call, +2.71 us/call) and total device time increased (209.50 -> 213.02 us/call, +3.52 us/call). Only the grid-programs observable matched the expectation (120 -> 60 programs, confirmed by code inspection). The hypothesis posited that halving the grid programs would halve per-program overhead and recover the 30.86 us/call device regression; instead, the larger BLOCK_V doubled per-program work (each program now processes 2048 vocab elements instead of 1024) and the increased per-program cost outweighed any launch-dispatch savings. The fused kernel remains slower than the 6 library kernels it replaced (102.42 us/call vs the 67.87 us/call combined library cost). Because the primary metric fails the threshold AND two observables contradict the expected causal chain, the verdict is `falsified`.

## Profiler Evidence

- profiler_level: `summary` (Level 1)
- iterations: `50` (forward calls per scope)
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/round_002_forward_50iter.pt.trace.json`
- profiler run also reported `PASS accuracy; v0=0.894074 ms, v1=0.609396 ms, speedup=1.467x` (diagnostic only; not the authoritative 3-pair median)

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations` before they are compared. The
profiler trace was produced with
`--profile --profile-reference-file sparse_pooler/triton_sparse_pooler_001.py --profile-mode forward --profile-warmup 20 --profile-iterations 50`.
The standard `scripts/summarize_trace.py` failed with `overlapping scope events`
because each scope has 2 events (`user_annotation` on CPU + `gpu_user_annotation`
on GPU) that overlap. Per the Verifier contract fallback, a custom Python
summarizer was used that scopes kernels using ONLY the `gpu_user_annotation`
device-side intervals (filtering to GPU scope only), matching the
`summarize_trace` internal logic.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_triton_sparse_pooler_001 | 10475.00 | 209.50 | 250 | 5.0 | 0.625936 | 0.3347 |
| candidate_triton_sparse_pooler_002 | 10651.25 | 213.02 | 250 | 5.0 | 0.621848 | 0.3425 |

```text
device_ratio (reference) = 209.50 / (0.625936 * 1000) = 0.3347
device_ratio (candidate) = 213.02 / (0.621848 * 1000) = 0.3425
```

The candidate's device ratio rose from 0.335 to 0.343 because device time
increased while wall time fell marginally. The class remains mixed. The
device-side regression in the fused kernel (+2.71 us/call) is the dominant
cause of the device-time increase; the four non-fused kernels are unchanged
within noise.

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _sparse_pooler_max_kernel (fused relu+log1p+per-segment max) | 50 | 1.00 | 4985.56 | 99.71 |
| MLUFusedMatMulGepm (decoder matmul 768->30522) | 50 | 1.00 | 4480.56 | 89.61 |
| MLUFusedMatMulGepdot (dense matmul 768->768) | 50 | 1.00 | 420.52 | 8.41 |
| layerNormForwardKernel (LayerNorm) | 50 | 1.00 | 360.44 | 7.21 |
| MLUBlockKernel3StagePipelineGeluHighAccCubic (GELU) | 50 | 1.00 | 227.92 | 4.56 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _sparse_pooler_max_kernel (fused relu+log1p+per-segment max) | 50 | 1.00 | 5121.04 | 102.42 |
| MLUFusedMatMulGepm (decoder matmul 768->30522) | 50 | 1.00 | 4516.72 | 90.33 |
| MLUFusedMatMulGepdot (dense matmul 768->768) | 50 | 1.00 | 426.64 | 8.53 |
| layerNormForwardKernel (LayerNorm) | 50 | 1.00 | 359.84 | 7.20 |
| MLUBlockKernel3StagePipelineGeluHighAccCubic (GELU) | 50 | 1.00 | 227.00 | 4.54 |

The candidate has exactly 5 kernel types per call, same as the reference. The
fused `_sparse_pooler_max_kernel` got slower by 2.71 us/call (99.71 -> 102.42)
with BLOCK_V=2048. The four non-fused kernels (decoder matmul, dense matmul,
LayerNorm, GELU) are unchanged within noise. The entire device-time increase
(3.52 us/call) is attributable to the fused kernel. The fused kernel remains
slower than the 6 library kernels it replaced (102.42 us/call vs the 67.87
us/call combined library cost recorded in report_001.md).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Round 002 verification | 62dc853db5423cb5d99ad53433f3fb35919abe901a64d6e3acb3d815ac678248 | 62dc853db5423cb5d99ad53433f3fb35919abe901a64d6e3acb3d815ac678248 | pass (correctness); no-improvement (wall) |

No Coder repair was required. The candidate passed the correctness gate on the
first Verifier attempt. At most one Verifier-to-Coder repair is allowed in the
same round; it was not used. The candidate simply does not clear the 5% adoption
threshold, so the terminal classification is `no-improvement`, not
`candidate-failed`.

## Upbound Gap

- upbound_source: `project.md#upbound (estimated semantic bound)`
- comparable_metric: `wall_time_ms`
- absolute_gap: `null (no measured upbound)`
- ratio_to_upbound: `null`
- interpretation: `The declared upbound is a semantic estimate (30-50% wall improvement plausible from a single fused kernel covering the MLM head tail), not a measured bound. The 5% adoption threshold uses measured wall time only. The observed 0.65% improvement does not clear the adoption threshold, so the upbound comparison is not the deciding factor; the measured wall result controls.`

## evidence_for_next_round

- The fused `_sparse_pooler_max_kernel` got SLOWER with BLOCK_V=2048 (99.71 -> 102.42 us/call, +2.71 us/call), falsifying the hypothesis that halving grid programs reduces per-program overhead. The larger BLOCK_V doubled per-program work (each program processes 2048 vocab elements instead of 1024) and the increased per-program cost outweighed any launch-dispatch savings. The fused kernel remains the dominant device kernel at 102.42 us/call (48.0% of candidate device time) and is still slower than the 6 library kernels it replaced (67.87 us/call combined) — a 34.55 us/call device regression persists.
- The grid-programs observable was confirmed (120 -> 60 programs) but the expected causal consequence (lower per-program overhead -> lower fused-kernel device time) did not materialize. This suggests the bottleneck in the fused kernel is per-program WORK (the inner loop over `seq_len` rows, processing BLOCK_V elements per row), not per-program launch dispatch overhead. Larger BLOCK_V increases the work per program without reducing the total work.
- The candidate device ratio is 0.343 (mixed). Wall time is 621.85 us/call and device time is 213.02 us/call, so roughly 409 us/call (~66%) is host-side. The host-side Python loop and D2H sync are gone (eliminated in Round 001); the remaining host time is launcher, wrapper, allocation, and harness-fixed cost.
- The four non-fused kernels (decoder matmul 90.33, dense matmul 8.53, LayerNorm 7.20, GELU 4.54 us/call) are unchanged within noise. The decoder matmul (MLUFusedMatMulGepm, 90.33 us/call) remains the second-largest device kernel and the largest non-fused kernel.
- The wall-time variance across the 3 reference samples (0.605 - 0.690 ms, a ~14% spread) and 3 candidate samples (0.613 - 0.632 ms, a ~3% spread) suggests measurement noise is non-trivial at this wall-time scale. The reference median (0.625936 ms) sits between the candidate median (0.621848 ms) and the candidate's worst sample (0.631596 ms), so the 0.65% "improvement" is within measurement noise.
- BLOCK_V=1024 (the Round 001 accepted value) remains the best-known tiling parameter for this kernel on this runtime. Increasing BLOCK_V to 2048 regressed device time; the decision's optional fallback probes (BLOCK_V=4096, other num_warps values) were not exercised by Coder because the normative BLOCK_V=2048 compiled and ran correctly, but the Verifier evidence shows BLOCK_V=2048 is worse than BLOCK_V=1024 on the device.

Record evidence only; do not select the next optimization.

## Stop Recommendation

- recommendation: `continue`
- evidence: `Candidate not adopted: 0.65% wall improvement is below the 5% adoption threshold and the 7.0% expected improvement. The hypothesis H-002 is falsified: increasing BLOCK_V from 1024 to 2048 increased the fused kernel's device time (99.71 -> 102.42 us/call) and total device time (209.50 -> 213.02 us/call), contrary to the expectation that halving grid programs would reduce per-program overhead. The grid-programs observable was confirmed (120 -> 60) but the expected causal consequence did not materialize. performance_miss_streak becomes 1 after this round. None of the 5 stop criteria trigger yet: measurements are not noise-bound (the 0.65% result is within noise but the falsified hypothesis provides clear directional evidence), the streak is 1 (not 3, so diminishing returns does not trigger), the estimated upbound is not a measured bound (upbound reached does not apply), resources are not exhausted, and there is no user intervention. The next round should target a different mechanism: the fused kernel's per-program work (inner loop over seq_len rows), the decoder matmul (the largest non-fused kernel at 90.33 us/call), or host-side launcher/allocation reduction.`
- applicable stop criteria: none (continue)

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness gate:

```bash
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_002.py \
  --warmup 50 --repeat 100
```

Authoritative wall timing (3 interleaved pairs; run the reference command and the candidate command alternately, 3 times each):

```bash
# accepted reference (run 3 times)
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_001.py \
  --warmup 50 --repeat 100

# candidate (run 3 times)
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_002.py \
  --warmup 50 --repeat 100
```

Level 1 profiler:

```bash
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_002.py \
  --profile --profile-reference-file sparse_pooler/triton_sparse_pooler_001.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output sparse_pooler/log/round_002_forward_50iter.pt.trace.json
```

For `Result: no-improvement`, this report contains the decision/candidate/accepted-reference/source
hashes; the correctness/guardrail matrix; all 6 raw samples and unrounded medians; improvement;
the Evaluation Contract mirror (3 observables with expectation/observation/verdict); hypothesis
verdict; Level 1 profiler data (scope summary + top kernels for each scope); retry history;
upbound gap; evidence_for_next_round; stop recommendation; and exact reproduction commands.
Verifier does not update `last_accepted_kernel`; that is Orchestrator's job.
