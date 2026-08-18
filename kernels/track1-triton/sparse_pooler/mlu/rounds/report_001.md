# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_sparse_pooler_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `0816c943dfcfd157c9c4268196f4779b9804b9107de5fff0ba135d66f4f5bc75`
- Candidate SHA256: `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`
- Accepted reference SHA256: `d7e69ed4b66a4193a4475cc307fc0929cef807f875785652df6cc36fb2c487e5`
- Base SHA256: `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `a0208c7da7e371d45c88f82ebddd3850d01669aa5d912f31db9234a7a56ebab7`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` between `base.py` Model and `triton_sparse_pooler_001.py` ModelNew outputs (list of 4 x Tensor[30522]) | `PASS accuracy` across the correctness gate, all 3 candidate timing runs, and the profiler run | pass | `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_001.py --warmup 50 --repeat 100` |
| output structure | list of 4 tensors, each shape [30522], dtype fp32, device mlu:0 | candidate returns `[out[i] for i in range(num_seq)]` from a `[num_seq, vocab_size]` fp32 mlu:0 tensor; confirmed by correctness gate and profiler run | pass | `triton_sparse_pooler_001.py` forward; harness `PASS accuracy` |
| numerical semantics | `log(1+relu(decoder_logits))` max-pooled per sequence within atol=1e-2 rtol=1e-2 | fused kernel computes `log(1+relu(x))` per element then per-segment max; correctness gate passed | pass | `_sparse_pooler_max_kernel` body; `PASS accuracy` |
| caller-selected device and current stream preserved | no explicit `torch.mlu.device()` introduced; kernel launches on current stream | candidate uses `device = x.device` for output allocation; no device context; kernel launch inherits stream | pass | `triton_sparse_pooler_001.py` forward |
| dense GELU LayerNorm decoder matmul pipeline unchanged | library ops preserved | `self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))` unchanged from baseline | pass | `triton_sparse_pooler_001.py` forward |
| ModelNew public constructor and forward signature unchanged | `ModelNew(hidden_size=768, vocab_size=30522, pooling="max")` and `forward(hidden_states, seq_lens) -> list[Tensor]` | signatures match baseline exactly | pass | `triton_sparse_pooler_001.py` |
| load_state_dict compatibility | candidate accepts reference state dict | harness runs `model_new.load_state_dict(model.state_dict())` before timing; `PASS accuracy` confirms | pass | harness behavior; `PASS accuracy` |

Conformance, correctness, and every declared guardrail pass before adoption.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`
- accepted reference: `baseline_adapter.py` (compared as v1 against `base.py` v0)
- candidate: `triton_sparse_pooler_001.py` (compared as v1 against `base.py` v0)
- reference_raw_samples_ms: `[0.899378, 0.910847, 0.919767]`
- candidate_raw_samples_ms: `[0.606758, 0.610001, 0.600999]`
- reference_median_ms: `0.910847`
- candidate_median_ms: `0.606758`
- improvement_pct: `33.38529961673036`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
                = (0.910847 - 0.606758) / 0.910847 * 100
                = 33.38529961673036
```

The unrounded improvement controls the 5% adoption threshold. Profiler time does
not replace this benchmark result. The improvement of 33.39% exceeds both the 5%
adoption threshold and the decision's 15% expected wall improvement.

Raw per-run harness output (v0 = base.py Model, v1 = accepted reference or candidate):

| Pair | Role | v0 ms | v1 ms | speedup | exit |
|---:|---|---:|---:|---:|---:|
| 1 | reference (baseline_adapter) | 0.909853 | 0.899378 | 1.012x | 0 |
| 1 | candidate (triton_sparse_pooler_001) | 0.900469 | 0.606758 | 1.484x | 0 |
| 2 | reference (baseline_adapter) | 0.905048 | 0.910847 | 0.994x | 0 |
| 2 | candidate (triton_sparse_pooler_001) | 0.898329 | 0.610001 | 1.473x | 0 |
| 3 | reference (baseline_adapter) | 0.909076 | 0.919767 | 0.988x | 0 |
| 3 | candidate (triton_sparse_pooler_001) | 0.887124 | 0.600999 | 1.476x | 0 |

## Evaluation Contract Mirror

- hypothesis_id: `H-001`
- intervention: fuse relu, log1p, and per-sequence max pooling into a single Triton kernel launched once per forward, eliminating the Python for-loop over `seq_lens.tolist()` and replacing six device kernels with one fused reduction kernel
- expected_causal_chain: Python for-loop and four per-chunk torch.max dispatches replaced by one Triton kernel launch; host-side D2H synchronization from `seq_lens.tolist()` eliminated; device kernel count per call drops from 10 to 5; host dispatch overhead and device kernel count decrease; wall time decreases by at least 5%
- primary_metric: `wall_time`, expected_improvement_pct 5.0
- profiling_level: `targeted`

### Mechanism observables

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `kernel_count_per_call` | decrease from 10 to 5 | reference 10.0 -> candidate 5.0 (exactly) | confirmed | profiler scope summary below |
| `device_us_per_call` | decrease by roughly 158 us/call (combined relu+log1p+4x max-pool cost) minus the fused kernel cost | reference 179.33 us/call -> candidate 210.12 us/call (increased by 30.79 us/call) | falsified | profiler scope summary below; the fused `_sparse_pooler_max_kernel` costs 98.73 us/call, more than the 6 baseline kernels it replaced (relu 13.67 + log1p 26.17 + 4x reduceKernelMaxIndex 28.03 = 67.87 us/call combined) |
| `host_sync_count_per_call` | decrease because `seq_lens.tolist()` D2H sync is eliminated | candidate's max path computes `seq_len` and `seq_offset` on-device via `tl.load(seq_lens_ptr + pid_s)` and a bounded prefix scan; `seq_lens.tolist()` is no longer called on the max path (only retained on the sum-pooling fallback path) | confirmed | `triton_sparse_pooler_001.py` forward + kernel body; the wall-time collapse with concurrent device-time increase isolates the gain to host-side loop and sync elimination |

### Hypothesis verdict: `partially-confirmed`

The wall improvement of 33.39% exceeds both the 5% adoption threshold and the 15% expected improvement, and 2 of 3 mechanism observables match their expectations (`kernel_count_per_call` and `host_sync_count_per_call`). However, `device_us_per_call` increased rather than decreased: the fused Triton kernel (98.73 us/call) is slower on the device than the 6 library kernels it replaced (67.87 us/call combined). The wall-time gain is therefore attributable to host-side Python-loop and D2H-sync elimination, not to device-side kernel cost reduction. Because one observable contradicts the expected causal chain while the primary metric clears the threshold, the verdict is `partially-confirmed`.

## Profiler Evidence

- profiler_level: `summary` (Level 1)
- iterations: `50` (forward calls per scope)
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/round_001_forward_50iter.pt.trace.json`
- profiler run also reported `PASS accuracy; v0=0.893909 ms, v1=0.592478 ms, speedup=1.509x` (diagnostic only; not the authoritative 3-pair median)

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations` before they are compared. The
profiler trace was produced with
`--profile --profile-reference-file sparse_pooler/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50`.
The standard `scripts/summarize_trace.py` failed with `overlapping scope events`
because each scope has 2 events (`user_annotation` on CPU + `gpu_user_annotation`
on GPU) that overlap. Per the Verifier contract fallback, a custom Python
summarizer was used that scopes kernels using ONLY the `gpu_user_annotation`
device-side intervals (filtering to GPU scope only), matching the
`summarize_trace` internal logic.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_baseline_adapter | 8966.35 | 179.33 | 500 | 10.0 | 0.910847 | 0.1969 |
| candidate_triton_sparse_pooler_001 | 10506.03 | 210.12 | 250 | 5.0 | 0.606758 | 0.3463 |

```text
device_ratio (reference) = 179.33 / (0.910847 * 1000) = 0.1969
device_ratio (candidate) = 210.12 / (0.606758 * 1000) = 0.3463
```

The candidate's device ratio rose from 0.197 to 0.346 because wall time fell
sharply while device time rose slightly. The class moved from the
host-bound/mixed boundary toward mixed, consistent with the host-side loop and
sync elimination being the dominant causal driver.

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| MLUFusedMatMulGepm (decoder matmul 768->30522) | 50 | 1.00 | 4519.88 | 90.40 |
| reduceKernelMaxIndex (per-sequence max pool) | 200 | 4.00 | 1401.35 | 28.03 |
| MLUBlockKernel5StagePipelineLog1pFast (log1p) | 50 | 1.00 | 1308.56 | 26.17 |
| MLUBlockKernel3StagePipelineClipFast (relu) | 50 | 1.00 | 683.56 | 13.67 |
| MLUFusedMatMulGepdot (dense matmul 768->768) | 50 | 1.00 | 422.56 | 8.45 |
| layerNormForwardKernel (LayerNorm) | 50 | 1.00 | 360.64 | 7.21 |
| MLUBlockKernel3StagePipelineGeluHighAccCubic (GELU) | 50 | 1.00 | 269.79 | 5.40 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _sparse_pooler_max_kernel (fused relu+log1p+per-segment max) | 50 | 1.00 | 4936.48 | 98.73 |
| MLUFusedMatMulGepm (decoder matmul 768->30522) | 50 | 1.00 | 4517.96 | 90.36 |
| MLUFusedMatMulGepdot (dense matmul 768->768) | 50 | 1.00 | 421.24 | 8.42 |
| layerNormForwardKernel (LayerNorm) | 50 | 1.00 | 360.28 | 7.21 |
| MLUBlockKernel3StagePipelineGeluHighAccCubic (GELU) | 50 | 1.00 | 270.08 | 5.40 |

The candidate has exactly 5 kernel types per call (down from 7 types, 10
invocations). The 6 baseline fusion-target kernels (relu, log1p, 4x
reduceKernelMaxIndex) are gone, replaced by one `_sparse_pooler_max_kernel`.
The fused kernel costs 98.73 us/call on the device, which is 30.86 us/call
more than the 67.87 us/call combined cost of the 6 kernels it replaced. The
non-fused kernels (decoder matmul, dense matmul, LayerNorm, GELU) are
unchanged within noise. Device time therefore increased slightly while wall
time decreased sharply, isolating the 33.39% wall gain to host-side Python-loop
and D2H-sync elimination.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Round 001 verification | 182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd | 182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd | pass |

No Coder repair was required. The candidate passed the correctness gate on the
first Verifier attempt. At most one Verifier-to-Coder repair is allowed in the
same round; it was not used.

## Upbound Gap

- upbound_source: `project.md#upbound (estimated semantic bound)`
- comparable_metric: `wall_time_ms`
- absolute_gap: `null (no measured upbound)`
- ratio_to_upbound: `null`
- interpretation: `The declared upbound is a semantic estimate (30-50% wall improvement plausible from a single fused kernel covering the MLM head tail), not a measured bound. The 5% adoption threshold uses measured wall time only. The observed 33.39% improvement falls within the estimated 30-50% semantic range, but this is not a comparison against a measured upbound.`

## evidence_for_next_round

- The fused `_sparse_pooler_max_kernel` is the new dominant device kernel at 98.73 us/call (47.0% of candidate device time). It is slower on the device than the 6 library kernels it replaced (67.87 us/call combined), so there is device-side headroom: a better-tuned fused kernel (e.g., larger BLOCK_V, different num_warps, or a reduction strategy that avoids re-reading the full [total_seq, vocab_size] logits tile per vocab tile) could recover the 30.86 us/call regression and add further device savings.
- The candidate device ratio is 0.346 (mixed). Wall time is 606.76 us/call and device time is 210.12 us/call, so roughly 396 us/call (~65%) is now host-side. The host-side Python loop and D2H sync are gone; the remaining host time is launcher, wrapper, allocation, and harness-fixed cost.
- The decoder matmul (MLUFusedMatMulGepm, 90.36 us/call) is now the second-largest device kernel and the largest non-fused kernel. It is a PyTorch library op; fusing it into Triton would require a `tl.dot` matmul primitive with shape [83,768]x[768,30522], a larger change boundary flagged as a future-round candidate in decision_001.
- `host_sync_count_per_call` is now effectively zero D2H syncs on the max path; further host-side gains must come from launcher reduction or allocation reuse, both of which require Host Plan lifecycle changes.
- The wall improvement already exceeds the decision's 15% expectation and the 5% adoption threshold; the remaining headroom is the 30.86 us/call device regression in the fused kernel plus the decoder matmul.

Record evidence only; do not select the next optimization.

## Stop Recommendation

- recommendation: `continue`
- evidence: `Candidate accepted with 33.39% wall improvement. Device ratio is 0.346 (mixed); the fused kernel is the new dominant device kernel at 98.73 us/call and is slower than the 6 library kernels it replaced, leaving clear device-side headroom. The decoder matmul (90.36 us/call) is the next largest non-fused kernel. None of the 5 stop criteria (measurement-bound, diminishing returns, upbound reached, resource exhausted, user intervention) apply: measurements are not noise-bound, improvement is substantial, the estimated upbound is not a measured bound so "upbound reached" does not apply, resources are not exhausted, and there is no user intervention.`
- applicable stop criteria: none (continue)

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness gate:

```bash
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_001.py \
  --warmup 50 --repeat 100
```

Authoritative wall timing (3 interleaved pairs; run the reference command and the candidate command alternately, 3 times each):

```bash
# accepted reference (run 3 times)
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/baseline_adapter.py \
  --warmup 50 --repeat 100

# candidate (run 3 times)
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_001.py \
  --warmup 50 --repeat 100
```

Level 1 profiler:

```bash
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_001.py \
  --profile --profile-reference-file sparse_pooler/baseline_adapter.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output sparse_pooler/log/round_001_forward_50iter.pt.trace.json
```

For `Result: accepted`, this report contains the decision/candidate/accepted-reference/source
hashes; the correctness/guardrail matrix; all 6 raw samples and unrounded medians; improvement;
the Evaluation Contract mirror (3 observables with expectation/observation/verdict); hypothesis
verdict; Level 1 profiler data (scope summary + top kernels for each scope); retry history;
upbound gap; evidence_for_next_round; stop recommendation; and exact reproduction commands.
Verifier does not update `last_accepted_kernel`; that is Orchestrator's job.
