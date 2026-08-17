# Report 003

Result: no-improvement

## Identity

- Round: `003`
- Decision: `rounds/decision_003.md`
- Candidate: `triton_sparse_pooler_003.py`
- Accepted reference: `triton_sparse_pooler_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `8f78d0425148e387ba82fc827012c63440e8d38edcdf19750a0e79825c8505bb`
- Candidate SHA256: `3406f7c9a731e1fd7560ab95bf1d903fd4d6f8287c5880d9801e9d41e5ed7860`
- Accepted reference SHA256: `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`
- Base SHA256: `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `a0208c7da7e371d45c88f82ebddd3850d01669aa5d912f31db9234a7a56ebab7`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` between `base.py` Model and `triton_sparse_pooler_003.py` ModelNew outputs (list of 4 x Tensor[30522]) | `PASS accuracy` across the correctness gate (warmup 50, repeat 100), all 4 v2 screening runs, and the profiler run | pass | `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_003.py --warmup 50 --repeat 100` |
| output structure | list of 4 tensors, each shape [30522], dtype fp32, device mlu:0 | candidate returns `[out[i] for i in range(num_seq)]` from a `[num_seq, vocab_size]` fp32 mlu:0 tensor; confirmed by correctness gate and profiler run | pass | `triton_sparse_pooler_003.py` forward; harness `PASS accuracy` |
| numerical semantics | `log(1+relu(decoder(LayerNorm(GELU(Dense(hidden)))))` max-pooled per sequence within atol=1e-2 rtol=1e-2 | fused matmul kernel computes `log(1+relu(tl.dot(...)+bias))` per element then per-segment max; correctness gate passed with `input_precision="ieee"` | pass | `_sparse_pooler_fused_matmul_max_kernel` body; `PASS accuracy` |
| caller-selected device and current stream preserved | no explicit `torch.mlu.device()` introduced; kernel launches on current stream | candidate uses `device = x.device` for output allocation; no device context; kernel launch inherits stream | pass | `triton_sparse_pooler_003.py` forward |
| dense GELU LayerNorm pipeline unchanged | library ops preserved | `self.layer_norm(self.act(self.dense(hidden_states)))` unchanged; decoder matmul moved into the fused Triton kernel as the decision authorizes | pass | `triton_sparse_pooler_003.py` forward |
| ModelNew public constructor and forward signature unchanged | `ModelNew(hidden_size=768, vocab_size=30522, pooling="max")` and `forward(hidden_states, seq_lens) -> list[Tensor]` | signatures match Round 001 exactly | pass | `triton_sparse_pooler_003.py` |
| load_state_dict compatibility | candidate accepts reference state dict | harness runs `model_new.load_state_dict(model.state_dict())` before timing; `PASS accuracy` confirms; decoder weight remains an `nn.Linear` parameter in canonical `[vocab, hidden]` layout | pass | harness behavior; `PASS accuracy` |
| `kernel_count_per_call` decreases (not increases) | guardrail from the decision's Evaluation Contract | reference 5.0 kernels/call -> candidate 4.0 kernels/call (decreased by 1); the library decoder matmul (`MLUFusedMatMulGepm`) and the existing fused `_sparse_pooler_max_kernel` are replaced by one fused `_sparse_pooler_fused_matmul_max_kernel` | pass | profiler scope summary below |
| `num_warps=2` is not used | known to fail on this runtime per `triton_mlu` target profile | candidate uses `num_warps=1` exactly as the decision requires | pass | `triton_sparse_pooler_003.py` dispatch |

Correctness and every declared guardrail pass. The candidate is a conforming implementation of the decision; it simply does not clear the 5% adoption threshold (it is screened-out before authoritative timing).

## v2 Screening

- protocol: after correctness passes, run exactly two short interleaved accepted-reference/candidate pairs (warmup 5, repeat 5). A correct candidate is `screened-out` only when BOTH pairs are at least 10% slower than the accepted reference. Otherwise proceed to authoritative 3-pair timing.
- accepted reference: `triton_sparse_pooler_001.py` (compared as v1 against `base.py` v0)
- candidate: `triton_sparse_pooler_003.py` (compared as v1 against `base.py` v0)

| Pair | Role | Command (abbreviated) | v0 ms | v1 ms | speedup | exit | Slower vs reference? |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | reference (triton_sparse_pooler_001) | `--v1_file .../triton_sparse_pooler_001.py --warmup 5 --repeat 5` | 0.918679 | 0.623747 | 1.473x | 0 | n/a |
| 1 | candidate (triton_sparse_pooler_003) | `--v1_file .../triton_sparse_pooler_003.py --warmup 5 --repeat 5` | 0.895909 | 0.835409 | 1.072x | 0 | +33.90% slower |
| 2 | reference (triton_sparse_pooler_001) | `--v1_file .../triton_sparse_pooler_001.py --warmup 5 --repeat 5` | 1.108457 | 0.658215 | 1.684x | 0 | n/a |
| 2 | candidate (triton_sparse_pooler_003) | `--v1_file .../triton_sparse_pooler_003.py --warmup 5 --repeat 5` | 0.970991 | 0.869966 | 1.116x | 0 | +32.16% slower |

```text
pair 1 slowdown_pct = (0.835409 - 0.623747) / 0.623747 * 100 = 33.90%
pair 2 slowdown_pct = (0.869966 - 0.658215) / 0.658215 * 100 = 32.16%
```

Both pairs are at least 10% slower than the accepted reference. The candidate is `screened-out`. Per the v2 screening protocol, the authoritative 3-pair timing was NOT run. The screening samples are short (warmup 5, repeat 5) and are persisted as diagnostic evidence only; they are not used to compute `improvement_pct`. The adoption decision is driven by the screening verdict: the candidate is at least 10% slower than the accepted reference on both short pairs, so it cannot clear the 5% adoption threshold even with a full 3-pair measurement.

## Interleaved Wall Timing (authoritative)

- status: `not run`
- reason: candidate is `screened-out` in v2 screening (both pairs >=10% slower than accepted reference). Per the v2 screening protocol, the authoritative 3-pair timing is not run.
- reference_raw_samples_ms: `n/a (screened-out)`
- candidate_raw_samples_ms: `n/a (screened-out)`
- reference_median_ms: `n/a (screened-out)`
- candidate_median_ms: `n/a (screened-out)`
- improvement_pct: `not computed (screened-out)`; the screening evidence shows the candidate is approximately 33% slower than the accepted reference, which is far below the 5% adoption threshold

## Evaluation Contract Mirror

- hypothesis_id: `H-003`
- intervention: fuse the decoder matmul (via `tl.dot` with K-dimension tiling), bias addition, relu, log1p, and per-sequence max reduction into a single Triton kernel launched once per forward, eliminating the library `MLUFusedMatMulGepm` decoder matmul kernel (90.36 us/call) and the existing fused `_sparse_pooler_max_kernel` (98.73 us/call) and avoiding materialization of the intermediate logits tensor `[total_seq, vocab_size]` in global memory
- expected_causal_chain: the decoder matmul is fused into the Triton kernel via `tl.dot`, eliminating the library matmul kernel; the intermediate logits tensor is no longer materialized, saving the matmul output write and the fused reduction kernel input read; the new fused kernel replaces two device kernels with one; total device kernel count per call decreases from 5 to 4; device time decreases from ~210.12 us/call; wall time decreases by at least 5%
- primary_metric: `wall_time`, expected_improvement_pct 5.0
- profiling_level: `targeted`

### Mechanism observables

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `decoder_matmul_kernel_count_per_call` | decrease from 1 to 0; the `MLUFusedMatMulGepm` library kernel is eliminated because the matmul is fused into the Triton kernel via `tl.dot` | reference 1.0 -> candidate 0.0 (exactly); the `MLUFusedMatMulGepm` kernel does not appear in the candidate profiler scope | confirmed | profiler candidate scope summary below (4 kernels, no `MLUFusedMatMulGepm`) |
| `total_kernel_count_per_call` | decrease from 5 to 4; the decoder matmul and the existing fused reduction kernel are replaced by one fused matmul+relu+log1p+max kernel; dense matmul, LayerNorm, and GELU are unchanged | reference 5.0 -> candidate 4.0 (exactly); dense matmul, LayerNorm, and GELU are unchanged within noise | confirmed | profiler scope summary below |
| `device_us_per_call` | decrease from ~210.12 us/call; the combined cost of the two replaced kernels is 189.09 us/call (90.36 + 98.73), and the new fused kernel is expected to cost less because it avoids materializing and re-reading the intermediate logits tensor | reference 212.32 us/call -> candidate 392.94 us/call (INCREASED by 180.63 us/call); the new fused kernel at 373.31 us/call is 184.22 us/call slower than the 191.44 us/call combined cost of the two kernels it replaced (99.52 + 91.92) | falsified | profiler scope summary below |
| `fused_kernel_us_per_call` | the new fused matmul+relu+log1p+max kernel costs less than the 189.09 us/call combined cost of the decoder matmul (90.36 us/call) and the existing fused reduction kernel (98.73 us/call) it replaces | new fused `_sparse_pooler_fused_matmul_max_kernel` costs 373.31 us/call; the two replaced kernels cost 191.44 us/call combined (99.52 + 91.92); the new fused kernel is 184.22 us/call SLOWER, not faster | falsified | profiler candidate scope top kernel below |

### Hypothesis verdict: `falsified`

Two of the four mechanism observables (`decoder_matmul_kernel_count_per_call` and `total_kernel_count_per_call`) are confirmed: the library decoder matmul is eliminated and the kernel count drops from 5 to 4. However, the two device-time observables (`device_us_per_call` and `fused_kernel_us_per_call`) are falsified in the strongest possible direction: the new fused `_sparse_pooler_fused_matmul_max_kernel` at 373.31 us/call is 184.22 us/call SLOWER than the 191.44 us/call combined cost of the two kernels it replaced (the existing fused reduction kernel at 99.52 us/call plus the library decoder matmul at 91.92 us/call). Total device time increased from 212.32 to 392.94 us/call — a 180.63 us/call device regression — and wall time increased by ~33% (screening evidence). The root cause is that `tl.dot` with small M (BLOCK_M=32, actual seq_len 18-25) is very inefficient on the MLU590-H8 architecture: the matmul-tiling overhead per program dominates the saved intermediate-tensor traffic. This is exactly the primary risk the decision recorded: "If the `tl.dot` path is slower than the library matmul for these shapes, the new fused kernel may not beat 189.09 us/call, and the hypothesis will be falsified." The profiler evidence confirms the risk materialized. Because the primary metric fails the threshold (screened-out before authoritative timing) AND two observables contradict the expected causal chain, the verdict is `falsified`.

## Profiler Evidence

- profiler_level: `summary` (Level 1)
- iterations: `50` (forward calls per scope)
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/round_003_forward_50iter.pt.trace.json`
- profiler run also reported `PASS accuracy; v0=0.888508 ms, v1=0.822277 ms, speedup=1.081x` (diagnostic only; not the authoritative median, which was not run because the candidate was screened-out)

Reference and candidate scopes are collected and summarized independently. All totals below are normalized by `iterations` before they are compared. The profiler trace was produced with `--profile --profile-reference-file sparse_pooler/triton_sparse_pooler_001.py --profile-mode forward --profile-warmup 20 --profile-iterations 50`. The standard `scripts/summarize_trace.py` failed with `overlapping scope events` because each scope has 2 events (`user_annotation` on CPU + `gpu_user_annotation` on GPU) that overlap. Per the Verifier contract fallback (documented in `state/verifier_context.md` and in `rounds/report_001.md`/`rounds/report_002.md`), a custom Python summarizer was used that scopes kernels using ONLY the `gpu_user_annotation` device-side intervals (filtering to GPU scope only), matching the `summarize_trace` internal logic.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Device ratio (diagnostic wall) |
|---|---:|---:|---:|---:|---:|
| reference_triton_sparse_pooler_001 | 10615.88 | 212.32 | 250 | 5.0 | 0.3312 |
| candidate_triton_sparse_pooler_003 | 19647.08 | 392.94 | 200 | 4.0 | 0.4608 |

```text
device_ratio (reference, diagnostic) = 212.32 / (0.640981 * 1000) = 0.3312
device_ratio (candidate, diagnostic) = 392.94 / (0.852688 * 1000) = 0.4608
```

The diagnostic wall values above are the means of the two v2 screening samples for each side (reference: (0.623747 + 0.658215) / 2 = 0.640981 ms; candidate: (0.835409 + 0.869966) / 2 = 0.852688 ms). These are short-sample diagnostics only; the authoritative 3-pair medians were not collected because the candidate was screened-out. The device_ratio is therefore diagnostic, not authoritative.

The candidate's device ratio rose from 0.331 to 0.461 because device time rose sharply (212.32 -> 392.94 us/call) while wall time also rose (screening +33%). The class remains mixed but moved further toward the device-bound boundary. The device-side regression in the new fused kernel (+184.22 us/call vs the two kernels it replaced) is the dominant cause of the device-time increase; the three non-fused kernels (dense matmul, LayerNorm, GELU) are unchanged within noise.

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `_sparse_pooler_max_kernel` (fused relu+log1p+per-segment max) | 50 | 1.00 | 4975.76 | 99.52 |
| `MLUFusedMatMulGepm` (decoder matmul 768->30522) | 50 | 1.00 | 4596.16 | 91.92 |
| `MLUFusedMatMulGepdot` (dense matmul 768->768) | 50 | 1.00 | 441.56 | 8.83 |
| `layerNormForwardKernel` (LayerNorm) | 50 | 1.00 | 374.12 | 7.48 |
| `MLUBlockKernel3StagePipelineGeluHighAccCubic` (GELU) | 50 | 1.00 | 228.28 | 4.57 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `_sparse_pooler_fused_matmul_max_kernel` (fused matmul+bias+relu+log1p+max via `tl.dot`) | 50 | 1.00 | 18665.40 | 373.31 |
| `MLUFusedMatMulGepdot` (dense matmul 768->768) | 50 | 1.00 | 391.32 | 7.83 |
| `layerNormForwardKernel` (LayerNorm) | 50 | 1.00 | 362.16 | 7.24 |
| `MLUBlockKernel3StagePipelineGeluHighAccCubic` (GELU) | 50 | 1.00 | 228.20 | 4.56 |

The candidate has exactly 4 kernel types per call (down from 5 types, 5 invocations). The library `MLUFusedMatMulGepm` decoder matmul (91.92 us/call in the reference) is gone, replaced by the matmul inside the new fused `_sparse_pooler_fused_matmul_max_kernel`. The new fused kernel costs 373.31 us/call on the device, which is 184.22 us/call MORE than the 191.44 us/call combined cost of the two kernels it replaced (99.52 + 91.92). The three non-fused kernels (dense matmul 7.83, LayerNorm 7.24, GELU 4.56 us/call) are unchanged within noise relative to the reference (8.83, 7.48, 4.57). The entire device-time increase (180.63 us/call) is attributable to the new fused kernel. The `tl.dot` matmul with small M (BLOCK_M=32, actual seq_len 18-25) is very inefficient on MLU590-H8 — the per-program matmul-tiling overhead dominates the saved intermediate-tensor traffic, exactly the primary risk the decision recorded.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Round 003 verification (correctness gate) | `3406f7c9a731e1fd7560ab95bf1d903fd4d6f8287c5880d9801e9d41e5ed7860` | `3406f7c9a731e1fd7560ab95bf1d903fd4d6f8287c5880d9801e9d41e5ed7860` | pass (correctness); no-improvement (screened-out in v2 screening; device-time hypothesis falsified) |

No Coder repair was required by Verifier. The candidate passed the correctness gate on the first Verifier attempt. Coder's own attempt ledger records one non-semantic tile-size accommodation (BLOCK_K=128 -> 64, BLOCK_V=1024 -> 512) before handoff; that is Coder's internal repair, not a Verifier-to-Coder repair. At most one Verifier-to-Coder repair is allowed in the same round; it was not used. The candidate simply does not clear the 5% adoption threshold (it is screened-out before authoritative timing), so the terminal classification is `no-improvement`, not `candidate-failed`.

## Upbound Gap

- upbound_source: `project.md#upbound (estimated semantic bound)`
- comparable_metric: `wall_time_ms`
- absolute_gap: `null (no measured upbound; candidate was screened-out before authoritative timing)`
- ratio_to_upbound: `null`
- interpretation: `The declared upbound is a semantic estimate (30-50% wall improvement plausible from a single fused kernel covering the MLM head tail), not a measured bound. The 5% adoption threshold uses measured wall time only. The candidate was screened-out in v2 screening (~33% slower than the accepted reference), so the upbound comparison is not the deciding factor; the measured screening result controls. The estimated upbound assumed a single fused kernel covering the MLM head tail; this round attempted only the decoder matmul + existing fused reduction (not the dense/GELU/LayerNorm), and the `tl.dot` matmul with small M proved inefficient on this runtime, so the semantic upbound remains unverified.`

## evidence_for_next_round

- The new fused `_sparse_pooler_fused_matmul_max_kernel` at 373.31 us/call is 184.22 us/call SLOWER than the 191.44 us/call combined cost of the two kernels it replaced (the existing fused `_sparse_pooler_max_kernel` at 99.52 us/call and the library `MLUFusedMatMulGepm` decoder matmul at 91.92 us/call). The hypothesis that fusing the decoder matmul via `tl.dot` would reduce the combined device cost is falsified on this runtime.
- Root cause: `tl.dot` with small M (BLOCK_M=32, actual seq_len 18-25) is very inefficient on MLU590-H8. The per-program matmul-tiling overhead (12 K-tiles of BLOCK_K=64, 60 vocab tiles of BLOCK_V=512, 4 sequences = 240 programs total) dominates the saved intermediate-tensor traffic. The `fused_moe` and `flexattention` evidence kernels cited in the target profile use `tl.dot` with larger M dimensions; this project's M (max 25, padded to 32) is too small for `tl.dot` to be competitive with the library `MLUFusedMatMulGepm` decoder matmul.
- Total device time increased from 212.32 to 392.94 us/call (a 180.63 us/call device regression). Wall time increased by ~33% in screening. The candidate is correct but is a device-side and wall-side regression.
- The `kernel_count_per_call` observable was confirmed (5 -> 4) and the `decoder_matmul_kernel_count_per_call` observable was confirmed (1 -> 0), but the device-time observables were falsified. Fewer kernels does not imply lower device time when the fused kernel is much slower per call.
- The accepted Round 001 reference (`triton_sparse_pooler_001.py`) remains the canonical kernel. Its device profile is: `_sparse_pooler_max_kernel` 99.52 us/call (47%), `MLUFusedMatMulGepm` decoder matmul 91.92 us/call (43%), dense matmul 8.83, LayerNorm 7.48, GELU 4.57 us/call. The two largest kernels (fused reduction + decoder matmul) together account for 191.44 us/call (90% of device time).
- The `kernel-matmul-fusion` change family via `tl.dot` with small M is falsified on this runtime. A future round targeting the decoder matmul should consider a different mechanism (e.g., a library op with a better small-M path, a different tiling strategy, or host-side launcher/allocation reduction). The dense matmul (8.83 us/call), LayerNorm (7.48), and GELU (4.57) are small and fusing them would require a much larger change boundary (the full MLM head) for limited device savings.
- The candidate device ratio is 0.461 (diagnostic, mixed). Wall time is ~853 us/call (screening mean) and device time is 392.94 us/call, so roughly 460 us/call (~54%) is host-side. The host-side Python loop and D2H sync are gone (eliminated in Round 001); the remaining host time is launcher, wrapper, allocation, and harness-fixed cost.

Record evidence only; do not select the next optimization.

## Stop Recommendation

- recommendation: `continue`
- evidence: `Candidate not adopted: the candidate is screened-out in v2 screening (both pairs >=10% slower than the accepted reference: +33.90% and +32.16% slower). The hypothesis H-003 is falsified: the new fused `_sparse_pooler_fused_matmul_max_kernel` at 373.31 us/call is 184.22 us/call slower than the 191.44 us/call combined cost of the two kernels it replaced, because `tl.dot` with small M (BLOCK_M=32, actual seq_len 18-25) is inefficient on MLU590-H8. Two of the four mechanism observables (`decoder_matmul_kernel_count_per_call` and `total_kernel_count_per_call`) were confirmed, but the two device-time observables (`device_us_per_call` and `fused_kernel_us_per_call`) were falsified. Device time increased from 212.32 to 392.94 us/call. performance_miss_streak becomes 2 after this round. None of the 5 stop criteria trigger yet: measurements are not noise-bound (the device-time regression is large and consistent, not noise), the streak is 2 (not 3, so diminishing returns does not trigger), the estimated upbound is not a measured bound (upbound reached does not apply), resources are not exhausted, and there is no user intervention. The `kernel-matmul-fusion` change family via `tl.dot` with small M is exhausted on this runtime; the next round should target a different change family (e.g., host-side launcher/allocation reduction, or a different device-side mechanism for the decoder matmul that does not rely on `tl.dot` with small M).`
- applicable stop criteria: none (continue)

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness gate:

```bash
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_003.py \
  --warmup 50 --repeat 100
```

v2 screening (two short interleaved accepted-reference/candidate pairs; warmup 5, repeat 5):

```bash
# pair 1 reference
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_001.py \
  --warmup 5 --repeat 5

# pair 1 candidate
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_003.py \
  --warmup 5 --repeat 5

# pair 2 reference
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_001.py \
  --warmup 5 --repeat 5

# pair 2 candidate
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_003.py \
  --warmup 5 --repeat 5
```

Authoritative wall timing (NOT run — candidate was screened-out; the command shape is recorded for completeness):

```bash
# accepted reference (would have been run 3 times)
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_001.py \
  --warmup 50 --repeat 100

# candidate (would have been run 3 times)
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_003.py \
  --warmup 50 --repeat 100
```

Level 1 profiler:

```bash
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/triton_sparse_pooler_003.py \
  --profile --profile-reference-file sparse_pooler/triton_sparse_pooler_001.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output sparse_pooler/log/round_003_forward_50iter.pt.trace.json
```

For `Result: no-improvement`, this report contains the decision/candidate/accepted-reference/source hashes; the correctness/guardrail matrix; all v2 screening raw samples and the screening verdict; the Evaluation Contract mirror (4 observables with expectation/observation/verdict); hypothesis verdict; Level 1 profiler data (scope summary + top kernels for each scope); retry history; upbound gap; evidence_for_next_round; stop recommendation; and exact reproduction commands. Verifier does not update `last_accepted_kernel`; that is Orchestrator's job.
