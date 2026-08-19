# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_mhc_head_compute_mix_backward_001.py`
- Accepted reference: `baseline_adapter.py` (canonical after Round 000)
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `dc0a4837cc8a5aeb867e9d71f8c1e4bc1930ee57d431a279f761329271e5371a`
- Candidate SHA256: `5d419f5d2e920abf3cf583a22f155e76047f9e5bc3a5cc36baca5477fae94349`
- Accepted reference SHA256: `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d`
- Base SHA256: `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `a03823074048c8cb5e8199b593c8c19aa3b259180969321015e5a1679461b71a`
- verification_tier: `authoritative`
- screening_pairs: `not-run` (correctness passed; proceeded directly to authoritative timing)

All hashes match the frozen project.md / decision / coder_result values exactly.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=0.344201 ms, v1=0.198203 ms, speedup=1.737x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| independent numerical probe | base `Model` vs candidate `ModelNew` fed identical inputs; allclose of three outputs | `grad_input_mix` max_abs_diff `1.49e-8`; `grad_mhc_scale` `1.67e-6`; `grad_mhc_base` `1.14e-5`; all `allclose=True` | pass | `/tmp/probe_mhcbwd_verify.py` |
| reduction contract (dim semantics) | `grad_mhc_base = grad_z.sum(dim=(0,1))` → `[4]`; `grad_mhc_scale = (grad_z*input_mix).sum(dim=(0,1,2))` → `[1]` | Manual recompute matched candidate exactly: `gb_ref=[2.2168, 8.3552, 2.6569, 19.1573]` vs cand; `gs_ref=[-2.0920]` vs cand | pass | probe output; `decision_001.md` invariants |
| output structure/shape/dtype | 3-tuple fp32 `[2,1024,4]`, `[1]`, `[4]` | Harness comparator accepted structure/shape/dtype; probe confirmed shapes `(2,1024,4)`, `(1,)`, `(4,)` | pass | correctness code `0`; probe output |
| frozen artifact identity | local hashes equal frozen values before measurement | all five files match | pass | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte | pass | round_status_001.md |

The correctness command's `v0=0.344201 ms` / `v1=0.198203 ms` are smoke timing and do not replace the authoritative interleaved samples below.

## Screening Evidence

Not run. Correctness passed on the first attempt, so the candidate proceeded
directly to authoritative timing (three interleaved pairs). No screening
classification was made.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[0.350071, 0.345161, 0.349112]`
- candidate_raw_samples_ms: `[0.199972, 0.198597, 0.196444]`
- reference_median_ms: `0.349112`
- candidate_median_ms: `0.198597`
- improvement_pct: `43.11367125736153`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
                = (0.349112 - 0.198597) / 0.349112 * 100
                = 43.1137%
```

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `0.350071` | `0.199972` | `0` |
| 2 | `0.345161` | `0.198597` | `0` |
| 3 | `0.349112` | `0.196444` | `0` |

The unrounded improvement `43.11%` exceeds the `5.0` adoption threshold and the
decision's `expected_wall_improvement_pct` of `20.0`. Correctness and every
guardrail pass.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the full sigmoid-backward chain (elementwise multiply/add/sigmoid plus the two sum reductions) into a single Triton kernel`
- expected_causal_chain: intermediate z/sigmoid/grad_z tensors stop being materialized; kernel count drops toward 1-2; the two dominant sum reductions absorbed into the fused kernel; device time decreases; wall time decreases
- primary_metric: `wall_time` (expected improvement `5.0`)

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `kernel_count_per_call` | decrease toward 1-2 | `9.74` → `~2.96` (1 fused Triton kernel + 2 `torch.zeros` accumulator fills per forward) | pass | profiler summarize |
| `device_us_per_call` | decrease | `186.057` → `~14.692` us/call (92% reduction) | pass | profiler summarize |
| `reduce_sum_kernel_us_per_call` | decrease toward zero as a standalone kernel | standalone `reduce_kernel<sum_functor>` fully eliminated (0 in candidate scope); the two sums absorbed into `_mhc_head_compute_mix_backward_kernel` | pass | profiler summarize |

- Hypothesis verdict: `confirmed`

The full expected causal chain is observed: intermediate tensors are no longer
materialized (single fused kernel), kernel count collapsed from 9.74 toward ~3,
the two dominant standalone `sum` reductions are eliminated (absorbed into the
fused kernel's `tl.sum`), device time dropped 92%, and wall time dropped 43%.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `reference_baseline_adapter`, `candidate_triton_mhc_head_compute_mix_backward_001`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `log/round_001_forward_50iter.pt.trace.json`, SHA256 `c13dbb5389a99f17cbdefb45955a21769c92da5220dff44942638e2e87e5d976`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `reference_baseline_adapter` | `9302.8671875` | `186.05734375` | `487` | `9.74` | `0.349112` | `0.5329445672162516` |
| `candidate_triton_mhc_head_compute_mix_backward_001` | `734.590` | `14.692` | `148` | `2.96` | `0.198597` | `0.07397896242138602` |

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000)
reference: 186.057 / 349.112 ≈ 0.533
candidate:  14.692 / 198.597 ≈ 0.074
```

### Profiler scope note (nested record_function artifact)

The candidate scope emitted two overlapping `record_function` X-events (a
PyTorch profiler artifact when a fused Triton launch is wrapped by the harness's
`record_function(label)`), which caused `summarize_trace.py --scope
candidate_triton_mhc_head_compute_mix_backward_001` to reject the scope as
"overlapping scope events". The candidate device totals above were therefore
summarized by manually filtering `cat=kernel` events inside the inner (clean)
candidate scope. This inner scope contains exactly the candidate's kernels:
`_mhc_head_compute_mix_backward_kernel` ×50 (one fused kernel per forward) plus
`FillFunctor` ×~98 (the `torch.zeros` initialization of the two accumulator
outputs). The outer candidate scope additionally captured a small number of
leftover reference kernels still in flight at the scope boundary (reduce/mul/
sigmoid/add), which are reference-scope events, not candidate work.

### Accepted Reference Top Kernels (reference_baseline_adapter scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `reduce_kernel<1024,1,ReduceOp<float,sum_functor...>>` | `96` | `1.92` | `7403.756` | `148.075` |
| `modern::elementwise_kernel<BinaryFunctor<float,MulFunctor<float>>>` | `146` | `2.92` | `715.945` | `14.319` |
| `elementwise_kernel<512,4,BinaryFunctor<float,MulFunctor<float>>>` | `98` | `1.96` | `462.859` | `9.257` |
| `modern::elementwise_kernel<sigmoid_kernel_cuda>` | `49` | `0.98` | `260.508` | `5.210` |
| `elementwise_kernel<512,4,CUDAFunctor_add<float>>` (lambda#2) | `49` | `0.98` | `242.564` | `4.851` |
| `modern::elementwise_kernel<CUDAFunctorOnOther_add<float>>` | `49` | `0.98` | `217.234` | `4.345` |

### Candidate Top Kernels (candidate_triton_mhc_head_compute_mix_backward_001 scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `modern::elementwise_kernel<FillFunctor<float>>` (torch.zeros accumulators) | `98` | `1.96` | `363.797` | `7.276` |
| `_mhc_head_compute_mix_backward_kernel` (fused Triton kernel) | `50` | `1.0` | `370.793` | `7.416` |

The candidate reduces device time from `186.057 us/call` to `14.692 us/call`
(92% reduction) and kernel count from `9.74/call` to `~2.96/call`. The two
dominant standalone `sum_functor` reductions (`148.075 us/call`, ~80% of
baseline device time) are completely eliminated and absorbed into the single
fused Triton kernel (`_mhc_head_compute_mix_backward_kernel`, `7.416 us/call`).
The remaining candidate device time is dominated by the `torch.zeros`
accumulator initialization (`FillFunctor`, ~`7.276 us/call`), which is a
host-side setup cost that a later round could further reduce (e.g. by
initializing accumulators inside the kernel or reusing a persistent buffer).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, independent probe, authoritative timing, targeted profiler | `5d419f5d2e920abf3cf583a22f155e76047f9e5bc3a5cc36baca5477fae94349` | same | correctness and timing passed; profiler summarized with a nested-scope note |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Candidate `triton_mhc_head_compute_mix_backward_001.py` is `accepted`: wall median `0.198597 ms` vs reference `0.349112 ms`, improvement `43.11%`, hypothesis `H-001` `confirmed`.
- Kernel fusion succeeded: kernel count `9.74` → `~2.96` per call; the two dominant `sum` reductions (`148.075 us/call`) are eliminated and absorbed into a single fused Triton kernel (`7.416 us/call`).
- Candidate device time `14.692 us/call` is now dominated by `torch.zeros` accumulator initialization (`FillFunctor`, ~`7.276 us/call`, ~half of device time), not by the actual computation (`_mhc_head_compute_mix_backward_kernel`, `7.416 us/call`).
- Candidate `device_ratio` is now only `0.074` — the operator has become strongly host-bound (~93% of wall time is host/launch overhead, including the two `torch.zeros` launches, `torch.empty` allocation, `view`/`contiguous` calls, and the Triton launch path). A future round targeting host-side overhead (persistent zero-initialized accumulator buffers, reduced Python-side per-call allocation, or a lower-overhead launch) would attack the remaining wall-time fraction.
- The profiler nested-scope artifact (two overlapping `record_function` events for a fused Triton launch) is worth recording: `summarize_trace.py --scope` rejects the candidate scope; candidate device totals were recovered by manual kernel filtering.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 001 is `accepted` with a `43.11%` wall improvement, but the operator is now strongly host-bound (`device_ratio ≈ 0.074`), leaving substantial remaining wall time (`0.199 ms`) dominated by host/launch overhead and `torch.zeros` accumulator initialization. No optional target is configured.

Orchestrator owns the stop transition and canonical pointer updates.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (run before and after measurement; all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/mhc_head_compute_mix_backward/base.py kernels/track1-triton/mhc_head_compute_mix_backward/bi150/baseline_adapter.py kernels/track1-triton/mhc_head_compute_mix_backward/bi150/triton_mhc_head_compute_mix_backward_001.py kernels/track1-triton/mhc_head_compute_mix_backward/bi150/rounds/decision_001.md auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/bi150/triton_mhc_head_compute_mix_backward_001.py --warmup 50 --repeat 100 --full-traceback
```

Authoritative timing (wrapper for baseline_adapter exposes `Model`; execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sed 's/ModelNew/Model/g' kernels/track1-triton/mhc_head_compute_mix_backward/bi150/baseline_adapter.py > /tmp/mhcmb_baseline_model_001.py && python3 auto_bench.py --v0_file /tmp/mhcmb_baseline_model_001.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/bi150/triton_mhc_head_compute_mix_backward_001.py --warmup 50 --repeat 100
```

Targeted profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/mhcmb_baseline_model_001.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/bi150/triton_mhc_head_compute_mix_backward_001.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/mhc_head_compute_mix_backward/bi150/baseline_adapter.py --profile-output kernels/track1-triton/mhc_head_compute_mix_backward/bi150/log/round_001_forward_50iter.pt.trace.json
```

Separately scoped reference summary (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_head_compute_mix_backward/bi150/log/round_001_forward_50iter.pt.trace.json --iterations 50 --scope reference_baseline_adapter --wall-ms 0.349112
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity |
| correctness 50/100 (base vs candidate) | `0` | report Correctness table |
| independent numerical probe | `0` | report Correctness table (reduction semantics) |
| authoritative pair 1, 50/100 | `0` | report Interleaved Wall Timing |
| authoritative pair 2, 50/100 | `0` | report Interleaved Wall Timing |
| authoritative pair 3, 50/100 | `0` | report Interleaved Wall Timing |
| targeted profiler 20/50 | `0` | `log/round_001_forward_50iter.pt.trace.json` |
| summarize `reference_baseline_adapter` | `0` | report Profiler Evidence |
| summarize candidate (manual, nested-scope) | `0` | report Profiler Evidence |
