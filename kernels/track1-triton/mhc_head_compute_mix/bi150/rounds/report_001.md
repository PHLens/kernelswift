# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `kernels/track1-triton/mhc_head_compute_mix/bi150/triton_mhc_head_compute_mix_001.py`
- Accepted reference: `kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `not-computed` (decision is read-only, not in the frozen hash set)
- Candidate SHA256: `a98b1b12593d858ca29c787afa939a3ae0061df4ec6b51aa9a0fe7fa43c6b473`
- Accepted reference SHA256: `ceebdc6185de4c980156a7833073678a0964fb7ccb5edf74b42be6156652eaed`
- Base SHA256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `4e4f0575e2251810e9be7667c98eb4923c6910787d4412abc2c4a976c2b26a8e`
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeded directly to authoritative timing`

The candidate, accepted-reference, base, and harness hashes all match the frozen
project.md / decision values exactly.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=1.429648 ms, v1=0.180503 ms, speedup=7.920x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| public loader contract | candidate exposes `ModelNew/get_init_inputs/get_inputs`; retained by AST loader `_filter_module_ast` | Harness loaded, compiled (`@triton.jit` FunctionDef retained), constructed, and launched without error | pass | correctness return code `0` |
| output structure/shape/dtype | 3-tuple float32 `pre[2,8,4]`, `post[2,8,4]`, `comb[2,8,4,4]` | Independent probe: shapes `[(2,8,4),(2,8,4),(2,8,4,4)]` on both sides | pass | independent probe (below) |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | Independent probe max abs diff: `pre=5.96e-08`, `post=0.0`, `comb=1.19e-07` | pass | independent probe (below) |
| Sinkhorn eps asymmetric placement | first row-normalize adds eps to matrix; loop row + all col-normalize add eps to denominator | exact match (comb max abs diff `1.19e-07`, far below tolerance) confirms asymmetric eps placement preserved | pass | independent probe `comb` diff |
| input not mutated | `forward` must not mutate `mixes/hc_scale/hc_base` | Candidate loads read-only via `tl.load`, writes only to fresh `torch.empty` outputs | pass | candidate source read |
| frozen artifact identity | local hashes equal project.md before measurement | base `4c5167f6...`, adapter `ceebdc61...`, candidate `a98b1b12...`, harness `3d4fa4ee...` all match | pass | SHA256 in round_status_001.md |
| measurement regime | device cuda:0, seed/tolerances defaults, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte | pass | round_status_001.md |

### Independent numerical probe

`base.py Model` vs candidate `ModelNew`, both loaded via the harness AST loader
(`auto_bench.load_ks_module`), fed identical inputs, outputs compared with
`torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`:

| Output | Shape v0 | Shape v1 | allclose | max abs diff |
|---|---|---|---|---|
| pre | `(2,8,4)` | `(2,8,4)` | True | `5.96e-08` |
| post | `(2,8,4)` | `(2,8,4)` | True | `0.0` |
| comb | `(2,8,4,4)` | `(2,8,4,4)` | True | `1.19e-07` |

The candidate reproduces the reference to machine precision, confirming the
Sinkhorn 20-round semantics (including the asymmetric eps placement) are exact.

## Screening Evidence

Not run. The candidate passed correctness and is a correct candidate; screening
applies only when a correct candidate is at least 10% slower than the reference.
The candidate is dramatically faster, so it proceeds directly to authoritative
timing.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation; v0 = baseline_adapter wrapper, v1 = candidate)
- reference_raw_samples_ms: `[1.420944, 1.462009, 1.433128]`
- candidate_raw_samples_ms: `[0.180531, 0.183889, 0.188159]`
- reference_median_ms: `1.433128`
- candidate_median_ms: `0.183889`
- improvement_pct: `87.16869672492618`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (1.433128 - 0.183889) / 1.433128 * 100 = 87.17%
```

The unrounded improvement (87.17%) far exceeds the 5% adoption threshold.

The timing wrapper for the accepted reference was generated with
`sed 's/^class ModelNew/class Model/' baseline_adapter.py` (SHA256
`bcf256b1a4d3d95c88815e65a60e63b7296e65e7940a0f70d5ecad5b2a218e2c`) so the
harness's `Model` entry point could drive the adapter; it was deleted after
measurement. The candidate path was used unmodified.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease | `132.88` (reference) → `1.12` (candidate, incl. 0.12 amortized one-time prologue; steady-state `1.0` fused kernel/call) | pass | profiler summarize + manual scope analysis |
| device_us_per_call | decrease | `924.793` us/call (reference) → `13.879` us/call (candidate total) | pass | profiler summarize + manual scope analysis |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the 20-round Sinkhorn iteration and the surrounding head-compute elementwise chain into a single Triton kernel using a compile-time tl.static_range loop (degraded to dynamic tl.range(19), semantic-equivalent)`
- expected_causal_chain: `per-call kernel count drops from 132.88 to ~1-2 → dominant reduce/div/add tiny kernels disappear, math kept in registers → device_us_per_call drops → wall time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `reference_baseline_adapter`, `candidate_triton_mhc_head_compute_mix_001`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `log/round_001_forward_50iter.pt.trace.json`, SHA256 `961335d11c644fa987f3c32f1d1be9e0f170b633070eea4f6b357afa83492b94`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (`reference_baseline_adapter`) | `46239.66259765625` | `924.793251953125` | `6644` | `132.88` | `1.433128` | `0.6452970369381695` |
| candidate (`candidate_triton_mhc_head_compute_mix_001`) | `693.969` | `13.879` | `56` | `1.12` | `0.183889` | `0.075471` |

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000)
reference = 924.793 / 1433.128 = 0.6453
candidate = 13.879 / 183.889 = 0.0755
```

### Scope-overlap note (candidate scope)

The `summarize_trace.py` scope partition for the candidate raised
`overlapping scope events` because the candidate's 50 forward calls execute in
~0.18 ms total, so the PyTorch `record_function` CPU-side scope markers coalesce
into two overlapping intervals (ts `...3043.453`/dur `9058` and
`...6880.715`/dur `5491`). The reference scope (a single clean interval) was
summarized normally. For the candidate, I partitioned kernels by their `ts` —
every kernel with `ts >= reference_scope_end` belongs to the candidate side —
which yields an unambiguous, non-overlapping assignment. This manual
partitioning produced the candidate totals above (56 kernels, `693.969` us
device total over 50 calls).

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `reduce_kernel<...sum_functor...>` | `1998` | `39.96` | `22023.690` | `440.474` |
| `elementwise_kernel<...DivFunctor...>` | `1998` | `39.96` | `12683.513` | `253.670` |
| `modern::elementwise_kernel<CUDAFunctorOnSelf_add<float>>` | `2048` | `40.96` | `8134.634` | `162.693` |
| `elementwise_kernel<...MulFunctor...>` | `150` | `3.0` | `996.052` | `19.921` |
| `reduce_kernel<...MaxNanFunctor...>` | `50` | `1.0` | `611.007` | `12.220` |
| `elementwise_kernel<...CUDAFunctor_add...> (lambda#2)` | `150` | `3.0` | `593.073` | `11.861` |
| `modern::elementwise_kernel<sigmoid_kernel_cuda>` | `100` | `2.0` | `427.985` | `8.560` |
| `elementwise_kernel<...CUDAFunctor_add...> (lambda#5)` | `50` | `1.0` | `351.712` | `7.034` |
| `modern::elementwise_kernel<exp_kernel_cuda>` | `50` | `1.0` | `216.619` | `4.332` |
| `modern::elementwise_kernel<AUnaryFunctor<MulFunctor>>` | `50` | `1.0` | `201.378` | `4.028` |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `_mhc_head_compute_mix_kernel` (fused Triton) | `50` | `1.0` | `649.790` | `12.996` |
| `reduce_kernel<...sum_functor...>` (one-time prologue) | `2` | `0.04` | `23.48` | `0.470` |
| `elementwise_kernel<...DivFunctor...>` (one-time prologue) | `2` | `0.04` | `12.77` | `0.255` |
| `modern::elementwise_kernel<CUDAFunctorOnSelf_add<float>>` (one-time prologue) | `2` | `0.04` | `7.92` | `0.158` |

The candidate's steady-state device work is a single `_mhc_head_compute_mix_kernel`
launch per forward call (~12.996 us/call). The 6 prologue kernels are a one-time
first-call cost (the `reshape`/`.to(float32)` materialization in `forward`), not
a per-call recurring cost — they appear exactly once at the start of the
candidate's 50-iteration profile window, before the first fused kernel.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial correctness + timing + profiler verification | `a98b1b12593d858ca29c787afa939a3ae0061df4ec6b51aa9a0fe7fa43c6b473` | same | correctness passed; authoritative timing and profiler collected without repair |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Kernel fusion of the 20-round Sinkhorn iteration (plus head-compute elementwise
  chain) collapsed per-call kernel count from `132.88` to `1.0` (steady-state),
  and device time from `924.79 us/call` to `12.996 us/call` (fused kernel), a
  ~71x device-time reduction.
- Wall time improved from `1.433128 ms` to `0.183889 ms` (unrounded median), an
  `87.17%` improvement, confirming the baseline's launch-count overhead was the
  dominant cost.
- Candidate device_ratio dropped to `0.0755`, meaning the operator is now
  host-bound: only ~7.5% of wall time is device kernel time, and ~92.5% is
  host/launch/synchronization overhead (largely the harness `set_seed` +
  `sync_devices` per-iteration floor). Further device-side fusion has limited
  headroom; the remaining bottleneck is the host-side per-call overhead and the
  single-kernel launch.
- The `tl.static_range(19)` compile-time unroll was confirmed infeasible on this
  runtime (>300s compile); the dynamic `tl.range(19)` fallback preserves exact
  semantics and achieves the fusion. This is now empirical evidence (not just a
  capability-risk prediction) that 19-iteration static unrolling does not lower
  on CoreX/Triton 3.1.0.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 001 accepted with `87.17%` wall improvement (well above 5%),
  kernel count `132.88 → 1.0`, device time `924.79 → 12.996 us/call`. The
  operator is now host-bound (`device_ratio 0.0755`). No optional target is
  configured and no terminal-round limit is reached; further rounds may target
  the remaining host-side overhead if a new decision is validated.

Orchestrator owns the canonical pointer update and workflow transition.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/bi150/triton_mhc_head_compute_mix_001.py --warmup 50 --repeat 100 --full-traceback
```

Interleaved wall timing (reference wrapper, execute three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sed 's/^class ModelNew/class Model/' kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py > /tmp/mhcm_baseline_model_001.py
python3 auto_bench.py --v0_file /tmp/mhcm_baseline_model_001.py --v1_file kernels/track1-triton/mhc_head_compute_mix/bi150/triton_mhc_head_compute_mix_001.py --warmup 50 --repeat 100
```

Separately scoped profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/mhcm_baseline_model_001.py --v1_file kernels/track1-triton/mhc_head_compute_mix/bi150/triton_mhc_head_compute_mix_001.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py --profile-output kernels/track1-triton/mhc_head_compute_mix/bi150/log/round_001_forward_50iter.pt.trace.json
```

Reference scope summary (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_head_compute_mix/bi150/log/round_001_forward_50iter.pt.trace.json --iterations 50 --scope reference_baseline_adapter --wall-ms 1.433128
```
