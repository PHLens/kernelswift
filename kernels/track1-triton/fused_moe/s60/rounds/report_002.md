# Report 002

Result: accepted

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_fused_moe_002.py`
- Accepted reference: `triton_fused_moe_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `31e2767c6b1a65aaa49b0c3a9711dd2f0a3c06a741ef042e1c60bf31a911df6d`
- Candidate SHA256: `e5d0058d6fb6f23f94e2623ae374d2776c4b2c6d4eb235b2c6c75524fb44eb73`
- Accepted reference SHA256: `444eb2fb3e14c48359b27137d11b7f57da22211ad4034e6e56f05af5b4561124`
- Base SHA256: `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `d8f8f6bf8965ab279eb59215a7cc0c6f24f7dd0ad5ea7d8436162336955af6c3`
- verification_tier: authoritative
- screening_pairs: `not-run (correctness gate passed on first attempt; proceeded directly to authoritative timing)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `PASS accuracy; v0=5.652439 ms, v1=0.386946 ms, speedup=14.608x`; `Summary: 1 passed, 0 failed, 1 total.` (atol=1e-2, rtol=1e-2) | pass | `python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/triton_fused_moe_002.py --warmup 5 --repeat 10 --full-traceback` |
| output dtype and shape unchanged | fp16 `[83,128]` on `gcu:0` | `out = torch.empty_like(hidden_states)` then cast back to `x.dtype` (fp16); shape `[T,H]` preserved | pass | candidate lines 123, 101 |
| top-2 semantics and renormalization match base | softmax + top-2 + renorm + fp16 cast match base routing | correctness PASS vs `base.py` confirms routing numerics match within tolerance | pass | correctness command above |
| int32 expert indexing preserved | no `tl.int64`, int32 offsets | `expert_id_scalar` stays int32; offsets `expert_id_scalar * TWO_I * H` etc. are int32 | pass | candidate lines 73, 76, 91 |
| selected GCU device and current stream preserved | no device/stream change | kernel-only fusion; no device or stream mutation | pass | candidate `fused_moe_v2` body (no stream/device calls) |

## Screening Evidence

Not run as a separate phase. Correctness passed on the first attempt, so the candidate proceeded directly to authoritative timing per the contract.

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | not-run | not-run | not-run | correctness passed; direct to authoritative timing |
| 2 | not-run | not-run | not-run | correctness passed; direct to authoritative timing |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `base.py (Model v0) vs candidate (ModelNew v1)` — harness benchmark requires v0 to define `Model`; the accepted reference `triton_fused_moe_001.py` defines `ModelNew` and cannot be a benchmark v0, so the wall comparison against the Round-1 reference is derived from the per-run candidate wall medians (see note below).
- reference_raw_samples_ms: `[0.498811, 0.571977, 0.470294, 0.531536, 0.547079]` (Round-1 reference `triton_fused_moe_001.py` v1 wall, 5 runs)
- candidate_raw_samples_ms: `[0.465872, 0.390131, 0.385763, 0.390289, 0.390325]` (candidate `triton_fused_moe_002.py` v1 wall, 5 runs)
- reference_median_ms: `0.531536`
- candidate_median_ms: `0.390289`
- improvement_pct: `26.57`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (0.531536 - 0.390289) / 0.531536 * 100 = 26.57%
```

Note on benchmark v0: the harness's `build_case` requires the v0 file to define a
`Model` class (`require_attr(v0_module, "Model", v0_path)`), and the v1 file to
define `ModelNew`. `triton_fused_moe_001.py` (Round-1 accepted reference) defines
only `ModelNew`, so it cannot be passed as `--v0_file` for a direct interleaved
wall comparison. The candidate is therefore benchmarked against `base.py` as v0
(`speedup=15.693x`, `v0=7.310893 ms, v1=0.465872 ms`), and the Round-1-vs-Round-2
wall comparison is computed from the independent per-run v1 wall medians of both
`ModelNew` candidates (both measured under identical `--warmup 50 --repeat 100`
regime). The Round-2 mechanism (launch count 8 -> 3) is confirmed directly by the
dual-scope profiler, which does place `reference_triton_fused_moe_001` and
`candidate_triton_fused_moe_002` in the same trace.

The unrounded improvement controls the 5% adoption threshold. Profiler time does
not replace this benchmark result. Only authoritative timing can yield `accepted` or `no-improvement`.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| runtime_launch_count_per_call | decrease from 8 to fewer launches | 8.0 (reference) -> 3.0 (candidate), a 62.5% reduction | pass | `summarize_trace.py ... --scope reference_triton_fused_moe_001` = `8.0`; `--scope candidate_triton_fused_moe_002` = `3.0` |
| runtime_launch_us_per_call | decrease in the GCU runtime-launch diagnostic | 81.73 us (reference) -> 30.24 us (candidate), a 63.0% reduction | pass | `runtime_launch_us_per_call`: reference `81.72662109375`, candidate `30.243955078125` |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: `fuse softmax, top-2, renormalize, and fp16 weight cast of routing into the per-token fused-MoE kernel so the kernel computes topk from raw router_logits in-place`
- expected_causal_chain: `["eager softmax, topk, renormalize, and routing cast kernels disappear", "runtime_launch_count_per_call decreases below 8", "host overhead per call decreases", "benchmark wall time decreases"]`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `unavailable: GCU trace exposes runtime launch events but no cat=kernel device durations`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels` (all `null` — device time unavailable)
- backend_runtime_fields: `runtime_launch_total_us`, `runtime_launch_us_per_call`, `runtime_launch_count_total`, `runtime_launch_count_per_call`, `runtime_launches`
- trace: `log/fused_moe_round_002_forward_50iter.pt.trace.json`

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations` before they are compared.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (001) | unavailable | unavailable | unavailable | unavailable | 0.531536 | unavailable |
| candidate (002) | unavailable | unavailable | unavailable | unavailable | 0.390289 | unavailable |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)  # not computable: device time unavailable on GCU exporter
```

### Accepted Reference Top Kernels (triton_fused_moe_001, runtime launches)

| Launch kind | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| topsLaunchKernel | 300 | 6.0 | 3049.11 | 60.98 |
| topsModuleLaunchKernel | 50 | 1.0 | 558.29 | 11.17 |
| topsLaunchCooperativeKernel | 50 | 1.0 | 478.93 | 9.58 |

(reference `runtime_launch_count_per_call` = 8.0; `runtime_launch_us_per_call` = 81.73)

### Candidate Top Kernels (triton_fused_moe_002, runtime launches)

| Launch kind | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| topsLaunchKernel | 100 | 2.0 | 995.29 | 19.91 |
| topsModuleLaunchKernel | 50 | 1.0 | 516.91 | 10.34 |

(candidate `runtime_launch_count_per_call` = 3.0; `runtime_launch_us_per_call` = 30.24)

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | `e5d0058d6fb6f23f94e2623ae374d2776c4b2c6d4eb235b2c6c75524fb44eb73` | `e5d0058d6fb6f23f94e2623ae374d2776c4b2c6d4eb235b2c6c75524fb44eb73` | correctness PASS; launch count 8 -> 3; wall 0.531536 -> 0.390289 ms; accepted |

No repair was required.

## evidence_for_next_round

- The H-002 mechanism is confirmed: in-kernel routing fusion reduced `runtime_launch_count_per_call` from 8 to 3 (62.5% reduction) and `runtime_launch_us_per_call` from 81.73 us to 30.24 us.
- Candidate wall median improved from 0.531536 ms (Round-1 reference) to 0.390289 ms (26.57% improvement, ~1.36x), exceeding the 5% threshold.
- The candidate now emits exactly 3 runtime launches/call: 2 `topsLaunchKernel` + 1 `topsModuleLaunchKernel`. The `topsLaunchCooperativeKernel` present in Round 1 has disappeared, confirming the eager `torch.topk` routing launch was absorbed into the fused kernel.
- Remaining launch surface: 3 launches/call are still present — one `topsModuleLaunchKernel` (the single fused Triton kernel) plus 2 `topsLaunchKernel` (the host-side fp16 weight casts `w1.to(dtype)` / `w2.to(dtype)`, which still run as eager torch ops before launch). The weight-cast eager ops are the remaining host-bound surface.
- Device time remains unavailable on the GCU exporter (`device_time_available: false`); runtime-launch fields remain the only profiler observables.
- Harness structural note: the benchmark `--v0_file` requires a `Model` class, so `ModelNew`-only candidates (001/002) cannot be directly interleaved as v0; Round-1-vs-Round-2 wall comparison is derived from independent per-run v1 wall medians under identical regime.

## Stop Recommendation

- recommendation: `continue`
- evidence: H-002 is confirmed (launch count 8 -> 3, wall 26.57% improvement). The remaining host-side fp16 weight casts (2 `topsLaunchKernel` eager ops) present a further, smaller optimization surface. No stop boundary is reached; this is a record-then-decide point for the Orchestrator.

## Exact Reproduction Commands

```bash
cd /root/kernelswift/.worktrees/fused-moe-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/triton_fused_moe_002.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift/.worktrees/fused-moe-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/triton_fused_moe_002.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/fused-moe-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/triton_fused_moe_002.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-reference-file kernels/track1-triton/fused_moe/s60/triton_fused_moe_001.py --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/fused_moe/s60/log/fused_moe_round_002_forward_50iter.pt.trace.json
```
