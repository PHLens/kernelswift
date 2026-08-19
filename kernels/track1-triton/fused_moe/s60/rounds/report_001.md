# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_fused_moe_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `f03ecaf64e86d3ae01303e5c5ae5390dde32ee2f60ebf3244fd34f9f6aa01c7c`
- Candidate SHA256: `444eb2fb3e14c48359b27137d11b7f57da22211ad4034e6e56f05af5b4561124`
- Accepted reference SHA256: `b939d91f0f85e299a1102bfceb00da0e38c484a81c8d23ec78777fce68a3ee6f`
- Base SHA256: `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `d8f8f6bf8965ab279eb59215a7cc0c6f24f7dd0ad5ea7d8436162336955af6c3`
- verification_tier: authoritative
- screening_pairs: `not-run (correctness gate failed on first submission; single same-round Coder repair, then authoritative timing)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `PASS accuracy; v0=5.072005 ms, v1=0.494510 ms, speedup=10.257x`; `Summary: 1 passed, 0 failed, 1 total.` | pass | `python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/triton_fused_moe_001.py --warmup 5 --repeat 10 --full-traceback` |
| output dtype and shape unchanged | fp16 `[83,128]` on `gcu:0` | `out = torch.empty_like(hidden_states)` then cast back to `x.dtype` (fp16); shape `[T,H]` preserved | pass | candidate `triton_fused_moe_001.py` line 100, 73 |
| state_dict keys unchanged | exactly `{w1, w2}` fp32 `nn.Parameter` | `ModelNew` exposes only `w1`/`w2` `nn.Parameter`; `load_state_dict` synchronization passed via `compare_case` correctness PASS | pass | candidate lines 132-139 |

## Screening Evidence

Not run as a separate phase. The first submission failed the correctness gate (compile-time `slice` indexing defect); a single same-round Coder repair fixed it, and the repaired candidate proceeded directly to authoritative timing per the contract.

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | not-run | not-run | not-run | initial correctness gate failed (`CompilationError`) |
| 2 | not-run | not-run | not-run | single repair then authoritative timing |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`
- reference_raw_samples_ms: `(median) 5.262259`
- candidate_raw_samples_ms: `(median) 0.498811`
- reference_median_ms: `5.262259`
- candidate_median_ms: `0.498811`
- improvement_pct: `90.52`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (5.262259 - 0.498811) / 5.262259 * 100 = 90.52%
```

The unrounded improvement controls the 5% adoption threshold. Profiler time does
not replace this benchmark result. Only authoritative timing can yield `accepted` or `no-improvement`.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| runtime_launch_count_per_call | decrease | 147.0 (baseline) -> 8.0 (candidate), a 94.6% reduction | pass | `summarize_trace.py ... --scope baseline_base` = `147.0`; `--scope candidate_triton_fused_moe_001` = `8.0` |
| runtime_launch_us_per_call | decrease | 1522.31 us (baseline) -> 82.95 us (candidate), a 94.6% reduction | pass | `runtime_launch_us_per_call`: baseline `1522.311865234375`, candidate `82.9491015625` |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the 8-expert loop, per-expert mask/gather/scatter, double GEMM, SiLU, and weighted reduction into a single per-token Triton kernel (grid=(T,)); routing (softmax/topk/renorm/cast) stays eager this round`
- expected_causal_chain: `["per-expert Python loop and scatter/gather eager ops disappear", "runtime launch count per call drops from 147 to a few routing kernels plus one fused kernel", "wall time decreases"]`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `unavailable: GCU trace exposes runtime launch events but no cat=kernel device durations`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels` (all `null` — device time unavailable)
- backend_runtime_fields: `runtime_launch_total_us`, `runtime_launch_us_per_call`, `runtime_launch_count_total`, `runtime_launch_count_per_call`, `runtime_launches`

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations` before they are compared.
Profiler evidence is required for baseline and accepted candidates, and is not
run for `screened-out` candidates.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference | unavailable | unavailable | unavailable | unavailable | 5.262259 | unavailable |
| candidate | unavailable | unavailable | unavailable | unavailable | 0.498811 | unavailable |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)  # not computable: device time unavailable on GCU exporter
```

### Accepted Reference Top Kernels

Device kernel durations unavailable (GCU runtime-launch-only exporter). Normalized
runtime launch evidence instead:

| Launch kind | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| topsLaunchKernel | 3700 | 74.0 | 39223.43 | 784.47 |
| topsLaunchCooperativeKernel | 3650 | 73.0 | 36892.16 | 737.84 |

(reference `runtime_launch_count_per_call` = 147.0; `runtime_launch_us_per_call` = 1522.31)

### Candidate Top Kernels

Device kernel durations unavailable (GCU runtime-launch-only exporter). Normalized
runtime launch evidence instead:

| Launch kind | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| topsLaunchKernel | 300 | 6.0 | 3160.40 | 63.21 |
| topsModuleLaunchKernel | 50 | 1.0 | 504.26 | 10.09 |
| topsLaunchCooperativeKernel | 50 | 1.0 | 482.80 | 9.66 |

(candidate `runtime_launch_count_per_call` = 8.0; `runtime_launch_us_per_call` = 82.95)

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | `9cb247591e84113625e3a63e34d6c40a8e03241620576c3612058addeb9fe45b` | `9cb247591e84113625e3a63e34d6c40a8e03241620576c3612058addeb9fe45b` | correctness FAIL — `unsupported tensor index: slice` on `gate_up[:I]` / `gate_up[I:]` |
| 2 | single same-round Coder repair (slice split -> two independent `[I]` GEMMs) | `9cb247591e84113625e3a63e34d6c40a8e03241620576c3612058addeb9fe45b` | `444eb2fb3e14c48359b27137d11b7f57da22211ad4034e6e56f05af5b4561124` | correctness PASS; speedup 10.550x; accepted |

At most one Verifier-to-Coder repair is allowed in the same round.

## evidence_for_next_round

- The H-001 mechanism is confirmed: per-token Triton fusion reduced `runtime_launch_count_per_call` from 147 to 8 (94.6% reduction), and `runtime_launch_us_per_call` from 1522.31 us to 82.95 us.
- Wall median improved from 5.262259 ms to 0.498811 ms (10.550x speedup, 90.52% improvement), far exceeding the 5% adoption threshold.
- The causal chain (eager expert-loop/dispatch ops disappear -> launch count drops -> wall time decreases) is fully observed.
- Remaining launch overhead: candidate still emits 8 runtime launches/call (6 `topsLaunchKernel` + 1 `topsModuleLaunchKernel` + 1 `topsLaunchCooperativeKernel`), dominated by the still-eager routing (softmax/topk/renorm/cast) and weight fp16 casts — a potential next-round target.
- Device time remains unavailable on the GCU exporter (`device_time_available: false`); normalized runtime-launch fields remain the only profiler observables.
- `tl.split` was not needed; splitting `gate_up` into two independent `[I]` GEMMs (gate/up projections loaded separately from `w1`) is semantically equivalent and uses only confirmed primitives (`tl.load`/`tl.sum`).

## Stop Recommendation

- recommendation: `continue`
- evidence: H-001 is confirmed with a 10.550x speedup (90.52% improvement), well above target. Remaining eager routing (8 launches/call) still presents an optimization surface, but this is a record-then-decide point for the Orchestrator; the accepted candidate is a strong first-step result.

## Exact Reproduction Commands

```bash
cd /root/kernelswift/.worktrees/fused-moe-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/triton_fused_moe_001.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift/.worktrees/fused-moe-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/triton_fused_moe_001.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/fused-moe-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/triton_fused_moe_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/fused_moe/s60/log/fused_moe_round_001_forward_50iter.pt.trace.json
```
