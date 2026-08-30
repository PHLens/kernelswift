# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_fused_moe_001.py`
- Accepted reference: `baseline_adapter.py` (Phase 0 canonical baseline)
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `0745c37ddc4a5e27811d9ad20845d8b168017033b3b61f59f253d2129d9f7681`
- Candidate SHA256: `8424c7a01bc1d293c2b0ef509dd895950112cfb71dedd145053b4ac3f7eb9ad6`
- Accepted reference SHA256: `8e5c70232e541a02d83343216376ece9127a1c3e6ea6af77dc77a2723783facf`
- Base SHA256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1), 16 SM, 16 GiB)
- Measurement fingerprint: `5c2a51ab3f3ebaab1123b9fa534d4e4b940f3334f80fac00252df780d3900150`
- verification_tier: `authoritative`
- screening_pairs: `not-run` (correctness passed; proceeded directly to authoritative timing)

All source hashes match the frozen `project.md` values; the candidate hash matches the
Coder-dispatched value. The decision hash is recorded above and verified locally.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=3.214486 ms, v1=2.474534 ms, speedup=1.299x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| output dtype and shape unchanged | `out[83,128]` fp16 | Independent probe: `shape base torch.Size([83,128]) torch.float16`, candidate identical | pass | independent probe (base vs ModelNew) |
| torch.topk tie order preserved | descending value, ascending index on equal scores | Independent probe with tie-constructed router (`router[0,2]==router[0,5]`, `router[1,0]==router[1,1]==router[1,3]`): candidate uses identical `torch.topk`; `allclose=True`, `max_abs_diff=7.629e-06` | pass | independent probe |
| routing and GEMM contraction semantics preserved | fp32 softmax, gate/up contraction 128, down contraction 64, SiLU | `allclose=True` at `7.63e-06` max abs diff (far below `1e-2` tol) | pass | independent probe |
| input not mutated | `hidden_states`, `router_logits`, `w1`, `w2` read-only | forward does not write inputs; output written to fresh `torch.empty` | pass | candidate source lines 62-120 |
| public contract | `ModelNew(num_experts, top_k, hidden_size, intermediate_size, renormalize=True)` + `get_init_inputs` + `get_inputs` | Loaded, constructed, moved, executed through AST loader without error | pass | correctness return code `0` |

Correctness PASS and all guardrails pass. No local repair was required.

## Screening Evidence

Not applicable. Correctness passed, so the candidate proceeded directly to
authoritative timing (three interleaved pairs), skipping screening.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[3.239021, 3.167858, 3.158865]`
- candidate_raw_samples_ms: `[2.525259, 2.450700, 2.488731]`
- reference_median_ms: `3.167858`
- candidate_median_ms: `2.488731`
- improvement_pct: `21.44`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (3.167858 - 2.488731) / 3.167858 * 100 = 21.44
```

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `3.239021` | `2.525259` | `0` |
| 2 | `3.167858` | `2.450700` | `0` |
| 3 | `3.158865` | `2.488731` | `0` |

The reference is `baseline_adapter.py` (canonical Phase 0 baseline), presented to the
harness as v0 via a temporary class-rename wrapper (see Exact Reproduction Commands).
The unrounded median improvement `21.44%` exceeds the `5%` adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease | `123.9` → `54.1` per call (56.3% decrease) | pass | Level 1 profiler summary (time-interval separation) |
| device_us_per_call | decrease | `968.16` → `504.31` us/call (47.9% decrease) | pass | Level 1 profiler summary |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: fuse the per-expert Python loop's non-GEMM dispatch (mask/gather/scatter/reduce/chunk) into fewer kernels
- expected_causal_chain: kernel count drops → dispatch overhead disappears → device_us decreases → wall time decreases
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

Both declared mechanism observables are confirmed: kernel count fell from 123.9 to
54.1 per call, and device time per call fell from 968.16 to 504.31 us, and the wall
time improved 21.44% (exceeding the 5% threshold). The causal chain is fully observed.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `reference_baseline_adapter`, `candidate_triton_fused_moe_001`
- raw trace: `log/round_001_forward_50iter.pt.trace.json`, SHA256 `574a83d0103f7f9e91cb51aec02f70fbff733230d7431dbe1a545eeab434a53d`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `reference_baseline_adapter` | `48433.856` | `968.677` | `6195` | `123.9` | `3.167858` | `0.3058` |
| `candidate_triton_fused_moe_001` | `25215.624` | `504.312` | `2705` | `54.1` | `2.488731` | `0.2026` |

```text
device_ratio(reference) = 968.677 / (3.167858 * 1000) ≈ 0.3058
device_ratio(candidate) = 504.312 / (2.488731 * 1000) ≈ 0.2026
```

### Measurement note — candidate scope attribution

The reference scope summarized cleanly through `summarize_trace.py`
(`--scope reference_baseline_adapter`), returning `968.677 us/call` and `123.9
kernels/call` — identical to the Phase 0 baseline.

The candidate scope (`candidate_triton_fused_moe_001`) could NOT be summarized by
`summarize_trace.py`: the tool raised `overlapping scope events:
candidate_triton_fused_moe_001`. The candidate's `record_function` scope produced two
overlapping CPU-side interval events (ts `4167826822747`/dur `377248` and ts
`4167826831044`/dur `369094`), an artifact of the Triton `_weighted_reduce_kernel`
launch on the CoreX profiler (the first interval is inflated by the warmup/JIT
boundary and overlaps the second). The `cat=kernel` events themselves are correctly
time-stamped on the shared timeline.

Because the reference interval `[4167826316672, 4167826822715.7]` and the candidate
interval `[4167826822747.1, 4167827200138.9]` are strictly sequential and non-
overlapping, the candidate's kernel totals were attributed by time-interval
separation over the `cat=kernel` events (2705 kernels, 25215.624 us device total).
This is the same attribution rule `summarize_trace.py` applies (kernel events fully
contained within the scope interval), applied with an explicitly-verified
non-overlapping interval. The result is reliable and reproducible; the only deviation
is that the candidate CPU-side `record_function` scope was non-unambiguous and the
interval was bounded by the kernel-event extremes instead.

### Reference Top Kernels (reference_baseline_adapter scope)

Identical to Phase 0 baseline (`report_000.md`). Summary of the dispatch-dominant
top kernels:

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `index_elementwise_kernel<...index_kernel_impl...>` (gather) | `400` | `8.0` | `6366.8` | `127.34` |
| `index_elementwise_kernel<...index_put_kernel_impl...>` (scatter) | `399` | `7.98` | `6395.9` | `127.92` |
| `cub::DeviceSelectSweepKernel` (mask select) | `799` | `15.98` | `6299.1` | `125.98` |
| `reduce_kernel<1024,1,ReduceOp<bool,or_kernel>>` (mask.any) | `400` | `8.0` | `4332.0` | `86.64` |
| `cub::DeviceReduceSingleTileKernel` (mask.any reduce) | `800` | `16.0` | `4067.6` | `81.35` |
| `Gemm_tcu_mr_kernel::gemm_tcu_h<128u,128u,...>` (gate/up) | `400` | `8.0` | `3058.7` | `61.17` |
| `Gemm_tcu_mr_kernel::gemm_tcu_h<32u,64u,...>` (down) | `400` | `8.0` | `2891.1` | `57.82` |
| `cub::DeviceCompactInitKernel` | `799` | `15.98` | `2823.4` | `56.47` |
| `elementwise_kernel_v3<silu_kernel>` (SiLU) | `400` | `8.0` | `2005.0` | `40.10` |
| `sbtopk::gatherTopK<float,unsigned int,2,false>` | `50` | `1.0` | `1127.4` | `22.55` |
| `bitonicSortKVInPlace<2,-1,16,16,...>` | `50` | `1.0` | `932.5` | `18.65` |

### Candidate Top Kernels (candidate_triton_fused_moe_001 scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `radixSortKVInPlace<-2,-1,32,32,long,long,unsigned int>` (argsort) | `50` | `1.0` | `5368.1` | `107.36` |
| `Gemm_tcu_mr_kernel::gemm_tcu_h<128u,128u,...>` (gate/up) | `400` | `8.0` | `3053.1` | `61.06` |
| `Gemm_tcu_mr_kernel::gemm_tcu_h<32u,64u,...>` (down) | `400` | `8.0` | `2878.9` | `57.58` |
| `elementwise_kernel_v3<BinaryFunctor<Half,...>>` (chunk) | `400` | `8.0` | `2402.3` | `48.05` |
| `elementwise_kernel_v3<silu_kernel>` (SiLU) | `400` | `8.0` | `2008.7` | `40.17` |
| `sbtopk::gatherTopK<float,unsigned int,2,false>` | `50` | `1.0` | `1128.5` | `22.57` |
| `index_elementwise_kernel<...index_kernel_impl...>` (argsort gather) | `50` | `1.0` | `995.9` | `19.92` |
| `bitonicSortKVInPlace<2,-1,16,16,...>` | `50` | `1.0` | `926.1` | `18.52` |
| `reduce_kernel<1024,1,ReduceOp<long,...>>` (bincount/cumsum) | `50` | `1.0` | `625.3` | `12.51` |
| `_weighted_reduce_kernel` (Triton weighted sum) | `50` | `1.0` | `274.6` | `5.49` |

### Fusion Outcome Observation

The intervention succeeded exactly as the Evaluation Contract predicted:

- **Eliminated per-expert dispatch kernels** (the ~263 us/call overhead): `cub::DeviceSelectSweepKernel` (15.98/call), `cub::DeviceReduceSingleTileKernel` (16/call), `cub::DeviceCompactInitKernel` (15.98/call), `index_elementwise` gather (8/call) + scatter (7.98/call), and `reduce_kernel<or_kernel>` (8/call) are all gone from the candidate scope. The per-expert `flat_ids == e` boolean-mask selection was replaced by a single `torch.argsort` bucketing (`radixSortKVInPlace`, 1/call) plus `bincount`/`cumsum` offsets.
- **GEMMs unchanged**: both `gemm_tcu_h` kernels remain at 8/call (~61 and ~58 us/call), confirming `torch.topk` and the TCU GEMMs were preserved as required.
- **Weighted reduction fused**: the scatter + scale + sum (`index_put` scatter, `MulFunctor` scale, `reduce_kernel` sum) collapsed into one Triton `_weighted_reduce_kernel` at 1/call, 5.49 us/call.
- **Net**: kernel count `123.9 → 54.1/call` (−56.3%), device time `968.16 → 504.31 us/call` (−47.9%), wall time `3.167858 → 2.488731 ms` (+21.44%).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `8424c7a01bc1d293c2b0ef509dd895950112cfb71dedd145053b4ac3f7eb9ad6` | same | correctness and guardrails passed; timing improved 21.44%; profiler confirmed mechanism |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Round 001 accepted: candidate `triton_fused_moe_001.py` wall median `2.488731 ms`, +21.44% vs baseline `3.167858 ms`. The kernel-fusion intervention (argsort bucketing + fused Triton weighted-reduce) is confirmed by the Level 1 profile: kernel count 123.9 → 54.1/call, device time 968.16 → 504.31 us/call.
- Remaining candidate bottleneck: `radixSortKVInPlace` (argsort) is now the single largest kernel at 107.36 us/call (1/call), followed by the two preserved GEMMs (~61 and ~58 us/call each) and the chunk elementwise (48.05 us/call). The sort was introduced by the argsort bucketing approach and is now the top dispatch cost.
- The two TCU GEMMs remain untouched at 8/call each; the down GEMM tile is still small (32x64x16). A round-002+ opportunity is to probe `tl.dot` for these GEMM shapes (decision_001 explicitly deferred this pending a matched local probe) and/or to eliminate the argsort via a per-token static `top_k` layout that avoids the global sort.
- `torch.topk` (bitonic sort, 1/call each for gatherTopK + bitonicSortKVInPlace) is still present and untouched; tie semantics remain inherited from `torch.topk`.
- device_ratio dropped to 0.2026, meaning the candidate is now even more host/launch bound (the remaining 54 kernels/call still carry launch overhead). Further fusion of the chunk/SiLU elementwise and the argsort is the natural next target.
- Profiler note for future rounds: the CoreX profiler emits overlapping CPU-side `record_function` scopes around Triton kernel launches; the candidate scope requires time-interval separation over `cat=kernel` events (reliable, as reference and candidate intervals are strictly sequential).

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 001 accepted with +21.44% wall improvement. The kernel-fusion mechanism is confirmed (kernel count −56.3%, device time −47.9%). The remaining bottleneck is the newly-introduced argsort (107 us/call) plus the preserved TCU GEMMs, indicating clear further optimization headroom (e.g. `tl.dot` GEMM fusion or sort elimination). No target-reached or budget-exhausted condition applies.

Orchestrator owns the terminal transition and canonical pointer updates.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/fused_moe/base.py kernels/track1-triton/fused_moe/bi150/baseline_adapter.py kernels/track1-triton/fused_moe/bi150/triton_fused_moe_001.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/triton_fused_moe_001.py --warmup 50 --repeat 100 --full-traceback
```

Authoritative timing wrapper (baseline_adapter exposes `ModelNew`; rename to `Model` for the v0 slot; wrapper SHA `dd7cb62d13f2637522fdaa5975a5a7818745efaa05d7ca3be2a2718089c3ecb3`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sed 's/^class ModelNew/class Model/' kernels/track1-triton/fused_moe/bi150/baseline_adapter.py > /tmp/fm_baseline_model_001.py
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/fm_baseline_model_001.py --v1_file kernels/track1-triton/fused_moe/bi150/triton_fused_moe_001.py --warmup 50 --repeat 100
```

Targeted profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/triton_fused_moe_001.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/fused_moe/bi150/baseline_adapter.py --profile-output kernels/track1-triton/fused_moe/bi150/log/round_001_forward_50iter.pt.trace.json
```

Reference scope summary (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/fused_moe/bi150/log/round_001_forward_50iter.pt.trace.json --iterations 50 --scope reference_baseline_adapter --wall-ms 3.167858
```

Candidate scope summary (time-interval separation; `summarize_trace.py` rejects the overlapping Triton record_function scope):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 -c "import json,collections; d=json.load(open('kernels/track1-triton/fused_moe/bi150/log/round_001_forward_50iter.pt.trace.json')); evs=d['traceEvents']; c=[e for e in evs if e.get('ph')=='X' and e.get('cat')!='kernel' and e.get('name')=='candidate_triton_fused_moe_001']; s=min(e['ts'] for e in c); t=max(e['ts']+e['dur'] for e in c); ks=[k for k in evs if k.get('cat')=='kernel' and k['ts']>=s and k['ts']+k['dur']<=t]; print(len(ks), len(ks)/50, sum(k['dur'] for k in ks), sum(k['dur'] for k in ks)/50)"
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity |
| correctness (base vs candidate) | `0` | Correctness table |
| independent numeric probe (tie + weighted sum) | `0` | allclose True, 7.63e-06 |
| wrapper generation | `0` | `/tmp/fm_baseline_model_001.py`, SHA recorded |
| timing pair 1 | `0` | Interleaved Wall Timing |
| timing pair 2 | `0` | Interleaved Wall Timing |
| timing pair 3 | `0` | Interleaved Wall Timing |
| targeted profiler | `0` | `log/round_001_forward_50iter.pt.trace.json` |
| summarize reference | `0` | 968.677 us/call, 123.9 kernels/call |
| candidate scope attribution | fallback | time-interval separation, 504.312 us/call, 54.1 kernels/call |
