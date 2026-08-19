# Report 002

Result: accepted

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_fused_moe_002.py`
- Accepted reference: `triton_fused_moe_001.py` (round 001 canonical)
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `2d44dd2c808bf27c20cdd4d6ca0aa0ecba422080394462f6d176ccc2c5a146a6`
- Candidate SHA256: `6ac1f44b111285f5bf746110c51f6486868b12beb2deae3390663d74233f8ae5`
- Accepted reference SHA256: `8424c7a01bc1d293c2b0ef509dd895950112cfb71dedd145053b4ac3f7eb9ad6`
- Base SHA256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1), 16 SM, 16 GiB)
- Measurement fingerprint: `5c2a51ab3f3ebaab1123b9fa534d4e4b940f3334f80fac00252df780d3900150`
- verification_tier: `authoritative`
- screening_pairs: `not-run` (correctness passed; proceeded directly to authoritative timing)

All source hashes match the frozen values; candidate and decision hashes match the
Coder-dispatched values.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=3.223496 ms, v1=0.506451 ms, speedup=6.365x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| output dtype and shape unchanged | `out[83,128]` fp16 | Independent probe: `shape torch.Size([83,128]) torch.float16` both sides | pass | independent probe |
| torch.topk tie order preserved | descending value, ascending index on equal scores | Independent probe with tie-constructed router: candidate uses identical `torch.topk`; `allclose=True`, `max_abs_diff=1.53e-05` | pass | independent probe |
| GEMM contraction dims + SiLU preserved | gate/up contraction 128, down contraction 64, SiLU activation | `allclose=True` at `1.53e-05` max abs diff (`tl.dot` fp16 in / fp32 accumulate) | pass | independent probe |
| weighted-sum reduction semantics | per-token sum of top-2 weighted expert outputs | `allclose=True`, `mean_abs_diff=1.29e-06` (atomic-add fp32 accumulate then fp16 cast) | pass | independent probe |
| input not mutated | inputs read-only | forward reads `hidden_states`/`router_logits`/`w1`/`w2`; writes only fresh `out` | pass | candidate source lines 95-127 |
| public contract | `ModelNew(...)` + `get_init_inputs` + `get_inputs` | Loaded, constructed, moved, executed through AST loader without error | pass | correctness return code `0` |

Correctness PASS and all guardrails pass. No local repair was required.

## Screening Evidence

Not applicable. Correctness passed, so the candidate proceeded directly to
authoritative timing (three interleaved pairs), skipping screening.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[2.443674, 2.464602, 2.474194]`
- candidate_raw_samples_ms: `[0.493893, 0.492143, 0.493474]`
- reference_median_ms: `2.464602`
- candidate_median_ms: `0.493474`
- improvement_pct: `79.98`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (2.464602 - 0.493474) / 2.464602 * 100 = 79.98
```

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `2.443674` | `0.493893` | `0` |
| 2 | `2.464602` | `0.492143` | `0` |
| 3 | `2.474194` | `0.493474` | `0` |

The reference is `triton_fused_moe_001.py` (round 001 canonical), presented to the
harness as v0 via a temporary class-rename wrapper (see Exact Reproduction Commands).
The unrounded median improvement `79.98%` far exceeds the `5%` threshold and the
`12.0%` central expectation.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease | `54.0` → `9.82` per call (81.8% decrease) | pass | Level 1 profiler (time-interval separation) |
| device_us_per_call | decrease | `500.65` → `140.84` us/call (71.9% decrease) | pass | Level 1 profiler |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: fuse the per-expert GEMM loop + argsort bucketing into a single `tl.dot` Triton kernel
- expected_causal_chain: argsort/per-expert-loop kernels disappear → device_us decreases → wall time decreases
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

Both declared mechanism observables are confirmed: kernel count fell from 54.0 to 9.82
per call, device time fell from 500.65 to 140.84 us/call, and wall time improved 79.98%.
The `tl.dot` capability risk did not materialize — the fp16 contraction-128/64 GEMMs
lowered correctly and the fusion collapsed ~44 kernels into one.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `reference_triton_fused_moe_001`, `candidate_triton_fused_moe_002`
- raw trace: `log/round_002_forward_50iter.pt.trace.json`, SHA256 `72e292d9e23e872d85d2dee80a69f8b33302f29168566b0f1bd8d5795c499ebb`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `reference_triton_fused_moe_001` | `25032.583` | `500.652` | `2700` | `54.0` | `2.464602` | `0.2031` |
| `candidate_triton_fused_moe_002` | `7042.0` | `140.840` | `491` | `9.82` | `0.493474` | `0.2854` |

```text
device_ratio(reference) = 500.652 / (2.464602 * 1000) ≈ 0.2031
device_ratio(candidate) = 140.840 / (0.493474 * 1000) ≈ 0.2854
```

### Measurement note — scope attribution (both scopes)

Both scopes contain Triton kernels (`_weighted_reduce_kernel` in the reference,
`_fused_moe_expert_kernel` in the candidate), and on this CoreX profiler each Triton
launch emits two overlapping CPU-side `record_function` interval events. Consequently
`summarize_trace.py` raises `overlapping scope events` for BOTH scopes. The
`cat=kernel` events themselves are correctly time-stamped on a shared timeline, and the
two scopes are strictly sequential (the reference's last `_weighted_reduce_kernel` ends
at ts `4168904882722`, the candidate's first `_fused_moe_expert_kernel` starts at ts
`4168904882818`, a ~95 us gap). Kernel totals were therefore attributed by
time-interval separation over the `cat=kernel` events, the same rule
`summarize_trace.py` applies, with an explicitly verified non-overlapping interval.

### Reference Top Kernels (reference_triton_fused_moe_001 scope)

Same structure as round 001 candidate (report_001.md): argsort-dominant.

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `radixSortKVInPlace<-2,-1,32,32,long,long,...>` (argsort) | `50` | `1.0` | `5368` | `107.36` |
| `Gemm_tcu_mr_kernel::gemm_tcu_h<128u,128u,...>` (gate/up) | `400` | `8.0` | `3053` | `61.06` |
| `Gemm_tcu_mr_kernel::gemm_tcu_h<32u,64u,...>` (down) | `400` | `8.0` | `2879` | `57.58` |
| `elementwise_kernel_v3<BinaryFunctor<Half,...>>` (chunk) | `400` | `8.0` | `2402` | `48.05` |
| `elementwise_kernel_v3<silu_kernel>` (SiLU) | `400` | `8.0` | `2009` | `40.17` |
| `sbtopk::gatherTopK<float,unsigned int,2,false>` | `50` | `1.0` | `1128` | `22.57` |
| `index_elementwise_kernel<...>` (argsort gather) | `50` | `1.0` | `996` | `19.92` |
| `bitonicSortKVInPlace<2,-1,16,16,...>` | `50` | `1.0` | `926` | `18.52` |
| `_weighted_reduce_kernel` (Triton weighted sum) | `50` | `1.0` | `275` | `5.49` |

### Candidate Top Kernels (candidate_triton_fused_moe_002 scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `_fused_moe_expert_kernel` (Triton, all experts) | `50` | `1.0` | `2789.8` | `55.80` |
| `sbtopk::gatherTopK<float,unsigned int,2,false>` | `49` | `0.98` | `1079.7` | `21.59` |
| `bitonicSortKVInPlace<2,-1,16,16,...>` | `49` | `0.98` | `892.3` | `17.85` |
| `vectorized_elementwise_kernel<float16_copy_kernel>` (w1/w2 cast) | `147` | `2.94` | `777.8` | `15.56` |
| `reduce_kernel<1024,1,ReduceOp<float,sum_functor>>` (renormalize sum) | `49` | `0.98` | `687.0` | `13.74` |
| `elementwise_kernel<512,4,DivFunctor<float>>` (renormalize div) | `49` | `0.98` | `335.5` | `6.71` |
| `softmax_warp_forward<float,float,float,3,...>` | `49` | `0.98` | `249.8` | `5.00` |
| `vectorized_elementwise_kernel<FillFunctor<Half>>` (out zeros) | `49` | `0.98` | `230.1` | `4.60` |

### Fusion Outcome Observation

The intervention succeeded exactly as the Evaluation Contract predicted:

- **Single fused Triton kernel**: the per-expert GEMM loop (8x gate/up GEMM + 8x chunk
  + 8x SiLU + 8x mul + 8x down GEMM ≈ 40 kernels) and the `torch.argsort` bucketing
  chain (`radixSortKVInPlace` + argsort gather + bincount/cumsum ≈ 3 kernels) collapsed
  into a single `_fused_moe_expert_kernel` (1 launch, grid=8 programs, one per expert).
  `tl.dot` with fp16 inputs and contraction dims 128/64 lowered correctly.
- **Eliminated**: `radixSortKVInPlace` (107 us/call), both `gemm_tcu_h` kernels
  (61+58 us/call), chunk elementwise (48 us/call), SiLU elementwise (40 us/call), and
  the argsort gather (20 us/call) — all replaced by the fused kernel.
- **Remaining**: `torch.topk` (`gatherTopK` + `bitonicSortKVInPlace`, preserved, 0.98/call each), the routing softmax, the renormalize sum/div, the `w1`/`w2` fp16 cast
  (`float16_copy_kernel` 2.94/call), and the `out` zero-init. Total 9.82 kernels/call.
- **Net**: kernel count `54.0 → 9.82/call` (−81.8%), device time `500.65 → 140.84
  us/call` (−71.9%), wall time `2.464602 → 0.493474 ms` (+79.98%).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `6ac1f44b111285f5bf746110c51f6486868b12beb2deae3390663d74233f8ae5` | same | correctness and guardrails passed; timing improved 79.98%; profiler confirmed mechanism |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Round 002 accepted: candidate `triton_fused_moe_002.py` wall median `0.493474 ms`, +79.98% vs round-001 canonical `2.464602 ms`. The single-kernel `tl.dot` fusion is confirmed: kernel count 54.0 → 9.82/call, device time 500.65 → 140.84 us/call.
- The `tl.dot` fp16 contraction-128/64 GEMM capability is now PROVEN on this BI150 profile (max_rel_err ~1.5e-05 at the output, far below the 1e-2 tolerance). This updates the target-profile understanding: `tl.dot` works for these skinny MoE GEMM shapes, not just the `(32,32)@(32,32)` probe.
- Remaining bottleneck: the single `_fused_moe_expert_kernel` is now the largest kernel at 55.80 us/call (1 launch, grid=8). The `torch.topk` (gatherTopK 21.59 + bitonicSort 17.85 us/call) and the routing softmax (5.0 us/call) are the next device-time contributors, but the operator is now strongly host/launch-bound (device_ratio 0.2854, wall 0.493 ms vs device 140.84 us — ~71% is host/launch/other).
- The `w1`/`w2` `.to(dtype)` fp16 cast (`float16_copy_kernel` 2.94/call, 15.56 us/call) and `out` zero-init (`FillFunctor` 4.60 us/call) are small remaining launch overheads that could be eliminated (e.g. keep weights pre-cast, or use the fused kernel to zero-init via the atomic-add reduction).
- device_ratio rose slightly (0.2031 → 0.2854) as wall time dropped faster than device time; the operator is approaching a host/launch floor. Further wall-time gains are bounded by the harness-fixed seed/synchronization and the remaining 9.82 kernels/call.
- Profiler note (recurring): BOTH Triton-containing scopes now emit overlapping CPU-side `record_function` events; summarize requires time-interval separation over `cat=kernel` events (reference and candidate intervals are strictly sequential).

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 002 accepted with +79.98% wall improvement, far exceeding the 12% expectation. The single-kernel `tl.dot` fusion is confirmed and the `tl.dot` capability risk resolved. Remaining gains are bounded (device_ratio 0.2854, approaching host/launch floor; topk and routing softmax are still torch kernels but are correctness-critical to preserve). No target-reached condition is set; Orchestrator may judge whether the diminishing-return boundary is near, but no explicit stop criterion has been met.

Orchestrator owns the terminal transition and canonical pointer updates.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/fused_moe/base.py kernels/track1-triton/fused_moe/bi150/triton_fused_moe_001.py kernels/track1-triton/fused_moe/bi150/triton_fused_moe_002.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/triton_fused_moe_002.py --warmup 50 --repeat 100 --full-traceback
```

Authoritative timing wrapper (canonical 001 exposes `ModelNew`; rename to `Model` for the v0 slot; wrapper SHA `f3cb21a574bb59f44815acdeeb45cc916b49337cdde82fdd8500fb44f40f3f46`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sed 's/^class ModelNew/class Model/' kernels/track1-triton/fused_moe/bi150/triton_fused_moe_001.py > /tmp/fm_canonical_model_002.py
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/fm_canonical_model_002.py --v1_file kernels/track1-triton/fused_moe/bi150/triton_fused_moe_002.py --warmup 50 --repeat 100
```

Targeted profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/triton_fused_moe_002.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/fused_moe/bi150/triton_fused_moe_001.py --profile-output kernels/track1-triton/fused_moe/bi150/log/round_002_forward_50iter.pt.trace.json
```

Scope summary (both scopes require time-interval separation over `cat=kernel` events; `summarize_trace.py` rejects the overlapping Triton `record_function` scopes):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 -c "import json; d=json.load(open('kernels/track1-triton/fused_moe/bi150/log/round_002_forward_50iter.pt.trace.json')); evs=d['traceEvents']; wr=[k for k in evs if k.get('cat')=='kernel' and k.get('name')=='_weighted_reduce_kernel']; fmk=[k for k in evs if k.get('cat')=='kernel' and k.get('name')=='_fused_moe_expert_kernel']; rs=min(k['ts'] for k in evs if k.get('cat')=='kernel'); re=max(k['ts']+k['dur'] for k in wr); cs=min(k['ts'] for k in fmk); ce=max(k['ts']+k['dur'] for k in evs if k.get('cat')=='kernel'); kr=[k for k in evs if k.get('cat')=='kernel' and k['ts']>=rs and k['ts']+k['dur']<=re]; kc=[k for k in evs if k.get('cat')=='kernel' and k['ts']>=cs and k['ts']+k['dur']<=ce]; print('ref', len(kr), len(kr)/50, sum(k['dur'] for k in kr)/50); print('cand', len(kc), len(kc)/50, sum(k['dur'] for k in kc)/50)"
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity |
| correctness (base vs 002) | `0` | Correctness table |
| independent numeric probe (tie + GEMM + SiLU + reduction) | `0` | allclose True, 1.53e-05 |
| wrapper generation | `0` | `/tmp/fm_canonical_model_002.py`, SHA recorded |
| timing pair 1 | `0` | Interleaved Wall Timing |
| timing pair 2 | `0` | Interleaved Wall Timing |
| timing pair 3 | `0` | Interleaved Wall Timing |
| targeted profiler | `0` | `log/round_002_forward_50iter.pt.trace.json` |
| scope attribution (both) | fallback | time-interval separation |
