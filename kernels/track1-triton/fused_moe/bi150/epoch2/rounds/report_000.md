# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `/root/CodeBuddy/20260818191200/kernelswift/kernels/track1-triton/fused_moe/bi150/epoch2/baseline_adapter.py`
- Accepted reference: `base.py for Phase 0` = `/root/CodeBuddy/20260818191200/kernelswift/kernels/track1-triton/fused_moe/base.py`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `752a25033b7629459c6eb128c60a4bdc3ab77b9c7cc97f5d3592bdff4cd45a47`
- Accepted reference SHA256: `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d` (3598 bytes)
- Base SHA256: `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d` (3598 bytes)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes)
- Runtime fingerprint: `project.md#runtime-fingerprint` — `triton 3.1.0 (/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton) / torch 2.7.1 / CoreX 4.4.0 nvcc V10.2.89 / Iluvatar BI-V150 capability major=7 minor=1 multi_processor_count=16 total_memory=17179869184` — rediscovered live, `target_profile_match: pass`
- Measurement fingerprint: `fe73bc58146d8c16f524be2a00fe99b31e1b9678bca6b3702f4284a3ac0a5bef`
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`

### Fingerprint confirmation

Live recompute over `base_bytes || NUL || harness_bytes || NUL || canonical_json` with
`sort_keys=True, separators=(',', ':')`:

```text
canonical_json = {"device":"cuda:0","dtype":"mixed(fp16-hidden/w1,w2; fp32-router)","profile_iterations":100,"profile_mode":"kernel","profile_warmup":20,"repeat":100,"shape":{"hidden_states":[83,128],"router_logits":[83,8],"w1":[8,128,128],"w2":[8,128,64]},"warmup":50}
recomputed     = fe73bc58146d8c16f524be2a00fe99b31e1b9678bca6b3702f4284a3ac0a5bef
project.md     = fe73bc58146d8c16f524be2a00fe99b31e1b9678bca6b3702f4284a3ac0a5bef
verdict        = MATCH
```

Positive control — two sibling fingerprints reproduced with the same algorithm and the
same harness bytes:

| Campaign | Expected | Recomputed | Verdict |
|---|---|---|---|
| flexattention @ bi150 epoch2 | `6dc07009177b649f7c2cad8f7be5e9aad74235bd9f50abfebc88bdb273e32af4` | `6dc07009177b649f7c2cad8f7be5e9aad74235bd9f50abfebc88bdb273e32af4` | reproduced |
| mm_encoder_attention @ bi150 epoch2 | `0c4c7d664c85e65d0580091ca5e3a77ff769a0d28f7e679f5bdf78fe5d0d966e` | `0c4c7d664c85e65d0580091ca5e3a77ff769a0d28f7e679f5bdf78fe5d0d966e` | reproduced |

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `PASS accuracy; v0=3.779870 ms, v1=3.379537 ms, speedup=1.118x` — `Summary: 1 passed, 0 failed, 1 total.` | pass | `log/round_000_correctness.txt` |
| output shape | `[83, 128]` | `Tensor(shape=(83,128))` | pass | inline probe, seed 42 |
| output dtype | `fp16` | `torch.float16` | pass | inline probe, seed 42 |
| output finiteness | all finite | `torch.isfinite(out).all() == True` | pass | inline probe, seed 42 |
| tolerance | `atol=rtol=1e-2` | `torch.allclose` satisfied | pass | correctness command |
| seed | `42` | `--seed 42` on every command | pass | all commands |
| base immutability | bytes unchanged | 3598 bytes, SHA `21e75853…` before and after | pass | `sha256sum` pre/post |
| Ast loader only | no base copy/edit | harness AST-loads base; no write | pass | harness `load_ks_module` |
| device | `cuda:0 (Iluvatar BI-V150)` | `torch.cuda.get_device_name(0) == "Iluvatar BI-V150"` | pass | runtime discovery |

Correctness gate passed before any timing. Conformance, correctness, and every declared
guardrail pass.

## Screening Evidence

Not run. Phase 0 baseline verification compares `base.py` against `baseline_adapter.py`
(adapter-of-base); the candidate is the accepted reference for subsequent rounds, so no
screen applies.

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | not-run | not-run | not-run | `not-applicable: Phase 0` |
| 2 | not-run | not-run | not-run | `not-applicable: Phase 0` |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` — three ordered pairs, each pair an
  independent `auto_bench.py` invocation timing v0 then v1 with byte-identical flags
- interpreter: `/usr/local/bin/python3` (python3.10), CoreX bootstrapped
  (`export COREX_VERSION=4.4.0; . /usr/local/corex/enable`) on every invocation
- device: `cuda:0`, default stream

| Pair | v0 (base.py) ms | v1 (baseline_adapter.py) ms | v1/v0 |
|---:|---:|---:|---:|
| 1 | 3.253012 | 3.245672 | 1.002x |
| 2 | 3.271220 | 3.280101 | 0.997x |
| 3 | 3.255288 | 3.278401 | 0.993x |

- reference_raw_samples_ms: `[3.253012, 3.271220, 3.255288]`
- candidate_raw_samples_ms: `[3.245672, 3.280101, 3.278401]`
- reference_median_ms: `3.255288`
- candidate_median_ms: `3.278401`
- improvement_pct: `-0.709971`

```text
improvement_pct = (3.255288 - 3.278401) / 3.255288 * 100 = -0.709971
```

**Baseline canonization (target 1).** The v0 median for this regime is
**3.255288 ms** (warmup 50 / repeat 100). The epoch-1 archive figure of 3.259 ms was
recorded under a different fingerprint; the number above is the canonical baseline for
the epoch-2 fingerprint `fe73bc58…`. v1 tracks v0 at 0.993x–1.002x, i.e. identity to
within run-to-run noise, exactly as expected for an adapter-of-base (the only change is
the `Model` → `ModelNew` rename plus trailing `__main__` scaffolding). This is a
`baseline` result, not an `accepted` or `no-improvement` result; no 5% gate applies.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `not-applicable`

No round decision exists, so no `mechanism_observables` are declared and no observable
rows are required. The mirror is empty by construction, not by omission.

## Profiler Evidence

- profiler_applicability: `required` (baseline)
- profiler_level: `summary`
- profiler_device_time: `available` (trace carries `cat=kernel` device durations)
- iterations: `100` forward calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`,
  `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: not applicable (device kernel time available; no
  `runtime_launch_*` fallback needed)
- profiler_mode: `forward` (see Deviations)

Reference and candidate scopes were collected in one trace and summarized independently
via `scripts/summarize_trace.py`. All totals below are normalized by `iterations`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (`reference_baseline_adapter`) | 96785.19 | 967.852 | 12395 | 123.95 | 3.255288 | 0.297317 |
| candidate (`candidate_baseline_adapter`) | 96861.22 | 968.612 | 12400 | 124.00 | 3.278401 | 0.295453 |

```text
device_ratio = device_us_per_call / (median_ms * 1000)
reference: 967.852 / (3.255288 * 1000) = 0.297317
candidate: 968.612 / (3.278401 * 1000) = 0.295453
```

**Device time per call (target 3):** 967.852 µs reference / 968.612 µs candidate,
device_ratio ≈ 0.2973, so **device share ≈ 29.7% and host share ≈ 70.3%**. The base path
is host-launch-dominated: roughly 2.26 ms of each ~3.26 ms call is host-side launch and
dispatch overhead, not device execution.

### Launch census (target 2)

Kernel count per call: **123.95 (reference) / 124.00 (candidate)** — the two scopes agree
to within half a launch over 100 iterations. **The epoch-1 prior of 123.9 kernels/call is
confirmed**, not corrected: the two regimes agree despite different fingerprints.

Top device-time contributors, reference scope, 21 distinct kernels, listed µs/call sum
reconciles exactly to `device_us_per_call = 967.852`:

| # | Kernel (abbreviated; full mangled name in `log/round_000_summary_reference.json`) | Count/call | Count total | Us/call | % device |
|---:|---|---:|---:|---:|---:|
| 1 | `index_elementwise_kernel<…index_put_kernel_impl…>` — scatter-store into `expert_out[mask]` | 7.99 | 799 | 127.402 | 13.16% |
| 2 | `index_elementwise_kernel<…index_kernel_impl…>` — boolean-mask gather `x_rep[mask]` | 8.00 | 800 | 127.312 | 13.15% |
| 3 | `cub::DeviceSelectSweepKernel` — `nonzero` / mask→indices | 15.99 | 1599 | 125.756 | 12.99% |
| 4 | `reduce_kernel<1024,1,ReduceOp<bool…>>` — `mask.any()` reduction | 8.00 | 800 | 86.502 | 8.94% |
| 5 | `cub::DeviceReduce*Kernel` | 16.00 | 1600 | 81.186 | 8.39% |
| 6 | `Gemm_tcu_mr_kernel::gemm_tcu_h<128,128,32,32,32,1,…>` — `x_e @ w1[e].T` (gate/up proj) | 8.00 | 800 | 61.114 | 6.31% |
| 7 | `Gemm_tcu_mr_kernel::gemm_tcu_h<32,64,32,16,32,2,…>` — `act @ w2[e].T` (down proj) | 8.00 | 800 | 57.717 | 5.96% |
| 8 | `cub::DeviceCompactInitKernel<ScanTileState<int…>>` | 15.99 | 1599 | 56.404 | 5.83% |
| 9 | `elementwise_kernel<MulFunctor<Half,…>>` — `silu(gate)*up` | 8.00 | 800 | 48.995 | 5.06% |
| 10 | silu kernel — `F.silu(gate)` | 8.00 | 800 | 40.001 | 4.13% |
| 11 | `vectorized_elementwise_kernel<AUnaryFunctor<long,long,bool…>>` — `flat_ids == e` | 8.00 | 800 | 30.751 | 3.18% |
| 12 | `sbtopk::gatherTopK<float,uint,2,false>` — `torch.topk` | 1.00 | 100 | 22.762 | 2.35% |
| 13 | `bitonicSortKVInPlace<2,-1,16,16,float,long,GTOp…>` — topk sort stage | 1.00 | 100 | 18.861 | 1.95% |
| 14 | `vectorized_elementwise_kernel<float16_copy_kernel_cuda>` — dtype copies | 3.00 | 300 | 16.763 | 1.73% |
| 15 | `reduce_kernel<512,2,ReduceOp<Half…>>` — final `sum(dim=1)` | 0.99 | 99 | 16.722 | 1.73% |
| 16 | `reduce_kernel<1024,1,ReduceOp<float…>>` — `topk_weights.sum(-1)` | 1.00 | 100 | 14.463 | 1.49% |
| 17 | `elementwise_kernel<MulFunctor>` — `expert_out * flat_w` | 0.99 | 99 | 11.211 | 1.16% |
| 18 | `elementwise_kernel<DivFunctor<float>>` — renormalize | 1.00 | 100 | 6.790 | 0.70% |
| 19 | `vectorized_elementwise_kernel<FillFunctor<Half>>` — `zeros_like` | 1.00 | 100 | 6.327 | 0.65% |
| 20 | `elementwise_kernel_v3<direct_copy_kernel_cuda>` — `topk_weights.to(dtype)` | 1.00 | 100 | 5.672 | 0.59% |
| 21 | `softmax_warp_forward<float,float,float,3,…>` — `torch.softmax` | 1.00 | 100 | 5.142 | 0.53% |

- Top 5 kernels account for **56.64%** of device time; the two GEMMs together are only
  **12.27%** (118.831 µs/call of 967.852).
- The three dispatch/indexing families (#1, #2, #3, #4, #5, #8, #11) — scatter-store,
  mask gather, `nonzero`, `mask.any()`, cub reduce/compact-init, and `flat_ids == e` —
  total **635.313 µs/call = 65.6% of device time** and ~95 of the 124 launches. The real
  math (GEMMs + silu + mul + reductions + softmax + topk) is under 320 µs/call.
- Kernel #1 shows 7.99/call (799 total, not 800): one scatter-store launch is missing
  across the 100 iterations, i.e. the trace-window boundary clipped one event whose
  duration fell outside the scope interval.

### Static vs data-dependent launch count (target 2, second half)

**The per-expert Python loop produces a data-dependent launch count, and `mask.any()`
does vary it between calls.** Measured directly:

| Router input | Active experts | Per-call CUDA ops | Ops per active expert |
|---|---:|---:|---:|
| seed-42 random (benchmark regime) | 8 (`[0..7]`, counts 21/19/19/20/22/22/18/25) | 148 | 18.5 |
| expert 7 forced out of top-2 (`rl[:,7] = -1e4`) | 7 (`[0..6]`) | 134 | 19.1 |
| only experts 0,1 reachable | 2 (`[0,1]`) | 64 | 32.0 |
| all-tie zeros | 2 (`[0,1]`) | 64 | 32.0 |

Within any fixed input the count is perfectly static (12 consecutive calls, all 148), but
it scales with the number of experts that win at least one token: each active expert
contributes ~14 launches, and 8 active experts gives 148 total (≈14×8 + 36 fixed
preamble). The `if not mask.any(): continue` early-exit is therefore live and
data-dependent.

Implication (informational — the campaign captures Triton candidates, not base): a
CUDA-graph capture over this base path would **not** be stable across arbitrary inputs;
it would only be valid for a fixed active-expert set. Under the seed-42 benchmark inputs
all 8 experts are always active, so the measured regime is stable at 148 ops/call, but
that stability is a property of the benchmark input, not of the code shape.

### Accepted Reference Top Kernels

See the census table above (reference scope). Full mangled names and unrounded values:
`log/round_000_summary_reference.json`.

### Candidate Top Kernels

| # | Kernel (abbreviated) | Count/call | Count total | Us/call | % device |
|---:|---|---:|---:|---:|---:|
| 1 | `index_elementwise_kernel<…index_put_kernel_impl…>` | 8.00 | 800 | 127.794 | 13.19% |
| 2 | `index_elementwise_kernel<…index_kernel_impl…>` | 8.00 | 800 | 127.421 | 13.15% |
| 3 | `cub::DeviceSelectSweepKernel` | 16.00 | 1600 | 125.825 | 12.99% |
| 4 | `reduce_kernel<1024,1,ReduceOp<bool…>>` | 8.00 | 800 | 86.491 | 8.93% |
| 5 | `cub::DeviceReduce*Kernel` | 16.00 | 1600 | 81.195 | 8.38% |
| 6 | `Gemm_tcu_mr_kernel::gemm_tcu_h<128,128,32,…>` | 8.00 | 800 | 61.124 | 6.31% |
| 7 | `Gemm_tcu_mr_kernel::gemm_tcu_h<32,64,32,…>` | 8.00 | 800 | 57.749 | 5.96% |
| 8 | `cub::DeviceCompactInitKernel<ScanTileState<int…>>` | 16.00 | 1600 | 56.449 | 5.83% |
| 9 | `elementwise_kernel<MulFunctor<Half,…>>` | 8.00 | 800 | 49.007 | 5.06% |
| 10 | silu kernel | 8.00 | 800 | 39.999 | 4.13% |

Full list (21 kernels): `log/round_000_summary_candidate.json`. Top-5 share 56.65%,
matching the reference scope within noise.

## Deviations

1. **Kernel-mode profiling unavailable → forward-mode dual-scope fallback.**
   `project.md` declares `profile_mode: kernel`, but the Phase-0 `baseline_adapter.py`
   defines no `ModelNew.run_out`, so `make_profile_call()` raises
   `KsCompareError: kernel profiling requires a callable ModelNew.run_out`. Per the
   sibling precedent, the harness's existing dual-scope interface was used unchanged with
   `--profile-mode forward --profile-warmup 20 --profile-iterations 100`. This measures
   the complete forward call rather than a preallocated `run_out`, so it includes the
   final weighted reduction and the `.view(...).sum(dim=1)` epilogue. No harness edit was
   made. The declared `profile_iterations: 100` was preserved.

2. **No change to `project.md` or the measurement fingerprint.** The fingerprint is
   computed from `profile_mode: kernel` even though profiling actually ran in `forward`
   mode. The fingerprint remains `fe73bc58…` and is unchanged; this is reported as a
   deviation rather than silently altered.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `752a25033b7629459c6eb128c60a4bdc3ab77b9c7cc97f5d3592bdff4cd45a47` | baseline, correctness pass, no repair needed |

No repair was requested or performed.

## evidence_for_next_round

- **Canonized baseline: 3.255288 ms** (v0, warmup 50 / repeat 100, fingerprint
  `fe73bc58…`). Epoch-1's 3.259 ms is a different-fingerprint prior; use 3.255288 as the
  denominator for all epoch-2 improvement percentages.
- **Launch census confirmed at ~124 kernels/call**, matching the epoch-1 123.9 prior
  across a fingerprint change. Device 967.852 µs/call, device_ratio 0.2973.
- **The base path is host-dominated, ~70.3% host / ~29.7% device.** ~2.26 ms per call is
  launch and dispatch overhead against only 968 µs of device work. Any purely
  device-time optimization is bounded by the 29.7% it can touch unless launch count also
  drops.
- **The dispatch/indexing machinery, not the GEMMs, is the device-time bottleneck.**
  Scatter-store, mask gather, `nonzero`, `mask.any()`, cub reduce/compact-init, and
  `flat_ids == e` together cost 635.313 µs/call = 65.6% of device time and ~95 of 124
  launches. The two GEMMs are only 118.831 µs/call = 12.27%.
- **The per-expert loop launch count is data-dependent** (~14 launches per active expert,
  148 ops/call at 8 active experts, 64 at 2). The `mask.any()` early-exit is live. A
  candidate that replaces the Python loop with a data-independent Triton launch pattern
  both cuts launches and removes this input sensitivity.
- **Two levers are quantitatively live for the designer.** The launch lever is large
  (≈124 launches/call against a ~85 µs/call Triton-launcher-tax prior means compressible
  overhead far exceeds device time). The device lever is also live: 635 µs/call of
  dispatch/indexing work is algorithmically eliminable by fusing dispatch into a Triton
  kernel, versus only ~119 µs/call of irreducible GEMM.
- **Graph-capture caveat:** capturing the base path itself would be input-fragile (launch
  count varies with active-expert count), so any graph replay must be built over a
  fixed-shape Triton candidate, not over base.
- **Fingerprint stability verified three ways** — this campaign plus two sibling positive
  controls — so wall comparisons within epoch-2 are on solid ground.

No next optimization is selected here; these are observations only.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline is canonized and clean — correctness PASS, three ordered
  pairs at 3.253012 / 3.271220 / 3.255288 ms (v0) with ~1.00x adapter identity, no
  incidents, fingerprints confirmed. `total_rounds` is 0 and `max_rounds` is 20, so the
  round budget is untouched. Both optimization levers (launch compression and
  dispatch-fusion device time) are quantitatively live.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness gate:

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/epoch2/baseline_adapter.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 0 --repeat 1
```

Interleaved benchmark (three ordered pairs):

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && for i in 1 2 3; do /usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/epoch2/baseline_adapter.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 50 --repeat 100; done
```

Separately scoped profiler:

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/epoch2/baseline_adapter.py --seed 42 --atol 1e-2 --rtol 1e-2 --profile --profile-reference-file kernels/track1-triton/fused_moe/bi150/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/fused_moe/bi150/epoch2/log/round_000_forward_100iter.pt.trace.json
```

Scope summarization (run once per scope):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/fused_moe/bi150/epoch2/log/round_000_forward_100iter.pt.trace.json --iterations 100 --scope reference_baseline_adapter --wall-ms 3.255288
cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/fused_moe/bi150/epoch2/log/round_000_forward_100iter.pt.trace.json --iterations 100 --scope candidate_baseline_adapter --wall-ms 3.278401
```

Measurement fingerprint recompute:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 -c "
import hashlib, json, pathlib
base = pathlib.Path('kernels/track1-triton/fused_moe/base.py').read_bytes()
harness = pathlib.Path('auto_bench.py').read_bytes()
settings = {'device':'cuda:0','dtype':'mixed(fp16-hidden/w1,w2; fp32-router)','profile_iterations':100,'profile_mode':'kernel','profile_warmup':20,'repeat':100,'shape':{'hidden_states':[83,128],'router_logits':[83,8],'w1':[8,128,128],'w2':[8,128,64]},'warmup':50}
cj = json.dumps(settings, sort_keys=True, separators=(',',':'))
print(hashlib.sha256(base + b'\x00' + harness + b'\x00' + cj.encode()).hexdigest())
"
```
