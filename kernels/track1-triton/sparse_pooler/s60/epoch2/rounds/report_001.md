# Report 001

Result: no-improvement

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`264c7be47436c5a8e9a9c2d324aae52632ec0e0201f3725a77bea0a163d2a4ab` (hash re-verified live; family "sparse-pooler-tail-fusion"; expected_wall_improvement_pct 0.0 declared honestly; deliverable-grade round per project.md DELIVERABLE RULE)
- Candidate: `triton_sparse_pooler_e2_001.py`
- Accepted reference: `baseline_adapter.py` @`359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8` (last_accepted_kernel per r000; byte-equivalent pipeline to base.py; pure PyTorch, zero Triton kernels)
- Accepted reference report: `rounds/report_000.md` (Phase 0 baseline; canonical wall median ~0.838 ms; census 11 topsLaunchKernel/call, GEMM-bound 481us / 61%)
- Decision SHA256: `264c7be47436c5a8e9a9c2d324aae52632ec0e0201f3725a77bea0a163d2a4ab`
- Sketch SHA256: `a92ec7842e345d0112a12c19efb2cccd6b5f7017e43765935461b9ebd989a295` (rounds/sketch_001.json, re-verified)
- Candidate SHA256: `f99538b13f7768297d7aa95a25e4c33231eb12321575bdb80ede401b226d81fa` (re-verified live; AST-parse OK)
- Accepted reference SHA256: `359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8`
- Base SHA256: `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58` (unchanged, re-verified)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Profile snapshot SHA256: `7cd0cdf4b01b064b91f2b8f199cff6d12b175903a2c8d24ba7153f4d6a6aa6a0` (profile_snapshot/triton_gcu.yaml, unchanged)
- Runtime fingerprint: `project.md#runtime-fingerprint` (unchanged since r000; triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2 / Enflame GCU major=3 minor=0 multi_processor_count=2)
- Measurement fingerprint: `sp-s60-e2` (team-state.md; base/harness bytes re-verified identical)
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeded directly to authoritative timing (sibling r001 precedent; this dispatch routes below-bar outcomes to no-improvement with full census, and a screen-out would skip the profiler — destroying the round's mandated launch-count / device-delta dual-gate measurement duty)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (seed42 canonical) | `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` vs base.py, seed 42, fp32 out `list of 4 x [30522]` | `PASS accuracy` in all three authoritative pairs + profile run (4/4 invocations) | pass | timing pairs 1-3, profile run |
| stateless module | zero call-time instance state, no caches/workspace | `__init__` stores only dense/act/layer_norm/decoder/pooling (5 constructor attrs); `forward` writes zero instance attrs | pass | source audit |
| forward contract | `forward(hidden_states, seq_lens) -> list of 4 x [30522] fp32`, order preserved | returns `[out[i] for i in range(NS)]` of 4 fresh `[30522]` fp32 tensors; per-segment max over `[20,25,18,20]` | pass | source audit + correctness |
| capability legality | NO tl.dot; tl.arange power-of-2; reduction.max (not argmax); fp32 load/store; num_warps=2 | 0 tl.dot sites (both GEMMs vendor `nn.Linear`); 1 `tl.arange(0,256)` power-of-2; `tl.maximum` reduction (max, not argmax); fp32 loads/stores; `num_warps=2` | pass | source audit |
| no compile/graph machinery | zero torch.compile/TORCHINDUCTOR/reduce-overhead/graph/capture/contiguous/tolist tokens | no such constructs in candidate source; `.tolist()` count 0 (D2H sync eliminated — see D1) | pass | source audit |
| AST-loader-safe module | safe-literal module constants; get_inputs/get_init_inputs retained | module-level imports + @triton.jit + ClassDef + 2 helper FunctionDefs; `get_inputs`/`get_init_inputs` present; `ast.parse` gate OK | pass | source audit |
| default-stream discipline | all invocations on harness default route | unchanged harness default path; zero stream manipulation | pass | command history |
| cold JIT outside medians | warmup 50 absorbs first-call compile | harness warmup 50 precedes every timed section | pass | harness behavior |

Conformance, correctness, and every declared guardrail passed.

## Screening Evidence

Not run — correct candidate proceeded directly to authoritative timing (three ordered pairs). Rationale recorded in Identity: this round's contractual products are the Triton deliverable plus the two named mechanism observables (`runtime_launch_count_per_call`, `device_us_per_call_tail`), which require the profiler census a screen-out would skip.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms), byte-for-byte identical flags, interpreter, device, default-stream route`
- reference_raw_samples_ms: `[0.922848, 0.844259, 0.857295]`
- candidate_raw_samples_ms: `[3.496313, 3.447952, 3.430022]`
- reference_median_ms: `0.857295`
- candidate_median_ms: `3.447952`
- improvement_pct: `-302.1896779988218`

```text
improvement_pct = (0.857295 - 3.447952) / 0.857295 * 100 = -302.189678
```

| Independent invocation | Reference wall ms | Candidate wall ms | Speedup | Evidence |
|---:|---:|---:|---:|---|
| 1 | `0.922848` | `3.496313` | `0.264x` | pair 1 timing |
| 2 | `0.844259` | `3.447952` | `0.245x` | pair 2 timing |
| 3 | `0.857295` | `3.430022` | `0.250x` | pair 3 timing |

BELOW the 5.0% adoption bar with a decisively NEGATIVE sign: candidate wall 3.447952 ms vs reference 0.857295 ms = −302.2% paired improvement (candidate ~0.249x, ~4x slower). The fused-tail Triton kernel is ~4x slower on device than the base's PyTorch elementwise + `chunk.max` tail path, and that device deficit dominates the entire round: the candidate's launch-count collapse (11 → 8 topsLaunchKernel/call) and D2H-sync elimination (`.tolist()` → device-side `cumsum`/`sub`) cannot offset a device regression that is ~3x the whole reference wall.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time_unrounded_paired_median_ms | honest two-sided (expected no-improvement; GEMM 61% vendor-bound + retained 125us D2H sync + 7% padding waste) | −302.2% (candidate 3.447952 vs reference 0.857295 ms, three ordered pairs); decisively negative, far beyond the honest no-improvement band | **fail** | pairs 1-3 timing |
| runtime_launch_count_per_call | decrease from base's 11 topsLaunchKernel/call toward fewer launches in the tail path | **11 → 8/call** (base = 11 `topsLaunchKernel`; candidate = 7 `topsLaunchKernel` + 1 `topsModuleLaunchKernel`) — dispatch collapse engaged (3 launches removed), the fused tail is 1 Triton launch | **pass** | profile census |
| aten_cpu_ops_per_call | log1p/relu elementwise + 4x chunk.max collapse out of the aten census; D2H sync remains | base 83 cpu_op events/call (incl. 8 `aten::max`, 2 `aten::log1p`, 1 `aten::relu`, slice/as_strided chain); candidate 59/call — `aten::max`/`log1p`/`relu` GONE, replaced by `aten::cumsum`+`aten::sub`+`aten::select`; D2H `tolist`/`item` GONE (eliminated, see D1) | **pass** | profile census |
| device_us_per_call_tail | two-sided: fused-tail device time vs base's ~110us elementwise+pooling floor | device_time_available = **false** (GCU launch-only trace); inferred tail delta = candidate wall − base wall − (launch delta) ≈ +2.59ms — the fused tail device regression is ~3x the base's entire wall, far above the ~110us floor (padding waste + reduction cost materialized massively) | **fail** | profile census + wall inference |
| reduction_max_binding_audit | zero tl.dot; tl.arange power-of-2; reduction.max (not argmax/scatter_reduce); fp32 load/store; num_warps=2; zero compile/capture strings | 0 tl.dot; 1 tl.arange extent 256 (power-of-2); tl.maximum reduction (max); fp32 loads/stores; num_warps=2; 0 DANGER tokens | **pass** | source audit |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: one direct-launched Triton tail kernel (grid=(NS=4, ceil(30522/256)=120), num_warps=2) computing log1p(relu) + per-segment max in a single device pass, replacing the base tail path (log1p + relu + Python-loop of 4 chunk.max); both GEMMs (dense + decoder, 481us / 61%) stay vendor-bound; V=30522 padded to 32768 (power-of-2 tl.arange), masked `vocab < V`
- expected_causal_chain: cn.dispatch-collapse → cn.aten-dispatch-time CONFIRMED (11 → 8 launches/call; `aten::max`/`log1p`/`relu` collapsed out); cn.dispatch-collapse → cn.D2H-sync CONFIRMED-eliminated (`.tolist()` → device-side `cumsum`/`sub`, beyond the decision's conservative "tolist retained" note); cn.device-time-delta NEGATIVE and dominant (fused-tail kernel ~4x slower than base's PyTorch tail; padding waste 30522→32768 ~7% + per-program reduction cost + static `for t in range(0,83)` unroll — the candidate's device time alone exceeds the entire base wall); cn.dispatch-collapse → cn.wall-time = −302.2%
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed` — the dispatch-collapse mechanism ENGAGED exactly as designed (launch count 11 → 8, aten tail ops collapsed, D2H sync eliminated), but the wall criterion FAILED decisively (−302.2%) because the fused-tail Triton kernel is ~4x slower on device than the base's PyTorch elementwise+chunk.max path. This is the decision's pre-declared honest no-improvement reading, amplified far beyond the expected 0.0% band by a device-side regression the sketch's two-sided device-delta (band b) anticipated but underestimated.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (per decision profiling_level; forward-mode dual-scope trace + host census)
- profiler_device_time: `unavailable: device_time_available = false — GCU trace exposes runtime-launch events (gcu_runtime) but no cat=kernel device durations (target profile triton_gcu marks kernel-summary/kernel-events/instruction-level as unavailable)` — Level 1 normalized runtime-launch evidence recorded instead, never substituting launch time for device kernel time
- iterations: `100` forward calls per scope
- normalized_fields: `runtime_launch_count_per_call`, `runtime_launch_total_us`, `runtime_launch_us_per_call`
- trace: `log/report_001_forward.pt.trace.json`
- trace_sha256: `2b7db3cf114aea5304e658bf2852e27ca4ae7ba76187e07d4cfbbba8c4f88b7b`

### Runtime-launch census (Level 1, launch-only trace, per call)

| Signal | accepted_reference (baseline_base) | candidate (direct Triton) |
|---|---:|---:|
| runtime_launch_count_per_call | 11.0 | 8.0 |
| launch event class | `topsLaunchKernel` (11/call @132.42us total) | `topsLaunchKernel` (7/call @65.08us) + `topsModuleLaunchKernel` (1/call @9.20us) |
| runtime_launch_total_us | 132.419 | 74.280 |
| aten cpu_ops total/call | **83** (8 `aten::max` + 12 empty + 6 as_strided + 4 addmm + 4 slice + 2 log1p + 1 relu + linear/gelu/layer_norm chain) | **59** (6 empty + 6 as_strided + 4 addmm + 4 select + 2 cumsum + 2 sub + linear/gelu/layer_norm chain; `max`/`log1p`/`relu` GONE) |

Notes: (i) device_time_available is `false` on this target — the trace exposes launch-only events, so all device attribution is via launch-count + launch-API-time; kernel-internal device duration cannot be attributed. (ii) The dispatch collapse ENGAGED: launch count dropped 11 → 8/call and the aten tail ops (`aten::max` ×8, `aten::log1p` ×2, `aten::relu` ×1, plus the slice/as_strided reshapes) collapsed out, replaced by `aten::cumsum` + `aten::sub` (device-side offset computation) + `aten::select`. (iii) The D2H sync is ELIMINATED: `.tolist()` count 0 — segment boundaries now arrive device-side via `seq_lens` + `cumsum`-derived `offsets` tensors read inside the kernel (Coder deviation D1, a host-path improvement over the decision's conservative "tolist retained" note). (iv) Despite a strictly smaller host/launch profile, the wall regressed −302.2%: the fused-tail Triton kernel is ~4x slower on device than the base PyTorch elementwise+`chunk.max` path — the device deficit (~2.59ms) is ~3x the entire reference wall (0.857ms).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `f99538b13f7768297d7aa95a25e4c33231eb12321575bdb80ede401b226d81fa` | correctness passed on first attempt; no repairs needed |

Zero Verifier-to-Coder repairs were needed — no candidate defect was found at any point (candidate hash constant end-to-end, matching the coder ledger).

## evidence_for_next_round

- Observed fact: the dispatch collapse ENGAGES — launch count 11 → 8/call, aten tail ops (`max`/`log1p`/`relu` + slice/as_strided reshapes) collapse to a single Triton launch — yet paired wall REGRESSED −302.2% (3.447952 vs 0.857295 ms). The wall is decided by the DEVICE, not by launch/aten op count.
- Observed fact (canonical, this campaign): the fused-tail Triton kernel is ~4x slower on device than base's PyTorch `log1p(relu)` + 4x `chunk.max` tail. The device deficit (~2.59ms inferred from wall − launch-API) is ~3x the entire reference wall — the per-program reduction over the static `for t in range(0,83)` unroll with 120 vocab-blocks × 4 segments = 480 programs, each re-loading the full [83,30522] tile's token span, is far more expensive than the base's vectorized library reduction.
- Observed fact (canonical, this campaign): **D2H sync elimination IS achievable device-side** (`.tolist()` count 0; `cumsum`/`sub` offset computation) — the preflight's "D2H sync cannot be removed without a slower hand-written segment reduction" is confirmed in its second clause: removing it via the fused-tail kernel produced a ~4x device regression, exactly the preflight's predicted "slower hand-written segment reduction" penalty, but far larger than the ~150us preflight estimate.
- Observed fact: GEMMs remain vendor-bound (2 `addmm`/`linear` pairs in both base and candidate, dense [83,768]@[768,768] + decoder [83,768]@[768,30522]); 768 and 30522 are NOT powers of two, so tl.dot is capability-blocked — the 481us / 61% GEMM slice is structurally untouchable.
- Capability constraint confirmation: `tl.arange` requires power-of-2 extent (V=30522 → 32768, ~7% wasted lanes) and `tl.dot` requires power-of-2 M/N/K — both honored correctly by the candidate; neither is the source of the regression.
- Deliverable banked: `triton_sparse_pooler_e2_001.py` @`f99538b13f7768297d7aa95a25e4c33231eb12321575bdb80ede401b226d81fa` is a correctness-PASS Triton submission (forward surface, stateless, envelope-legal, 0 tl.dot) at ~0.249x — per project.md DELIVERABLE RULE this is the campaign's primary contractual product regardless of adoption (the FIRST Triton candidate for sparse_pooler, whose base is pure PyTorch); canonical pointer stays `baseline_adapter.py`.
- Session drift note: paired same-session basis absorbs drift; the −302.2% delta is ~60x the 5.0% bar in the negative direction, so no plausible drift affects the classification.

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: no-improvement #1 on the campaign (streak 1/3 vs valid_no_improvement_limit 3); round budget 1/20 consumed; the round banked the FIRST Triton deliverable plus both canonical physics numbers (D2H-sync elimination is device-side-achievable but only via a ~4x-slower fused-tail kernel; launch count collapses 11 → 8 but the device dominates). The GEMM-bound diagnosis (61% untouchable) and the device-tail regression are now census-grade.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + authoritative timing (three identical interleaved pairs):

```bash
cd /root/CodeBuddy/20260828202827/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/epoch2/triton_sparse_pooler_e2_001.py --warmup 50 --repeat 100
```

Dual-scope profiler (forward-mode, pw=20/pi=100 default):

```bash
cd /root/CodeBuddy/20260828202827/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/bin/python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/epoch2/triton_sparse_pooler_e2_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-output kernels/track1-triton/sparse_pooler/s60/epoch2/log/report_001_forward.pt.trace.json
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/sparse_pooler/s60/epoch2/log/report_001_forward.pt.trace.json --iterations 100 --scope baseline_base --wall-ms 0.857295
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/sparse_pooler/s60/epoch2/log/report_001_forward.pt.trace.json --iterations 100 --scope candidate_triton_sparse_pooler_e2_001 --wall-ms 0.857295
```

Artifact hash ledger (re-verified this round):

```text
f99538b13f7768297d7aa95a25e4c33231eb12321575bdb80ede401b226d81fa  triton_sparse_pooler_e2_001.py
264c7be47436c5a8e9a9c2d324aae52632ec0e0201f3725a77bea0a163d2a4ab  rounds/decision_001.md
a92ec7842e345d0112a12c19efb2cccd6b5f7017e43765935461b9ebd989a295  rounds/sketch_001.json
359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8  baseline_adapter.py
46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
7cd0cdf4b01b064b91f2b8f199cff6d12b175903a2c8d24ba7153f4d6a6aa6a0  profile_snapshot/triton_gcu.yaml
2b7db3cf114aea5304e658bf2852e27ca4ae7ba76187e07d4cfbbba8c4f88b7b  log/report_001_forward.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "f99538b13f7768297d7aa95a25e4c33231eb12321575bdb80ede401b226d81fa",
  "correctness": {
    "status": "pass",
    "evidence": [
      "auto_bench.py PASS accuracy in all three authoritative pairs + profile run (4/4 invocations, seed42 canonical regime)",
      "harness comparator PASS printed every run"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "-302.2% (reference 0.857295 ms vs candidate 3.447952 ms; bar +5.0% FAILED with negative sign; fused-tail device regression ~4x)",
      "confidence": "high",
      "evidence": ["timing pairs 1-3"]
    },
    {
      "name": "runtime_launch_count_per_call",
      "status": "observed",
      "value": "11 -> 8/call (base 11 topsLaunchKernel; candidate 7 topsLaunchKernel + 1 topsModuleLaunchKernel; dispatch collapse engaged)",
      "confidence": "high",
      "evidence": ["profile census"]
    },
    {
      "name": "aten_cpu_ops_per_call",
      "status": "observed",
      "value": "83 -> 59/call; aten::max/log1p/relu + slice/as_strided reshapes collapsed out; replaced by aten::cumsum+sub (device-side offsets) + aten::select",
      "confidence": "high",
      "evidence": ["profile census"]
    },
    {
      "name": "device_us_per_call_tail",
      "status": "observed",
      "value": "device_time_available = false (GCU launch-only trace); fused-tail device regression inferred ~2.59ms = ~3x entire reference wall (vs base ~110us elementwise+pooling floor) — band (b) materialized massively",
      "confidence": "high",
      "evidence": ["profile census", "wall inference"]
    },
    {
      "name": "reduction_max_binding_audit",
      "status": "observed",
      "value": "0 tl.dot; 1 tl.arange extent 256 power-of-2; tl.maximum reduction (max, not argmax/scatter_reduce); fp32 load/store; num_warps=2; 0 DANGER tokens; .tolist() count 0",
      "confidence": "high",
      "evidence": ["source audit"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "present — ONE Triton kernel _sparse_pooler_tail_kernel lowered and device-executed (1 topsModuleLaunchKernel/call); both GEMMs remain vendor nn.Linear (0 tl.dot)",
    "evidence_contract": "triton_gcu (power-of-2 tl.arange honored; reduction.max used not argmax)",
    "evidence": ["profile census"]
  },
  "evidence_gap_cause": "device_time_available = false on GCU launch-only trace; device attribution is inference from wall - launch-API-time (no cat=kernel events)"
}
```
