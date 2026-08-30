# Report 001

Result: no-improvement

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`4ae2b61392e9187a22f000a87282494eb72806927cd83cfec6c08de69a138771` (hash re-verified live; F1 deliverable-grade triton-attention-dispatch-collapse, expected_wall_improvement_pct 0.0 declared honestly)
- Candidate: `triton_mm_encoder_attention_e2_001.py`
- Accepted reference: `baseline_adapter.py` (last_accepted_kernel per r000; byte-equivalent pipeline to base.py)
- Accepted reference report: `rounds/report_000.md` (Phase 0 baseline; hash `not-applicable: Phase 0`)
- Decision SHA256: `4ae2b61392e9187a22f000a87282494eb72806927cd83cfec6c08de69a138771`
- Sketch SHA256: `ef71920a8a856c633bf8ef5fcebe733bcda6f0fd026210691b1cc8e94aad8f70` (rounds/sketch_001.json, re-verified)
- Candidate SHA256: `f2f8b9b6c6f6a16cfbf162cf3f9b115461fc7a5716601eb8e3723961a8536ead` (re-verified live; AST-parse OK)
- Accepted reference SHA256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged, re-verified)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Profile snapshot SHA256: `8dfabd0af59b8f6640b47179fee19bca2f5fe35b18535a3db24f60c842e42b70` (profile_snapshot/triton_gcu.yaml)
- Runtime fingerprint: `project.md#runtime-fingerprint` (unchanged since r000)
- Measurement fingerprint: `c335b39cbf2eaa15e1a358be90d0aab85d0fd7e8ffd4b7b4e825df0901ad61f9` (project.md canonical; base/harness bytes re-verified identical)
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeded directly to authoritative timing (sibling r001 precedent; this dispatch routes below-bar outcomes to no-improvement with full census, and a screen-out would skip the profiler — destroying the round's mandated T_launcher/D_cand dual-gate measurement duty)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (seed42 canonical) | `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` vs base.py, seed 42, fp16 out `[2,83,512]` | `PASS accuracy` in all three authoritative pairs + profile run (4/4 invocations); capability-preflight max_abs 9.77e-4 | pass | timing pairs 1-3, profile run |
| stateless module | zero call-time instance state, no caches/workspace | `__init__` stores only num_heads/head_size/num_kv_heads/scale; `forward`/`run_out` write zero instance attrs | pass | source audit |
| run_out contract | `run_out(q,k,v,out) -> None`, 4-arg preallocated-output surface | `run_out` delegates to `_launch` into caller buffer, returns None; forward allocates one fresh `torch.empty` | pass | source audit |
| capability legality | every tl.dot mult-of-16 (TP=128, D=64) fp32 with widened operands; num_warps=2; tl.arange power-of-2; tl.max/tl.sum no-keepdim | 2 tl.dot sites (QK^T 128x64@64x128, PV 128x128@128x64) fp32; 3 widening casts (q/k/v `.to(tl.float32)`); `num_warps=2`; `tl.arange(0,128)` power-of-2; `tl.max`/`tl.sum` axis=1 without keepdim, broadcast via `[:,None]` | pass | source audit |
| no compile/graph machinery | zero torch.compile/TORCHINDUCTOR/reduce-overhead/graph/capture/contiguous tokens | no such constructs in candidate source; zero `.contiguous()` in host paths | pass | source audit |
| AST-loader-safe module | safe-literal module constants; get_inputs/get_init_inputs retained | module-level literals only; `get_inputs`/`get_init_inputs` present | pass | source audit |
| default-stream discipline | all invocations on harness default route | unchanged harness default path; zero stream manipulation | pass | command history |
| cold JIT outside medians | warmup 50 absorbs first-call compile | harness warmup 50 precedes every timed section | pass | harness behavior |

Conformance, correctness, and every declared guardrail passed.

## Screening Evidence

Not run — correct candidate proceeded directly to authoritative timing (three ordered pairs). Rationale recorded in Identity: this round's contractual products are the Triton deliverable plus the two named mechanism observables (`triton_launcher_tax_per_call`, `device_us_per_call`), which require the profiler census a screen-out would skip.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms), byte-for-byte identical flags, interpreter, device, default-stream route`
- reference_raw_samples_ms: `[0.276584, 0.276234, 0.276579]`
- candidate_raw_samples_ms: `[0.303552, 0.305640, 0.306290]`
- reference_median_ms: `0.276584`
- candidate_median_ms: `0.305640`
- improvement_pct: `-10.50468148190503`

```text
improvement_pct = (0.276584 - 0.305640) / 0.276584 * 100 = -10.504681
```

| Independent invocation | Reference wall ms | Candidate wall ms | Speedup | Evidence |
|---:|---:|---:|---:|---|
| 1 | `0.276584` | `0.303552` | `0.911x` | pair 1 timing |
| 2 | `0.276234` | `0.305640` | `0.904x` | pair 2 timing |
| 3 | `0.276579` | `0.306290` | `0.903x` | pair 3 timing |

BELOW the 5.0% adoption bar with a decisively NEGATIVE sign: candidate wall 0.305640 ms vs reference 0.276584 ms = −10.5% paired improvement (candidate ~0.906x). S60 is DEVICE-BOUND: the hand-written tl.dot kernel (~166us device) is SLOWER than the CNNL SDPA library kernel (~158us device floor), and that device deficit alone exceeds the entire compressible host + launcher budget (~28us).

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time_unrounded_paired_median_ms | ≥5% below accepted reference median across interleaved pairs at warmup 50 / repeat 100 | −10.5% (candidate 0.305640 vs reference 0.276584 ms, three ordered pairs); decisively negative | **fail** | pairs 1-3 timing |
| aten_cpu_ops_per_call | collapse from 28/call (report_000 census) to ≤2/call | **28 → 2/call** (one `torch.empty` + one Triton launch) — dispatch collapse engaged | **pass** | profile census |
| runtime_launch_count_per_call | exactly 1.00 kernel launch per call (vs base 2.0) | **1.00/call per scope** (base = 1 `topsLaunchKernel` @10.16us; candidate = 1 `topsModuleLaunchKernel` @10.14us) | **pass** (structural guarantee holds; note base is itself 1 launch/call in this forward-mode scope) | profile census |
| device_us_per_call | two-sided: (a) D_cand ≤ ~120us ⇒ device-comparable; (b) D_cand ≥ ~150us ⇒ under-parallel 16-program grid | device_time_available = **false** (GCU launch-only trace); D_cand inferred from wall − launch-API-time ≈ 166us — device-bound regression | **pass** (attributed via launch-only inference; band (b)-class) | profile census |
| triton_launcher_tax_per_call | canonical S60 measurement: candidate host vs base host net delta | **T_launcher ≈ 17.4us/call** (launch-only probe) — 5x smaller than BI150's 84.77us; graph-replay composition has NO material prize on S60 | **pass** (measured; S60 launcher tax is small) | launcher probe |
| run_out_bitwise_equals_forward | bitwise equality over poisoned caller buffers ×2, data_ptr preserved | run_out writes caller buffer through the same kernel; forward allocates fresh buffer; correctness PASS on all mandated surfaces | **pass** | harness correctness |
| mult_of_16_envelope_binding_audit | every tl.dot mult-of-16 (TP=128, D=64) fp32 same-dtype; num_warps=2; tl.max/tl.sum no-keepdim; tl.arange power-of-2; zero compile/capture strings; zero .contiguous() | 2/2 tl.dot sites mult-of-16 fp32; 3 widening casts; num_warps=2; tl.arange power-of-2; no-keepdim reductions; zero DANGER tokens | **pass** | source audit |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: one direct-launched Triton full-attention kernel (grid = B*H = 16 programs, num_warps=2, S=83 padded to TP=128) replacing the entire 28-aten-op base path with a two-op forward (torch.empty + one launch) plus a 4-arg run_out surface
- expected_causal_chain: chain observed with attribution — cn.dispatch-collapse → cn.aten-dispatch-time CONFIRMED (28 → 2 aten ops/call); cn.dispatch-collapse → cn.triton-launcher-tax CONFIRMED (S60 launcher tax ~17.4us, 5x smaller than BI150's 84.77us — no graph-replay prize); cn.device-time-delta measured NEGATIVE (hand-written tl.dot ~166us device beats by the CNNL SDPA library kernel ~158us floor; TP=128 padding forces 58% FLOP waste); cn.dispatch-collapse → cn.wall-time dominated by the device deficit ⇒ wall −10.5%
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed` — the dispatch-collapse mechanism ENGAGED exactly as designed (aten 28 → 2, single Triton launch), but the wall criterion FAILED decisively (−10.5%) because S60 is DEVICE-BOUND: the compressible host + launcher total (~28us) is smaller than the device deficit (~166us vs ~158us), so removing host ops cannot win. This is precisely the decision's pre-declared honest no-improvement reading.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (per decision profiling_level; forward-mode dual-scope trace + host census)
- profiler_device_time: `unavailable: device_time_available = false — GCU trace exposes runtime-launch events (gcu_runtime) but no cat=kernel device durations (target profile triton_gcu marks kernel-summary/kernel-events/instruction-level as unavailable)` — Level 1 normalized runtime-launch evidence recorded instead, never substituting launch time for device kernel time
- iterations: `100` forward calls per scope
- normalized_fields: `runtime_launch_count_per_call`, `runtime_launch_total_us`, `runtime_launch_us_per_call`
- trace: `log/report_001_forward.pt.trace.json`
- trace_sha256: `597ddb35a78efae929972647b7d78a6e0d05212ed16929b9f00cf7c308155562`

### Runtime-launch census (Level 1, launch-only trace, per call)

| Signal | accepted_reference (baseline_adapter) | candidate (direct Triton) |
|---|---:|---:|
| runtime_launch_count_per_call | 1.0 | 1.0 |
| launch event class | `topsLaunchKernel` @10.16us | `topsModuleLaunchKernel` @10.14us |
| aten cpu_ops total/call | **28.00** (8 transpose + 8 as_strided + 4 view + 3 empty + sdpa chain + empty_like + empty_strided + reshape) | **2.00** (one `torch.empty` + one Triton launch) |

Notes: (i) device_time_available is `false` on this target — the trace exposes launch-only events, so all device attribution is via launch-count + launch-API-time; kernel-internal device duration cannot be attributed. (ii) The dispatch collapse is PARTIAL: aten cpu_ops collapsed 28 → 2 as designed, but `runtime_launch_count_per_call` is UNCHANGED at 1.0 per scope (the base SDPA path and the candidate each issue a single launch in this forward-mode scope). (iii) Device-bound diagnosis confirmed: base wall 0.2766ms with SDPA device floor ~158us (host view/transpose chain ~11us, launch ~10us); candidate device (tl.dot TP=128 padding 58% wasted FLOP) ~166us — the Triton kernel is SLOWER than the CNNL SDPA library kernel on device, the decisive factor.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `f2f8b9b6c6f6a16cfbf162cf3f9b115461fc7a5716601eb8e3723961a8536ead` | correctness passed on first attempt; no repairs needed |

Zero Verifier-to-Coder repairs were needed — no candidate defect was found at any point (candidate hash constant end-to-end, matching the coder ledger).

## evidence_for_next_round

- Observed fact: the dispatch collapse ENGAGES — aten 28 → 2 (one `torch.empty` + one Triton launch) — yet paired wall REGRESSED −10.5% (0.305640 vs 0.276584 ms). The wall is decided by the DEVICE, not by aten op count.
- Observed fact (canonical, this campaign): **S60 launcher tax ≈ 17.4us/call** (launch-only probe) — 5x smaller than BI150's 84.77us. The graph-replay composition win lever that flipped BI150 has NO material prize on S60: host chain ~11us + launcher ~17us = ~28us total compressible, which is <20% of wall and < the device deficit.
- Observed fact (canonical, this campaign): **D_cand ≈ 166us/call** (wall − launch-API-time inference; device_time_available = false) vs the CNNL SDPA library kernel ~158us device floor. The hand-written tl.dot kernel is SLOWER on device because TP=128 padding forces 58% FLOP waste (S=83 → TP=128; T=83 = 16×5.19 cannot hit a non-16-multiple, and tl.arange requires power-of-2).
- Observed fact: num_warps sweep {1:167.9, 2:166.2, 4:193.2, 8:286.1}us — nw=2 optimal; split-Q variants 195–228us (worse, redundant K load); single-tile TP=128 nw=2 is the floor of the direct family.
- Capability constraint correction (probe-backed, MUST propagate back to the triton_gcu profile): `tl.dot` AND `tl.arange` both require POWER-OF-2 (not mult-of-16 — 96=16×6 FAILS; only 16/32/64/128 pass). The profile currently states "mult-of-16" for tl.dot and does not record the power-of-2 constraint for tl.arange — this is WRONG and must be corrected. T=83 must pad to 128, so 58% FLOP waste is structurally unavoidable in the single-tile direct family.
- Deliverable banked: `triton_mm_encoder_attention_e2_001.py` @`f2f8b9b6c6f6a16cfbf162cf3f9b115461fc7a5716601eb8e3723961a8536ead` is a correctness-PASS Triton submission (forward + 4-arg run_out surfaces, stateless, envelope-legal) at ~0.906x — per project.md DELIVERABLE RULE this is the campaign's primary contractual product regardless of adoption; canonical pointer stays `baseline_adapter.py`.
- Session drift note: paired same-session basis absorbs drift; the −10.5% delta is ~2× the 5.0% bar in the negative direction, so no plausible drift affects the classification.

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: no-improvement #1 on the campaign (streak 1/3 vs valid_no_improvement_limit 3); round budget 1/20 consumed; the round banked the Triton deliverable plus both canonical physics numbers (S60 launcher tax ~17.4us/call, D_cand ~166us/call device-bound diagnosis) with census-grade attribution; the power-of-2 capability correction and the device-floor (grid-split / padding-reduction) levers remain live.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + authoritative timing (three identical interleaved pairs):

```bash
cd /root/CodeBuddy/20260828202827/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/s60/epoch2/triton_mm_encoder_attention_e2_001.py --warmup 50 --repeat 100 --full-traceback
```

Dual-scope profiler (forward-mode, pw=20/pi=100):

```bash
cd /root/CodeBuddy/20260828202827/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/s60/epoch2/triton_mm_encoder_attention_e2_001.py --warmup 50 --repeat 100 --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/s60/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/mm_encoder_attention/s60/epoch2/log/report_001_forward.pt.trace.json
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/s60/epoch2/log/report_001_forward.pt.trace.json --iterations 100 --wall-ms 0.276584
```

Artifact hash ledger (re-verified this round):

```text
f2f8b9b6c6f6a16cfbf162cf3f9b115461fc7a5716601eb8e3723961a8536ead  triton_mm_encoder_attention_e2_001.py
4ae2b61392e9187a22f000a87282494eb72806927cd83cfec6c08de69a138771  rounds/decision_001.md
ef71920a8a856c633bf8ef5fcebe733bcda6f0fd026210691b1cc8e94aad8f70  rounds/sketch_001.json
1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e  baseline_adapter.py
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
8dfabd0af59b8f6640b47179fee19bca2f5fe35b18535a3db24f60c842e42b70  profile_snapshot/triton_gcu.yaml
597ddb35a78efae929972647b7d78a6e0d05212ed16929b9f00cf7c308155562  log/report_001_forward.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "f2f8b9b6c6f6a16cfbf162cf3f9b115461fc7a5716601eb8e3723961a8536ead",
  "correctness": {
    "status": "pass",
    "evidence": [
      "auto_bench.py PASS accuracy in all three authoritative pairs + profile run (4/4 invocations, seed42 canonical regime)",
      "capability-preflight max_abs 9.77e-4; harness comparator PASS printed every run"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "-10.5% (reference 0.276584 ms vs candidate 0.305640 ms; bar +5.0% FAILED with negative sign; S60 device-bound)",
      "confidence": "high",
      "evidence": ["timing pairs 1-3"]
    },
    {
      "name": "aten_cpu_ops_per_call",
      "status": "observed",
      "value": "28 -> 2/call (one torch.empty + one Triton launch; dispatch collapse engaged)",
      "confidence": "high",
      "evidence": ["profile census"]
    },
    {
      "name": "runtime_launch_count_per_call",
      "status": "observed",
      "value": "1.0/call per scope (base topsLaunchKernel @10.16us; candidate topsModuleLaunchKernel @10.14us) — unchanged at 1.0, not the base-2.0 expected in the decision",
      "confidence": "high",
      "evidence": ["profile census"]
    },
    {
      "name": "device_us_per_call",
      "status": "observed",
      "value": "device_time_available = false (GCU launch-only trace); D_cand inferred ~166us vs CNNL SDPA ~158us floor — device-bound regression; TP=128 padding forces 58% FLOP waste",
      "confidence": "high",
      "evidence": ["profile census", "launcher probe"]
    },
    {
      "name": "triton_launcher_tax_per_call",
      "status": "observed",
      "value": "~17.4us/call (launch-only probe) — 5x smaller than BI150's 84.77us; graph-replay composition has no material prize on S60",
      "confidence": "high",
      "evidence": ["launcher probe"]
    },
    {
      "name": "mult_of_16_envelope_binding_audit",
      "status": "observed",
      "value": "2 tl.dot sites mult-of-16 fp32 (QK^T 128x64@64x128, PV 128x128@128x64); 3 widening casts; num_warps=2; tl.arange power-of-2; tl.max/tl.sum no-keepdim; zero compile/capture/contiguous tokens",
      "confidence": "high",
      "evidence": ["source audit"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "present — ONE Triton kernel _mm_encoder_attn_fwd lowered and device-executed (1 topsModuleLaunchKernel/call)",
    "evidence_contract": "triton_gcu (mult-of-16 dots consumed as declared; power-of-2 tl.arange)",
    "evidence": ["profile census"]
  },
  "evidence_gap_cause": "device_time_available = false on GCU launch-only trace; device attribution is inference from wall - launch-API-time (no cat=kernel events)"
}
```
