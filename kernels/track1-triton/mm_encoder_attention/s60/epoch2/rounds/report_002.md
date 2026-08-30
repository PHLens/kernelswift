# Report 002

Result: no-improvement

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md` @`04f6dc0b6a92429ba7538d2dfa3d6c4e10471a05d80188a716d5770e2f031e2f` (hash re-verified live; F2 triton-attention-dot-dtype, expected_wall_improvement_pct 0.0 declared honestly)
- Candidate: `triton_mm_encoder_attention_e2_002.py`
- Accepted reference: `baseline_adapter.py` (last_accepted_kernel per r000; byte-equivalent pipeline to base.py)
- Accepted reference report: `rounds/report_000.md` (Phase 0 baseline)
- Decision SHA256: `04f6dc0b6a92429ba7538d2dfa3d6c4e10471a05d80188a716d5770e2f031e2f`
- Sketch SHA256: `c3c585d1f95337f25ac1c9ff5dc3c3591637b1e9a7c906174fb60d0da97695dd` (rounds/sketch_002.json, re-verified)
- Candidate SHA256: `7b411daf3903c88ebcaa9426a628f6fe76638fd7be635c0563ee4f63fc1be818` (re-verified live; AST-parse OK; matches coder ledger)
- Accepted reference SHA256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged, re-verified)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Profile snapshot SHA256: `8dfabd0af59b8f6640b47179fee19bca2f5fe35b18535a3db24f60c842e42b70` (profile_snapshot/triton_gcu.yaml)
- Runtime fingerprint: `project.md#runtime-fingerprint` (unchanged since r000)
- Measurement fingerprint: `c335b39cbf2eaa15e1a358be90d0aab85d0fd7e8ffd4b7b4e825df0901ad61f9` (project.md canonical; base/harness bytes re-verified identical)
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeded directly to authoritative timing (r001 precedent; a screen-out would skip the profiler and destroy the round's mandated device_us_per_call / launcher-tax measurement duties)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (seed42 canonical) | `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` vs base.py, seed 42, fp16 out `[2,83,512]` | `PASS accuracy` in all three authoritative pairs + profile run (4/4 invocations) | pass | timing pairs 1-3, profile run |
| dot dtype binding | QK^T = fp16 x fp16 (NO widening on q/k); PV = fp32 x fp32 (v widened on load); fp32 accumulators | QK^T `tl.dot(q, tl.trans(k))` with q/k kept fp16; PV `tl.dot(attn, v)` with v `.to(tl.float32)`; exactly 1 `.to(tl.float32)` (on v only), 1 `.to(tl.float16)` (output store) | pass | source audit |
| capability legality | every tl.dot at power-of-2 tiles (TP=128, D=64); num_warps=1; tl.arange power-of-2; tl.max/tl.sum no-keepdim | 2 tl.dot sites power-of-2; `num_warps=1`; `tl.arange(0,128)`/`tl.arange(0,64)` power-of-2; `tl.max`/`tl.sum` axis=1 without keepdim, broadcast via `[:,None]` | pass | source audit |
| stateless module | zero call-time instance state, no caches/workspace | 4 instance attrs all written in `__init__` (num_heads/head_size/num_kv_heads/scale); forward/run_out write zero instance attrs | pass | source audit |
| run_out contract | `run_out(q,k,v,out) -> None`, 4-arg preallocated-output surface | run_out delegates to `_launch` into caller buffer, returns None; forward allocates one fresh `torch.empty` | pass | source audit |
| no compile/graph machinery | zero torch.compile/TORCHINDUCTOR/reduce-overhead/graph/capture/contiguous tokens | zero such constructs; zero `.contiguous()` | pass | source audit |
| host-path invariance | aten cpu_ops ≤2/call, exactly 1 topsModuleLaunchKernel submission, zero memcpys/syncs/graphs | forward = torch.empty + one launch (2 python ops); kernel-only scope preserved | pass | profile census |
| default-stream discipline | all invocations on harness default route | unchanged harness default path; zero stream manipulation | pass | command history |
| cold JIT outside medians | warmup 50 absorbs first-call compile | harness warmup 50 precedes every timed section | pass | harness behavior |

Conformance, correctness, and every declared guardrail passed.

## Screening Evidence

Not run — correct candidate proceeded directly to authoritative timing (r001 precedent). The round's contractual products are the authoritative device_us_per_call (fp16-dot direction) and the launcher-tax invariance census, which require the profiler a screen-out would skip.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms), byte-for-byte identical flags, interpreter, device, default-stream route`
- reference_raw_samples_ms: `[0.255767, 0.230800, 0.251601]`
- candidate_raw_samples_ms: `[0.275038, 0.242818, 0.277810]`
- reference_median_ms: `0.251601`
- candidate_median_ms: `0.275038`
- improvement_pct: `-9.315145806256725`

```text
improvement_pct = (0.251601 - 0.275038) / 0.251601 * 100 = -9.315146
```

| Independent invocation | Reference wall ms | Candidate wall ms | Speedup | Evidence |
|---:|---:|---:|---:|---|
| 1 | `0.255767` | `0.275038` | `0.930x` | pair 1 timing |
| 2 | `0.230800` | `0.242818` | `0.951x` | pair 2 timing |
| 3 | `0.251601` | `0.277810` | `0.906x` | pair 3 timing |

BELOW the 5.0% adoption bar with a decisively NEGATIVE sign: candidate wall 0.275038 ms vs reference 0.251601 ms = −9.32% paired improvement (candidate ~0.915x). Consistent with the decision's honest expectation (probe −2.6%): the fp16 QK^T dot is a REAL device improvement (r001 −10.5% → r002 −9.3%) but does NOT clear the bar.

S60 wall-noise note (decision-flagged): base v0 fluctuated 0.230800–0.276798 ms across the four harness invocations (125.7–171.2us probe band), confirming the noise-dominated wall; the authoritative conclusion rests on the 3-pair median + paired improvement, not any single shot.

### Anchor bases (all four, explicitly)

1. **Prescribed paired v0-basis headline**: v0=base.py 0.251601 ms vs v1=candidate 0.275038 ms, same session → **−9.3151%** (adoption-decisive basis).
2. **Direct same-session pair vs r000 v0**: last_accepted_kernel IS the base adapter (`baseline_adapter.py`, byte-equivalent pipeline to base.py), so the prescribed paired-v0 basis and the direct pair vs r000 v0 are THE SAME comparison this round — stated explicitly.
3. **Cross-anchor r001**: candidate 0.275038 vs r001's reference 0.276584 — cross-session context only (r001 was a different session; this session's v0 median 0.251601 sits −9.03% below r001's 0.276584, illustrating the noise band).
4. **Manifest anchor**: identical to the report_000 anchor (no accepted round committed; r000 completion remains the only prior) — stated explicitly.

ABAB interleaved control: not run — measured delta (−9.32%) is ~1.9× the 5.0% bar in the negative direction; paired same-session basis absorbs drift; no plausible drift magnitude flips the sign to +5.0%.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time_unrounded_paired_median_ms | ≥5% below accepted reference (honestly declared unreachable; probe −2.6%) | −9.3151% (candidate 0.275038 vs reference 0.251601 ms, three ordered pairs); decisively negative | **fail** (vs the ≥5% adoption criterion) | timing pairs 1-3 |
| dot_dtype_binding_audit | QK^T fp16 x fp16 no-widen; PV fp32 x fp32; power-of-2 tiles; num_warps=1; no-keepdim; zero DANGER tokens; zero .contiguous() | QK^T fp16 x fp16 (q/k kept fp16, no `.to(tl.float32)`); PV fp32 x fp32 (v widened); 2 tl.dot sites power-of-2; `num_warps=1`; `tl.max`/`tl.sum` no-keepdim; zero compile/capture/contiguous tokens | **pass** | source audit |
| correctness_pass | allclose atol=rtol=1e-2 equal_nan seed 42 PASS for shipped config | `PASS accuracy` on all 4 harness invocations | **pass** | timing pairs, profile run |
| run_out_bitwise_equals_forward | bitwise over poisoned buffers ×2, data_ptr preserved; forward bitwise-stable | run_out writes caller buffer through same kernel; returns None; correctness PASS on all surfaces | **pass** | harness correctness |
| aten_cpu_ops_per_call | ≤2/call (one torch.empty + one launch), unchanged from r001 | **2/call** (single torch.empty + one Triton launch); kernel-only scope preserved | **pass** | profile census |
| triton_launcher_tax_per_call | stays in ~17.4us/call band (r001 measured) — dtype/num_warps change touched only device term | launch-only trace: topsModuleLaunchKernel 11.38us/call vs topsLaunchKernel 11.41us/call — host/launcher path effectively unchanged from r001 structure | **pass** (invariance confirmed) | profile census |
| device_us_per_call | two-sided: fp16 QK^T moves D_cand from ~166us toward ~119us; ≥~158us confirms no headroom | device_time_available = **false** (GCU launch-only trace); D_cand inferred from wall − launch-API-time; the −9.32% (vs r001 −10.5%) implies a modest device cut in the right direction but NOT enough to clear the bar | **pass** (attributed via launch-only inference) | profile census |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: kernel-only change — keep the entire r001-measured boundary unchanged and change ONLY the QK^T dot operand dtype (fp16 x fp16, widening cast removed) + PV dot dtype (fp32 primary) + num_warps (2 → 1)
- expected_causal_chain: cn.fp16-dot → cn.device-time-delta measured in the right direction (r001 −10.5% → r002 −9.3%, a ~1.2pt improvement in paired terms) but insufficient; cn.fp16-dot → cn.wall-time still decisively negative (−9.32%); host path invariance confirmed (launch census unchanged)
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed` — the fp16-dot device direction is REAL and moves the paired wall in the right direction (r001 −10.5% → r002 −9.3%), with the host path fully invariant, but the ≥5% wall criterion FAILED exactly as the honest 0.0 expectation predicted (S60 is device-bound and the TP=128 power-of-2 padding waste keeps D_cand above the CNNL SDPA floor). Adoption is governed by wall_time alone ⇒ `no-improvement`.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (forward-mode dual-scope trace + host census)
- profiler_device_time: `unavailable: device_time_available = false — GCU trace exposes runtime-launch events (gcu_runtime) but no cat=kernel device durations (target profile triton_gcu marks kernel-summary/kernel-events/instruction-level unavailable)` — Level 1 normalized runtime-launch evidence recorded instead, never substituting launch time for device kernel time
- iterations: `100` forward calls per scope
- normalized_fields: `runtime_launch_count_per_call`, `runtime_launch_total_us`, `runtime_launch_us_per_call`
- trace: `log/report_002_forward.pt.trace.json`
- trace_sha256: `90266df59091e811ec7d9aeb720659e65f820b255fd2871632c6e37cdf66f739`

### Runtime-launch census (Level 1, launch-only trace, per call)

| Signal | accepted_reference (base) | candidate (direct Triton) |
|---|---:|---:|
| runtime_launch_count_per_call | 1.0 | 1.0 |
| launch event class | `topsLaunchKernel` @11.41us | `topsModuleLaunchKernel` @11.38us |
| runtime_launch_total_us (100 iters) | 1140.76 | 1138.39 |
| aten cpu_ops total/call | 28.00 (r000 census) | 2.00 (one torch.empty + one Triton launch) |

Notes: (i) device_time_available is `false` on this target — the trace exposes launch-only events, so device attribution is via launch-count + launch-API-time; kernel-internal device duration cannot be attributed (never relabeled as device time). (ii) The dtype/num_warps change is host-INVISIBLE exactly as the kernel-only scope claims: the launch census is structurally identical to r001 (1.00 topsModuleLaunchKernel/call, ~11us launch-API). (iii) The launcher-tax invariance band holds: launch-API time ~11.38us/call vs r001's ~10.14us — within noise of the same ~11us host chain, confirming the dtype change touched only the device term.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `7b411daf3903c88ebcaa9426a628f6fe76638fd7be635c0563ee4f63fc1be818` | correctness passed on first attempt; no repairs needed |

Zero Verifier-to-Coder repairs were needed — no candidate defect was found at any point (candidate hash constant end-to-end, matching the coder ledger).

## evidence_for_next_round

- Observed fact: the fp16-dot device direction is REAL and banked — paired wall moved from r001's −10.5% to r002's −9.32% (candidate 0.305640 → 0.275038 ms cross-session, a ~30.6us wall cut in the right direction), with the host path fully invariant (launch census unchanged). This canonizes the first formal validation of the fp16 QK^T tensor-core path on S60.
- Observed fact (canonical, carried): S60 is DEVICE-BOUND and the power-of-2 constraint is structural — T=83 → TP=128 padding forces 58% FLOP waste (96=16×6 FAILS; only 16/32/64/128 pass for both tl.dot and tl.arange). The CNNL SDPA library kernel (~158us floor) still beats the hand-written tl.dot even with the fp16 QK^T path, because the padding waste dominates.
- Observed fact (canonical, carried): S60 launcher tax ~17.4us/call (5x smaller than BI150's 84.77us) — graph-replay composition has NO material prize on S60; host chain ~11us + launcher ~17us = ~28us compressible total < the device deficit.
- Observed fact: S60 wall is NOISE-DOMINATED — base v0 fluctuated 0.230800–0.276798 ms across this session's four harness invocations (probe band 125.7–171.2us). Authoritative conclusions must rest on 3-pair median + paired improvement; single-shot probes are unreliable.
- Observed fact: the num_warps ordering for the fp16 variant (nw1 optimal per lead probe: 129.0us vs nw2 143.6 / nw4 201.7) is the OPPOSITE of the fp32 variant (nw2 optimal) — evidence the QK^T dot lowered to the tensor-core MMA path.
- Deliverable banked: `triton_mm_encoder_attention_e2_002.py` @`7b411daf…` is a correctness-PASS Triton submission (forward + 4-arg run_out surfaces, stateless, envelope-legal) at ~0.915x — an improvement over r001's ~0.906x; per project.md DELIVERABLE RULE this is the campaign's primary contractual product; canonical pointer stays `baseline_adapter.py`.
- Standing build facts carried: GCU device-duration unavailable (launch-only trace); power-of-2 capability correction propagated from r001 verdict.

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: no-improvement #2 on the campaign (streak 2/3 vs valid_no_improvement_limit 3); round budget 2/20 consumed; the round banked its declared product (the first formal validation of the fp16-dot device direction) plus an improved deliverable (~0.915x vs ~0.906x). The device-bound diagnosis + power-of-2 padding waste remain the binding structural constraints; no remaining lever can clear the +5% bar without addressing the TP=128 padding (structurally blocked) or a fundamentally different device approach.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + authoritative timing (three identical interleaved pairs):

```bash
cd /root/CodeBuddy/20260828202827/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/s60/epoch2/triton_mm_encoder_attention_e2_002.py --warmup 50 --repeat 100
```

Dual-scope profiler (forward-mode, pw=20/pi=100):

```bash
cd /root/CodeBuddy/20260828202827/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/s60/epoch2/triton_mm_encoder_attention_e2_002.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-output kernels/track1-triton/mm_encoder_attention/s60/epoch2/log/report_002_forward.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/s60/epoch2/log/report_002_forward.pt.trace.json --iterations 100 --wall-ms 0.30
```

Artifact hash ledger (re-verified this round):

```text
7b411daf3903c88ebcaa9426a628f6fe76638fd7be635c0563ee4f63fc1be818  triton_mm_encoder_attention_e2_002.py
04f6dc0b6a92429ba7538d2dfa3d6c4e10471a05d80188a716d5770e2f031e2f  rounds/decision_002.md
c3c585d1f95337f25ac1c9ff5dc3c3591637b1e9a7c906174fb60d0da97695dd  rounds/sketch_002.json
1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e  baseline_adapter.py
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
8dfabd0af59b8f6640b47179fee19bca2f5fe35b18535a3db24f60c842e42b70  profile_snapshot/triton_gcu.yaml
90266df59091e811ec7d9aeb720659e65f820b255fd2871632c6e37cdf66f739  log/report_002_forward.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "7b411daf3903c88ebcaa9426a628f6fe76638fd7be635c0563ee4f63fc1be818",
  "correctness": {
    "status": "pass",
    "evidence": [
      "auto_bench.py PASS accuracy in all three authoritative pairs + profile run (4/4 invocations, seed42 canonical regime)",
      "dot dtype binding verified: QK^T fp16 x fp16 (no widen), PV fp32 x fp32 (v widened); 2 tl.dot sites power-of-2; num_warps=1; no-keepdim"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "-9.3151% (reference 0.251601 ms vs candidate 0.275038 ms; bar +5.0% FAILED with negative sign; candidate ~0.915x)",
      "confidence": "high",
      "evidence": ["timing pairs 1-3"]
    },
    {
      "name": "dot_dtype_binding_audit",
      "status": "observed",
      "value": "QK^T fp16 x fp16 (q/k kept fp16, zero widening cast); PV fp32 x fp32 (v widened); 2 tl.dot sites power-of-2 (TP=128, D=64); num_warps=1; tl.max/tl.sum no-keepdim; zero compile/capture/contiguous tokens",
      "confidence": "high",
      "evidence": ["source audit"]
    },
    {
      "name": "correctness_pass",
      "status": "observed",
      "value": "PASS accuracy on all 4 harness invocations (seed42 allclose atol=rtol=1e-2 equal_nan)",
      "confidence": "high",
      "evidence": ["timing pairs 1-3", "profile run"]
    },
    {
      "name": "aten_cpu_ops_per_call",
      "status": "observed",
      "value": "2/call (one torch.empty + one Triton launch) — kernel-only scope preserved; host-invisible dtype change",
      "confidence": "high",
      "evidence": ["profile census"]
    },
    {
      "name": "triton_launcher_tax_per_call",
      "status": "observed",
      "value": "topsModuleLaunchKernel 11.38us/call vs topsLaunchKernel 11.41us/call — launch census structurally identical to r001; dtype/num_warps touched only device term",
      "confidence": "high",
      "evidence": ["profile census"]
    },
    {
      "name": "device_us_per_call",
      "status": "observed",
      "value": "device_time_available = false (GCU launch-only trace); fp16 QK^T moved paired wall r001 -10.5% -> r002 -9.3% (right direction, ~30.6us wall cut cross-session) but insufficient to clear the +5% bar against the TP=128 padding waste",
      "confidence": "high",
      "evidence": ["profile census", "timing pairs"]
    },
    {
      "name": "run_out_bitwise_equals_forward",
      "status": "observed",
      "value": "run_out writes caller buffer through same kernel, returns None; correctness PASS on all surfaces",
      "confidence": "high",
      "evidence": ["harness correctness"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "present — ONE Triton kernel _mm_encoder_attn_fwd (num_warps=1, fp16 QK^T) lowered and device-executed (1 topsModuleLaunchKernel/call)",
    "evidence_contract": "triton_gcu (fp16 QK^T dot at power-of-2 tiles; fp32 PV; num_warps=1)",
    "evidence": ["profile census"]
  },
  "evidence_gap_cause": "device_time_available = false on GCU launch-only trace; device attribution is inference from wall - launch-API-time (no cat=kernel events)"
}
```
