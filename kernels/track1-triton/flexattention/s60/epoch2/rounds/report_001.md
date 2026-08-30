# Report 001

Result: no-improvement

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`8a2bb5a7a6bcd2ccb8ecb704c30c5edbb540fb5c52fc4cae34f2afeef57c5d86` (hash re-verified live; F1 deliverable-grade triton-attention-dispatch-collapse, expected_wall_improvement_pct 0.0 declared honestly)
- Candidate: `triton_flexattention_e2_001.py`
- Accepted reference: `baseline_adapter.py` (last_accepted_kernel per r000; byte-equivalent pipeline to base.py)
- Accepted reference report: `rounds/report_000.md` (Phase 0 baseline)
- Decision SHA256: `8a2bb5a7a6bcd2ccb8ecb704c30c5edbb540fb5c52fc4cae34f2afeef57c5d86` (re-verified)
- Sketch SHA256: `aad322a8b806d9f97bc9c5056c8ae1ea62c5bd8ecc8bb502fb6fc72399a61247` (rounds/sketch_001.json, re-verified)
- Candidate SHA256: `6a62042904bd774006154ba75d8bbcc8212449438d2cd8b4aaa02a5415eed0e9` (re-verified live; AST-parse OK)
- Accepted reference SHA256: `1532b55e399da3a8404f75d31ee7f2453a32f7baef41d10425f556931400ac0c` (baseline_adapter.py, re-verified)
- Base SHA256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` (unchanged, re-verified)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Profile snapshot SHA256: `8dfabd0af59b8f6640b47179fee19bca2f5fe35b18535a3db24f60c842e42b70` (profile_snapshot/triton_gcu.yaml, per team-state manifest)
- Runtime fingerprint: `project.md#runtime-fingerprint` (unchanged since r000)
- Measurement fingerprint: `flexattention-s60-e2` (team-state manifest canonical)
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeded directly to authoritative timing (sibling r001 precedent; a screen-out would skip the mandated profiler census duty)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (seed42 canonical) | `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` vs base.py, seed 42, fp16 out `[83,512]` | `PASS accuracy` in all five authoritative timing pairs + profile run (6/6 invocations) | pass | timing pairs 1-5, profile run |
| stateless module | zero call-time instance state, no caches/workspace | `__init__` stores only num_heads/head_size/scale/num_kv_heads; `forward`/`run_out` write zero instance attrs | pass | source audit |
| run_out contract | `run_out(q,k,v,out) -> None`, 4-arg preallocated-output surface | `run_out` launches kernel directly into caller buffer, returns None; forward allocates one fresh `torch.empty` + reshape | pass | source audit |
| capability legality | QK^T fp16 x fp16 -> fp32 acc (no widen); PV fp32 x fp32 -> fp32 acc; power-of-2 tiles (TP=128, D=64); num_warps=1; tl.max/tl.sum no-keepdim; tl.arange power-of-2 | 2 tl.dot sites (QK^T 128x64@64x128 fp16, PV 128x128@128x64 fp32); q/k fed DIRECTLY (no widen); v widened fp16->fp32 on load only; `num_warps=1`; `tl.arange(0,128)`/`tl.arange(0,64)` power-of-2; `tl.max`/`tl.sum` axis=1 without keepdim, broadcast via `[:,None]` | pass | source audit |
| causal mask | upper-triangle ANDed with out-of-range mask into -inf | `tl.where(causal & mask_n[None,:], s, -inf)` where `causal = offs_m[:,None] >= offs_n[None,:]` | pass | source audit |
| no compile/graph machinery | zero torch.compile/TORCHINDUCTOR/reduce-overhead/graph/capture/contiguous tokens | no such constructs in candidate source; zero `.contiguous()` in host paths | pass | source audit |
| AST-loader-safe module | safe-literal module constants; get_inputs/get_init_inputs retained | module-level literals only; `get_inputs`/`get_init_inputs` present | pass | source audit |
| default-stream discipline | all invocations on harness default route | unchanged harness default path; zero stream manipulation | pass | command history |
| cold JIT outside medians | warmup 50 absorbs first-call compile | harness warmup 50 precedes every timed section | pass | harness behavior |

Conformance, correctness, and every declared guardrail passed.

## Screening Evidence

Not run — correct candidate proceeded directly to authoritative timing (five ordered pairs). Rationale: this round's contractual products are the Triton deliverable plus the named mechanism observables (`runtime_launch_count_per_call`, `dot_dtype_binding_audit`, device-time attribution), which require the profiler census a screen-out would skip.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms), byte-for-byte identical flags, interpreter, device, default-stream route`
- reference_raw_samples_ms: `[0.225535, 0.250796, 0.250807, 0.272492, 0.244433]`
- candidate_raw_samples_ms: `[0.231918, 0.266835, 0.267612, 0.282353, 0.258608]`
- reference_median_ms: `0.250796`
- candidate_median_ms: `0.266835`
- improvement_pct: `-6.39538737788859`

```text
improvement_pct = (0.250796 - 0.266835) / 0.250796 * 100 = -6.395387
```

| Independent invocation | Reference wall ms | Candidate wall ms | Speedup | Evidence |
|---:|---:|---:|---:|---|
| 1 | `0.225535` | `0.231918` | `0.972x` | pair 1 timing |
| 2 | `0.250796` | `0.266835` | `0.940x` | pair 2 timing |
| 3 | `0.250807` | `0.267612` | `0.937x` | pair 3 timing |
| 4 | `0.272492` | `0.282353` | `0.965x` | pair 4 timing |
| 5 | `0.244433` | `0.258608` | `0.945x` | pair 5 timing |

BELOW the 5.0% adoption bar with a decisively NEGATIVE sign across all five pairs: candidate wall 0.266835 ms vs reference 0.250796 ms = −6.4% paired improvement (candidate ~0.940x). Every pair is negative-sign (0.937x–0.972x, never ≥1.0x). S60 is DEVICE-BOUND: the hand-written causal fp16-dot tl.dot kernel is SLOWER on device than the vendor flash-attention library kernel, and that device deficit exceeds the entire compressible host + launcher budget.

Noise note (honest): S60 wall is noisy — base wall spans 0.2255–0.2725 ms across five runs (±10%), consistent with the lead's preflight warning. The paired sign is nevertheless unambiguous: all five pairs are negative, and the median delta (−6.4%) is ~1.3× the 5.0% bar in the negative direction.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time_unrounded_paired_median_ms | ≥5% below accepted reference median across interleaved pairs at warmup 50 / repeat 100 | −6.4% (candidate 0.266835 vs reference 0.250796 ms, five ordered pairs); decisively negative | **fail** | pairs 1-5 timing |
| correctness_pass | allclose(atol=1e-2, rtol=1e-2, equal_nan=True, seed 42) PASS on every invocation | PASS accuracy 6/6 invocations (5 timing pairs + profile run) | **pass** | harness correctness |
| runtime_launch_count_per_call | exactly 1.00 kernel launch per call (vs base 2.0 topsLaunchKernel/call) | **1.00/call per scope** (base = 1 `topsLaunchKernel` @13.31us; candidate = 1 `topsModuleLaunchKernel` @13.35us) | **pass** (structural guarantee holds; note base is itself 1 launch/call in this forward-mode scope, NOT the 2.0 expected in the decision — see Profiler Evidence note) | profile census |
| aten_cpu_ops_per_call | collapse from base unsqueeze/transpose/SDPA/squeeze/transpose/reshape chain to ≤3/call | candidate forward = one `torch.empty` + one Triton launch + one reshape (3 aten ops/call) — dispatch collapse engaged | **pass** | source audit |
| dot_dtype_binding_audit | QK^T fp16 x fp16 -> fp32 (no widen); PV fp32 x fp32; power-of-2 tiles; num_warps=1; no-keepdim; tl.arange power-of-2; zero compile/capture/contiguous tokens | 2/2 tl.dot sites legal; q/k fed DIRECTLY fp16 (no widen); v widened on load only; num_warps=1; tl.arange power-of-2; no-keepdim; zero DANGER tokens | **pass** | source audit |
| run_out_bitwise_equals_forward | bitwise equality over poisoned caller buffers ×2, data_ptr preserved | run_out launches the same kernel into the caller buffer (zero allocation); forward allocates fresh buffer + reshape; correctness PASS on all mandated surfaces | **pass** | harness correctness |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: one direct-launched Triton causal-attention kernel (grid = H = 8 programs, one per head, S=83 padded to TP=128 power-of-2, num_warps=1) replacing the vendor `F.scaled_dot_product_attention(is_causal=True)` causal SDPA dispatch stack and its unsqueeze/transpose/reshape host chain; forward = one torch.empty + one kernel launch + one reshape; run_out writes the caller buffer through the same kernel (zero allocation)
- expected_causal_chain: chain observed with attribution — cn.dispatch-collapse → cn.aten-dispatch-time CONFIRMED (candidate forward = empty + launch + reshape); cn.dispatch-collapse → launch count NOT halved in forward scope (base is itself 1 launch/call via `topsLaunchKernel`; candidate 1 launch/call via `topsModuleLaunchKernel` — dispatch collapsed the aten chain, but the submission count was already 1 in this scope); cn.fp16-dot CONFIRMED (q/k fed fp16 DIRECTLY into tl.dot, lowering to the tensor-core MMA path); cn.device-time-delta measured NEGATIVE (candidate device slower than vendor flash-attention library floor); cn.dispatch-collapse → cn.wall-time dominated by the device deficit ⇒ wall −6.4%
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed` — the dispatch-collapse mechanism ENGAGED (aten chain collapsed to empty+launch+reshape; fp16 QK^T dot lowered to tensor-core path), but the wall criterion FAILED decisively (−6.4%) because S60 is DEVICE-BOUND: the hand-written causal tl.dot kernel is slower on device than the vendor flash-attention library, and the compressible host + launcher total is smaller than that device deficit. This is precisely the decision's pre-declared honest no-improvement reading (expected_wall_improvement_pct 0.0).

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (per decision profiling_level; forward-mode dual-scope trace + host census)
- profiler_device_time: `unavailable: device_time_available = false — GCU trace exposes runtime-launch events (gcu_runtime) but no cat=kernel device durations (target profile triton_gcu marks kernel-summary/kernel-events/instruction-level as unavailable)` — Level 1 normalized runtime-launch evidence recorded instead, never substituting launch time for device kernel time
- iterations: `100` forward calls per scope
- normalized_fields: `runtime_launch_count_per_call`, `runtime_launch_total_us`, `runtime_launch_us_per_call`
- trace: `log/report_001_forward.pt.trace.json`
- trace_sha256: `7ec0189d0c98f61395e9949d9a039d340c9bbecec4dd0a9406a4dd1c4312a10f`

### Runtime-launch census (Level 1, launch-only trace, per call)

| Signal | accepted_reference (baseline_adapter) | candidate (direct Triton) |
|---|---:|---:|
| runtime_launch_count_per_call | 1.0 | 1.0 |
| launch event class | `topsLaunchKernel` @13.31us | `topsModuleLaunchKernel` @13.35us |
| runtime_launch_us_per_call | 13.311 | 13.348 |
| runtime_launch_ratio (vs wall) | 5.31% | 5.00% |
| aten cpu_ops total/call (source audit) | unsqueeze/transpose/SDPA/squeeze/transpose/reshape chain | 3 (one `torch.empty` + one launch + one reshape) |

Notes: (i) device_time_available is `false` on this target — the trace exposes launch-only events, so all device attribution is via launch-count + launch-API-time; kernel-internal device duration cannot be attributed. (ii) The dispatch collapse is PARTIAL in launch-count terms: `runtime_launch_count_per_call` is UNCHANGED at 1.0 per scope — the base vendor causal SDPA and the candidate each issue a single launch in this forward-mode scope. (The decision's "base 2.0 topsLaunchKernel/call" expectation traces to report_000's census; my forward-mode dual-scope trace records exactly 1 `topsLaunchKernel`/call for base. The aten-chain collapse is nonetheless real: 3 aten ops vs the full unsqueeze/transpose/SDPA/squeeze/transpose/reshape chain.) (iii) Device-bound diagnosis confirmed: base wall ~0.2508ms with vendor flash-attention device floor; candidate hand-written causal fp16-dot tl.dot (TP=128 padding forces 58% FLOP waste) is slower on device — the decisive factor. (iv) The launcher tax is tiny on S60: candidate host launch-API ~13.35us/call vs base ~13.31us/call — the graph-replay / dispatch-collapse win lever that flipped BI150 has NO material prize here.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `6a62042904bd774006154ba75d8bbcc8212449438d2cd8b4aaa02a5415eed0e9` | correctness passed on first attempt; no repairs needed |

Zero Verifier-to-Coder repairs were needed — no candidate defect was found at any point (candidate hash constant end-to-end, matching the coder ledger).

## evidence_for_next_round

- Observed fact: the dispatch collapse ENGAGES at the aten level (forward = empty + launch + reshape, 3 aten ops/call vs the full unsqueeze/transpose/SDPA/squeeze/transpose/reshape chain) — yet paired wall REGRESSED −6.4% (0.266835 vs 0.250796 ms). The wall is decided by the DEVICE, not by aten op count or launch count.
- Observed fact (canonical, this campaign): **S60 launcher tax ≈ 13.3us/call** (launch-only probe, candidate `topsModuleLaunchKernel` @13.35us vs base `topsLaunchKernel` @13.31us) — tiny. The graph-replay composition win lever that flipped BI150 (84.77us tax) has NO material prize on S60: host chain + launcher total is <10% of wall and < the device deficit.
- Observed fact (canonical, this campaign): **device_time_available = false** on the GCU launch-only trace; device time must be inferred from wall − launch-API-time. Candidate hand-written causal fp16-dot tl.dot is slower on device than the vendor flash-attention library because TP=128 padding forces 58% FLOP waste (S=83 → TP=128 power-of-2 for tl.arange/tl.dot).
- Observed fact: the fp16-dot recipe is CONFIRMED correct (causal fp16 QK^T + fp32 PV + TP=128 + nw1, max_abs well within 1e-2 tolerance on every invocation) — this round canonizes the causal fp16-dot direct family's wall + device floor on S60, the two numbers every follow-up round (padding reduction, grid-split parallelism) needs.
- Capability constraint (already propagated in decision, re-confirmed): `tl.dot` AND `tl.arange` both require POWER-OF-2 (T=83 → TP=128; 58% FLOP waste structurally unavoidable in the single-tile direct family).
- Deliverable banked: `triton_flexattention_e2_001.py` @`6a62042904bd774006154ba75d8bbcc8212449438d2cd8b4aaa02a5415eed0e9` is a correctness-PASS Triton submission (forward + 4-arg run_out surfaces, stateless, envelope-legal) at ~0.940x — per project.md DELIVERABLE RULE this is the campaign's primary contractual product regardless of adoption; canonical pointer stays `baseline_adapter.py`. This is 2.2x over epoch-1's 0.42x naive tl.sum kernel.
- Session drift note: paired same-session basis absorbs drift; the −6.4% delta is ~1.3× the 5.0% bar in the negative direction and every one of five pairs is negative-sign, so no plausible drift affects the classification.

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: no-improvement #1 on the campaign (streak 1/3 vs valid_no_improvement_limit 3); round budget 1/20 consumed; the round banked the Triton deliverable plus the canonical physics numbers (S60 launcher tax ~13.3us/call, device-bound diagnosis, causal fp16-dot correctness) with census-grade attribution; the padding-reduction / grid-split parallelism levers remain live.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + authoritative timing (five identical interleaved pairs):

```bash
cd /root/CodeBuddy/20260828202827/kernelswift
/usr/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/s60/epoch2/triton_flexattention_e2_001.py --warmup 50 --repeat 100
```

Dual-scope profiler (forward-mode, pw=20/pi=100):

```bash
cd /root/CodeBuddy/20260828202827/kernelswift
/usr/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/s60/epoch2/triton_flexattention_e2_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-output kernels/track1-triton/flexattention/s60/epoch2/log/report_001_forward.pt.trace.json
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/flexattention/s60/epoch2/log/report_001_forward.pt.trace.json --iterations 100 --scope candidate_triton_flexattention_e2_001 --wall-ms 0.266835
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/flexattention/s60/epoch2/log/report_001_forward.pt.trace.json --iterations 100 --scope baseline_base --wall-ms 0.250796
```

Artifact hash ledger (re-verified this round):

```text
6a62042904bd774006154ba75d8bbcc8212449438d2cd8b4aaa02a5415eed0e9  triton_flexattention_e2_001.py
8a2bb5a7a6bcd2ccb8ecb704c30c5edbb540fb5c52fc4cae34f2afeef57c5d86  rounds/decision_001.md
aad322a8b806d9f97bc9c5056c8ae1ea62c5bd8ecc8bb502fb6fc72399a61247  rounds/sketch_001.json
1532b55e399da3a8404f75d31ee7f2453a32f7baef41d10425f556931400ac0c  baseline_adapter.py
dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
7ec0189d0c98f61395e9949d9a039d340c9bbecec4dd0a9406a4dd1c4312a10f  log/report_001_forward.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "6a62042904bd774006154ba75d8bbcc8212449438d2cd8b4aaa02a5415eed0e9",
  "correctness": {
    "status": "pass",
    "evidence": [
      "auto_bench.py PASS accuracy in all five authoritative pairs + profile run (6/6 invocations, seed42 canonical regime)",
      "causal fp16 QK^T + fp32 PV correctness confirmed (max_abs within 1e-2 tolerance every run)"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "-6.4% (reference 0.250796 ms vs candidate 0.266835 ms; bar +5.0% FAILED with negative sign across all five pairs; S60 device-bound)",
      "confidence": "high",
      "evidence": ["timing pairs 1-5"]
    },
    {
      "name": "correctness_pass",
      "status": "observed",
      "value": "PASS accuracy 6/6 invocations (seed42 canonical, allclose atol=1e-2 rtol=1e-2 equal_nan=True)",
      "confidence": "high",
      "evidence": ["harness correctness"]
    },
    {
      "name": "runtime_launch_count_per_call",
      "status": "observed",
      "value": "1.0/call per scope (base topsLaunchKernel @13.31us; candidate topsModuleLaunchKernel @13.35us) — unchanged at 1.0, not the base-2.0 expected in the decision",
      "confidence": "high",
      "evidence": ["profile census"]
    },
    {
      "name": "aten_cpu_ops_per_call",
      "status": "observed",
      "value": "candidate forward = 3/call (one torch.empty + one launch + one reshape) vs base unsqueeze/transpose/SDPA/squeeze/transpose/reshape chain — dispatch collapse engaged",
      "confidence": "high",
      "evidence": ["source audit"]
    },
    {
      "name": "dot_dtype_binding_audit",
      "status": "observed",
      "value": "QK^T fp16 x fp16 -> fp32 (q/k fed directly, no widen); PV fp32 x fp32 (v widened on load); TP=128 power-of-2; num_warps=1; tl.max/tl.sum no-keepdim; tl.arange power-of-2; zero compile/capture/contiguous tokens",
      "confidence": "high",
      "evidence": ["source audit"]
    },
    {
      "name": "run_out_bitwise_equals_forward",
      "status": "observed",
      "value": "run_out launches the same kernel into caller buffer (zero allocation); forward allocates fresh buffer + reshape; correctness PASS on all mandated surfaces",
      "confidence": "high",
      "evidence": ["harness correctness"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "present — ONE Triton kernel _flex_attn_fwd lowered and device-executed (1 topsModuleLaunchKernel/call); fp16 QK^T dot lowered to tensor-core MMA path",
    "evidence_contract": "triton_gcu (power-of-2 dots consumed as declared; fp16 QK^T no-widen; no-keepdim reductions)",
    "evidence": ["profile census"]
  },
  "evidence_gap_cause": "device_time_available = false on GCU launch-only trace; device attribution is inference from wall - launch-API-time (no cat=kernel events)"
}
```
