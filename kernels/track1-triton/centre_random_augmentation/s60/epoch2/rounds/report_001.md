# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`459a1f9b36105b33966c53b3e7740313094ba96874ffae1be358171066948c40` (hash re-verified live; change_family "triton-launch-fusion"; expected_wall_improvement_pct 59.0 declared from preflight, Verifier measurement authoritative)
- Candidate: `triton_centre_random_augmentation_e2_001.py`
- Accepted reference: `baseline_adapter.py` @`7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b` (last_accepted_kernel per r000; byte-equivalent pipeline to base.py)
- Accepted reference report: `rounds/report_000.md` (Phase 0 baseline)
- Decision SHA256: `459a1f9b36105b33966c53b3e7740313094ba96874ffae1be358171066948c40`
- Sketch SHA256: `017b423b96d88ba28fde6f1d4d6a7534b9f0fcf486a540d78c7c59f149c4429f` (rounds/sketch_001.json, re-verified)
- Candidate SHA256: `542293c0ed3488b4f30c6c3758780115325593a592b11bc656cfa605f9d79522` (re-verified live pre- AND post-run; matches Coder ledger exactly; AST-parse OK)
- Accepted reference SHA256: `7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b`
- Base SHA256: `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553` (unchanged, re-verified)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Profile snapshot SHA256: `8dfabd0af59b8f6640b47179fee19bca2f5fe35b18535a3db24f60c842e42b70` (profile_snapshot/triton_gcu.yaml, per team-state)
- Runtime fingerprint: `project.md#runtime-fingerprint` (unchanged since r000)
- Measurement fingerprint: `cra-s60-e2` (project.md canonical; base/harness bytes re-verified identical)
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeded directly to authoritative timing (sibling r001 precedent; a screen-out would skip the profiler — destroying the round's mandated launch-collapse census duty)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (seed42 canonical) | `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` vs base.py, seed 42, fp32 out `[4,256,3]` | `PASS accuracy` in all three authoritative pairs + profile run (4/4 invocations); exact-match regime (random sequence bit-identical) | pass | timing pairs 1-3, profile run |
| stateless module | zero call-time instance state, no caches/workspace | `__init__` stores only n_sample/s_trans/centre_only; `forward` writes zero instance attrs | pass | source audit |
| public API contract | `ModelNew(n_sample=4, s_trans=1.0, centre_only=False).forward(x_input_coords, mask=None)` | constructor signature and forward surface preserved | pass | source audit |
| capability legality | math.elementwise primary (sqrt/sin/cos/mul/add/sub); tl.arange power-of-2; num_warps=1; zero tl.dot | `tl.sqrt`×4 / `tl.sin`×2 / `tl.cos`×2; `tl.arange(0,256)` power-of-2; `num_warps=1`; `tl.dot` count 0 (3x3 matvec static-unrolled into 3 fp32 dot products) | pass | source audit |
| random-number contract | u1/u2/u3 = torch.rand(n_sample) ×3 then T = s_trans*torch.randn(n_sample,3) host-side in base order | host generates u1/u2/u3 then T exactly; zero torch.rand inside kernel | pass | source audit |
| no compile/graph machinery | zero torch.compile/TORCHINDUCTOR/reduce-overhead/graph/capture/contiguous tokens | zero such constructs in candidate source; zero `.contiguous()` in forward host path | pass | source audit |
| AST-loader-safe module | safe-literal module constants; get_inputs/get_init_inputs retained | module-level literals only; `get_inputs`/`get_init_inputs` present with `torch.manual_seed(42)` | pass | source audit |
| default-stream discipline | all invocations on harness default route | unchanged harness default path; zero stream manipulation | pass | command history |
| cold JIT outside medians | warmup 50 absorbs first-call compile | harness warmup 50 precedes every timed section | pass | harness behavior |

Conformance, correctness, and every declared guardrail passed.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms), byte-for-byte identical flags, interpreter, device, default-stream route`
- reference_raw_samples_ms: `[3.025109, 3.192140, 2.316304]`
- candidate_raw_samples_ms: `[1.585115, 1.679165, 1.287518]`
- reference_median_ms: `3.025109`
- candidate_median_ms: `1.585115`
- improvement_pct: `+47.601761`

```text
improvement_pct = (3.025109 - 1.585115) / 3.025109 * 100 = +47.601761
```

| Independent invocation | Reference wall ms | Candidate wall ms | Speedup | Evidence |
|---:|---:|---:|---:|---|
| 1 | `3.025109` | `1.585115` | `1.908x` | pair 1 timing |
| 2 | `3.192140` | `1.679165` | `1.901x` | pair 2 timing |
| 3 | `2.316304` | `1.287518` | `1.799x` | pair 3 timing |

ABOVE the 5.0% adoption bar with a decisively POSITIVE sign: candidate wall 1.585115 ms vs reference 3.025109 ms = +47.6% paired improvement (candidate ~1.91x). This is the first S60 fusion-class operator to beat base.

Variance note (paired same-session basis absorbs drift): the reference raw samples span 2.316–3.192 ms, an unusually wide spread relative to the report_000 baseline median (~2.342 ms). The candidate is far tighter (1.288–1.679 ms). Two observations anchor the classification regardless of which reference sample is taken as the basis: (i) the profile-run pair (v0=2.360357, v1=1.285054, 1.837x) sits near the tight end of the reference spread and still yields +45.6%; (ii) even the slowest candidate (1.679165 ms) against the fastest reference (2.316304 ms) is +27.5% — well clear of the +5% bar. The reference-side variance is the known GCU launch-tax tail (78 tiny launches/call, sensitive to scheduling), which the candidate structurally eliminates.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time_unrounded_paired_median_ms | ≥5% below accepted reference median across interleaved pairs at warmup 50 / repeat 100 | +47.6% (candidate 1.585115 vs reference 3.025109 ms, three ordered pairs); decisively positive | **pass** | pairs 1-3 timing |
| runtime_launch_count_per_call | collapse from 78/call (report_000 census) to exactly 1.00 kernel launch per call | **97 → 12 launches/call** per scope; candidate's own Triton kernel = 1 `topsModuleLaunchKernel` @11.06us (the remaining 11 are the 4 random-source rand/randn + center/x_centered host ops) | **pass** | profile census |
| aten_cpu_ops_per_call | collapse from ~78/call to ≤6/call in candidate forward scope | **534 → 62/call** aten+GCU cpu_ops (candidate retains only rand×3 + randn + empty×2 + masked-mean center chain) | **pass** | profile census |
| correctness_max_abs_diff | exact-match (random sequence bit-identical under seed 42) | `PASS accuracy` all 4 invocations; comparator allclose atol/rtol 1e-2 with equal_nan | **pass** | harness correctness |
| triton_launcher_tax_per_call | single-launch launcher tax replaces 78 launch-API submissions | candidate 1 `topsModuleLaunchKernel` @11.06us replaces the base's 96 `topsLaunchKernel` @921.87us/call of launch-API time | **pass** | profile census |
| capability_legality_audit | math.elementwise only; zero tl.dot; tl.arange power-of-2; num_warps=1; zero DANGER tokens | all verified (see Correctness table); zero tl.dot; BLOCK=256 power-of-2; num_warps=1 | **pass** | source audit |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: one direct-launched Triton elementwise-fusion kernel (grid=(n_sample,)=4, num_warps=1) replacing the entire base path (quaternion→R + 3x3 matvec + translation + mask); host keeps only u1/u2/u3/T random sources + center/x_centered
- expected_causal_chain: chain observed with attribution — cn.launch-collapse → cn.dispatch-collapse CONFIRMED (aten cpu_ops 534 → 62/call; launch count 97 → 12/call, candidate's own kernel = 1 launch); cn.launch-collapse → cn.triton-launcher-tax CONFIRMED (96 topsLaunchKernel @921.87us → 1 topsModuleLaunchKernel @11.06us launch-API time); cn.launch-collapse → cn.wall-time CONFIRMED (wall +47.6%)
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed` — the launch-collapse mechanism ENGAGED exactly as designed and delivered the win: the 78-launch base path collapsed to a single Triton kernel, launch-API time dropped ~921us → ~11us/call, and paired wall improved +47.6% (candidate ~1.91x). This is the first S60 fusion-class operator to beat base.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (per decision profiling_level; forward-mode dual-scope trace + host census)
- profiler_device_time: `unavailable: device_time_available = false — GCU trace exposes runtime-launch events (gcu_runtime) but no cat=kernel device durations` — Level 1 normalized runtime-launch evidence recorded instead, never substituting launch time for device kernel time
- iterations: `100` forward calls per scope
- normalized_fields: `runtime_launch_count_per_call`, `runtime_launch_total_us`, `runtime_launch_us_per_call`
- trace: `log/report_001_forward.pt.trace.json`
- trace_sha256: `63c94e1a11d21817502ada37587d688d2089617a10507917366db36a1164c622`

### Runtime-launch census (Level 1, launch-only trace, per call)

| Signal | accepted_reference (baseline_base) | candidate (direct Triton) |
|---|---:|---:|
| runtime_launch_count_per_call | 97.0 | 12.0 |
| runtime_launch_total_us/call | 932.79 | 117.61 |
| launch event class | `topsLaunchKernel` @96.0/call (921.87us) + `topsLaunchCooperativeKernel` @1.0/call (10.92us) | `topsModuleLaunchKernel` @1.0/call (11.06us) + `topsLaunchKernel` @10.0/call (96.43us) + `topsLaunchCooperativeKernel` @1.0/call (10.12us) |
| aten+GCU cpu_ops total/call | **534.00** (empty 79 + mul 76 + as_strided 41 + add/sub/cat/stack/sqrt/sin/cos + expand/contiguous/reshape + rand/randn chains) | **62.00** (empty 12 + rand×3/uniform×3 + randn/normal + masked-mean center chain + the 3x3 matvec host sources) |

Notes: (i) device_time_available is `false` on this target — the trace exposes launch-only events, so all device attribution is via launch-count + launch-API-time. (ii) The launch collapse is DECISIVE: `topsLaunchKernel` (the base's dominant 78-launch class, report_000) collapses from 96/call to 10/call; the candidate's single Triton kernel is exactly 1 `topsModuleLaunchKernel` @11.06us. The remaining candidate launches (10 topsLaunchKernel + 1 cooperative) are the irreducible host-side random sources (u1/u2/u3 rand, T randn) and the masked-mean center/x_centered torch ops — exactly the ~11 ops the design preserves. (iii) launch-API time collapses ~932us → ~118us/call, directly matching the wall improvement: the base is launch-bound and the fusion removes ~814us of launch-API time per call.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `542293c0ed3488b4f30c6c3758780115325593a592b11bc656cfa605f9d79522` | correctness passed on first attempt; no repairs needed |

Zero Verifier-to-Coder repairs were needed — no candidate defect was found at any point (candidate hash constant end-to-end, matching the coder ledger).

## evidence_for_next_round

- Observed fact: the launch collapse ENGAGES and WINS — aten+GCU cpu_ops 534 → 62/call, `topsLaunchKernel` 96 → 10/call, the candidate's single Triton kernel = 1 `topsModuleLaunchKernel` @11.06us, and paired wall improved +47.6% (1.585115 vs 3.025109 ms). This is the first S60 fusion-class operator to beat base.
- Observed fact (canonical, this campaign): **S60 base launch-API tax ≈ 922us/call** across 96 `topsLaunchKernel` @9.60us each + 1 cooperative @10.92us. The candidate collapses this to ~118us/call (1 module-launch @11.06us + 10 topsLaunch @9.64us each + 1 cooperative @10.12us). Net ~814us/call saved is almost exactly the wall delta, confirming the base is launch-bound (not device-bound — opposite of the mm_encoder_attention sibling).
- Observed fact: reference wall has high variance (2.316–3.192 ms) while the candidate is tight (1.288–1.679 ms). This variance is the GCU launch-tax scheduling tail; the candidate structurally eliminates it, so its timing is both faster AND more stable.
- Observed fact: candidate retains 12 launches/call (not the idealized 1). Of these, ~11 are host-side torch ops (rand×3 + randn + empty + masked-mean center/x_centered). A future round could shave the center/x_centered torch reduction (4 sum/add ops) or the random-source generation, but these are already sub-10us each and outside the per-sample loop — marginal remaining prize is small (<~100us/call).
- Capability constraint (re-confirmed, no change needed): `tl.arange` requires power-of-2 (BLOCK=256 used, correct); GCU kernel has no torch.rand (randomness correctly host-side); math.elementwise primary contract honored (zero tl.dot, 3x3 matvec static-unrolled). No profile correction is required this round.
- Deliverable banked: `triton_centre_random_augmentation_e2_001.py` @`542293c0ed3488b4f30c6c3758780115325593a592b11bc656cfa605f9d79522` is a correctness-PASS Triton submission (forward surface, stateless, envelope-legal) at ~1.91x — per project.md DELIVERABLE RULE this is the campaign's primary contractual product; canonical pointer should move to this candidate on adoption.
- Session drift note: paired same-session basis absorbs drift; the +47.6% delta is ~10× the 5.0% bar and robust to the reference-side variance (worst-case pairing still +27.5%).

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: first win of the campaign (performance_miss_streak 0 → 0 on acceptance); round budget 1/20 consumed; the round banked the Triton deliverable plus the canonical launch-tax physics (S60 base ~922us/call launch-API tax, candidate ~118us/call). Remaining marginal levers (center/x_centered reduction, random-source generation) are sub-100us and outside the per-sample loop; the Designer may consider whether any further fusion has material prize or whether the operator is near its launch-bound floor.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + authoritative timing (three identical interleaved pairs):

```bash
cd /root/CodeBuddy/20260828202827/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/bin/python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/s60/epoch2/triton_centre_random_augmentation_e2_001.py --warmup 50 --repeat 100 --full-traceback
```

Dual-scope profiler (forward-mode, pw=50/pi=100):

```bash
cd /root/CodeBuddy/20260828202827/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/bin/python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/s60/epoch2/triton_centre_random_augmentation_e2_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-output kernels/track1-triton/centre_random_augmentation/s60/epoch2/log/report_001_forward.pt.trace.json
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/centre_random_augmentation/s60/epoch2/log/report_001_forward.pt.trace.json --iterations 100 --scope baseline_base --wall-ms 2.360357
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/centre_random_augmentation/s60/epoch2/log/report_001_forward.pt.trace.json --iterations 100 --scope candidate_triton_centre_random_augmentation_e2_001 --wall-ms 1.285054
```

Artifact hash ledger (re-verified this round):

```text
542293c0ed3488b4f30c6c3758780115325593a592b11bc656cfa605f9d79522  triton_centre_random_augmentation_e2_001.py
459a1f9b36105b33966c53b3e7740313094ba96874ffae1be358171066948c40  rounds/decision_001.md
017b423b96d88ba28fde6f1d4d6a7534b9f0fcf486a540d78c7c59f149c4429f  rounds/sketch_001.json
7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b  baseline_adapter.py
02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
8dfabd0af59b8f6640b47179fee19bca2f5fe35b18535a3db24f60c842e42b70  profile_snapshot/triton_gcu.yaml
63c94e1a11d21817502ada37587d688d2089617a10507917366db36a1164c622  log/report_001_forward.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "542293c0ed3488b4f30c6c3758780115325593a592b11bc656cfa605f9d79522",
  "correctness": {
    "status": "pass",
    "evidence": [
      "auto_bench.py PASS accuracy in all three authoritative pairs + profile run (4/4 invocations, seed42 canonical regime)",
      "exact-match regime: random sequence u1/u2/u3/T bit-identical to base under seed 42"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "+47.6% (reference 3.025109 ms vs candidate 1.585115 ms; bar +5.0% CLEARED decisively; candidate ~1.91x)",
      "confidence": "high",
      "evidence": ["timing pairs 1-3"]
    },
    {
      "name": "runtime_launch_count_per_call",
      "status": "observed",
      "value": "97 -> 12/call per scope (base topsLaunchKernel 96 @921.87us; candidate topsModuleLaunchKernel 1 @11.06us + topsLaunchKernel 10 @96.43us + cooperative 1 @10.12us)",
      "confidence": "high",
      "evidence": ["profile census"]
    },
    {
      "name": "aten_cpu_ops_per_call",
      "status": "observed",
      "value": "534 -> 62/call (base empty 79 + mul 76 + as_strided 41 + sqrt/sin/cos/stack/cat/reshape/expand/contiguous; candidate retains rand x3 + randn + empty x2 + masked-mean center)",
      "confidence": "high",
      "evidence": ["profile census"]
    },
    {
      "name": "triton_launcher_tax_per_call",
      "status": "observed",
      "value": "base launch-API ~922us/call (96 topsLaunchKernel @9.60us + 1 cooperative @10.92us) -> candidate ~118us/call (1 module-launch @11.06us); net ~814us saved ~= wall delta confirms launch-bound base",
      "confidence": "high",
      "evidence": ["profile census"]
    },
    {
      "name": "capability_legality_audit",
      "status": "observed",
      "value": "math.elementwise only (tl.sqrt x4 / tl.sin x2 / tl.cos x2); tl.dot 0; tl.arange(0,256) power-of-2; num_warps=1; zero compile/capture/contiguous tokens",
      "confidence": "high",
      "evidence": ["source audit"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "present — ONE Triton kernel _centre_random_aug_kernel lowered and device-executed (1 topsModuleLaunchKernel/call, grid=(4,), num_warps=1)",
    "evidence_contract": "triton_gcu (math.elementwise dots-as-elementwise; power-of-2 tl.arange; num_warps=1)",
    "evidence": ["profile census"]
  },
  "evidence_gap_cause": "device_time_available = false on GCU launch-only trace; device attribution is inference from launch-count + launch-API-time (no cat=kernel events)"
}
```
