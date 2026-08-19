# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_centre_random_augmentation_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `ad2f891ebb8929b7c8b290388081573f25dbb78dc39ab04585cf258e99a1156b`
- Candidate SHA256: `4e33276ec28f3695aa08462aa6cb796a160aca47dad889168a7cdd8aa8e16036`
- Accepted reference SHA256: `012754740961f6ec10d515563e51cd07eeaf35caefe33731d5c1e9a88387fe9b`
- Base SHA256: `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`
- verification_tier: `authoritative`
- screening_pairs: `not-run` (correct candidate proceeded directly to authoritative timing)

All candidate, decision, reference, base, and harness hashes match their recorded values.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=1.012969 ms, v1=0.709612 ms, speedup=1.427x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| RNG consumption order | candidate must draw 3×`torch.rand` + 1×`torch.randn` in exact reference order inside `forward` | Independent probe: per-sample translation mean (≈T) matches base to ~1e-7; max_abs_diff `4.77e-07` | pass | independent numerical probe (base vs candidate via AST loader) |
| output dtype/shape | single float32 tensor `out[4,256,3]` | both base and candidate return `(4,256,3)` fp32 on cuda:0 | pass | probe output; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | `allclose=True`; `bit-exact=False` (expected Triton vs torch float rounding) | pass | probe `allclose` result |
| centering formula | `sum / (sum + eps)`, `eps=1e-12` | candidate kernel uses `tl.sum(x0*m)/(msum+eps)` with `eps=1e-12` | pass | candidate source lines 107-110 |
| frozen artifact identity | local hashes equal decision/coder_result before measurement | candidate `4e33276e...`, decision `ad2f891e...`, reference/base/harness all match | pass | SHA256 commands in Exact Reproduction Commands |

Correctness passed on the first attempt; no repair was required.

## Screening Evidence

Not run: the correct candidate proceeded directly to authoritative timing (three interleaved pairs).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (v0 = baseline_adapter wrapper, v1 = candidate)
- independent invocations: `3`
- reference_raw_samples_ms: `[1.023173, 1.029014, 1.006014]`
- candidate_raw_samples_ms: `[0.726311, 0.712600, 0.707727]`
- reference_median_ms: `1.023173`
- candidate_median_ms: `0.712600`
- improvement_pct: `30.35390886976103`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (1.023173 - 0.712600) / 1.023173 * 100 ≈ 30.35%
```

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `1.023173` | `0.726311` | `0` |
| 2 | `1.029014` | `0.712600` | `0` |
| 3 | `1.006014` | `0.707727` | `0` |

The unrounded improvement `30.35%` exceeds the `5.0` adoption threshold by a wide margin.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease | reference `78.94` → candidate `54.8` (decreased by ~24, but NOT to single digits) | pass | profiler summary (reference tool, candidate manual) |
| device_us_per_call | decrease | reference `411.05` us → candidate `237.95` us (decreased 42%) | pass | profiler summary |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: fuse the deterministic rigid-transform chain (centering subtraction, quaternion-to-rotation-matrix construction, 3x3-by-3 vector product, translation, and mask multiply) into a small number of Triton kernels while leaving the random number draws (torch.rand/randn) unchanged on the host
- expected_causal_chain:
  1. "the ~70 deterministic elementwise/reduce/transcendental/cat/copy kernels collapse into a few fused Triton kernels" → **partially** realized: the fused `_centre_aug_kernel` collapsed the `rot_vec_mul`/centering/translation/mask chain, but the quaternion→matrix conversion (sqrt/sin/cos/mul/add/stack/cat) was kept on the host, so ~50 host kernels remain.
  2. "kernel count per call drops from ~79 toward single digits" → **not fully** realized: dropped to ~55, not single digits.
  3. "device launch overhead and device kernel time both decrease" → **confirmed**: device time 411→238 us/call.
  4. "wall time decreases" → **confirmed**: 1.023→0.713 ms.
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed` (wall time improved 30.35%, exceeding the 5% threshold, with both mechanism observables decreasing as expected — though kernel count did not reach single digits)

The two declared mechanism observables both decreased as expected (`kernel_count_per_call` 78.94→54.8; `device_us_per_call` 411.05→237.95). The causal chain's "single digits" target was only partially realized because the quaternion→rotation-matrix construction was left on the host (see conformance note below), but the primary metric (wall time) improved decisively, so the hypothesis is `confirmed`.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `reference_baseline_adapter`, `candidate_triton_centre_random_augmentation_001`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `kernels/track1-triton/centre_random_augmentation/bi150/log/round_001_forward_50iter.pt.trace.json`, SHA256 `aa1da42ee52a9475f59ef575f251ff078f8c56f1a7a97385f3d13428884ca932`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `reference_baseline_adapter` | `20552.675` | `411.054` | `3947` | `78.94` | `1.023173` | `0.40174` |
| `candidate_triton_centre_random_augmentation_001` | `11897.695` | `237.954` | `2740` | `54.8` | `0.712600` | `0.33392` |

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000)
reference: 411.054 / 1023.173 ≈ 0.402
candidate: 237.954 / 712.600  ≈ 0.334
```

Note: `summarize_trace.py` reported "overlapping scope events" for the candidate
scope because the Triton-launched kernel projects a device-side `record_function`
event (pid=0, tid=1) that overlaps the CPU-side scope event. This is a known
Triton profiler artifact, not a data problem. The candidate-scope kernel totals
were computed by filtering kernel events to the CPU-side scope interval
(pid==tid), which yields `device_us_per_call=237.95`, `kernel_count_per_call=54.8`,
and `device_ratio=0.334`. The reference scope summarized cleanly via the
unmodified tool.

### Accepted Reference Top Kernels (reference_baseline_adapter scope)

| Kernel (semantic label) | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| elementwise mul (binary, `MulFunctor`) | `549` | `10.98` | `3896.13` | `77.92` |
| elementwise add (`CUDAFunctor_add`, 3-ptr) | `750` | `15.0` | `3062.21` | `61.24` |
| elementwise mul (unary, `AUnaryFunctor<MulFunctor>`) | `700` | `14.0` | `2770.51` | `55.41` |
| elementwise mul (binary, `MulFunctor` variant) | `650` | `13.0` | `2641.77` | `52.84` |
| reduce sum (`sum_functor`) | `100` | `2.0` | `1843.46` | `36.87` |
| elementwise add (other, `CUDAFunctorOnOther_add`) | `250` | `5.0` | ~`1020` | `~20` |
| sqrt / sin / cos / cat / copy / rand / randn / div | — | — | — | — |

(Full reference-scope kernel list is identical in composition to the Phase 0
baseline in `report_000.md`; the top five elementwise/reduce kernels above carry
~85% of the reference device time.)

### Candidate Top Kernels (candidate_triton_centre_random_augmentation_001 scope)

| Kernel (semantic label) | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| elementwise mul (unary, `AUnaryFunctor<MulFunctor>`) | `695` | `13.90` | `2737.22` | `54.74` |
| elementwise mul (binary, `MulFunctor`) | `650` | `13.00` | `2621.16` | `52.42` |
| elementwise add (`CUDAFunctor_add`) | `446` | `8.92` | `1816.90` | `36.34` |
| elementwise add (other, `CUDAFunctorOnOther_add`) | `249` | `4.98` | `985.59` | `19.71` |
| sqrt (`sqrt_kernel_cuda`) | `200` | `4.00` | `802.61` | `16.05` |
| rand (uniform) distribution kernel | `150` | `3.00` | `768.59` | `15.37` |
| sin (`sin_kernel_cuda`) | `100` | `2.00` | `565.13` | `11.30` |
| cos (`cos_kernel_cuda`) | `100` | `2.00` | `563.72` | `11.27` |
| cat (batched copy) | `49` | `0.98` | `421.66` | `8.43` |
| **FUSED Triton `_centre_aug_kernel`** | `49` | `0.98` | `327.96` | `6.56` |
| randn (normal) distribution kernel | `49` | `0.98` | `258.74` | `5.17` |

### Key Profiler Observation (conformance note)

The fused Triton kernel `_centre_aug_kernel` (one per sample, ~0.98/call) now
carries the centering + `rot_vec_mul` + translation + mask-multiply chain in only
`6.56 us/call` — collapsing the reference's dominant `mul`/`add`/`reduce`/
`copy`/`cat` mass. However, the quaternion→rotation-matrix construction
(`sqrt`/`sin`/`cos`/`mul`/`add`/`stack`/`cat`) was intentionally **left on the
host** inside `random_rotation_matrices` (to preserve bit-identical `R` and avoid
Triton-vs-torch transcendental divergence). As a result the candidate still
launches ~50 host-side torch kernels for that conversion, and `kernel_count_per_call`
fell only from `78.94` to `54.8`, not to the single digits the decision's causal
chain anticipated.

This is a documented conformance note, not a correctness defect: the primary
metric (wall time) improved 30.35%, and both declared observables decreased. The
remaining host-side quaternion→matrix conversion (~55 of the ~55 kernels) is the
obvious next-round fusion target.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `4e33276ec28f3695aa08462aa6cb796a160aca47dad889168a7cdd8aa8e16036` | same | correctness and wall timing passed; profiler summarized (candidate scope via manual filter) |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Round 001 `accepted`: wall median `1.023173 ms` → `0.712600 ms`, improvement `30.35%`, against `baseline_adapter.py` under fingerprint `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`.
- The fused Triton `_centre_aug_kernel` collapsed the centering + `rot_vec_mul` + translation + mask chain into `6.56 us/call`; device time fell 411→238 us/call.
- The quaternion→rotation-matrix conversion (`sqrt`/`sin`/`cos`/`mul`/`add`/`stack`/`cat`) still runs on the host, leaving ~50 host-side torch kernels and ~55 kernels/call. This is the dominant remaining device/launch cost (≈230 us/call) and the natural next fusion target.
- `device_ratio` remains low (~0.33), so the operator is still host/launch-bound even after fusion; further kernel-count reduction should continue to improve wall time.
- RNG boundary is now well-characterized: keeping `torch.rand`/`torch.randn` host-side is safe and bit-comparable; moving the deterministic quaternion→matrix math into the kernel (using `tl.sqrt`/`tl.sin`/`tl.cos`, which are `Unknown`/unproven on this profile) would require validating transcendental bit-compatibility against the reference within `atol=1e-2`.

## Stop Recommendation

- recommendation: `continue`
- evidence: 30.35% wall improvement confirms the kernel-fusion hypothesis; the operator remains host-bound with a clear remaining fusion target (the host-side quaternion→matrix conversion). No optional target is configured and the round budget (`max_rounds=20`) is not exhausted.

Orchestrator owns the stop transition and canonical pointer update.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py kernels/track1-triton/centre_random_augmentation/bi150/rounds/decision_001.md kernels/track1-triton/centre_random_augmentation/bi150/baseline_adapter.py kernels/track1-triton/centre_random_augmentation/base.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py --warmup 50 --repeat 100 --full-traceback
```

Authoritative wall timing — baseline_adapter wrapper (ModelNew→Model) then three interleaved pairs:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sed 's/^class ModelNew/class Model/' kernels/track1-triton/centre_random_augmentation/bi150/baseline_adapter.py > /tmp/cra_baseline_model_001.py
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/cra_baseline_model_001.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py --warmup 50 --repeat 100
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/cra_baseline_model_001.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py --warmup 50 --repeat 100
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/cra_baseline_model_001.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py --warmup 50 --repeat 100
```

Targeted forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/cra_baseline_model_001.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/centre_random_augmentation/bi150/baseline_adapter.py --profile-output kernels/track1-triton/centre_random_augmentation/bi150/log/round_001_forward_50iter.pt.trace.json
```

Reference scope summary (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/centre_random_augmentation/bi150/log/round_001_forward_50iter.pt.trace.json --iterations 50 --scope reference_baseline_adapter --wall-ms 1.023173
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity |
| correctness 50/100 | `0` | round_status_001.md; report Correctness table |
| independent numerical probe | `0` | max_abs_diff=4.77e-07, allclose=True |
| wall pair 1, 50/100 | `0` | report Interleaved Wall Timing |
| wall pair 2, 50/100 | `0` | report Interleaved Wall Timing |
| wall pair 3, 50/100 | `0` | report Interleaved Wall Timing |
| forward profiler 20/50 | `0` | `log/round_001_forward_50iter.pt.trace.json` |
| summarize `reference_baseline_adapter` | `0` | report Profiler Evidence |
| summarize candidate scope | `0` (manual filter) | report Profiler Evidence |
