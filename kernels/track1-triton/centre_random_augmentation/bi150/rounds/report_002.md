# Report 002

Result: accepted

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_centre_random_augmentation_002.py`
- Accepted reference: `triton_centre_random_augmentation_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `2290e37b81072b794ca5735dddba52ed19805c943a8e7109b598e5fd1f65af8e`
- Candidate SHA256: `efac6ee782e859701bb14aca04b7f56516a575a5f74507958e1930a95005a530`
- Accepted reference SHA256: `4e33276ec28f3695aa08462aa6cb796a160aca47dad889168a7cdd8aa8e16036`
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
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=1.014323 ms, v1=0.243855 ms, speedup=4.160x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| RNG consumption order | candidate must draw 3×`torch.rand` + 1×`torch.randn` in exact reference order inside `forward` | Independent probe: per-sample translation mean (≈T) matches base to ~1e-7; `u1/u2/u3/T` drawn host-side in order | pass | independent numerical probe |
| transcendental correctness | `tl.sqrt`/`tl.sin`/`tl.cos` must match torch within tolerance | max_abs_diff `4.77e-07` (≈1 ulp fp32) on the full output; quaternion construction numerically compatible | pass | probe `max_abs_diff` |
| output dtype/shape | single float32 tensor `out[4,256,3]` | both base and candidate return `(4,256,3)` fp32 on cuda:0 | pass | probe output; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | `allclose=True`; `bit-exact=False` (expected Triton transcendental rounding) | pass | probe `allclose` result |
| centering formula | `sum / (sum + eps)`, `eps=1e-12` | candidate kernel uses `tl.sum(x0*m)/(msum+eps)` with `eps=1e-12` | pass | candidate source lines 97-99 |
| frozen artifact identity | local hashes equal decision/coder_result before measurement | candidate `efac6ee7...`, decision `2290e37b...`, reference/base/harness all match | pass | SHA256 commands in Exact Reproduction Commands |

Correctness passed on the first attempt; no repair was required. The `tl.sqrt`/`tl.sin`/`tl.cos` primitives (marked `Unknown` on the target profile) lowered successfully and are numerically equivalent to torch within `atol=1e-2` (max abs diff 4.77e-07), confirming the decision's capability-risk assessment: not a capability-miss.

## Screening Evidence

Not run: the correct candidate proceeded directly to authoritative timing (three interleaved pairs).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (v0 = canonical 001 wrapper, v1 = candidate 002)
- independent invocations: `3`
- reference_raw_samples_ms: `[0.711623, 0.724253, 0.711154]`
- candidate_raw_samples_ms: `[0.239284, 0.244788, 0.237824]`
- reference_median_ms: `0.711623`
- candidate_median_ms: `0.239284`
- improvement_pct: `66.37489232360393`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (0.711623 - 0.239284) / 0.711623 * 100 ≈ 66.37%
```

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `0.711623` | `0.239284` | `0` |
| 2 | `0.724253` | `0.244788` | `0` |
| 3 | `0.711154` | `0.237824` | `0` |

The unrounded improvement `66.37%` exceeds the `5.0` adoption threshold by a wide margin (and the decision's `expected_wall_improvement_pct` of `12.0`).

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease | reference `54.92` → candidate `5.52` (collapsed to single digits) | pass | profiler summary (manual filter) |
| device_us_per_call | decrease | reference `238.19` us → candidate `29.24` us (decreased 88%) | pass | profiler summary (manual filter) |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: extend the fused Triton kernel to absorb the quaternion-to-rotation-matrix construction (sqrt/sin/cos and the 9-entry matrix arithmetic) from u1/u2/u3, so the ~48 host-side transcendental/elementwise/stack kernels collapse into the single existing `_centre_aug_kernel`; the random draws remain unchanged host-side calls
- expected_causal_chain:
  1. "the ~48 host-side transcendental/elementwise/stack/copy kernels for the quaternion-to-matrix conversion collapse into the single fused kernel" → **confirmed**: candidate has only `_centre_aug_kernel` + the unavoidable RNG kernels + one `s_trans` mul.
  2. "kernel count per call drops from ~55 toward single digits" → **confirmed**: `54.92 → 5.52`.
  3. "device launch overhead and device kernel time both decrease" → **confirmed**: device time `238.19 → 29.24 us/call`.
  4. "wall time decreases" → **confirmed**: `0.711623 → 0.239284 ms`.
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed` (wall time improved 66.37%; both observables decreased decisively; kernel count reached single digits)

All four causal-chain steps realized. Both mechanism observables (`kernel_count_per_call`, `device_us_per_call`) decreased as expected, and the primary metric improved far beyond the threshold. The hypothesis is `confirmed`.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `reference_triton_centre_random_augmentation_001`, `candidate_triton_centre_random_augmentation_002`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `kernels/track1-triton/centre_random_augmentation/bi150/log/round_002_forward_50iter.pt.trace.json`, SHA256 `7e70eec09eb4e9240f99644726c80e851b4bb9c9d21e8691bf4b9ec0321d368c`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `reference_triton_centre_random_augmentation_001` | `11909.719` | `238.194` | `2746` | `54.92` | `0.711623` | `0.33472` |
| `candidate_triton_centre_random_augmentation_002` | `1462.110` | `29.242` | `276` | `5.52` | `0.239284` | `0.12221` |

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000)
reference: 238.194 / 711.623 ≈ 0.335
candidate: 29.242 / 239.284 ≈ 0.122
```

Note: `summarize_trace.py` reports "overlapping scope events" on **both** scopes
this round, because both the reference (001) and candidate (002) now launch a
Triton kernel, and each projects a device-side `record_function` event
(pid=0, tid=1) that overlaps the CPU-side scope event (pid==tid, non-zero). This
is the same known Triton profiler artifact documented in `report_001.md`. Both
scope summaries were computed by filtering kernel events to the CPU-side scope
interval (pid==tid); this yields the totals above.

### Accepted Reference Top Kernels (reference_triton_centre_random_augmentation_001 scope)

| Kernel (semantic label) | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| elementwise mul (unary, `AUnaryFunctor<MulFunctor>`) | `699` | `13.98` | `2751.05` | `55.02` |
| elementwise mul (binary, `MulFunctor`) | `650` | `13.00` | `2623.98` | `52.48` |
| elementwise add (`CUDAFunctor_add`) | `450` | `9.00` | `1825.72` | `36.51` |
| elementwise add (other, `CUDAFunctorOnOther_add`) | `250` | `5.00` | `990.15` | `19.80` |
| sqrt (`sqrt_kernel_cuda`) | `200` | `4.00` | `806.05` | `16.12` |
| rand (uniform) distribution | `150` | `3.00` | `766.73` | `15.33` |
| sin (`sin_kernel_cuda`) | `100` | `2.00` | `568.12` | `11.36` |
| cos (`cos_kernel_cuda`) | `100` | `2.00` | `566.77` | `11.34` |
| cat (batched copy) | `49` | `0.98` | `424.73` | `8.49` |
| FUSED `_centre_aug_kernel` (001) | `49` | `0.98` | `327.73` | `6.55` |
| randn (normal) distribution | `49` | `0.98` | `258.68` | `5.17` |

The reference (Round 001 canonical) still carries the host-side quaternion→matrix
conversion: the `mul`/`add`/`sqrt`/`sin`/`cos`/`cat` kernels (~48 of the ~55
kernels/call) that Round 002 eliminated.

### Candidate Top Kernels (candidate_triton_centre_random_augmentation_002 scope)

| Kernel (semantic label) | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| rand (uniform) distribution | `137` | `2.74` | `692.10` | `13.84` |
| FUSED `_centre_aug_kernel` (002) | `46` | `0.92` | `340.65` | `6.81` |
| randn (normal) distribution | `46` | `0.92` | `240.96` | `4.82` |
| elementwise mul (unary) | `46` | `0.92` | `179.97` | `3.60` |
| cat (batched copy) | `1` | `0.02` | `8.42` | `0.17` |

### Key Profiler Observation

The fusion fully achieved the decision's goal. The candidate (002) collapsed the
entire deterministic transform — quaternion construction (`sqrt`/`sin`/`cos`),
9-entry rotation-matrix arithmetic, centering, `rot_vec_mul`, translation, and
mask multiply — into the single `_centre_aug_kernel` (0.92/call, 6.81 us/call).
The remaining kernels are only the **unavoidable RNG draws** that the decision
mandates stay host-side:
- `rand` uniform (2.74/call): the three `torch.rand` draws for `u1/u2/u3`.
- `randn` normal (0.92/call): the single `torch.randn` draw for `T`.
- `elementwise mul` unary (0.92/call): the `s_trans * torch.randn(...)` scalar
  multiply (with `s_trans=1.0`, a numerical no-op but still a host launch).
- one stray `cat` (0.02/call): scope-boundary artifact.

`kernel_count_per_call` fell from `54.92` to `5.52`, and `device_us_per_call`
from `238.19` to `29.24` (an 88% device-time reduction). `device_ratio` dropped
from `0.335` to `0.122`, meaning the operator is now even more strongly
host/launch-bound — the tiny remaining device work (RNG + one fused kernel) is
dwarfed by the host-side dispatch of the four random draws and the fused-kernel
launch. The residual `s_trans` multiply (a scalar × tensor launch, numerically a
no-op for `s_trans=1.0`) is a minor further-fusion target but is unlikely to
yield meaningful wall-time gain given the operator is already dominated by the
irreducible RNG-draw + single-kernel launch floor.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `efac6ee782e859701bb14aca04b7f56516a575a5f74507958e1930a95005a530` | same | correctness and wall timing passed; profiler summarized (manual filter for both scopes) |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Round 002 `accepted`: wall median `0.711623 ms → 0.239284 ms`, improvement `66.37%`, against `triton_centre_random_augmentation_001.py` under fingerprint `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`.
- The full deterministic transform is now fused into a single `_centre_aug_kernel` (6.81 us/call); `kernel_count_per_call` collapsed `54.92 → 5.52`, `device_us_per_call` `238.19 → 29.24`.
- The remaining kernels are the irreducible host-side RNG draws (3×`torch.rand` + 1×`torch.randn`) plus a numerical no-op `s_trans * randn` multiply and a stray `cat`. `device_ratio ≈ 0.122`, so the operator is now strongly host/launch-bound at a floor set by the four mandatory RNG draws and the single fused-kernel launch.
- `tl.sqrt`/`tl.sin`/`tl.cos` lower correctly on the CoreX Triton 3.1.0 BI150 backend and are numerically equivalent to torch (max abs diff 4.77e-07), so the transcendental capability question is now resolved in favor of availability (can be recorded as locally-proven, pending profile update by Designer/Orchestrator).
- Further wall-time gains would require reducing the host-side RNG-draw dispatch overhead (which the decision forbids changing the RNG consumption order of) or reducing the single fused-kernel launch; diminishing returns are likely.

## Stop Recommendation

- recommendation: `valid-no-improvement-limit`
- evidence: The deterministic compute has been fully fused into a single kernel (kernel count 54.92→5.52, device 238→29 us/call, wall 0.7116→0.2393 ms, +66.37%). The remaining wall time is dominated by the irreducible host-side RNG draws (3×`torch.rand` + 1×`torch.randn`, which the decision mandates stay host-side in exact order) and the single fused-kernel launch; `device_ratio` is now ~0.12. The decision's optimization intent (fuse the transcendental chain) is fully realized, and further kernel-fusion rounds would target the mandatory RNG boundary with diminishing returns.

Orchestrator owns the stop transition and canonical pointer update.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py kernels/track1-triton/centre_random_augmentation/bi150/rounds/decision_002.md kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py kernels/track1-triton/centre_random_augmentation/base.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py --warmup 50 --repeat 100 --full-traceback
```

Authoritative wall timing — canonical 001 wrapper (ModelNew→Model) then three interleaved pairs:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sed 's/^class ModelNew/class Model/' kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py > /tmp/cra_canonical_model_002.py
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/cra_canonical_model_002.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py --warmup 50 --repeat 100
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/cra_canonical_model_002.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py --warmup 50 --repeat 100
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/cra_canonical_model_002.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py --warmup 50 --repeat 100
```

Targeted forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/cra_canonical_model_002.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py --profile-output kernels/track1-triton/centre_random_augmentation/bi150/log/round_002_forward_50iter.pt.trace.json
```

Reference and candidate scope summaries (manual CPU-side-interval filter; both scopes contain Triton kernels, so the unmodified `summarize_trace.py` reports overlapping device-side scope events):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 - <<'PY'
import json
d = json.load(open('kernels/track1-triton/centre_random_augmentation/bi150/log/round_002_forward_50iter.pt.trace.json'))
def summarize(scope, wall_ms):
    s = e2 = None
    for e in d['traceEvents']:
        if e.get('ph')=='X' and e.get('cat')!='kernel' and e.get('name')==scope and e.get('pid')==e.get('tid') and e.get('pid')!=0:
            s = e['ts']; e2 = e['ts']+e['dur']
    cnt = 0; tot = 0.0
    for e in d['traceEvents']:
        if e.get('cat')=='kernel':
            st = e['ts']; en = e['ts']+e['dur']
            if st >= s and en <= e2:
                cnt += 1; tot += e['dur']
    print(scope, 'kernel_count_per_call=', cnt/50, 'device_us_per_call=', round(tot/50,3), 'device_ratio=', round((tot/50)/(wall_ms*1000),5))
summarize('reference_triton_centre_random_augmentation_001', 0.711623)
summarize('candidate_triton_centre_random_augmentation_002', 0.239284)
PY
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity |
| correctness 50/100 | `0` | round_status_002.md; report Correctness table |
| independent numerical probe | `0` | max_abs_diff=4.77e-07, allclose=True |
| wall pair 1, 50/100 | `0` | report Interleaved Wall Timing |
| wall pair 2, 50/100 | `0` | report Interleaved Wall Timing |
| wall pair 3, 50/100 | `0` | report Interleaved Wall Timing |
| forward profiler 20/50 | `0` | `log/round_002_forward_50iter.pt.trace.json` |
| summarize reference(001) | `0` (manual filter) | report Profiler Evidence |
| summarize candidate(002) | `0` (manual filter) | report Profiler Evidence |
