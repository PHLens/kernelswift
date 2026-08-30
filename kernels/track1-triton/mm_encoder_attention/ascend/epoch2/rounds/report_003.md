# Report 003

Result: accepted

## Identity

- Round: `003`
- Decision: `rounds/decision_003.md`
- Decision SHA256: `a4956891de5fef4b9bd629fb3cceb270db5a247ba18b591aecee9480d96c5455`
- Decision kind: `optimization`; change scope `host` / change family `allocation-reuse`
- Hypothesis ID: `H-003`
- Sketch: `rounds/sketch_003.json` (sha256 `51ebe3a735c7659309e781fd2f35286fd4e67acc86b5d0a9f6676f08f08af69c`)
- Candidate: `triton_mm_encoder_attention_e2_003.py`
- Candidate SHA256: `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe`
- Accepted reference (campaign): `triton_mm_encoder_attention_e2_001.py`
- Accepted reference SHA256: `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124`
- Accepted reference report: `rounds/report_001.md`
- Paired `--v0_file`: `base.py` (the harness requires it to define `Model`)
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged)
- Coder result SHA256: `d60e74e94f5e87ffbe2c535f8caea8d58c1fc7d4b104e1b0351fb9d854ac948d`
- Profile snapshot: `state/implementation_profile_snapshot/profile.yaml` (sha256 `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321`)
- Capability claim: `state/project_capability_claim.json` (sha256 `a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d`)
- Runtime fingerprint: `project.md#runtime-fingerprint` (Ascend910B4, torch 2.7.1+cpu / torch_npu 2.7.1.post4 / triton 3.2.0 / CANN 9.0.0)
- Measurement fingerprint: `1b1822d7b74a8cd41411a27fcbc18a89cb50b1cfefb9fdac2585cdd520e9a79a`
- verification_tier: candidate
- screening_pairs: `3` (used as the authoritative timing pairs, per the round-001 convention)
- Level 2: `targeted` host decomposition (run; requested by decision section 3)

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | `atol=1e-2`, `rtol=1e-2`, `equal_nan=True` | `PASS accuracy` on the smoke run and on all three interleaved pairs; `Summary: 1 passed, 0 failed, 1 total` | pass | paired runs, `--warmup 50 --repeat 100` |
| output shape | `[2, 83, 512]` | `(2, 83, 512)` | pass | guardrail probe |
| output dtype | fp16 | `torch.float16` | pass | guardrail probe |
| output device / contiguity | `npu:0`, contiguous | `npu:0`, `is_contiguous() == True` | pass | guardrail probe |
| numerical tolerance | within `1e-2` | max abs diff vs accepted kernel `0.0`; `torch.equal` is `True` (bit-identical) | pass | guardrail probe |
| no aliasing of q/k/v | returned tensor shares storage with none of them | `shares storage with q/k/v: False` | pass | guardrail probe |
| cached buffer fully overwritten | kernel store must cover every element | buffer filled with NaN then reused: `NaN leaked into output: False`, `84992/84992` finite, pointer stable, output still bit-equal to the accepted kernel | pass | guardrail probe |
| cache invalidation | a key change is a miss, never a reinterpretation | stride change `(42496,512,1)` -> `(512,1024,1)` reallocates; returning to the original shape reallocates; restored output matches the reference | pass | guardrail probe |
| public contract | constructor and forward signature unchanged | `__init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8)` and `forward(self, query, key, value) -> Tensor` identical to round 001; `get_init_inputs() == [8, 64, 8]` | pass | guardrail probe, `inspect.signature` |
| kernel body unchanged | byte-identical to the accepted kernel | `diff` of lines 1-74 vs `triton_mm_encoder_attention_e2_001.py` -> exit 0, no output | pass | `diff` |
| launch count | one kernel launch | exactly one `_fused_attention_kernel[grid](...)` site | pass | `grep -c` |
| launch configuration | `BLOCK_M`/`BLOCK_N`/`HEAD_DIM`/`num_warps`/`num_stages` unchanged | `128`/`128`/`64`/`4`/`1` | pass | `grep` |
| buffer is not module state | never serialized | `state_dict keys: []` | pass | level-2 probe |
| base.py bytes | immutable | `86ac5703…` unchanged | pass | `sha256sum` |

## Screening Evidence

Three paired runs were executed in one Verifier turn, `base.py` (reference)
against the candidate, at `--warmup 50 --repeat 100`:

| Pair | Reference median ms | Candidate median ms | Speedup | Correctness |
|---:|---:|---:|---:|---|
| 1 | 0.361050 | 0.298240 | 1.211x | PASS |
| 2 | 0.362085 | 0.302220 | 1.198x | PASS |
| 3 | 0.346350 | 0.296060 | 1.170x | PASS |

Correctness passed on every pair. The candidate is ahead on all three, so the
effect is stable and is not a warm-up artifact. Reference medians span
`0.346350`-`0.362085` (4.5%), which is the drift band of this machine in a
single turn; this is why the reference is re-measured in the same turn rather
than taken from history.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: three paired reference/candidate runs in one Verifier turn
- reference_median_ms: `0.361050` (median of `0.346350`, `0.361050`, `0.362085`)
- candidate_median_ms: `0.298240` (median of `0.296060`, `0.298240`, `0.302220`)
- improvement_pct: `17.3965`

```text
improvement_pct = (0.361050 - 0.298240) / 0.361050 * 100 = 17.3965
```

Improvement clears the 5% adoption threshold.

### Control comparison against the last accepted kernel

The harness always pairs `base.py` against `--v1_file`, so the number above is
not a direct candidate-versus-accepted measurement. An interleaved control was
run in the same turn, three blocks of `e2_001` then `e2_003`, each at
`--warmup 50 --repeat 100`:

| Block | base.py with e2_001 | e2_001 | base.py with e2_003 | e2_003 |
|---:|---:|---:|---:|---:|
| 1 | 0.345875 | 0.325660 | 0.349320 | 0.290600 |
| 2 | 0.358765 | 0.329810 | 0.348050 | 0.292845 |
| 3 | 0.365885 | 0.334445 | 0.366220 | 0.297770 |
| **median** | 0.358765 | **0.329810** | 0.349320 | **0.292845** |

```text
raw             = (0.329810 - 0.292845) / 0.329810 * 100           = 11.2080%
base-normalized = (0.919293 - 0.838329) / 0.919293 * 100           =  8.8072%
```

The base-normalized figure (each candidate divided by the `base.py` median
measured in the same process) is the conservative one, because `base.py` ran
1.6% faster in the `e2_003` blocks than in the `e2_001` blocks. Both figures
clear 5% comfortably. A second, independent control at `warmup 200 / repeat 500`
gives `10.1299%` raw and `11.5476%` base-normalized.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `applicable`
- hypothesis_id: `H-003`
- intervention: eliminate the per-call output allocation in `ModelNew.forward` by caching the output buffer on the `ModelNew` instance under an explicit cache key, so the steady-state forward performs no allocation and no per-call launch-constant reconstruction while the kernel body and launch configuration stay byte-identical
- expected_causal_chain: the steady-state forward performs zero output allocations instead of one → per-call host work inside `ModelNew.forward` decreases → device time and kernel count stay fixed so the wall delta is attributable to host → synchronized wall median decreases by at least five percent
- primary_metric: `wall_time`, expected improvement `5.0%`
- Hypothesis verdict: **`confirmed`**

### Mechanism observables

| Observable | Expectation | Accepted kernel (e2_001) | Candidate (e2_003) | Verdict |
|---|---|---|---|---|
| `output_allocations_per_call` | decrease from 1 to 0 on a cache hit | **1.00** (`torch.empty_like`, 20/20 forwards) | **0.00** (`torch.empty` 0/20, `torch.empty_like` 0/20) | confirmed |
| `host_us_per_call` | decrease | **233.645** us (forward alone) | **206.375** us | confirmed, `-27.270` us (`-11.67%`) |
| `device_us_per_call` | unchanged at ~13.4064 | **13.4096** us (re-measured this round) | **13.4224** us | confirmed unchanged, `+0.0128` us (`+0.095%`) |
| `kernel_count_per_call` | unchanged at 1.00 | **1.00** | **1.00** | confirmed |

Every declared mechanism observable moved as predicted, and the primary metric
cleared its threshold.

`output_allocations_per_call` needed care. A `TorchDispatchMode` count showed
`{'aten.empty.memory_format': 20, 'aten.empty_like.default': 20}` for e2_001 and
`{'aten.empty.memory_format': 20}` for e2_003 over 20 forwards, which naively
reads as one allocation per call for both. It is not. A **bare direct kernel
launch** with a preallocated output also produces exactly
`aten.empty.memory_format` 20/20, while a Python-level counter on `torch.empty`
/ `torch.empty_like` records zero. So the `aten.empty.memory_format` belongs to
the Triton launch path, not to the output. The output allocation is carried by
`torch.empty_like`, which is `1.00` per call for e2_001 and `0.00` for e2_003.
This is also a new bottleneck fact; see `evidence_for_next_round`.

## Profiler Evidence

- profiler_applicability: `required` (both control observables are device-side)
- profiler_level: `targeted`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable`
- mode: `forward`; warmup `20`

Reference and candidate scopes are captured in separate CANN captures
(`ASCEND_WORK_PATH` per scope) and summarized independently. Each summary was
given the explicit `ai_core_op_summary.db` path rather than the scope directory.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_triton_mm_encoder_attention_e2_001 | 670.48 | 13.4096 | 50 | 1.00 | 0.335740 | 0.0399 |
| candidate_triton_mm_encoder_attention_e2_003 | 671.12 | 13.4224 | 50 | 1.00 | 0.301730 | 0.0445 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
```

Wall values: both scopes were timed at `warmup 200 / repeat 500`, the regime the
harness uses when `--warmup`/`--repeat` are not passed to the profile command.
The candidate value `0.301730` is the profiler process's own `time_forward`
result. The reference value `0.335740` is **not** the `v0` printed by that run
(the `v0` there is `base.py` at `0.370825`, which is not this scope); it comes
from a dedicated `warmup 200 / repeat 500` run of `e2_001`. The chrome trace
cannot supply it, because `export_chrome_trace` is called once per scope inside
the loop and the candidate export overwrites the reference one — only the
candidate `record_function` survives in `round_003_forward_50iter.pt.trace.json`.

Control check against `rounds/report_001.md`: device `13.4064` -> `13.4224`
(`+0.16` us, `+0.12%`) and kernel count `1.00` -> `1.00`. Both control
observables held.

### Reference Top Kernels (reference_triton_mm_encoder_attention_e2_001)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _fused_attention_kernel | 50 | 1.0 | 670.48 | 13.4096 |

### Candidate Top Kernels (candidate_triton_mm_encoder_attention_e2_003)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _fused_attention_kernel | 50 | 1.0 | 671.12 | 13.4224 |

Both scopes are a single kernel, identical in name and count. The device picture
is unchanged, as the decision required.

## Level 2 Host Decomposition

Requested by decision section 3. All figures are medians in us/call, measured in
**one process and one regime** (`warmup 50`, `repeat 100`, median over 3 blocks),
with the NPU queue drained before each timed call.

| Quantity | e2_001 (accepted) | e2_003 (candidate) |
|---|---:|---:|
| (a) harness wall (`auto_bench.time_forward`) | 327.535 | 297.410 |
| (b) `ModelNew.forward` alone, no synchronize | 233.645 | 206.375 |
| (c) `forward` + `torch.npu.synchronize()` | 288.290 | 258.190 |
| (c) - (b) synchronize term | 54.645 | 51.815 |
| (a) - (b) everything outside `forward` | 93.890 | 91.035 |

Single quantities:

| Quantity | us/call |
|---|---:|
| (d) allocation-free direct launch, preallocated output, no Python wrapper | 183.740 |
| (e) `torch.npu.synchronize()` on an idle queue | 22.350 |

```text
(b_e2_001) - (b_e2_003) = 233.645 - 206.375 =  27.270 us   <- this round's lever
(b_e2_001) - (d)        = 233.645 - 183.740 =  49.905 us   <- total host work in e2_001's forward above the bare launch
(b_e2_003) - (d)        = 206.375 - 183.740 =  22.635 us   <- residual wrapper work left in e2_003
(a_e2_003) - (b_e2_003) = 297.410 - 206.375 =  91.035 us   <- harness-fixed term
```

Sanity: the Level-2 harness wall for the candidate, `297.410` us, is within
`0.28%` of the independently measured benchmark median `298.240` us, so the probe
is in the same regime as the adoption measurement.

### Where the harness-fixed 91.035 us goes

```text
(a) - (c) = 297.410 - 258.190 = 39.220 us
(c) - (b) = 258.190 - 206.375 = 51.815 us
```

- **`51.815 us` is the synchronize term**: draining the 13.4 us kernel plus the
  fixed cost of a synchronize on this runtime.
- **`39.220 us` is seed drain plus harness dispatch.** `set_seed` is called
  before `start = time.perf_counter()` and is therefore untimed, but
  `mod.manual_seed_all(seed)` enqueues device work that the timed
  `sync_devices()` then waits for, so the seed op is billed inside the timed
  region. On top of that, `sync_devices()` costs more than a bare synchronize:

| Piece (idle queue) | us/call |
|---|---:|
| `torch.npu.synchronize()` | 21.910 |
| `auto_bench.sync_devices()` | 33.870 |
| `list(_iter_accelerators())` | 11.320 |
| `torch.npu.is_available()` | 1.020 |
| pre-resolved accelerator list | 0.600 |

`sync_devices()` spends `11.96 us` per call probing for accelerator backends
before synchronizing; a pre-resolved list costs `0.600 us`. This is harness code
and is out of scope for any host round.

### The ceiling, stated plainly

Against the candidate's `297.410 us` wall:

| Slice | us/call | Share of wall | Reachable by a host round? |
|---|---:|---:|---|
| harness-fixed (outside `ModelNew.forward`) | 91.035 | 30.61% | **no** |
| Triton launch path (bare launch) | 183.740 | 61.78% | only by `launch-path-reduction`, capability Unknown |
| residual `forward` wrapper | 22.635 | 7.61% | yes, this is what is left |
| device kernel time | 13.4224 | 4.51% | only by device work, bounded at 4.09% |

The two host slices nest and add up exactly:
`183.740 + 22.635 = 206.375 = (b)` and `206.375 + 91.035 = 297.410 = (a)`.
Device time is **not** a fourth independent slice — the `13.4224 us` of device
work is already contained inside the `51.815 us` synchronize term. The shares
therefore sum above 100%; read the device row as a sub-component, not an
addition.

**No host round can touch the 91.035 us harness-fixed term.** Even driving
`ModelNew.forward` to zero host cost leaves 30.61% of wall time standing.

The residual host work this round did not remove is `22.635 us/call`. Another
accepted round needs `0.05 * 297.410 = 14.871 us`. The residual is only
**1.52x** that, so a further round in this family must capture about 66% of
everything left in `forward` merely to clear the threshold.

## Attribution

**The wall gain is attributable to the host lever, and device movement does not
confound it.**

- Wall fell `36.965 us/call` against the accepted kernel at `warmup 50 /
  repeat 100` (`0.329810` -> `0.292845`) and `34.010 us/call` at
  `warmup 200 / repeat 500` (`0.335740` -> `0.301730`).
- Device time moved `+0.0128 us/call` (`13.4096` -> `13.4224`), i.e. `+0.095%`,
  in the *wrong* direction. As a fraction of the wall change, device movement is
  `0.0128 / 36.965 = 0.035%`. Device cannot be the explanation.
- Kernel count stayed at `1.00`, so no launch-count story is available either.
- The host link moved directly and by the predicted amount: `ModelNew.forward`
  alone fell `27.270 us/call`, and the mechanism observable
  `output_allocations_per_call` went `1.00 -> 0.00`. Of the `49.905 us` of host
  work that e2_001's `forward` carried above a bare kernel launch, this round
  removed `54.6%`.
- The `27.270 us` forward-level lever is slightly smaller than the `36.965 us`
  wall delta. The gap is consistent with the candidate's synchronize term also
  shrinking (`54.645` -> `51.815 us`) once the allocator round-trip no longer
  perturbs the launch, plus ordinary drift.

The decision's `expected_wall_improvement_pct` of `8.0` was a judgment; the
observed lever of `27.270 us` on a `327.535 us` wall is `8.33%`, so the judgment
was accurate even though the measured adoption figure against `base.py` is
larger.

**The epoch's ceiling, from the Level 2 work: `91.035 us/call` (30.61% of wall)
is harness-fixed and unreachable.** The largest single remaining term is the
Triton launch path at `183.740 us/call` (61.78% of wall), which is a different
family and requires an Ascend launch-ABI probe before it can be attempted.
Inside the current family, only `22.635 us/call` (7.61%) remains.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification of round 003 | `not-applicable` | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` | correctness pass on first attempt; no repair needed |

No environment incident occurred. No `incident_003_*.md` was written. Three
diagnostic probes under `log/` were written by Verifier; none is a tracked
campaign artifact and none modifies `base.py`, the harness, the candidate, or any
decision.

## evidence_for_next_round

- **Adoption cleared.** `0.361050` -> `0.298240` ms against `base.py`,
  `improvement_pct = 17.3965%`. Against the accepted kernel the gain is
  `11.2080%` raw / `8.8072%` base-normalized at `warmup 50 / repeat 100`, and
  `10.1299%` / `11.5476%` at `warmup 200 / repeat 500`.
- **The allocation lever is real and now measured, not inferred:**
  `output_allocations_per_call` `1.00 -> 0.00` and `ModelNew.forward` alone
  `233.645 -> 206.375 us` (`-27.270 us`).
- **Control observables held.** `device_us_per_call` `13.4096 -> 13.4224`
  (`+0.095%`), `kernel_count_per_call` `1.00 -> 1.00`. The round is a genuine
  host win, not a device win leaking into wall.
- **New, and the most important fact this round produced: the Triton launch path
  itself costs `183.740 us/call`** — `61.78%` of wall — for a bare
  `_fused_attention_kernel[grid](...)` with a preallocated output and no Python
  wrapper. It also issues exactly one `aten.empty.memory_format` per launch
  (proven: a bare launch counts 20/20 with zero Python-level `torch.empty`
  calls). So the launch path allocates, once per call, on its own. That is the
  concrete target for `launch-path-reduction`, and it is roughly seven times
  larger than everything still left in `ModelNew.forward`.
- **The harness-fixed floor is `91.035 us/call` (30.61% of wall).** It decomposes
  into `51.815 us` of synchronize plus `39.220 us` of seed drain and
  `sync_devices()` accelerator probing (`sync_devices()` costs `11.96 us` more
  than a bare `torch.npu.synchronize()` because `_iter_accelerators()` calls
  `torch.npu.is_available()` every time). If a future round is ever blocked by a
  threshold it cannot clear, this term — not the kernel — is why.
- **Remaining headroom inside the current family is `22.635 us/call` (7.61% of
  wall)**, and clearing 5% again needs `14.871 us`, i.e. about 66% of it. What is
  still in `forward` at a cache hit: `query.shape` unpacking, the four-component
  cache-key tuple including a fresh `torch.device` construction per call
  (`query.device`), the key comparison, and the grid tuple. The `query.device`
  construction is the item decision risk note 5 flagged; it is now inside the
  measured `22.635 us`, not outside it.
- **Drift is material and must keep being controlled.** Within this single turn
  `base.py` medians ranged `0.346350`-`0.370825` (`~7%`). Always re-measure the
  reference in the same turn; never compare against a historical baseline.
- **Bottleneck ordering after this round:** Triton launch path (`183.740 us`) >
  harness-fixed (`91.035 us`) > `forward` wrapper (`22.635 us`) > device
  (`13.4224 us`). Device is now the smallest term and is bounded at `4.09%`.
- Carried forward unchanged: the kernel requires `S <= 128`; the campaign shape
  is `S=83`. The second `tl.dot` tile `(128,64,128)` compiles and is numerically
  correct here but was not one of the eleven probed tiles. The reuse invariant
  depends on full store coverage and must be revisited if a future round
  introduces a masked or partial store.

## Stop Recommendation

- recommendation: `continue`
- evidence: an accepted round just advanced the canonical kernel. The no-improvement
  streak resets to 0, the failed-attempt streak is 0, and the round budget is
  `2` of `20` terminal rounds consumed, far from `round-budget-exhausted`. No
  target is set, so `target-reached` does not apply. Measured headroom still
  exists: `22.635 us/call` (7.61% of wall) in the current host family and
  `183.740 us/call` (61.78%) behind `launch-path-reduction`. No stop condition is
  met.

## Exact Reproduction Commands

Correctness gate:

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py --warmup 5 --repeat 10 --full-traceback
```

Authoritative timing (three pairs, one turn):

```bash
cd /workspace/kernelswift-dev-4ff2094
for i in 1 2 3; do
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py \
    --warmup 50 --repeat 100
done
```

Interleaved control against the accepted kernel:

```bash
cd /workspace/kernelswift-dev-4ff2094
for i in 1 2 3; do
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_001.py --warmup 50 --repeat 100
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py --warmup 50 --repeat 100
done
```

Profiler:

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
  --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py \
  --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_001.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_003_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/profiling_data/reference_triton_mm_encoder_attention_e2_001/profiling_data/16458e336fc3_68268_20260830061802484_ascend_pt/PROF_000001_20260830061802510_00068268LNFJPPBQ/device_0/sqlite/ai_core_op_summary.db" --iterations 50 --scope reference_triton_mm_encoder_attention_e2_001 --wall-ms 0.335740

python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/profiling_data/candidate_triton_mm_encoder_attention_e2_003/profiling_data/16458e336fc3_68268_20260830061806332_ascend_pt/PROF_000002_20260830061806355_00068268HGKBPNLO/device_0/sqlite/ai_core_op_summary.db" --iterations 50 --scope candidate_triton_mm_encoder_attention_e2_003 --wall-ms 0.301730
```

Reference-scope wall (the profiler run's own `v0` is `base.py`, not this scope):

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
  --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_001.py --warmup 200 --repeat 500
```

Level 2 and guardrail diagnostics (Verifier-owned, under `log/`, not campaign artifacts):

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_003_host_decomposition.py
python3 kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_003_alloc_probe.py
python3 kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_003_harness_overhead.py
python3 kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_003_guardrails.py
```

Kernel-body identity check:

```bash
cd /workspace/kernelswift-dev-4ff2094/kernels/track1-triton/mm_encoder_attention/ascend/epoch2
diff <(sed -n '1,74p' triton_mm_encoder_attention_e2_001.py) <(sed -n '1,74p' triton_mm_encoder_attention_e2_003.py) && echo KERNEL_BODY_IDENTICAL=yes
```
