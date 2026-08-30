# Report 005

Result: accepted

## Headline

The governing bar is cleared by a very large margin and the verdict is not
marginal in any sense.

| Test | Result | Verdict |
|---|---|---|
| **Governing: `speedup(e2_005) / speedup(e2_003) - 1 >= 5%`**, strict pair-by-pair alternation, one window | **+41.5498%** | **CLEARS by 36.55 points** |
| Wall-time improvement over canonical `e2_003` (same window, raw) | `28.8329%` | CLEARS |
| Wall-time improvement over canonical `e2_003` (in-process, paired) | `28.724%` | CLEARS |
| Standard protocol vs same-turn `base.py` | `40.7148%` | CLEARS (cumulative, see note) |

All five alternating pairs clear the bar; the **weakest** pair is `+34.3058%`,
roughly seven times the threshold.

## Identity

- Round: `005`
- Decision: `rounds/decision_005.md`
- Decision SHA256: `1fdd16d7ddca961760260b9e6130c7e6d2fb17b689728474ee9e5bea9b8ce551`
- Decision kind: `optimization`; change scope `host` / change family `launch-path-reduction`
- Hypothesis ID: `H-005`
- Sketch: `rounds/sketch_005.json` (sha256 `f44ed2bfbef80e9dc603494221bbc2cd47db40a9d8d48d85ee2ae344cd11c4ee`)
- Candidate: `triton_mm_encoder_attention_e2_005.py`
- Candidate SHA256: `bf54cea2a1fcdafd8916c2e0bf607766a6e7ffc2981fd956e18e92bf51b88b26`
- Accepted reference (campaign): `triton_mm_encoder_attention_e2_003.py`
- Accepted reference SHA256: `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe`
- Accepted reference report: `rounds/report_003.md`
- Paired `--v0_file`: `base.py` (the harness requires it to define `Model`)
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged)
- Coder result SHA256: `b8f8a06fddaa4328dc340ece21af997acab7250b6a4a0db33df335f68a087268`
- Capability gate: `lifecycle.fast-launcher` — `round_local_status: proven`, `new_probe_required: false`
- Selected mechanism: **M2, cached `CompiledKernel`**, shipped as the **M2b per-call-stream** variant
- Profile snapshot: `state/implementation_profile_snapshot/profile.yaml` (sha256 `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321`, **unamended**, still `Unknown`)
- Capability claim: `state/project_capability_claim.json` (sha256 `a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d`)
- Runtime fingerprint: `project.md#runtime-fingerprint` (Ascend910B4, torch 2.7.1+cpu / torch_npu 2.7.1.post4 / triton 3.2.0 / CANN 9.0.0)
- Measurement fingerprint: `1b1822d7b74a8cd41411a27fcbc18a89cb50b1cfefb9fdac2585cdd520e9a79a`
- verification_tier: candidate
- screening_pairs: `3` (used as the authoritative timing pairs, per campaign convention)
- Level 2: `targeted` host decomposition (run)

## Mechanism Verification (done before any timing)

Legality is Coder's to carry by citation of the retained round-004 artifacts;
Verifier does not re-adjudicate it. Verifier did independently confirm the
identity of the shipped mechanism, which the team lead asked for explicitly.

| Check | Observation | Verdict |
|---|---|---|
| kernel `triton.jit` definition byte-identical to `e2_003` | `diff` lines 1-76, exit `0`, no output | pass |
| **shipped path is M2b, not M2a** | the fast-path launch is `kernel[grid](query, key, value, out, stride0, stride1, S, scale)` with **no `stream=` argument**; `grep` for `stream=` at the launch site returns nothing | **confirmed** |
| stream genuinely resolved per call at runtime | instrumented `driver.active.get_current_stream`: **1.00 resolutions per call** for `e2_005` (and for `e2_003`) | **confirmed** |
| no `LibEntry` / `libentry` present | `grep` returns nothing | pass |
| two launch sites, mutually exclusive | fast path returns early; proven launch is the fall-through | pass |
| frozen snapshot not amended | `a2c3e2e4…` unchanged | pass |

### The 88-versus-67 discrepancy is real, expected, and now reproduced

Decision 005 quoted M2 at `66.895 us`. That figure is **M2a**, the
cached-stream variant. The Host Plan's `device_stream_behavior` requires the
stream be resolved per call, which is **M2b**. Verifier measured both in one
process, against a reproduced M0 baseline:

| Bare launch path | us/call | saving vs M0 |
|---|---:|---:|
| M0 proven `kernel[grid](...)` | 178.915 | — |
| **M2b, per-call stream — SHIPPED** | **89.220** | **89.695** |
| M2a, cached stream (not shipped) | 66.220 | 112.695 |
| **per-call stream resolution cost** | **23.000** | |

This reproduces Coder's finding independently: M2b `85.770`/`88.720`,
M2a `63.065`/`64.670`, stream cost `22.705`/`24.050`. M2a at `66.220` is
consistent with round 004's quoted `66.895` within this machine's drift. **A
`launch_path_us_per_call` near `89` is correct behaviour, not a regression.**

A further observation that strengthens the choice: M2a cannot be written with
`torch.npu.current_stream()` on this runtime — it raises
`TypeError: argument 4 must be int, not Stream`. The cached-stream form requires
reaching into `triton.runtime.driver.active.get_current_stream(device)` for a raw
handle, which is exactly the hand-marshalling decision 005 section 3 uses to
reject M3. M2b is both the conformant and the more robust form.

Recovering the `23.000 us` needs a decision amending `device_stream_behavior`.
The supporting measurement now exists from two independent probes.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict |
|---|---|---|---|
| correctness | `atol=1e-2`, `rtol=1e-2`, `equal_nan=True` | `PASS accuracy` on the smoke run and on all three interleaved pairs; `Summary: 1 passed, 0 failed, 1 total` | pass |
| bit-identity vs accepted kernel | `max_abs_diff = 0.0` | `torch.equal(cand, e2_003_out) == True` | pass |
| output shape / dtype / device / contiguity | unchanged | `(2, 83, 512)`, `torch.float16`, `npu:0`, `is_contiguous() == True` | pass |
| no aliasing of q/k/v | returned tensor shares storage with none | `False` | pass |
| cached buffer fully overwritten | kernel store covers every element | buffer filled with NaN then reused: `NaN leaked = False`, `84992/84992` finite, pointer stable, output still bit-equal | pass |
| kernel launch count | stays at one | `1.00` for `e2_003` and `1.00` for `e2_005`, counted through the real launcher class | pass |
| kernel definition unchanged | byte-identical to `e2_003` | `diff` lines 1-76 → exit `0` | pass |
| device kernel name | `_fused_attention_kernel` | same name in both profiler scopes | pass |
| launch configuration | `BLOCK_M`/`BLOCK_N`/`HEAD_DIM`/`num_warps`/`num_stages` unchanged | `128`/`128`/`64`/`4`/`1`; asserted from `CompiledKernel.metadata` at resolution | pass |
| forced `CompiledKernel.__getitem__` failure | correct output in the same call, sticky disable | `CompiledKernel.__getitem__` patched to raise → output bit-equal, no NaN, `_launcher_disabled == True`, handle cleared, subsequent call correct | pass |
| `_proven_kernel` swapped for a sentinel | correct output, sticky disable | output bit-equal, `_launcher_disabled == True`, handle cleared | pass |
| `S` change | handle **re-proves**, not disables | `S=40` → finite correct output, `_kernel is not None`, `_launcher_disabled == False`; restored output bit-equal | pass |
| stride change | handle **re-proves**, not disables | strides `(512,1024,1)` → `_kernel is not None`, `_launcher_disabled == False`; restored output bit-equal | pass |
| fast path genuinely taken | identity check has not silently fired | counting proxy on `_kernel` and `_proven_kernel`: **20 of 20 forwards** used the fast path; `_launcher_disabled == False`, `_kernel is not None` | pass |
| handle is not module state | never serialized | `state_dict() == []` | pass |
| public contract | constructor and forward signature unchanged | `__init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8)` and `forward(self, query, key, value) -> Tensor` identical to `e2_003`; `get_init_inputs() == [8, 64, 8]` | pass |
| base.py bytes | immutable | `86ac5703…` unchanged | pass |

### On the two instruments that matter

**Launch counting.** `triton.backends.ascend.driver.NPULauncher` is shadowed and
patching it silently reports zero. All counts here were taken by patching
`triton.runtime.driver.active.launcher_cls.__call__`, the real compiled
`ascend.NPULauncher`. Counts are `1.00` for both candidates.

**Fast-path detection.** The identity check `kernel is self._proven_kernel` is a
zero-cost `is` comparison and cannot be observed by counting launches. It was
observed by installing a counting proxy on **both** `_kernel` and
`_proven_kernel` simultaneously — they must be the same object or the
candidate's own check would disable the path. Result: 20 of 20 forwards took the
fast path with `_launcher_disabled == False`.

## Screening Evidence

Three paired runs in one Verifier turn, `base.py` (reference) against the
candidate, at `--warmup 50 --repeat 100`:

| Pair | Reference median ms | Candidate median ms | Speedup | Improvement | Correctness |
|---:|---:|---:|---:|---:|---|
| 1 | 0.358445 | 0.212025 | 1.690579 | 40.8487% | PASS |
| 2 | 0.358600 | 0.213610 | 1.678760 | 40.4322% | PASS |
| 3 | 0.352430 | 0.212505 | 1.658455 | 39.7029% | PASS |

Correctness passed on every pair and the candidate is ahead on all three.

## Interleaved Wall Timing (standard protocol)

- warmup: `50`
- repeat: `100`
- order: three paired reference/candidate runs in one Verifier turn
- reference_median_ms: `0.358445` (median of `0.352430`, `0.358445`, `0.358600`)
- candidate_median_ms: `0.212505` (median of `0.212025`, `0.212505`, `0.213610`)
- improvement_pct: `40.7148`

```text
improvement_pct = (0.358445 - 0.212505) / 0.358445 * 100 = 40.7148
```

Speedup spread across the three pairs: `1.658455` – `1.690579`, a range of
`1.937%` of ratio.

**This is a cumulative number, not this round's contribution.** As established in
round 004, the harness always pairs `base.py` against `--v1_file`, so this figure
includes round 003's allocation-reuse win. Roughly `17` of the `40.7` points are
inherited. The decisive measurement is the next section.

## Governing Bar: +5% versus the Previous Accepted Candidate

The previous accepted candidate is `e2_003`. Both candidates were measured in
**strict pair-by-pair alternation inside one window**, per the campaign rule
established in round 004: a speedup cancels drift only against a speedup drawn
from the same reference measurements.

| Pair | e2_005 ref ms | e2_005 cand ms | speedup | e2_003 ref ms | e2_003 cand ms | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.353115 | 0.210710 | 1.675834 | 0.364575 | 0.292180 | 1.247775 |
| 2 | 0.352980 | 0.209905 | 1.681618 | 0.352875 | 0.296900 | 1.188531 |
| 3 | 0.370555 | 0.214640 | 1.726402 | 0.353560 | 0.288395 | 1.225957 |
| 4 | 0.364895 | 0.212420 | 1.717800 | 0.355295 | 0.303285 | 1.171489 |
| 5 | 0.355475 | 0.211295 | 1.682364 | 0.342280 | 0.303145 | 1.129097 |
| **median** | | | **1.682364** | | | **1.188531** |

```text
ratio = 1.682364 / 1.188531 = 1.415498
gain  = +41.5498%
bar   = +5.00%
```

### Per-pair gains — the spread question

| Pair | ratio | gain |
|---:|---:|---:|
| 1 | 1.343058 | +34.3058% |
| 2 | 1.414870 | +41.4870% |
| 3 | 1.408207 | +40.8207% |
| 4 | 1.466339 | +46.6339% |
| 5 | 1.490008 | +49.0008% |
| **median** | **1.414870** | **+41.4870%** |

- **5 of 5 pairs clear the bar.**
- The **weakest** pair is `+34.3058%` — about seven times the threshold.
- The spread is `+34.31%` to `+49.00%`, and the bar sits far below the entire
  distribution. **There is no overlap with 5% and no plausible reading under
  which this is marginal.**

### Two ways of expressing the same win — do not confuse them

The bar is defined as a ratio of speedups, so its natural output is
`+41.5498%`. The underlying **wall-time** improvement is smaller and different:

```text
raw candidate medians, same window: e2_003 0.296900 -> e2_005 0.211295
raw wall improvement = 28.8329%

in-process paired wall lever: 298.130 -> 212.445 us
in-process wall improvement = 28.724%
```

Both are far above 5%. The difference is definitional: a ratio of ratios
amplifies, because `speedup` is itself a reciprocal of wall time. The governing
bar asks for the former; the physical win is the latter. Both are reported so
neither can be mistaken for the other.

### Non-decisive context: derived cross-turn bar

Reported for context only; the campaign rule forbids using it for the verdict.

```text
speedup(e2_003) in round 003's turn = 0.361050 / 0.298240 = 1.210602
derived bar                         = 1.210602 x 1.05      = 1.271132
e2_005 protocol median speedup      = 1.678760              -> clears

speedup(e2_003) in THIS turn        = 1.188531   (1.8% below round 003's reading)
```

The derived bar is cleared, but note that `e2_003`'s own speedup measured
`1.8%` lower in this turn than in round 003's turn, which is precisely the
cross-turn instability that makes this form non-decisive.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `applicable`
- hypothesis_id: `H-005`
- intervention: drive the resolved `CompiledKernel` object directly in `ModelNew.forward` instead of going through per-call `JITFunction` dispatch
- expected_causal_chain: the cached `CompiledKernel` is resolved once and reused → the steady-state forward stops paying per-call specialization, cache-key construction, and dispatch → `launch_path_us_per_call` falls from `183.740` toward the M2 value → `host_us_per_call` falls below `206.375` → device time and kernel count stay fixed → synchronized wall median decreases by at least five percent
- primary_metric: `wall_time`, expected improvement `5.0%`
- **Hypothesis verdict: `confirmed`**

### Mechanism observables

| Observable | Expectation | Reference (e2_003, this turn) | Candidate (e2_005) | Verdict |
|---|---|---|---|---|
| `launch_path_us_per_call` | decrease from the `183.740 us` baseline toward the M2 value | M0 `178.915` (in-turn baseline; report_003 read `183.740`, drift `-2.6%`) | **M2b `89.220`** | **confirmed**, `-89.695 us` (`-50.1%`) |
| `host_us_per_call` | decrease below the `206.375 us` forward-alone baseline | `203.485` (in-turn; report_003 read `206.375`, drift `-1.4%`) | **`115.185`** | **confirmed**, `-88.300 us` (`-43.4%`) |
| `device_us_per_call` | unchanged at approximately `13.4224` | `13.4816` | `13.4780` | **confirmed unchanged**, `-0.0036 us` (`-0.027%`) |
| `kernel_count_per_call` | unchanged at `1.00` | `1.00` | `1.00` | **confirmed** |

Every link of the causal chain held, including the final wall link. Unlike round
004, this round is `confirmed` rather than `partially-confirmed`.

Note on `launch_path_us_per_call`: the decision's expected value of `66.895`
corresponds to M2a, not to the shipped M2b. The shipped value `89.220` is
correct; see the Mechanism Verification section.

## Profiler Evidence

- profiler_applicability: `required` (both control observables are device-side)
- profiler_level: `targeted`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable`
- mode: `forward`; warmup `20`

Scopes captured separately and summarized independently. The reference scope
directory has now accumulated round-004 and round-005 captures, so each summary
was given its explicit round-005 `ai_core_op_summary.db` path.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_triton_mm_encoder_attention_e2_003 | 674.08 | 13.4816 | 50 | 1.00 | 0.299880 | 0.0450 |
| candidate_triton_mm_encoder_attention_e2_005 | 673.90 | 13.4780 | 50 | 1.00 | 0.210760 | 0.0639 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
```

Wall values are both at `warmup 200 / repeat 500`. The candidate value
`0.210760` is the profiler process's own `time_forward` result; the reference
value `0.299880` comes from a dedicated `200/500` run of `e2_003`, because the
`v0` the profiler run prints is `base.py` (`0.351085`), not this scope. The
chrome trace cannot supply it — `export_chrome_trace` runs per scope inside the
loop, so the candidate export overwrites the reference one.

### Reference Top Kernels (reference_triton_mm_encoder_attention_e2_003)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _fused_attention_kernel | 50 | 1.0 | 674.08 | 13.4816 |

### Candidate Top Kernels (candidate_triton_mm_encoder_attention_e2_005)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _fused_attention_kernel | 50 | 1.0 | 673.90 | 13.4780 |

**Control observables held.** Device time moved `-0.0036 us/call` (`-0.027%`)
and kernel count stayed at `1.00`, with the same kernel name. Against
`report_003`'s `13.4224`, this turn's re-measured `e2_003` reads `13.4816`
(`+0.44%` machine drift), so the in-turn reference is the right comparison and
it is flat.

## Level 2 Host Decomposition

One process, one regime (`warmup 50`, `repeat 100`, median of 5 blocks), queue
drained before each timed call, `e2_003` and `e2_005` interleaved at block level.

| Quantity | e2_003 (canonical) | e2_005 (candidate) |
|---|---:|---:|
| (a) harness wall (`auto_bench.time_forward`) | 298.130 | 212.445 |
| (b) `ModelNew.forward` alone, no synchronize | 203.485 | 115.185 |
| (c) `forward` + `torch.npu.synchronize()` | 256.185 | 162.605 |
| (c) - (b) synchronize term | 52.700 | 47.420 |
| (a) - (b) everything outside `forward` | 94.645 | 97.260 |

Bare launch paths, measured outside `ModelNew.forward` with a preallocated
output:

| Quantity | us/call | per-block |
|---|---:|---|
| M0 proven `kernel[grid](...)` | 178.915 | 179.4, 179.0, 178.9, 172.1, 173.5 |
| **M2b per-call stream — shipped** | **89.220** | 89.2, 91.0, 91.3, 86.1, 87.2 |
| M2a cached stream — not shipped | 66.220 | 66.2, 67.7, 67.9, 64.6, 65.2 |
| M0 − M2b saving | **89.695** | negative in all 5 blocks |
| per-call stream resolution cost | 23.000 | — |

```text
forward lever (b)  median-of-medians -88.300 us  (-43.394%)
forward lever (b)  paired median     -86.900 us  (-42.706%)
wall lever    (a)  median-of-medians -85.685 us  (-28.741%)
wall lever    (a)  paired median     -85.635 us  (-28.724%)
adoption threshold (report_003 wall)  14.871 us   -> cleared ~5.8x
residual wrapper e2_003 (b - M0)      24.570 us
residual wrapper e2_005 (b - M2b)     25.965 us
```

Per-block paired differences (`e2_005 - e2_003`), all negative:

```text
(a) -85.7, -81.7, -86.1, -84.3, -85.6   median -85.635 us
(b) -88.3, -87.4, -86.9, -85.1, -85.6   median -86.900 us
(c) -93.6, -92.3, -92.6, -89.0, -90.8   median -92.325 us
```

**Coder's lever reproduces.** Coder reported `219.610 → 128.655 us`
(`-90.955 us`); Verifier measures `203.485 → 115.185 us` (`-88.300 us`
median-of-medians, `-86.900 us` paired). Absolute levels differ by `~7%` between
turns, which is normal drift; the lever agrees within `2.7 us`.

Three further observations:

1. **Wall conversion is much better than round 004's `~75%`.** Forward fell
   `-88.300 us` and wall fell `-85.685 us`, i.e. `97.0%` propagation. The
   `+2.6 us` remainder is a slightly larger non-forward term
   (`94.645 → 97.260 us`).
2. **The residual wrapper is unchanged** (`24.570` vs `25.965 us`), consistent
   with the round's declared scope: it removed launch cost, not wrapper cost.
3. **The harness-fixed term is now the largest single slice.** At `97.260 us` it
   is `45.8%` of the `212.445 us` wall, up from `30.6%` before this round,
   simply because everything else shrank.

## Attribution

**The wall movement is essentially entirely host. Device contributes nothing
measurable.**

- Wall fell `-85.685 us/call` in-process (`298.130 → 212.445`), and `-85.635 us`
  on the paired median.
- Device time moved `-0.0036 us/call` (`13.4816 → 13.4780`), i.e. `-0.027%`. As
  a fraction of the wall change that is `0.004%`.
- Device's **maximum possible** contribution is bounded by the entire device
  budget: eliminating device time completely would remove `13.478 us`, which is
  `6.34%` of the new `212.445 us` wall — that is the `4.09%` device ceiling
  recomputed against the new, smaller wall. Observed device movement is three
  orders of magnitude below that bound.
- Kernel count stayed at `1.00` and the device kernel name is unchanged, so
  there is no launch-count story either.
- The host link is directly measured and large: `ModelNew.forward` alone fell
  `203.485 → 115.185 us`, and the bare launch path fell `178.915 → 89.220 us`.

The mechanism is confirmed and the attribution is unambiguous: this is a host
win of `-85.7 us`, of which `-89.7 us` is the launch-path saving, partially
offset by a `+2.6 us` growth in the non-forward term.

## Robustness Assessment

The team lead asked whether the 5% verdict is robust given the observed spread.
It is, emphatically.

| Evidence | Reading |
|---|---|
| 5 of 5 alternating pairs clear the bar | unanimous |
| weakest pair `+34.3058%` vs a `+5%` bar | ~7x the threshold |
| spread `+34.31%` to `+49.00%` | the bar is outside the distribution entirely |
| in-process per-block diffs | all 5 negative, range `-81.7` to `-86.1 us`, no sign change |
| in-process paired wall lever `-85.635 us` vs a `14.871 us` threshold | ~5.8x |
| round-004 comparison | there, the closest estimator missed by `1.056 us`, i.e. `0.33%` of wall; here the miss would have to be `70 us` |

For context on scale: round 004 was a real `2-4.5%` effect whose verdict sat
inside the noise band, and it was correctly recorded as `no-improvement`. This
round's effect is roughly an order of magnitude larger than that noise band. On
this machine, which drifts `5-7%` within a turn, an effect of `28.8%` wall
(`41.5%` on the bar metric) is not something drift can plausibly manufacture or
hide.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification of round 005 | `not-applicable` | `bf54cea2a1fcdafd8916c2e0bf607766a6e7ffc2981fd956e18e92bf51b88b26` | correctness pass on first attempt; no repair needed |

No environment incident occurred; no `incident_005_*.md` was written. Coder
reported four in-round defects of its own (attempts 1, 3 and 4 of its ledger:
a probe `KeyError`, and a design gap where an identity mismatch self-healed
instead of disabling sticky), all found and fixed before `candidate-ready`.
None was a Verifier retry.

Two diagnostic probes under `log/` were written by Verifier
(`round_005_host_decomposition.py`, `round_005_guardrails.py`); neither is a
tracked campaign artifact and neither modifies `base.py`, the harness, the
candidate, or any decision.

## evidence_for_next_round

- **Adoption cleared decisively.** Governing bar `+41.5498%`; wall improvement
  `28.83%` over the canonical kernel; in-process paired wall lever `-85.635 us`
  against a `14.871 us` threshold.
- **M2 works and M2b is the right form.** `89.220 us/call` bare, `-89.695 us`
  versus M0, bit-identical, `1.00` launches, NaN-clean, and the stream is
  resolved per call as the Host Plan requires.
- **The single largest remaining host lever is `23.000 us`: the per-call stream
  resolution.** M2a measures `66.220 us` against M2b's `89.220 us`. Two
  independent probes (Coder's and mine) agree within `1 us`. Taking it requires
  a decision amending `device_stream_behavior`; the measurement is already on
  disk in `log/probes/round_005_mechanism_probe.json`. Note the caveat I found:
  the cached-stream form cannot be written with `torch.npu.current_stream()` on
  this runtime, so the amendment must also specify how the handle is obtained.
- **The bottleneck has changed shape.** Against the new `212.445 us` wall:
  harness-fixed `97.260 us` (`45.8%`), launch path `89.220 us` (`42.0%`),
  residual wrapper `25.965 us` (`12.2%`), device `13.478 us` (inside the sync
  term). **The harness-fixed term is now the largest slice and it is
  unreachable**, so further rounds will see sharply diminishing returns: every
  further microsecond removed is a larger fraction of a smaller host budget, but
  a smaller fraction of a wall that is now nearly half harness overhead.
- **Wall propagation was `97%` this round**, not the `~75%` seen in round 004.
  Useful for predicting the next round, though the reason is not established and
  may be specific to how much of the saving sits before the synchronize.
- **Device is closed and now nearly irrelevant.** `13.478 us/call` at `1.00`
  kernels. The device-only ceiling recomputed against the new wall is `6.34%`,
  still the same absolute `13.478 us` budget; device work remains a dead end.
- **Methodological note carried forward:** the `base.py`-referenced protocol
  number is cumulative (`40.71%` here, of which roughly `17` points are round
  003's) and must never be read as a round's contribution. The strict
  pair-by-pair alternation in one window remains the only decisive form; the
  cross-turn derived bar is reported for context only and is non-decisive.
- **A definitional caution for whoever measures next:** the bar's output
  (`+41.55%`, a ratio of ratios) is not the wall-time improvement
  (`+28.83%`). Both clear 5% here by a wide margin, but on a smaller effect they
  can diverge enough to matter. Report both.
- **Carried forward unchanged:** the kernel requires `S <= 128` (campaign shape
  `S=83`); the second `tl.dot` tile `(128,64,128)` compiles and is numerically
  correct but was not one of the eleven probed tiles; the buffer-reuse invariant
  depends on full store coverage and must be revisited if a later round
  introduces a masked or partial store; `../triton_attn_001.py` is preserved
  intact.

## Stop Recommendation

- recommendation: `continue`
- evidence: an accepted round just advanced performance substantially, so the
  no-improvement streak resets from `1` to `0` against a limit of `3`, and the
  failed-attempt streak stays at `0`. The round budget is `4` of `20`, far from
  `round-budget-exhausted`, and no target is set, so `target-reached` does not
  apply. Measured headroom still exists and is now precisely identified:
  `23.000 us` behind a `device_stream_behavior` amendment, then the remaining
  launch path, then a `25.965 us` wrapper. Nothing is blocked.
- caveat worth recording: the reachable fraction of wall is shrinking. The
  harness-fixed term is now `45.8%` of wall and is unreachable, so the epoch
  should expect each subsequent round to deliver a smaller wall improvement even
  when its mechanism is just as effective.

## Exact Reproduction Commands

Correctness gate:

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_005.py --warmup 5 --repeat 10 --full-traceback
```

Authoritative timing (three pairs, one turn):

```bash
cd /workspace/kernelswift-dev-4ff2094
for i in 1 2 3; do
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_005.py \
    --warmup 50 --repeat 100
done
```

Decisive measurement — strict pair-by-pair alternation in one window:

```bash
cd /workspace/kernelswift-dev-4ff2094
for i in 1 2 3 4 5; do
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_005.py --warmup 50 --repeat 100
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py --warmup 50 --repeat 100
done
```

Profiler:

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
  --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_005.py \
  --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_005_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/profiling_data/reference_triton_mm_encoder_attention_e2_003/profiling_data/16458e336fc3_129637_20260830082241201_ascend_pt/PROF_000001_20260830082241226_00129637HBOFNMMA/device_0/sqlite/ai_core_op_summary.db" --iterations 50 --scope reference_triton_mm_encoder_attention_e2_003 --wall-ms 0.299880

python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/profiling_data/candidate_triton_mm_encoder_attention_e2_005/profiling_data/16458e336fc3_129637_20260830082245015_ascend_pt/PROF_000002_20260830082245038_00129637FMDBJRGG/device_0/sqlite/ai_core_op_summary.db" --iterations 50 --scope candidate_triton_mm_encoder_attention_e2_005 --wall-ms 0.210760
```

Reference-scope wall (the profiler run's own `v0` is `base.py`, not this scope):

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
  --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py --warmup 200 --repeat 500
```

Level 2 and guardrail diagnostics (Verifier-owned, under `log/`, not campaign artifacts):

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_005_host_decomposition.py
python3 kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_005_guardrails.py
```

Kernel-definition identity and mechanism check:

```bash
cd /workspace/kernelswift-dev-4ff2094/kernels/track1-triton/mm_encoder_attention/ascend/epoch2
diff <(sed -n '1,76p' triton_mm_encoder_attention_e2_003.py) <(sed -n '1,76p' triton_mm_encoder_attention_e2_005.py) && echo KERNEL_DEF_IDENTICAL=yes
grep -n "kernel\[grid\]\|stream" triton_mm_encoder_attention_e2_005.py
```
