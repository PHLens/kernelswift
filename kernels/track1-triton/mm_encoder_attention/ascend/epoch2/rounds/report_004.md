# Report 004

Result: no-improvement

> **Read this first.** Three different bars were applied to this round. **All
> three adoption tests fail**; only the (superseded) epoch-1 bar clears, and it
> does so for the incumbent too.
>
> | Bar | Result | Verdict |
> |---|---|---|
> | **+5% vs previous round's accepted candidate** (governing, per Orchestrator advisory 2) | **`+2.8874%`** ratio-of-speedups, same-turn interleaved | **FAILS** |
> | +5% over canonical `e2_003` (discriminating form) | `2.5939%` raw / `1.9317%` normalized / `4.35%` in-process | **FAILS** |
> | Derived cross-turn bar `1.271132` | `1.270171` (protocol window) / `1.217744` (interleaved window) | **FAILS** both |
> | ~~epoch-1 deliverable, bar `1.02626`~~ *(superseded, retained below)* | median speedup `1.270171` | clears, but non-discriminating |
>
> Classified **`no-improvement`**. The mechanism is real and confirmed; the
> magnitude is roughly `2-4.5%` against a `5%` bar. See the two bar sections and
> the Margin and Stability Assessment for the full reasoning.

## Identity

- Round: `004`
- Decision: `rounds/decision_004.md`
- Decision SHA256: `30758ad4dd30ccb0087534e47f61ea0443bdeead40ba64d41c28dd052c397088`
- Decision kind: `optimization`; change scope `host` / change family `launch-path-reduction`
- Hypothesis ID: `H-004`
- Sketch: `rounds/sketch_004.json` (sha256 `d3e52f6af032014381908e03e87a6b1c3f5694090686df2af3bfe3a6d9474dbf`)
- Candidate: `triton_mm_encoder_attention_e2_004.py`
- Candidate SHA256: `f5aa1d709e4deeb1562757d795dd4da41217238dd15c53d33c2c338da1938020`
- Accepted reference (campaign): `triton_mm_encoder_attention_e2_003.py`
- Accepted reference SHA256: `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe`
- Accepted reference report: `rounds/report_003.md`
- Paired `--v0_file`: `base.py` (the harness requires it to define `Model`)
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged)
- Coder result SHA256: `9c8c46ef1b58233e464a30022fd2b0dedf2fce7b95410a501d95e2e24ac59e0e`
- Capability gate: `lifecycle.fast-launcher` — probe outcome **`proven`** (Coder), mechanism `M1 fast_libentry`
- Profile snapshot: `state/implementation_profile_snapshot/profile.yaml` (sha256 `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321`, **unamended**, still `Unknown`)
- Capability claim: `state/project_capability_claim.json` (sha256 `a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d`)
- Runtime fingerprint: `project.md#runtime-fingerprint` (Ascend910B4, torch 2.7.1+cpu / torch_npu 2.7.1.post4 / triton 3.2.0 / CANN 9.0.0)
- Measurement fingerprint: `1b1822d7b74a8cd41411a27fcbc18a89cb50b1cfefb9fdac2585cdd520e9a79a`
- verification_tier: candidate
- screening_pairs: `3` (used as the authoritative timing pairs, per the round-001/003 convention)
- Level 2: `targeted` host decomposition (run)

## Capability Gate

The Decision-scoped probe is Coder's to run and adjudicate; Verifier does not
re-litigate it. Verifier confirms the artifacts exist and re-verifies the two
invariants the gate rests on that are cheap to check independently:

| Gate item | Observation | Verdict |
|---|---|---|
| probe artifacts retained under `log/probes/` | all five present (`round_004_launch_abi_probe.{py,json}`, `round_004_candidate_conformance.{py,json}`, `round_004_probe_evidence.md`) | pass |
| the fast path launches the same compiled kernel | bit-identical output to `e2_003`, `max_abs_diff = 0.0`; kernel name `_fused_attention_kernel` in both profiler scopes | pass |
| the fast path is actually taken in steady state | counting proxy: **20 of 20 forwards** used the fast path; `_launcher is not None`, `_launcher_disabled == False` | pass |
| exactly one launch per call | counted through the **real** launcher class: `1.00` for both `e2_003` and `e2_004` | pass |
| frozen snapshot not amended | `a2c3e2e4…` unchanged; `lifecycle.fast-launcher` still `Unknown` | pass |

The gate is therefore `proven`, and only the `accepted` / `no-improvement` rows
of the decision's table are live.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict |
|---|---|---|---|
| correctness | `atol=1e-2`, `rtol=1e-2`, `equal_nan=True` | `PASS accuracy` on the smoke run and on all three interleaved pairs; `Summary: 1 passed, 0 failed, 1 total` | pass |
| output shape / dtype / device / contiguity | unchanged | `(2, 83, 512)`, `torch.float16`, `npu:0`, `is_contiguous() == True` | pass |
| bit-identity vs accepted kernel | `max_abs_diff = 0.0` | `torch.equal(cand, e2_003_out) == True` | pass |
| no aliasing of q/k/v | returned tensor shares storage with none | `False` | pass |
| cached buffer fully overwritten | kernel store covers every element | buffer filled with NaN then reused: `NaN leaked = False`, `84992/84992` finite, pointer stable, output still bit-equal | pass |
| kernel launch count | stays at one | `1.00` for `e2_003` and `1.00` for `e2_004`, counted through the real launcher class | pass |
| kernel definition unchanged | byte-identical to `e2_003` | `diff` lines 1-76 → exit `0`, no output | pass |
| launch configuration | `BLOCK_M`/`BLOCK_N`/`HEAD_DIM`/`num_warps`/`num_stages` unchanged | `128`/`128`/`64`/`4`/`1`; device kernel name `_fused_attention_kernel` in both scopes | pass |
| fallback on forced fast-path failure | degrades to the proven launch, never a wrong answer | forced exception → output bit-equal to `e2_003`, no NaN, `_launcher_disabled == True`, handle cleared, subsequent call still correct | pass |
| fallback on kernel-identity mismatch | same | sentinel `_proven_kernel` → output bit-equal, sticky disable set, handle cleared | pass |
| cache-key change | clears and re-proves the handle | stride change → key changed, handle re-proven, restored output bit-equal, not disabled | pass |
| public contract | constructor and forward signature unchanged | `__init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8)` and `forward(self, query, key, value) -> Tensor` identical to `e2_003`; `get_init_inputs() == [8, 64, 8]` | pass |
| handle is not module state | never serialized | `state_dict() == []` | pass |
| base.py bytes | immutable | `86ac5703…` unchanged | pass |

### On the launch-count instrument

Coder reported that patching `triton.backends.ascend.driver.NPULauncher` reports
zero launches because that class is shadowed. Verifier confirmed this directly:

```text
shadowed Python NPULauncher : <class 'triton.backends.ascend.driver.NPULauncher'>
real launcher class         : <class 'ascend.NPULauncher'>
same object?                : False
```

Every launch count in this report was taken by patching
`triton.runtime.driver.active.launcher_cls.__call__`, which is the class actually
in use. Counts are `1.00` for both candidates.

## Screening Evidence

Three paired runs in one Verifier turn, `base.py` (reference) against the
candidate, at `--warmup 50 --repeat 100`:

| Pair | Reference median ms | Candidate median ms | Speedup | Improvement | Correctness |
|---:|---:|---:|---:|---:|---|
| 1 | 0.362315 | 0.287335 | 1.260950 | 20.6947% | PASS |
| 2 | 0.380105 | 0.299255 | 1.270171 | 21.2704% | PASS |
| 3 | 0.376040 | 0.295850 | 1.271050 | 21.3249% | PASS |

Correctness passed on every pair. The candidate is ahead on all three.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: three paired reference/candidate runs in one Verifier turn
- reference_median_ms: `0.376040` (median of `0.362315`, `0.376040`, `0.380105`)
- candidate_median_ms: `0.295850` (median of `0.287335`, `0.295850`, `0.299255`)
- improvement_pct: `21.3249`

```text
improvement_pct = (0.376040 - 0.295850) / 0.376040 * 100 = 21.3249
```

**This number is not attributable to this round's change.** It is measured
against `base.py`, which is not the canonical kernel. `base.py` drifted up
`4.15%` between the round-003 turn (`0.361050`) and this turn (`0.376040`),
while the candidate moved only `0.298240` -> `0.295850`. Roughly `17.4` of the
`21.3` points are inherited from round 003; this round's own contribution is the
subject of the next section.

## Epoch-1 Deliverable Bar (Orchestrator advisory 1 — superseded, retained for completeness)

**Superseded.** Advisory 2 replaced this bar with "+5% versus the previous
round's accepted candidate". This section is kept because the measurement costs
nothing to retain, but it is not the test to classify on.

Reported separately and clearly labelled, because it is a different test from
the 5% threshold.

**The bar.** Read from the epoch-1 artifact `../rounds/report_001.md`, not from
memory: `reference_median_ms 0.348605`, `candidate_median_ms 0.339685`,
`improvement_pct 2.5588`, per-pair speedups `1.026x / 1.028x / 1.005x`.

```text
epoch-1 bar = 0.348605 / 0.339685 = 1.02626
```

**Round 004 against the bar:**

| Pair | Reference ms | Candidate ms | Speedup | vs bar 1.02626 |
|---:|---:|---:|---:|---|
| 1 | 0.362315 | 0.287335 | 1.260950 | +22.87% |
| 2 | 0.380105 | 0.299255 | 1.270171 | +23.77% |
| 3 | 0.376040 | 0.295850 | 1.271050 | +23.86% |
| **median** | — | — | **1.270171** | **+23.77%** |

- median speedup across the three pairs: **`1.270171`**
- speedup spread: `1.260950` – `1.271050`, a range of **0.80% of ratio**
- all three pairs clear the bar; the **worst** pair clears it by `22.87%`
- margin is `23.77%` of ratio against a spread of `0.80%` — about **30x** the spread

**Assessment: this test is not marginal in any way.** The Orchestrator predicted
as much, and the data agrees. Even the single worst of the three pairs beats the
epoch-1 bar by more than twenty-two points of ratio. No plausible drift on this
machine (observed `~5-7%` within a turn) comes close to closing a `23.77%` gap.

**One objection I am obliged to raise.** The bar is cleared, but it does not
discriminate. The incumbent `e2_003` also clears it, by a wide margin:

| Candidate | Speedup vs same-turn `base.py` | vs bar 1.02626 |
|---|---:|---|
| `e2_003` (incumbent, canonical) | 1.189874 | CLEARS |
| `e2_004` (this round) | 1.213311 (control) / 1.270171 (protocol) | CLEARS |

So "beat the epoch-1 deliverable" is satisfied whether or not this round is
adopted. It tells us the *submission* is good; it cannot tell us whether
`e2_004` should *replace* `e2_003`. Only the 5%-over-canonical test answers that,
and that test fails. I flag this because a number that cannot change the
decision is not a sound basis for the decision, however robust it is.

## Controlled Comparison against the Canonical e2_003 — the discriminating test

The harness always pairs `base.py` against `--v1_file`, so the protocol number
above cannot isolate this round's effect. Twelve interleaved blocks were run in
this turn, six with `e2_003` measured first and six with `e2_004` measured first,
all at `--warmup 50 --repeat 100`.

### Per-block samples

| Block | order | `base.py` with e2_003 | e2_003 | `base.py` with e2_004 | e2_004 |
|---:|---|---:|---:|---:|---:|
| 1 | 003 first | 0.343685 | 0.291680 | 0.347740 | 0.292300 |
| 2 | 003 first | 0.352000 | 0.298590 | 0.354085 | 0.288730 |
| 3 | 003 first | 0.360560 | 0.299920 | 0.371695 | 0.299970 |
| 4 | 003 first | 0.352685 | 0.307580 | 0.360205 | 0.297150 |
| 5 | 003 first | 0.360785 | 0.304000 | 0.350135 | 0.291925 |
| 6 | 003 first | 0.364430 | 0.306615 | 0.348600 | 0.294360 |
| 7 | 004 first | 0.369430 | 0.300425 | 0.377065 | 0.295555 |
| 8 | 004 first | 0.353210 | 0.291945 | 0.353610 | 0.287700 |
| 9 | 004 first | 0.369515 | 0.296420 | 0.368150 | 0.287115 |
| 10 | 004 first | 0.367675 | 0.300620 | 0.354740 | 0.279275 |
| 11 | 004 first | 0.346165 | 0.299950 | 0.354210 | 0.292010 |
| 12 | 004 first | 0.333825 | 0.290795 | 0.362050 | 0.292760 |

### Aggregate

```text
e2_003 candidate median  = 0.299935 ms   (base median 0.356885)
e2_004 candidate median  = 0.292155 ms   (base median 0.354475)

RAW improvement             = (0.299935 - 0.292155) / 0.299935 * 100 =  2.5939%   (7.780 us)
BASE-NORMALIZED improvement = (0.840607 - 0.824372) / 0.840607 * 100 =  1.9317%   (5.794 us)
```

Both are **below 5%**.

### Paired per-block differences

| Statistic | Value |
|---|---:|
| median of 12 paired diffs (`e2_004 - e2_003`) | **-8.623 us** |
| mean of 12 paired diffs | -7.474 us |
| range | -21.35 us to +1.96 us |
| blocks where `e2_004` is faster | 11 of 12 |
| blocks clearing 5% (base-normalized) | 2 of 12 |
| blocks negative | 1 of 12 |

### Ordering-bias check

| Ordering | Median paired diff |
|---|---:|
| blocks 1-6, `e2_003` measured first | -10.145 us |
| blocks 7-12, `e2_004` measured first | -6.405 us |

Both orderings favour `e2_004`, so the effect is not an artifact of which
candidate ran first. The two orderings do disagree in magnitude by `3.7 us`,
which is itself a measure of how noisy this comparison is.

### Is 5% inside the spread?

The per-block base-normalized improvements range `-0.36%` to `+7.17%`, so the
5% line **does** fall inside the per-block spread, and 2 of 12 individual blocks
exceed it. But the central estimate is far below it: median `+3.30%`, mean
`+3.13%`. A threshold that only the top sixth of a distribution clears is not a
threshold the distribution's centre reaches.

## Adoption Bar: +5% versus the Previous Round's Accepted Candidate (Orchestrator advisory 2 — the governing test)

The stated bar:

```text
accept when  speedup(candidate) / speedup(last_accepted_kernel) - 1  >=  5%
where speedup = reference_median / candidate_median   (base.py as v0, same turn)
```

The previous accepted candidate is `triton_mm_encoder_attention_e2_003.py`, so
it was measured as a control block under the identical protocol, **strictly
alternating with `e2_004`, four pairs each, in this turn**:

| Pair | e2_004 ref ms | e2_004 cand ms | speedup(e2_004) | e2_003 ref ms | e2_003 cand ms | speedup(e2_003) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.350730 | 0.288005 | 1.217791 | 0.350265 | 0.292450 | 1.197692 |
| 2 | 0.358325 | 0.295770 | 1.211499 | 0.345970 | 0.295295 | 1.171608 |
| 3 | 0.355535 | 0.289480 | 1.228185 | 0.358745 | 0.302240 | 1.186954 |
| 4 | 0.351695 | 0.288820 | 1.217696 | 0.357425 | 0.302855 | 1.180185 |
| **median** | | | **1.217744** | | | **1.183570** |

```text
ratio = 1.217744 / 1.183570 = 1.028874
gain  = +2.8874%
bar   = +5.00%
```

**Verdict: FAILS, by 2.11 points of margin.**

### Robustness of this verdict — 5% is not inside the spread

Per-pair ratios, which is where the spread question is answered:

| Pair | ratio | gain |
|---:|---:|---:|
| 1 | 1.016782 | +1.678% |
| 2 | 1.034049 | +3.405% |
| 3 | 1.034736 | +3.474% |
| 4 | 1.031784 | +3.178% |
| **median** | **1.032915** | **+3.292%** |

Every one of the four pairs is below 5%, and the **best** pair reaches only
`+3.474%`. The 5% bar is therefore **not** inside this spread at all — the whole
distribution sits `1.5` to `3.3` points below it. This is a considerably more
robust verdict than the 12-block raw comparison, whose per-block spread did
reach past 5% in 2 of 12 blocks.

### The cross-turn derived bar also fails — and the near-miss should be disregarded

```text
speedup(e2_003) from round 003 = 0.361050 / 0.298240 = 1.210602
derived bar                    = 1.210602 x 1.05      = 1.271132

e2_004, protocol window   (3 pairs) = 1.270171  -> -0.0756% vs bar
e2_004, interleaved window (4 pairs) = 1.217744  -> -4.2001% vs bar
```

Both fail. The protocol-window figure misses by only `0.000961` of ratio
(`-0.0756%`), which looks like a photo finish. **It is not one, and it should
not be read as a near-pass.** The reason is below.

### Important methodological finding: speedup does not fully cancel drift here

The premise of the speedup metric is that reference and candidate drift
together, so their ratio is stable. On this machine in this turn they did not:

| Window | `base.py` median | e2_004 candidate median | speedup |
|---|---:|---:|---:|
| protocol (3 pairs, earlier in turn) | 0.376040 | 0.295850 | 1.270171 |
| interleaved (4 pairs, later in turn) | 0.353615 | 0.289150 | 1.217744 |
| movement | **-6.0%** | **-2.3%** | **-4.3%** |

`e2_004`'s own measured speedup swung **`4.31%` between two windows of the same
turn**, purely because `base.py` moved `6.0%` while the candidate moved `2.3%`.
Two consequences:

1. **Any cross-window speedup comparison is unsafe at the ~4% level**, and that
   includes the cross-turn derived bar. A `0.0756%` gap is two orders of
   magnitude smaller than the method's own window-to-window instability, so it
   carries no information.
2. **The only defensible form of this test is the strictly interleaved
   same-window ratio**, which is the `+2.8874%` reported above.

### Why this is below the ~+6.60% implied by Coder's lever

Coder's in-process forward lever of `-18.470 us` on a `~300 us` wall implies
about `+6.6%` before conversion losses. Two things close most of that gap:

- the wall conversion is lossy: only `-14.330 us` of the `-18.965 us` forward
  lever reaches the synchronized wall (about `75%`);
- the speedup-ratio metric amplifies `base.py` movement, as shown above, adding
  several points of instability in either direction.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `applicable`
- hypothesis_id: `H-004`
- intervention: replace the per-call Triton `JITFunction` dispatch in `ModelNew.forward` with the cheaper `M1 fast_libentry` launch path established by the Decision-scoped probe
- expected_causal_chain: a probe establishes a cheaper launch path → the steady-state forward stops paying per-call specialization, cache-key construction, and dispatch → `launch_path_us_per_call` falls below `183.740 us` → `host_us_per_call` falls below `206.375 us` → device time and kernel count stay fixed → synchronized wall median decreases by at least five percent
- primary_metric: `wall_time`, expected improvement `5.0%`
- **Hypothesis verdict: `partially-confirmed`**

### Mechanism observables

| Observable | Expectation | Reference (e2_003, this turn) | Candidate (e2_004) | Verdict |
|---|---|---|---|---|
| `launch_path_us_per_call` | decrease below the `183.740 us` bare-launch baseline | M0 `192.255` (in-turn baseline; report_003 read `183.740`, drift `+4.63%`) | M1 **`172.950`** | **confirmed**, `-19.305 us` (`-10.04%`); below `183.740` on the absolute test too |
| `host_us_per_call` | decrease below the `206.375 us` forward-alone baseline | `221.605` (in-turn; report_003 read `206.375`, drift `+7.4%`) | **`202.640`** | **confirmed**, `-18.965 us` (`-8.56%`); below `206.375` on the absolute test too |
| `device_us_per_call` | unchanged at approximately `13.4224` | `13.3272` | `13.3228` | **confirmed unchanged**, `-0.0044 us` (`-0.03%`) |
| `kernel_count_per_call` | unchanged at `1.00` | `1.00` | `1.00` | **confirmed** |

All four mechanism observables moved as predicted. The launch path is genuinely
cheaper, host time genuinely fell, and device stayed fixed. The final link of
the causal chain — a ≥5% synchronized wall decrease — did **not** hold against
the canonical reference. Hence `partially-confirmed`, not `confirmed`.

## Profiler Evidence

- profiler_applicability: `required` (both control observables are device-side)
- profiler_level: `targeted`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable`
- mode: `forward`; warmup `20`

Scopes captured separately (`ASCEND_WORK_PATH` per scope) and summarized
independently, each given its explicit `ai_core_op_summary.db` path.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_triton_mm_encoder_attention_e2_003 | 666.36 | 13.3272 | 50 | 1.00 | 0.297245 | 0.0448 |
| candidate_triton_mm_encoder_attention_e2_004 | 666.14 | 13.3228 | 50 | 1.00 | 0.280915 | 0.0474 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
```

Wall values are both at `warmup 200 / repeat 500`. The candidate value
`0.280915` is the profiler process's own `time_forward` result; the reference
value `0.297245` comes from a dedicated `200/500` run of `e2_003`, because the
`v0` the profiler run prints is `base.py` (`0.364495`), not this scope. The
chrome trace cannot supply it — `export_chrome_trace` runs per scope inside the
loop, so the candidate export overwrites the reference one.

### Reference Top Kernels (reference_triton_mm_encoder_attention_e2_003)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _fused_attention_kernel | 50 | 1.0 | 666.36 | 13.3272 |

### Candidate Top Kernels (candidate_triton_mm_encoder_attention_e2_004)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _fused_attention_kernel | 50 | 1.0 | 666.14 | 13.3228 |

**Control observables held.** Device time moved `-0.0044 us/call` (`-0.03%`)
and kernel count stayed at `1.00`, with the same kernel name. Against
`report_003`'s `13.4224`, this turn's re-measured `e2_003` reads `13.3272`
(`-0.71%` machine drift), so the in-turn reference is the right comparison and
it is flat. Attribution to the host launch link is intact: device cannot explain
any part of the wall movement.

## Level 2 Host Decomposition

One process, one regime (`warmup 50`, `repeat 100`, median of 5 blocks), queue
drained before each timed call, `e2_003` and `e2_004` interleaved at block level.

| Quantity | e2_003 (canonical) | e2_004 (candidate) |
|---|---:|---:|
| (a) harness wall (`auto_bench.time_forward`) | 317.325 | 302.995 |
| (b) `ModelNew.forward` alone, no synchronize | 221.605 | 202.640 |
| (c) `forward` + `torch.npu.synchronize()` | 273.350 | 257.090 |
| (c) - (b) synchronize term | 51.745 | 54.450 |
| (a) - (b) everything outside `forward` | 95.720 | 100.355 |

Bare launch paths, measured outside `ModelNew.forward` with a preallocated
output, giving the `launch_path_us_per_call` observable:

| Quantity | us/call | per-block |
|---|---:|---|
| (d0) bare M0 `_fused_attention_kernel[grid](...)` | 192.255 | 190.1, 189.0, 194.5, 195.2, 192.3 |
| (d1) bare M1 `LibEntry(...)[grid](...)` | 172.950 | 170.8, 174.6, 172.9, 173.3, 171.8 |
| (d0) - (d1) launch-path saving | **19.305** | consistently negative in all 5 blocks |

```text
forward-level lever (b, median of medians) = 202.640 - 221.605 = -18.965 us  (-8.56%)
forward-level lever (b, paired median)     =                      -16.260 us  (-7.34%)
wall lever          (a, median of medians) = 302.995 - 317.325 = -14.330 us  (-4.52%)
wall lever          (a, paired median)     =                      -13.815 us  (-4.35%)
adoption threshold (from report_003 wall)  =                       14.871 us
residual wrapper e2_003 (b - d0)           =                       29.350 us
residual wrapper e2_004 (b - d1)           =                       29.690 us
```

### Two observations that matter

1. **Coder's lever reproduces.** Coder reported `-18.470 us` at the forward
   level; Verifier measures `-18.965 us` (median of medians) and `-16.260 us`
   (paired median). Independent agreement.

2. **The forward lever does not fully reach the wall.** Forward time fell
   `-18.965 us` but wall fell only `-14.330 us`. The `+4.635 us` difference is
   the non-forward term growing: `(a)-(b)` went `95.720` -> `100.355`. Caution:
   this is a median-of-medians difference on a term that is ~31% of wall, and it
   is not established as systematic rather than noise. It is reported because it
   explains why the forward-level lever overstates the wall gain, and because
   anyone planning the next round should know the wall conversion is lossy.

The residual wrapper is **unchanged** (`29.350` vs `29.690 us`). This round
reduced launch cost, not wrapper cost — consistent with its declared scope.

## Attribution

**The mechanism worked. The magnitude is too small.**

- Device time is flat: `13.3272` -> `13.3228 us/call`, `-0.03%`. Kernel count
  `1.00` -> `1.00`, same kernel name. Device explains none of the movement, so
  attribution to the host launch link is sound.
- The launch path is genuinely cheaper: bare M0 `192.255` -> bare M1
  `172.950 us`, a `-19.305 us` (`-10.04%`) saving, stable across all 5 blocks.
  Coder's independent probe measured `-22.030 us` on the same comparison.
- Host work genuinely fell: `forward` alone `221.605` -> `202.640 us`.
- But only `-14.330 us` of that reached the synchronized wall, and the threshold
  is `14.871 us`.

So this is not a case of a broken mechanism or a device story leaking into wall.
It is a real host win that lands **just under** the bar.

## Margin and Stability Assessment (Orchestrator-requested)

Three questions were asked. Direct answers.

**1. Spread of the three pairs, and whether 5% is inside it.**

The three protocol pairs are extremely tight *as a ratio*: speedups
`1.260950 / 1.270171 / 1.271050`, a range of `0.80%`. But the three-pair
protocol measures against `base.py`, and its spread is not what decides the 5%
question. The 5% question is decided by the 12-block controlled comparison,
whose per-block base-normalized improvements range `-0.36%` to `+7.17%`.
**5% is inside that spread** — 2 of 12 blocks exceed it — but the centre of that
distribution is `+3.30%` median / `+3.13%` mean.

**2. Stable, or a coin-flip on this machine?**

The **direction** is stable and is not in doubt. `11 of 12` paired blocks favour
`e2_004`, both orderings favour it, all `4` of the strictly interleaved pairs
favour it, and the in-process launch-path saving is negative in all 5 blocks.
`e2_004` is genuinely faster than `e2_003`.

The **magnitude** is between `2%` and `4.5%` depending on estimator, and every
estimator is below 5%:

| Estimator | Gain | How far from 5% |
|---|---:|---|
| in-process, paired, harness wall (tightest for wall) | `-4.35%` | marginal: misses `14.871 us` by `1.056 us` |
| strict interleaved cross-process, ratio-of-speedups | `+2.89%` | clear: best pair `+3.47%` |
| 12-block cross-process, raw | `+2.59%` | clear |
| 12-block cross-process, base-normalized | `+1.93%` | clear |

**Verdict: below the line on every estimator, and not rescuable by more
sampling.** The in-process number is the closest at `-4.35%`, a `1.056 us`
shortfall that is `0.33%` of wall; but the strictly interleaved cross-process
measurement — the one that directly implements the governing bar — puts the gain
at `+2.89%` with the **best of four pairs** at `+3.47%`. To reach 5% the
estimator would have to move by more than its own observed spread. **I do not
think this can be stretched into a defensible `accepted`, and I have not tried
to.** The honest record is a real `2-4.5%` improvement that does not clear the
bar.

**3. Forward-level lever versus Coder's 18.470 us.**

Measured independently: `-18.965 us` (median of medians) / `-16.260 us` (paired
median), against Coder's `-18.470 us`. Agreement within about `2.7 us`. Both sit
above the `14.871 us` threshold, which is precisely why the forward-level number
is misleading on its own: the wall conversion is lossy (`-14.330 us` of
`-18.965 us` reaches wall).

**4. Does it clear the governing bar (+5% versus the previous accepted candidate)?**

**No — `+2.8874%`, a miss of `2.11` points.**

This is answered with the strictly interleaved same-turn control requested in
advisory 2, in which `e2_003` and `e2_004` alternate pair by pair under an
identical protocol. The full work is in the dedicated section above; the verdict
is robust because **all four pairs land between `+1.68%` and `+3.47%`**, so the
5% line is outside the spread rather than inside it.

One caution on the alternative form of this bar. The cross-turn derived bar
(`1.271132`) is missed by only `0.0756%` if `e2_004`'s protocol-window speedup
is used, which superficially looks like a photo finish. It should be
disregarded: `e2_004`'s measured speedup moved `4.31%` between two windows of
this same turn, so a `0.08%` gap is far below the method's resolution. The
Orchestrator's expectation of roughly `+6.60%` was based on the forward-level
lever, which overstates wall gain for the two reasons given above (lossy wall
conversion, and speedup sensitivity to `base.py` movement).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification of round 004 | `not-applicable` | `f5aa1d709e4deeb1562757d795dd4da41217238dd15c53d33c2c338da1938020` | correctness pass on first attempt; no repair needed |

No environment incident occurred; no `incident_004_*.md` was written. Coder
reported three in-round probe/candidate defects (attempts 2-4 of its own
ledger), all found and fixed before `candidate-ready`; none was a Verifier retry.

Two diagnostic probes under `log/` were written by Verifier
(`round_004_host_decomposition.py`, `round_004_guardrails.py`); neither is a
tracked campaign artifact and neither modifies `base.py`, the harness, the
candidate, or any decision.

## evidence_for_next_round

- **The capability question is answered once and for all, and the answer is
  large.** `lifecycle.fast-launcher` is proven on this runtime. M1
  (`fast_libentry`) saves `19.305 us/call` on the bare launch path — Verifier's
  independent figure, matching Coder's `22.030`. **The probe also measured M2
  (cached `CompiledKernel`) at `-119.360 us` and M3 (`NPULauncher` C entry) at
  `-139.580 us`, roughly 6x and 7x M1's saving.** All three launch the same
  compiled kernel, all bit-identical, all `1.00` launches per call. M1 was used
  only because the decision listed it first and the frozen profile names it.
- **The reason to prefer M2 or M3 is now measured, not speculative.** M1's
  entire saving is `19.305 us` on a `~303-317 us` wall, i.e. `6.1-6.4%` before
  the lossy wall conversion — which is exactly why it landed at `4.35%` and
  missed. M2 or M3 would not be marginal. The probe evidence to authorize either
  already exists in `log/probes/round_004_launch_abi_probe.json`; a new decision
  naming one of them by name is the cheapest possible next step.
- **The wall conversion is lossy and should be budgeted for.** Forward time fell
  `-18.965 us`; wall fell `-14.330 us`. About `-4.6 us` was absorbed by a larger
  non-forward term. Any future round should predict wall gain from forward gain
  at roughly `75%`, not 100%.
- **The residual wrapper is untouched at `~29.7 us`** and is unchanged by this
  round. It remains the fallback family, and it is small.
- **The harness-fixed term is now `95.7-100.4 us`** (31-33% of wall) and still
  unreachable. It is growing slightly as wall shrinks, which raises its share.
- **Device is closed.** `13.3228 us/call` at `1.00` kernels, `device_ratio`
  `0.0474`, against a `4.0902%` device-only ceiling.
- **Methodological note for whoever measures next:** the protocol's
  `base.py`-referenced number (`21.32%` here, `17.40%` in round 003) will keep
  overstating each round's contribution, because it is cumulative and because
  `base.py` drifts. **Always run the interleaved control against the current
  canonical kernel; that is the only number that can decide an adoption.** Run it
  in both orders, as was done here, to expose ordering bias.
- **Carried forward unchanged:** the kernel requires `S <= 128` (campaign shape
  `S=83`); the second `tl.dot` tile `(128,64,128)` compiles and is numerically
  correct but was not one of the eleven probed tiles; the output-buffer reuse
  invariant depends on full store coverage and must be revisited if a later
  round introduces a masked or partial store; the epoch-1 deliverable
  `triton_attn_001.py` is preserved intact at `../`.

## Stop Recommendation

- recommendation: `continue`
- evidence: no stop condition is met. The no-improvement streak moves to `1`
  against a limit of `3`; the round budget is `3` of `20`; no target is set, so
  `target-reached` does not apply; nothing is blocked. More importantly, this
  round produced the most actionable evidence of the epoch: a proven fast-launch
  capability whose *unused variants* are `6-7x` larger than the one that was
  tried. That is a live, measured lever, not a dead end.

## Exact Reproduction Commands

Correctness gate:

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_004.py --warmup 5 --repeat 10 --full-traceback
```

Authoritative timing (three pairs, one turn):

```bash
cd /workspace/kernelswift-dev-4ff2094
for i in 1 2 3; do
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_004.py \
    --warmup 50 --repeat 100
done
```

Interleaved control against the canonical kernel (run in both orders):

```bash
cd /workspace/kernelswift-dev-4ff2094
# blocks 1-6: e2_003 first
for i in 1 2 3 4 5 6; do
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py --warmup 50 --repeat 100
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_004.py --warmup 50 --repeat 100
done
# blocks 7-12: reversed order
for i in 7 8 9 10 11 12; do
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_004.py --warmup 50 --repeat 100
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py --warmup 50 --repeat 100
done
```

Strict pair-by-pair alternation — the control behind the governing bar
(`speedup(candidate) / speedup(last_accepted) - 1 >= 5%`):

```bash
cd /workspace/kernelswift-dev-4ff2094
for i in 1 2 3 4; do
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_004.py --warmup 50 --repeat 100
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py --warmup 50 --repeat 100
done
```

Take the median speedup of each set of four runs and divide. This is the only
form of the ratio test that is safe on this machine; see the methodological
finding above on why cross-window speedup comparisons are not.

Profiler:

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
  --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_004.py \
  --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_004_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/profiling_data/reference_triton_mm_encoder_attention_e2_003/profiling_data/16458e336fc3_106265_20260830073803678_ascend_pt/PROF_000001_20260830073803704_00106265HFQBQARR/device_0/sqlite/ai_core_op_summary.db" --iterations 50 --scope reference_triton_mm_encoder_attention_e2_003 --wall-ms 0.297245

python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/profiling_data/candidate_triton_mm_encoder_attention_e2_004/profiling_data/16458e336fc3_106265_20260830073807505_ascend_pt/PROF_000002_20260830073807529_00106265EEFRCOBB/device_0/sqlite/ai_core_op_summary.db" --iterations 50 --scope candidate_triton_mm_encoder_attention_e2_004 --wall-ms 0.280915
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
python3 kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_004_host_decomposition.py
python3 kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_004_guardrails.py
```

Kernel-definition identity check:

```bash
cd /workspace/kernelswift-dev-4ff2094/kernels/track1-triton/mm_encoder_attention/ascend/epoch2
diff <(sed -n '1,76p' triton_mm_encoder_attention_e2_003.py) <(sed -n '1,76p' triton_mm_encoder_attention_e2_004.py) && echo KERNEL_DEF_IDENTICAL=yes
```
