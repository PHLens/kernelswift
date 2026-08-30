# Round Status 002

- phase: `verifying`
- measurement_exclusive: `true`
- verifier_owns_machine: `true`
- round: `002`
- started_at: `2026-08-28T16:05Z`
- last_updated: `2026-08-28T16:35Z`
- result: `no-improvement`

## START

Preflight — all immutable inputs re-verified before any measurement:

| Artifact | SHA-256 | Expected | Verdict |
|---|---|---|---|
| `triton_fused_moe_e2_002.py` (C3 candidate) | `781d341cae2236917da988988fbe2754fc808ea0f016d7dff82fd142822d1b2d` | matches team-lead | OK |
| `rounds/decision_002.md` | `dc782254a54331454290fac6791b7f583fff81d8de9699f03f5d06722fd7637e` | `dc782254…` | OK |
| `rounds/sketch_002.json` | `015da3456f18582ad6114d3f5a0bfd14c5122a365bfbdd8031b1e543ecfe7ebe` | `015da345…` | OK |
| `rounds/binding_002.json` | `8be91ccae9c3887c480451698d6bd02f1d1eb2b5c8c0d8ea08c55570f6b4e876` | `8be91cca…` | OK |
| `triton_fused_moe_e2_001.py` (round-001 canonical) | `da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7` | unchanged | OK |
| `../../base.py` | `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d` (3598 B) | unchanged | OK |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | unchanged | OK |
| `profile_snapshot/triton_cuda.yaml` | `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae` | unchanged | OK |

- anchor: `rounds/report_001.md`, canon **0.219792 ms**; adoption gate 10.99 µs/call (5%)
- measurement fingerprint: `fe73bc58…` (unchanged)
- superseded two-hop variant `ffd4dac3…` NOT measured, per instruction
- CoreX bootstrap applied to every command

Next safe action: correctness gate before timing.

## CORRECTNESS

Harness comparator (seed 42, atol=rtol=1e-2): PASS on all three pairs plus the
profiler run.

Independent Verifier suite (`log/round_002_correctness_suite.py`, 13 suites) —
**all PASS**, and every suite is additionally **bitwise-equal to the round-001
accepted source** on identical input bits:

| Suite | max_abs vs base | bitwise-equal to r001 | verdict |
|---|---:|---|---|
| seed42 | 1.526e-05 | yes | PASS |
| fp16-extreme | 3.906e-03 | yes | PASS |
| activation / 8 active | 1.526e-05 | yes | PASS |
| activation / 7 active | 1.526e-05 | yes | PASS |
| activation / 2 active | 1.526e-05 | yes | PASS |
| activation / all-tie zeros | 1.526e-05 | yes | PASS |
| activation / all rows to expert 0 | 9.537e-06 | yes | PASS |
| run_out call 1 vs base | 1.526e-05 | — | PASS |
| run_out call 2 vs base | 1.526e-05 | — | PASS |
| run_out no stale carry-over | (differs, correct) | — | PASS |
| non-target T=128 E=16 (tier-3) | 1.526e-05 | yes | PASS |
| determinism 20 calls | bitwise-identical | — | PASS |
| tier1 vs tier3 eager | bitwise-equal | — | PASS |

**Coverage correction (documentation only, no verdict impact):** the
activation ladder is **8/7/2/2/2**, not 8/7/2/2/1. `all_rows_expert0` sets
`rl[:,0]=10` and `rl[:,1:]=-1e4`, but `torch.topk(k=2)` still returns two
distinct indices, so expert 1 is picked as the second slot on every row
(measured: `{0: 83, 1: 83}`). **With `top_k=2` a single active expert is
structurally impossible.** `coder_result_002.md` records "1 active" for this
variant; the true count is 2. The suite still exercises the intended regime
(all rows to expert 0, 7 empty experts).

Next safe action: three ordered interleaved wall pairs.

## PAIRS

Three ordered interleaved pairs, warmup 50 / repeat 100, default stream:

| Pair | v0 (base.py) ms | v1 (C3 candidate) ms | speedup |
|---:|---:|---:|---:|
| 1 | 3.208186 | 0.223927 | 14.327x |
| 2 | 3.217895 | 0.216378 | 14.872x |
| 3 | 3.229549 | 0.220177 | 13.740x → 14.668x |

- v0 median: `3.217895` ms
- v1 median: `0.220177` ms
- vs paired v0 median: `93.157732`%
- vs round-000 canon (3.255288): `93.236328`%
- **vs round-001 canon (0.219792): `-0.175166`%** — i.e. **+0.385 µs/call**,
  against a 10.99 µs/call gate. **FR-5 fires; the 5% gate is not met.**
- raw log: `log/round_002_wall_pairs.txt`

Next safe action: determine whether the +0.385 µs is a regression or noise.

## REGRESSION-VERSUS-NOISE DISCRIMINATION

The harness cannot compare two candidates (v0 must define `Model`; both
candidates define `ModelNew`), so a matched paired A/B was written
(`log/round_002_paired_ab.py`) replicating `auto_bench.time_forward`
(459-475) exactly, 6 replicates, alternating A/B order to cancel drift:

| | median-of-medians | spread |
|---|---:|---:|
| r001 accepted | 0.209003 ms | 1.403 µs |
| r002 C3 | 0.209106 ms | 0.801 µs |

- paired delta (r002 − r001): `[-0.182, +0.364, -0.305, -0.078, +0.691, +0.264]` µs
- **paired delta median `+0.093` µs, mean `+0.126` µs, signs MIXED (3 neg / 3 pos)**
- bitwise-equal on identical inputs: `True`

**Verdict: statistically indistinguishable — cost-neutral, NOT a regression.**
Mixed signs across replicates and a delta two orders of magnitude below the
10.99 µs gate mean there is no systematic difference to detect. This
corroborates the Coder's −0.022 µs and the Orchestrator's ruling.

Next safe action: the retention test, which is this round's actual product.

## RETENTION (the round's product)

Independently re-implemented p12 changing-data protocol
(`log/round_002_retention.py`):

| Suite | Further calls | Byte-identical | Max abs drift | Verdict |
|---|---:|---|---:|---|
| retention | 50 | yes | 0.0 | PASS |
| retention | 150 (harness parity) | yes | 0.0 | PASS |
| retention | 300 | yes | 0.0 | PASS |

Structural properties over 60 calls: `out_dest` has **1 distinct data_ptr**
(allocated once, reused), is **still all zeros** (never written on the served
path — C3's literal claim confirmed), is **never returned**, `out_ws` is never
returned, and zero returned tensors alias either buffer. Active tier:
`tier1_direct`.

**Negative control proves the test is non-vacuous:** a deliberately broken
rotating-pool model (pool 8) is detected corrupting the retained tensor at
**exactly call 8** = the pool size, under the same changing-data protocol. A
constant-data protocol would have passed it.

Next safe action: census and FR-3/FR-4.

## CENSUS

Host API census, replay route, 100 calls (`log/diagnostic_scope_census_002.json`):

| Observable | r001 | r002 C3 | Reading |
|---|---:|---:|---|
| `cudaGraphLaunch`/call | 1.00 | 1.00 | hold |
| `aten::copy_`/call | 1.00 | 1.00 | hold |
| `cudaMemcpyAsync` + `Memcpy DtoD` records/call | 2 records | 2 records | **one** copy, doubly recorded |
| **FR-3 submission count/call** | **2.0** | **2.0** | **PASS** |
| `aten::empty_like`/call | 1.00 | 1.00 | **unchanged** |
| `aten::empty_strided`/call | 1.00 | 1.00 | **unchanged** |
| **FR-1 alloc CPU µs/call** | 18.45 | 14.49 | **FIRES** (needs < 2.0) |
| python launcher executions/call | 0.000 | 0.000 | hold at 0 |
| recaptures in timed segment | 0 | 0 | hold |
| kernel events in replay interior | 0 | 0 | kineto-blind, as declared |

**FR-3 submission counting caveat:** the profiler records the single copy-out
twice — once as the host API `cudaMemcpyAsync` and once as the device activity
`Memcpy DtoD (Device -> Device)`. Summing the raw names gives a spurious 3.00.
The true count is 1 graph launch + 1 copy = **2.0**, identical to round 001.
(This double-recording is also present in round 001's raw census; round 001's
reported conclusion of 2.0 was right, but its intermediate "memcpy 1.00/call"
line was an under-count of the raw records.)

**FR-1 fires honestly.** `torch.empty_like` was **re-targeted** (`out_ws` →
`out_dest`), not removed — `forward()` must still return a fresh non-aliased
tensor. Alloc CPU time does improve 18.45 → 14.49 µs/call (≈ −4 µs,
corroborating the Orchestrator's 4.13 µs re-measurement), but it does not fall
below the 2.0 µs threshold, and the saving is invisible at the wall because it
is off the critical path.

**FR-4 passes with the pre-bind fix applied.** Both candidates pre-bound
(asserted all five workspace buffers non-None before the guards were disabled):
r001 252.716 µs/call, r002 253.554 µs/call, **delta +0.837 µs**, threshold
±15 µs. The two `@triton.jit` bodies are byte-identical between the rounds
(digest `61d16bde3d12fb12`, 2 kernels, non-vacuous).

Harness dual-scope trace: reference scope 123.95 kernels/call @ 970.492 µs
(reproduces rounds 000/001); candidate scope **0.05 kernels/call, 0.549 µs** —
the declared UNAVAILABLE-not-zero artifact.

Next safe action: write the report, verdict, and ledger.

## END

- terminal classification: **no-improvement**
- improvement vs round-001 canon: `-0.175166`% (gate 5%)
- falsification: FR-1 **fires**, FR-2 pass, FR-3 pass, FR-4 pass, FR-5 **fires**
- artifacts: `rounds/report_002.md`, `rounds/verdict_002.json`,
  `state/verifier_context.md`, this file
- no repair requested; candidate hash frozen and unchanged throughout
- the round is **correctness-hardening by design**; round 001 remains canonical
- next safe action: Orchestrator increments `performance_miss_streak` to 1/3,
  keeps `last_accepted_kernel` at `triton_fused_moe_e2_001.py`, sets
  `last_result: no-improvement`, and clears `measurement_exclusive`
