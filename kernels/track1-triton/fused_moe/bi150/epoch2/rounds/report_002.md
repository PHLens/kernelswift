# Report 002

Result: no-improvement

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md` @ `dc782254a54331454290fac6791b7f583fff81d8de9699f03f5d06722fd7637e`
- Candidate: `/root/CodeBuddy/20260818191200/kernelswift/kernels/track1-triton/fused_moe/bi150/epoch2/triton_fused_moe_e2_002.py`
- Accepted reference: `triton_fused_moe_e2_001.py` (round-001 accepted source)
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `dc782254a54331454290fac6791b7f583fff81d8de9699f03f5d06722fd7637e`
- Candidate SHA256: `781d341cae2236917da988988fbe2754fc808ea0f016d7dff82fd142822d1b2d`
- Accepted reference SHA256: `da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7`
- Base SHA256: `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d` (3598 bytes, unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes, unchanged)
- Sketch SHA256: `015da3456f18582ad6114d3f5a0bfd14c5122a365bfbdd8031b1e543ecfe7ebe`
- Binding SHA256: `8be91ccae9c3887c480451698d6bd02f1d1eb2b5c8c0d8ea08c55570f6b4e876`
- Profile snapshot SHA256: `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae`
- Runtime fingerprint: `project.md#runtime-fingerprint` — `triton 3.1.0 / torch 2.7.1 / CoreX 4.4.0 nvcc V10.2.89 / Iluvatar BI-V150 sm_71 / 16 SM` — matches
- Measurement fingerprint: `fe73bc58146d8c16f524be2a00fe99b31e1b9678bca6b3702f4284a3ac0a5bef` (unchanged)
- verification_tier: `authoritative`
- screening_pairs: `not-run: candidate cleared on authoritative timing`
- superseded variant `ffd4dac3…`: **not measured**, per Orchestrator instruction

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (harness) | pass | PASS on all three pairs + profiler run | pass | `log/round_002_wall_pairs.txt` |
| seed 42 vs base | pass | max_abs 1.526e-05 | pass | suite |
| fp16-extreme | pass | max_abs 3.906e-03 | pass | suite |
| activation / 8 active | pass | 1.526e-05 | pass | suite |
| activation / 7 active | pass | 1.526e-05 | pass | suite |
| activation / 2 active | pass | 1.526e-05 | pass | suite |
| activation / all-tie zeros | pass | 1.526e-05 | pass | suite |
| activation / all rows to expert 0 | pass | 9.537e-06 | pass | suite |
| run_out poisoned ×2 | no stale carry-over | call1 ≠ call2, both match base | pass | suite |
| non-target T=128 E=16 (tier-3) | pass | 1.526e-05 | pass | suite |
| determinism 20 calls | identical | bitwise-identical | pass | suite |
| tier1 vs tier3 eager | equal | bitwise-equal | pass | suite |
| **bitwise-equal to round-001 source** | yes | **true on every suite** | pass | suite |
| output shape | `[83, 128]` | confirmed | pass | suite |
| output dtype | `fp16` | confirmed | pass | suite |
| all finite | yes | base and candidate finite on every suite | pass | suite |
| base.py unchanged | sha `21e75853…` | 3598 bytes before/after | pass | `sha256sum` |
| decision/sketch/binding unchanged | sha match | all verified post-run | pass | `sha256sum` |
| r001 canonical unchanged | `da623fa9…` | verified post-run | pass | `sha256sum` |
| seed | 42 | on every command | pass | all commands |

Conformance, correctness, and every declared guardrail pass.

### Activation-coverage correction (documentation only)

The activation ladder actually measured is **8/7/2/2/2**, not 8/7/2/2/1.
`all_rows_expert0` sets `rl[:,0]=10` and `rl[:,1:]=-1e4`, but `torch.topk(k=2)`
still returns two distinct indices, so expert 1 takes the second slot on every
row (measured distribution `{0: 83, 1: 83}`). **With `top_k=2` a single active
expert is structurally unreachable** — every row contributes two distinct
experts, so the union is ≥ 2. `coder_result_002.md` records "1 active" for this
variant; the correct value is 2. The suite still exercises the intended regime
(all rows to expert 0, seven empty experts), and no verdict depends on the
label.

## Screening Evidence

Not run — the candidate cleared correctness and proceeded directly to
authoritative timing.

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | not-run | not-run | not-run | proceeded to authoritative |
| 2 | not-run | not-run | not-run | proceeded to authoritative |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`, three ordered pairs
- device: `cuda:0`, default stream

| Pair | v0 (base.py) ms | v1 (candidate) ms | speedup |
|---:|---:|---:|---:|
| 1 | 3.208186 | 0.223927 | 14.327x |
| 2 | 3.217895 | 0.216378 | 14.872x |
| 3 | 3.229549 | 0.220177 | 14.668x |

- reference_raw_samples_ms: `[3.208186, 3.217895, 3.229549]`
- candidate_raw_samples_ms: `[0.223927, 0.216378, 0.220177]`
- reference_median_ms: `3.217895`
- candidate_median_ms: `0.220177`
- improvement_pct: `93.157732` (vs paired v0 median)
- **vs round-001 canon (0.219792 ms): `-0.175166`**

```text
vs round-001 canon = (0.219792 - 0.220177) / 0.219792 * 100 = -0.175166
delta              = +0.385 us/call against a 10.990 us/call gate
```

**FR-5 fires: the 5% gate is not met.** This is the expected and predicted
outcome, not a defect — see `coder_result_002.md` and the Orchestrator's
framing. The round was shipped as correctness-hardening after the G1 premise
was falsified.

### Regression-versus-noise discrimination

The harness cannot compare two candidates (v0 must define `Model`; both
candidates define `ModelNew`), so a matched paired A/B was written
(`log/round_002_paired_ab.py`) replicating `auto_bench.time_forward`
(459-475) exactly — 6 replicates, alternating A/B order to cancel drift:

| | median-of-medians | spread |
|---|---:|---:|
| r001 accepted | 0.209003 ms | 1.403 µs |
| r002 C3 | 0.209106 ms | 0.801 µs |

- paired deltas (r002 − r001), µs: `[-0.182, +0.364, -0.305, -0.078, +0.691, +0.264]`
- **paired delta median `+0.093` µs, mean `+0.126` µs, signs MIXED (3 negative / 3 positive)**
- bitwise-equal on identical inputs: `True`

**Cost-neutral, not a regression.** Mixed signs across replicates and a delta
two orders of magnitude below the gate mean there is no systematic difference
to detect. This independently corroborates the Coder's −0.022 µs and the
Orchestrator's expectation. Classified `no-improvement`, **not** `regression`.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `host_output_alloc_us_per_call` | decrease below 2.0 µs/call | r001 18.45 → r002 **14.49** µs/call; `aten::empty_like` and `aten::empty_strided` still **1.00/call each** | **fail** | `log/diagnostic_scope_census_002.json` |
| `host_submission_count_per_call` | hold at 2.0 | **2.0** (1 `cudaGraphLaunch` + 1 copy-out), identical to round 001 | pass | census |
| `host_triton_launcher_executions_per_call` | hold at 0 | **0.000**/call over 100 timed calls | pass | census |
| `retained_output_unchanged` | pass across 50 further forwards | **byte-identical at 50, 150, and 300** further calls, max drift 0.0 | pass | `log/round_002_retention.json` |
| `device_us_per_call_nonreplay_tier` | hold | r001 252.716 → r002 253.554 µs/call, **delta +0.837 µs**, threshold ±15 µs; both pre-binds asserted | pass | census |
| `best_block_m` | directional argmin | reported by Coder p16: BLOCK_M 16, margin 9.829 µs, bitwise tie-break | pass (as reported) | `coder_result_002.md` |
| `best_num_stages` | exploratory, adopt only outside 0.5 µs band | Coder p16: argmin `default` by **0.031 µs**, inside the tie band ⇒ **not recorded, not adopted** | pass (as reported) | `coder_result_002.md` |
| `graph_capture_stability` | hold at round-001 quality | 0 recaptures in 100 timed calls, budget 4→4, 1 bound set, 0 capture errors, 0 diverged outputs | pass | census |
| `device_us_per_call` | directional cross-check only | candidate scope **0.05 kernels/call, 0.549 µs/call** — UNAVAILABLE-not-zero | missing (declared) | trace |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: reuse one persistent non-graph-pool output tensor as the fixed copy-out destination, removing the per-call allocation
- primary_metric: `wall_time`
- **Hypothesis verdict: `falsified`**

The causal chain breaks at its second link. `out_dest` IS allocated once
(`per_call_alloc_removed` depends on it), but the per-call allocation was
**re-targeted, not removed**: `forward()` must still hand back a fresh,
non-aliased tensor, so `torch.empty_like(out_dest)` remains at 1.00/call
alongside `aten::empty_strided`. The allocation does get cheaper (18.45 →
14.49 µs/call, ≈ −4 µs, which independently corroborates the Orchestrator's
4.13 µs re-measurement of `empty_like`), but it does not disappear, it stays
far above the 2.0 µs threshold, and the saving is invisible at the wall
because it sits off the critical path.

- `out_dest_allocated_once` → **true** (1 distinct data_ptr over 60 calls)
- `per_call_alloc_removed` → **false** (`empty_like`/`empty_strided` unchanged at 1.00/call)
- `wall_time_falls_16us` → **false** (+0.385 µs vs canon; paired A/B +0.093 µs)
- `retained_output_unchanged` → **true** (the round's product, and it works)
- `adoption_gate_cleared` → **false**

### Falsification rules

| Rule | Observable | Threshold | Reading | Fired |
|---|---|---|---|---|
| FR-1 | `host_output_alloc_us_per_call` | did not fall below 2.0 µs/call | the per-call allocation was not actually removed | **YES** |
| FR-2 | `retained_output_unchanged` | retained tensor changed at all | aliasing introduced; unsafe regardless of speed | no |
| FR-3 | `host_submission_count_per_call` | does not hold at 2.0 | the replay boundary was restructured | no |
| FR-4 | `device_us_per_call_nonreplay_tier` | moves by more than 15 µs | unintended kernel/pipeline edit | no |
| FR-5 | `mean_wall_ms` | did not improve ≥ 5% | global conservative guard | **YES** |

FR-1 and FR-5 both fire. Neither is a surprise: the Coder stated in advance
that FR-5 would not be met, and the Orchestrator's own re-measurement
(`empty_like` ≈ 4.13 µs against a 10.99 µs gate) established that the G1
ceiling sits below the gate regardless of implementation. **FR-2 — the rule
that would have made this change unsafe rather than merely unrewarding —
passes.**

## Retention Test (this round's product)

Independently re-implemented p12 changing-data protocol
(`log/round_002_retention.py`). Retain a tensor returned by `forward()`, then
run N further forwards driven by **genuinely different data every call**, and
assert byte-identity.

| Further calls | Byte-identical | Max abs drift | Returned ptr collided with retained | Verdict |
|---:|---|---:|---|---|
| 50 | yes | 0.0 | no | PASS |
| 150 (harness parity) | yes | 0.0 | no | PASS |
| 300 | yes | 0.0 | no | PASS |

Structural properties over 60 calls:

| Property | Observation |
|---|---|
| distinct `out_dest` data_ptrs | **1** (allocated once, reused forever) |
| `out_dest` still all-zero | **true** — never written on the served path, confirming C3's literal claim |
| `out_dest` ever returned | **0** |
| `out_ws` ever returned | **0** |
| returned tensors aliasing `out_dest` | **0** |
| returned tensors aliasing `out_ws` | **0** |
| active tier | `tier1_direct` |

**Negative control — the test is non-vacuous.** A deliberately broken
rotating-pool model (pool 8) was subjected to the same changing-data protocol.
It corrupts the retained tensor and is detected at **exactly call 8** — the
pool size. A constant-data protocol would have passed it, which is precisely
the false-pass failure mode the changing-data requirement exists to prevent.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available on the reference and eager-control scopes; UNAVAILABLE (declared) on the replay scope`
- iterations: `100` forward calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: not applicable — device kernel time is available on
  the scopes that matter; the replay scope is substituted by the host API
  census per the branch-B contract

| Scope | Device us/call | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|
| accepted_reference (`reference_baseline_adapter`) | 970.492 | 123.95 | 3.255288 | 4.407782 |
| candidate replay | **0.549** | **0.05** | 0.220177 | 0.002494 |
| candidate eager control (pre-bound) | 253.554 | 56.02 | — | — |
| r001 eager control (pre-bound) | 252.716 | 56.02 | — | — |

The candidate replay row is **UNAVAILABLE, not zero** — a replayed CUDA graph
emits zero `cat=kernel` events in its interior. The 5 events / 0.549 µs are
stray outside-the-graph events. No falsification rule references
`kernel_count_per_call`.

### Host API census (Level-2 substitute), 100 timed calls

| Host API | r001 | r002 C3 |
|---|---:|---:|
| `cudaGraphLaunch` | 1.00 | 1.00 |
| `aten::copy_` | 1.00 | 1.00 |
| `cudaMemcpyAsync` (host API record) | 1.00 | 1.00 |
| `Memcpy DtoD` (device activity record) | 1.00 | 1.00 |
| `aten::empty_like` | 1.00 | 1.00 |
| `aten::empty_strided` | 1.00 | 1.00 |
| python Triton launcher executions | 0.000 | 0.000 |
| **true submission count** | **2.0** | **2.0** |

**Submission-count counting caveat.** The profiler records the single copy-out
twice: once as the host API `cudaMemcpyAsync` and once as the device activity
`Memcpy DtoD (Device -> Device)`. Summing the raw memcpy names yields a
spurious **3.00** and would falsely fire FR-3. The true count is 1 graph launch
+ 1 copy-out = **2.0**, unchanged from round 001, corroborated by
`aten::copy_` sitting at exactly 1.00/call. The same double-recording is
present in round 001's raw census; round 001's reported conclusion of 2.0 was
correct, but its intermediate "memcpy 1.00/call" line under-counted the raw
records.

### FR-4 device control with the pre-bind fix

Round 001's forced-eager control was contaminated by ~49 µs of `aten::fill_`
churn because disabling the tier guards also bypasses `_alloc_workspace`. Both
controls here **pre-bind** (one real tier-1 serve) and **assert** the pre-bind
(all five workspace buffers non-None) before the guards are disabled:

| | r001 | r002 C3 |
|---|---:|---:|
| pre-bind asserted | yes | yes |
| device µs/call | 252.716 | 253.554 |
| kernel count/call | 56.02 | 56.02 |

**Delta +0.837 µs, threshold ±15 µs — WITHIN.** The two `@triton.jit` bodies
are byte-identical between the rounds (digest `61d16bde3d12fb12`, 2 kernels,
non-vacuous).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `781d341cae2236917da988988fbe2754fc808ea0f016d7dff82fd142822d1b2d` | no-improvement; correctness and retention pass, wall gate not met |

No repair was requested or performed. The candidate hash was frozen for this
round and is unchanged by verification.

## evidence_for_next_round

- **Round-001 canonical stands: 0.219792 ms.** Round 002 is cost-neutral
  (paired A/B +0.093 µs, mixed signs) and is classified `no-improvement`, not
  `regression`. `triton_fused_moe_e2_001.py` @ `da623fa9…` remains the
  canonical kernel.
- **G1 is closed as a line of work, with evidence.** "No per-call allocation"
  and "the returned tensor never aliases across calls" are mutually exclusive
  below ~150 forwards: `forward()` must produce a fresh tensor, so
  `torch.empty_like` survives any re-targeting of the copy destination. The
  Coder's V3 rotating pool is the only allocation-free shape and it fails
  retention exactly at the pool size. Do not revisit G1.
- **The allocation saving, though real, is off the critical path.** Alloc CPU
  time improved 18.45 → 14.49 µs/call (≈ −4 µs, matching the Orchestrator's
  4.13 µs re-measurement) with **zero** wall movement. Host-side allocator
  costs at this size do not convert to wall on this rig.
- **The retention guarantee is now explicit and independently tested**, at 50 /
  150 / 300 further calls with changing data, with a negative control proving
  the test non-vacuous. This is durable value even though the wall did not
  move. Any future candidate that returns a persistent buffer will now be
  caught.
- **Remaining addressable budget is G2, the routing prelude** (~34-42 µs/call;
  topk ~39.9 µs alone and frozen by the tie-semantics invariant, so the
  realistic reclaim is ~20 µs — still under the 10.99 µs gate only if nearly
  all of it lands). The harness `cudaDeviceSynchronize` at ~122 µs/call is
  non-addressable and sets a practical floor near 214 µs.
- **Convergence is the honest near-term call.** With G1 closed and G2's
  addressable part sub-gate, the campaign has little room left. A third
  marginal round is not indicated; if G2 is chartered it should be understood
  as the last item.
- **Carry the methodology fixes:** (a) always pre-bind workspaces before an
  eager device control; (b) always assert a finite BASE before reading a
  candidate FAIL on fp16-extreme (cap the operand tier at 32, not 1024); (c)
  sum submissions, not raw profiler records — the copy-out is doubly recorded.
- **Activation ladder is 8/7/2/2/2, not 8/7/2/2/1.** With `top_k=2` a single
  active expert is structurally unreachable. Do not plan a "1 active" case.

No next optimization is selected here; these are observations only.

## Stop Recommendation

- recommendation: `converge`
- evidence: round 002 is `no-improvement` at −0.175% against the round-001
  canon, taking `performance_miss_streak` to 1/3. G1 is now closed by direct
  measurement rather than by argument: the per-call allocation cannot be
  removed without breaking non-aliasing, and the ~4 µs it does cost does not
  convert to wall. The only remaining item the decision identified (G2, the
  routing prelude, ~20 µs addressable against a 10.99 µs gate) is marginal and
  touches a frozen tie-semantics invariant, while ~122 µs/call of harness
  synchronisation is not addressable at all. Round 002 delivered its real
  product — an explicit, independently-tested retention guarantee — and that
  value does not depend on further rounds.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness gate (harness comparator):

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/epoch2/triton_fused_moe_e2_002.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 0 --repeat 1
```

Interleaved benchmark (three ordered pairs):

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && for i in 1 2 3; do /usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/epoch2/triton_fused_moe_e2_002.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 50 --repeat 100; done
```

Retention test (this round's product):

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 kernels/track1-triton/fused_moe/bi150/epoch2/log/round_002_retention.py
```

Paired A/B against the round-001 accepted source:

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 kernels/track1-triton/fused_moe/bi150/epoch2/log/round_002_paired_ab.py
```

Correctness parity suite:

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 kernels/track1-triton/fused_moe/bi150/epoch2/log/round_002_correctness_suite.py
```

Census (host API + pre-bound FR-4 control + kernel-body digest):

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 kernels/track1-triton/fused_moe/bi150/epoch2/log/round_002_census.py
```

Separately scoped profiler:

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/epoch2/triton_fused_moe_e2_002.py --seed 42 --atol 1e-2 --rtol 1e-2 --profile --profile-reference-file kernels/track1-triton/fused_moe/bi150/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/fused_moe/bi150/epoch2/log/round_002_forward_100iter.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "781d341cae2236917da988988fbe2754fc808ea0f016d7dff82fd142822d1b2d",
  "correctness": {
    "status": "pass",
    "evidence": [
      "log/round_002_correctness_suite.json",
      "log/round_002_retention.json",
      "log/round_002_wall_pairs.txt"
    ]
  },
  "observables": [
    {
      "name": "host_output_alloc_us_per_call",
      "status": "observed",
      "value": "r001 18.45 -> r002 14.49 us/call; aten::empty_like and aten::empty_strided still 1.00/call each; not below 2.0",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_002.json"]
    },
    {
      "name": "host_submission_count_per_call",
      "status": "observed",
      "value": "2.0 (1 cudaGraphLaunch + 1 copy-out); raw memcpy records double-count to a spurious 3.00",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_002.json"]
    },
    {
      "name": "host_triton_launcher_executions_per_call",
      "status": "observed",
      "value": "0.000 over 100 timed calls",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_002.json"]
    },
    {
      "name": "retained_output_unchanged",
      "status": "observed",
      "value": "byte-identical at 50, 150 and 300 further calls, max drift 0.0; negative control detects corruption at call 8",
      "confidence": "high",
      "evidence": ["log/round_002_retention.json"]
    },
    {
      "name": "device_us_per_call_nonreplay_tier",
      "status": "observed",
      "value": "r001 252.716 -> r002 253.554 us/call, delta +0.837 us, within +/-15 us; both pre-binds asserted",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_002.json"]
    },
    {
      "name": "best_block_m",
      "status": "observed",
      "value": "16 (Coder p16: 9.829 us margin, bitwise tie-break); not re-swept by Verifier",
      "confidence": "medium",
      "evidence": ["coder_result_002.md"]
    },
    {
      "name": "best_num_stages",
      "status": "observed",
      "value": "not recorded: argmin margin 0.031 us is inside the 0.5 us tie band (Coder p16); no value adopted or written into source",
      "confidence": "medium",
      "evidence": ["coder_result_002.md"]
    },
    {
      "name": "graph_capture_stability",
      "status": "observed",
      "value": "0 recaptures in 100 timed calls, budget 4->4, 1 bound set, 0 capture errors, 0 diverged outputs",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_002.json"]
    },
    {
      "name": "device_us_per_call",
      "status": "missing",
      "value": "UNAVAILABLE-not-zero on the replay scope: 0.05 kernels/call, 0.549 us/call (kineto graph-interior blindness)",
      "confidence": "high",
      "evidence": ["log/round_002_forward_100iter.pt.trace.json"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "manual-graph-replay-fused",
    "evidence_contract": "branch-B host-API-census-v1",
    "evidence": [
      "log/diagnostic_scope_census_002.json",
      "log/round_002_forward_100iter.pt.trace.json"
    ]
  },
  "evidence_gap_cause": "environment"
}
```
