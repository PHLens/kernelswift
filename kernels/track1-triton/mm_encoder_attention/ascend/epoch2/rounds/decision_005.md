# Decision 005

## Metadata

```json
{"schema_version":2,"decision":"proceed","decision_kind":"optimization","round":"005","reference_implementation":"triton_mm_encoder_attention_e2_003.py","reference_report":"rounds/report_003.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"launch-path-reduction","sketch_ref":"rounds/sketch_005.json","sketch_sha256":"f44ed2bfbef80e9dc603494221bbc2cd47db40a9d8d48d85ee2ae344cd11c4ee","implementation_profile_snapshot_ref":"state/implementation_profile_snapshot/profile.yaml","implementation_profile_snapshot_sha256":"a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321","project_capability_claim_ref":"state/project_capability_claim.json","project_capability_claim_sha256":"a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"drive the resolved CompiledKernel object directly in ModelNew.forward instead of going through per-call JITFunction dispatch, using the cached-CompiledKernel launch path measured at 66.895 us/call by the retained round-004 capability probe, so the steady-state forward no longer pays per-call argument specialization, cache-key construction, and dispatch for the same compiled kernel","allowed_changes":["ModelNew.forward launch invocation site","ModelNew.__init__ launcher cache state","cached CompiledKernel handle"],"invariants":["the _fused_attention_kernel triton.jit definition stays byte-identical","kernel launch count stays at one","BLOCK_M BLOCK_N HEAD_DIM accumulator_dtype num_warps and num_stages unchanged","device kernel name stays _fused_attention_kernel","the fast path is used only when it resolves the same CompiledKernel object the proven path produced","ModelNew public contract","output shape dtype device and contiguity","numerical tolerance atol=1e-2 rtol=1e-2","no aliasing of query key or value","cached output buffer still fully overwritten every call","base.py bytes unchanged"],"expected_wall_improvement_pct":30.0,"capability_gate":{"capability_id":"lifecycle.fast-launcher","frozen_status":"unknown","round_local_status":"proven","new_probe_required":false,"selected_mechanism":"M2 cached CompiledKernel","evidence_refs":["log/probes/round_004_launch_abi_probe.json","log/probes/round_004_probe_evidence.md","log/probes/round_004_candidate_conformance.json"],"legality_reestablished_by":"citation of the retained round-004 Decision-scoped probe artifacts on disk, which discharge all four decision-004 criteria for M2","ordering_rule_corrected":"decision 004's stop-at-first-passing-mechanism rule settled legality only; with magnitudes now measured, mechanism selection is a separate magnitude-and-robustness judgment made here","profile_amendment":"none; the frozen snapshot stays Unknown and hash-pinned, and this round-local evidence does not license any later round"}}
```

## Unified Sketch

```json
{"artifact":"rounds/sketch_005.json","sha256":"f44ed2bfbef80e9dc603494221bbc2cd47db40a9d8d48d85ee2ae344cd11c4ee","rendering":"# D Declarations\ntensor query shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor key shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor value shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor out shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\nscalar out_shape shape=[3] dtype=int64 layout=scalar memory=host\ntile q_tile shape=[BLOCK_M,HEAD_DIM] dtype=fp16 layout=blocked memory=register\ntile k_tile shape=[BLOCK_N,HEAD_DIM] dtype=fp16 layout=blocked memory=register\ntile v_tile shape=[BLOCK_N,HEAD_DIM] dtype=fp16 layout=blocked memory=register\nscalar scale shape=[1] dtype=fp32 layout=scalar memory=register\n\n# O Operations\nalloc out <- out_shape on the ModelNew instance; a cache hit performs no allocation\nload q_tile <- query[b, 0:BLOCK_M, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask row_idx < S\nload k_tile <- key[b, 0:BLOCK_N, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask offs_n < S\nload v_tile <- value[b, 0:BLOCK_N, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask offs_n < S\ncompute qk = dot(q_tile, trans(k_tile)) * scale  # fp32 accumulate, conversion declared\ncompute p = masked_softmax(qk)\ncompute acc = dot(p.to(fp16), v_tile)\ncompute acc_norm = acc / rowsum(p)\nstore out[b, 0:BLOCK_M, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] <- acc_norm mask row_idx < S\n\n# C Control\nparallel bh over B*NH\nguard row_idx < S\nguard offs_n < S\n\n# H Target Hints\ntarget=triton_ascend\nBLOCK_M=128\nBLOCK_N=128\nHEAD_DIM=64\naccumulator_dtype=fp32\nnum_warps=4\nnum_stages=1\n"}
```

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward launch invocation site","ModelNew.__init__ launcher cache state","cached CompiledKernel handle"],"state_owner":"ModelNew instance; the handle and the proven-kernel reference are ordinary instance attributes and are not module state, so neither is ever serialized into state_dict","lifetime":"model lifetime; the CompiledKernel is resolved through the proven launch on the first call after construction or after any cache-key change, then reused until a cache-key component changes","allocation_reuse":"the resolved CompiledKernel object is reused across calls; no per-call allocation is added and the round-003 output-buffer cache is retained unchanged with its own key","cache_key":["query dtype and rank","output shape tuple","output device","query stride tuple","key and value stride tuple","S (q_len)","scale value","grid tuple","BLOCK_M BLOCK_N HEAD_DIM NH","num_warps and num_stages"],"invalidation":"any change to a cache-key component discards the handle and routes that call through the proven kernel[grid](...) launch, re-proving the handle for the new key; an unproven, mismatched, or failed resolution is a miss that never launches","concurrency":"one ModelNew instance is not shared across concurrent forwards; the benchmark drives a single sequential stream from one thread and no lock or thread-local is introduced","device_stream_behavior":"the stream is resolved per call by CompiledKernel.__getitem__ exactly as the proven path does; the same device and the caller's current stream are used, and no stream is created, captured, or switched","unchanged_behavior":["returned shape","returned dtype","returned device","returned contiguity","numerical semantics within atol=1e-2 rtol=1e-2","no aliasing of query key or value","kernel launch count stays at one","public constructor and forward signature"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-005","intervention":"drive the resolved CompiledKernel object directly in ModelNew.forward instead of going through per-call JITFunction dispatch, using the cached-CompiledKernel launch path measured at 66.895 us/call by the retained round-004 capability probe, so the steady-state forward no longer pays per-call argument specialization, cache-key construction, and dispatch for the same compiled kernel","expected_causal_chain":["the cached CompiledKernel is resolved once and reused across calls","the steady-state forward stops paying per-call argument specialization, cache-key construction, and JITFunction dispatch","launch_path_us_per_call falls from the 183.740 us proven baseline toward the 66.895 us measured M2 value","per-call host time inside ModelNew.forward falls below the 206.375 us baseline","device time and kernel count stay fixed so the wall delta is attributable to host","synchronized wall median decreases by at least five percent"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"launch_path_us_per_call","expectation":"decrease from the 183.740 us proven baseline toward the 66.895 us M2 value"},{"name":"host_us_per_call","expectation":"decrease below the 206.375 us forward-alone baseline"},{"name":"device_us_per_call","expectation":"unchanged at approximately 13.4224"},{"name":"kernel_count_per_call","expectation":"unchanged at 1.00"}],"guardrails":["correctness:pass","output bit-identical to the accepted kernel","device kernel name stays _fused_attention_kernel","resolved CompiledKernel is the same object the proven path produced","BLOCK_M BLOCK_N HEAD_DIM num_warps and num_stages unchanged","kernel launch count stays at one","any fast-path failure degrades sticky to the proven launch","no aliasing of query key or value","cached buffer fully overwritten every call","public constructor and forward signature unchanged","base.py bytes unchanged"],"profiling_level":"targeted","causal_graph":{"nodes":["n_launch_dispatch","n_host_work","n_device_unchanged","n_wall"],"edges":[["n_launch_dispatch","n_host_work"],["n_host_work","n_wall"],["n_device_unchanged","n_wall"]]}}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`: entries 011 through 016 are grouped-top-k selection on MLU590-H8 under a different compiler lowering. None matches this operator, backend, or host-side launch mechanism, so no recorded failure invalidates this path.
- Consulted `references/bottleneck-judgment.md`. `device_ratio` is `0.0445`, so the classification is `host-bound`, and the launcher row applies: "potentially compressible", with the required check being "target profile and same-regime wall evidence". The retained probe supplies exactly that check, with the baseline reproduced in-process at `186.255 us` against Verifier's `183.740 us` (`+1.37%`).
- **I am correcting an error I made in round 004.** I wrote one ordering rule to serve two different questions: whether a fast path is *legal*, and *which* fast path to use. "Stop at the first mechanism satisfying all four criteria" is a sound rule for legality — it answers "does one exist?" — but it is the wrong rule for selection once magnitudes are known, because it is indifferent to a `5.4x` difference in the lever. Coder correctly refused to reorder a normative list and flagged the gap. Selecting among three mechanisms that all pass all four criteria is a magnitude-and-robustness judgment, and it belongs here, in a new decision, not in Coder's discretion.
- **This is not a new capability claim.** All three mechanisms are realizations of the same `lifecycle.fast-launcher` capability that decision 004 required to be probed and that the probe discharged. The probe, not the `implementation_symbol` field, is the authority for what exists on this runtime. `fast_libentry` is the symbol the frozen profile happens to name; it is an example binding, not an exhaustive enumeration, and it is not the reason to prefer M1 now that M1's measured lever is known to be the smallest of the three.
- The probe is **round-local evidence and is reused by citation only**. It does not amend the frozen snapshot, which stays `Unknown` and hash-pinned, and it does not license a later round. A later round must re-establish legality on its own evidence.
- **Fallback must stay sticky, and M2 preserves the property that makes sticky work.** For M1 and M2 a failure surfaces as a Python exception or a kernel-identity mismatch, both of which the round-004 design catches and degrades on. That property is part of why M2 is chosen over M3 (see section 3).
- **Do not trust a Python-level launch counter.** Coder's attempt ledger records that patching `triton.backends.ascend.driver.NPULauncher` reported `0.00` launches because that class is shadowed by the compiled C++ `ascend.NPULauncher` reachable through `triton.runtime.driver.active.launcher_cls`. `kernel_count_per_call` must be confirmed through the active launcher class or the profiler.
- Device time is not a fourth independent slice. Per `report_003.md`, the `13.4224 us` of device work sits inside the `51.815 us` synchronize term. `device_us_per_call` and `kernel_count_per_call` are unchanged control observables so the device story stays out of the attribution, which is required given the `4.0902%` device ceiling.

## Rationale and Evidence

Runtime facts come from `rounds/report_003.md` and the retained round-004 probe artifacts under `log/probes/`. No value here is a Designer measurement.

### 1. The measured mechanism table

In-process, warmup 50 / repeat 100 / 3 blocks, M0 baseline reproduced in the same script:

| Mechanism | us/call | saving vs M0 | bit-identical | same kernel object |
|---|---:|---:|---|---|
| M0 proven `kernel[grid](...)` | 186.255 | — | control | control |
| M1 `fast_libentry` | 164.225 | 22.030 | yes (`0.0`) | yes |
| **M2 cached `CompiledKernel`** | **66.895** | **119.360** | **yes (`0.0`)** | **yes** |
| M3 `NPULauncher.launch` C entry | 46.675 | 139.580 | yes (`0.0`) | yes |

All four share kernel hash `18db9f0320830a397f740d02078551aeea898355fd7e06d59bb3a7bca2e1c903`, so criterion 2 holds for every mechanism. M0 reproduces Verifier's `183.740 us` at `186.255 us` (`+1.37%`), so the regime matches and no new probe is needed.

### 2. Why not M1 — the margin is inside the noise band

M1 is the mechanism round 004 implemented, and its lever measured at the forward level is `-18.470 us/call` against a `14.871 us` threshold. That is a margin of `3.599 us`, or `1.21%` of wall, on a machine whose `base.py` medians ranged `0.346350`-`0.370825` (~7%) within a single turn in round 003.

The propagation arithmetic makes the fragility explicit:

```text
M1 bare-launch saving   22.030 us   -> needs 67.5% propagation to clear 14.871 us
M1 observed propagation 18.470 / 22.030 = 83.8%
```

M1 clears the threshold only because `83.8%` exceeds `67.5%`, and it does so with `3.6 us` to spare. Coder's attempt ledger shows how thin that really is: before hoisting the launch keyword bundle to `__init__`, M1's forward lever was only `-11.715 us`, i.e. below the threshold. Roughly `10 us` of M1's lever was nearly consumed by ordinary per-call host work. A lever that a `10 us` accident can erase is not a defensible adoption bet.

### 3. Why M2 and not M3 — M3 is dominated

M3 is the fastest at `46.675 us`, but it buys only `20.220 us` more than M2:

```text
M3 saving 139.580 us
M2 saving 119.360 us
difference 20.220 us  (6.8% of wall)
```

Against that marginal `20 us`, M3 carries the largest coupling cost of the three:

1. **It reaches past Triton's runtime into backend codegen.** M3 calls the generated per-kernel C entry point through the compiled C++ `ascend.NPULauncher`, hand-marshalling `kernel0.function` and `kernel0.packed_metadata`. Those are compiler-internal artifacts whose layout is fixed by the Ascend backend's code generator. A CANN or triton-ascend patch can change the entry signature or the packed-metadata layout with no deprecation and no Python-level error.
2. **Its failure mode is the one the fallback cannot catch.** A marshalling mismatch produces a *successful* call with wrong arguments, not an exception. The sticky-disable design only helps when the failure is detectable, and M3's worst failure is not.
3. **It perturbs the instrument.** Coder's risk note 6 records that M3 does not pass `launch_metadata`; with profiler hooks enabled, the profiler's `record_function` metadata could change — affecting the very evidence Verifier uses to decide adoption.
4. **It is backend-private, so it is the least likely to survive a toolchain patch.** M2 uses `CompiledKernel.__getitem__`, which is backend-agnostic Triton runtime; M3 is Ascend-specific.

M2 needs only `12.5%` propagation to clear the threshold (`14.871 / 119.360`). Paying maximum coupling risk for the last `20 us` when M2 already overshoots by roughly `8x` is a bad trade. **M3 is dominated: it adds risk faster than it adds time.**

M2 sits at the right point. It drives the object `JITFunction.run` itself returns, keeps Triton's own grid and argument handling rather than hand-marshalling into a C entry, and was proven bit-identical, NaN-clean, and kernel-hash-identical in the same process and regime as the baseline.

### 4. Expected magnitude

Against Verifier's `183.740 us` baseline, M2's `66.895 us` is a `116.845 us` saving. The `22.635 us` residual wrapper is unaffected, so:

```text
new forward         = 22.635 + 66.895          =  89.530 us
forward lever       = 206.375 - 89.530         = 116.845 us
wall improvement    = 116.845 / 297.410        =  39.29%   (full propagation)
wall improvement    = 116.845 * 0.838 / 297.410 =  32.94%   (M1's observed 83.8%)
```

`expected_wall_improvement_pct` is `30.0`, the conservative end of that range. The decisive number is the break-even: **M2 needs only `12.5%` of its bare-launch saving to reach the wall, against M1's `67.5%`.** Even at a quarter of M1's observed propagation — about `21%` — M2 clears the threshold with room to spare, and unlike M1 it does so without depending on a `10 us` implementation detail.

### 5. Fallback under the named mechanism

The round-004 sticky-disabled design carries over unchanged and applies to M2 as follows:

| Event | Behaviour |
|---|---|
| handle not yet resolved | proven `_fused_attention_kernel[grid](...)` launch, then resolve for this key |
| any cache-key component changes | buffer reallocated, handle cleared, **proven launch for that call**, handle re-proven for the new key |
| fast path raises | exception swallowed, handle cleared and disabled, proven launch in the **same** call |
| fast path resolves a different `CompiledKernel` | handle cleared and disabled, proven launch in the **same** call |
| resolution or import fails at construction | `_launcher_disabled` set, proven launch for the remainder of the instance lifetime |

`_launcher_disabled` is sticky because a structural failure will not recur-fix itself, and because retrying a failing fast path every call would pay exception-handling cost forever. The worst case at runtime is therefore always the accepted behaviour — never a wrong answer and never a per-call penalty.

The kernel-identity check is the whole safety net, exactly as in round 004: the fast path is trusted only when it is the same `CompiledKernel` object the proven path produced. That check is zero-cost and adds no launch.

### 6. Scope is unchanged

`change_scope` stays `host` and no `mixed` round is needed. Coder confirmed the `_fused_attention_kernel` `triton.jit` definition stays byte-identical under every mechanism, so the computation boundary is untouched and the Sketch declares it as unchanged. What changes inside the Host Plan is the *identity* of the cached handle: a resolved `CompiledKernel` object rather than a `LibEntry` wrapper. Every other field — state owner, lifetime, allocation reuse, concurrency, stream behaviour, unchanged behaviour — carries forward, and the cache key is extended from the round-003 output key to cover the launch-specialization inputs (`S`, `scale`, grid, block sizes, warp and stage counts) because those determine which `CompiledKernel` is correct.

### 7. Reference and round-004 interaction

`reference_implementation` is `triton_mm_encoder_attention_e2_003.py` because round 004 has not yet terminated and `e2_003` remains the canonical kernel. This is deliberate and robust to either outcome. M2 is not "M1 plus more" — it is a different launch path — so if round 004 lands `accepted`, round 005 replaces the M1 path wholesale rather than stacking on it, and if it lands `no-improvement` the reasoning in section 2 is confirmed. In either case Verifier should compare the round-005 candidate against whichever kernel is canonical at dispatch time; the mechanism and the observables are unaffected.

### 8. What this round leaves alone

The `91.035 us` harness-fixed term is unreachable and is not targeted. The `22.635 us` residual wrapper is left intact: it is the remaining fallback family if this round somehow fails, and squeezing it would require capturing `65.7%` of it. M3 is deliberately not adopted; if a future round wants its last `20.220 us`, that round must re-establish legality on its own evidence and should expect to justify the coupling cost that section 3 sets out.
