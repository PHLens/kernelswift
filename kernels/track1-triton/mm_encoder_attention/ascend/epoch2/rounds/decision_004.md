# Decision 004

## Metadata

```json
{"schema_version":2,"decision":"proceed","decision_kind":"optimization","round":"004","reference_implementation":"triton_mm_encoder_attention_e2_003.py","reference_report":"rounds/report_003.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"launch-path-reduction","sketch_ref":"rounds/sketch_004.json","sketch_sha256":"d3e52f6af032014381908e03e87a6b1c3f5694090686df2af3bfe3a6d9474dbf","implementation_profile_snapshot_ref":"state/implementation_profile_snapshot/profile.yaml","implementation_profile_snapshot_sha256":"a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321","project_capability_claim_ref":"state/project_capability_claim.json","project_capability_claim_sha256":"a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"replace the per-call Triton JITFunction dispatch in ModelNew.forward with a cheaper launch path established by a Decision-scoped Ascend launch-ABI probe, so the steady-state forward no longer pays per-call argument specialization, cache-key construction, and dispatch for the same compiled kernel","allowed_changes":["ModelNew.forward launch invocation site","ModelNew.__init__ launcher cache state","lazily cached compiled-kernel handle"],"invariants":["the _fused_attention_kernel triton.jit definition stays byte-identical","kernel launch count stays at one","BLOCK_M BLOCK_N HEAD_DIM accumulator_dtype num_warps and num_stages unchanged","device kernel name stays _fused_attention_kernel","ModelNew public contract","output shape dtype device and contiguity","numerical tolerance atol=1e-2 rtol=1e-2","no aliasing of query key or value","cached output buffer still fully overwritten every call","base.py bytes unchanged"],"expected_wall_improvement_pct":15.0,"capability_gate":{"capability_id":"lifecycle.fast-launcher","frozen_status":"unknown","probe_owner":"coder","probe_scope":"decision-scoped","probe_output_dir":"log/probes/","candidate_mechanisms":["fast_libentry fast launcher","cached CompiledKernel direct invocation","vendor precompiled launch entry point"],"probe_must_establish":["the alternative launch path exists on Ascend910B4 with triton 3.2.0, torch_npu 2.7.1.post4, and CANN 9.0.0","it launches the same compiled kernel with the same grid and the same BLOCK_M BLOCK_N HEAD_DIM num_warps num_stages","its output is bit-identical to the accepted kernel under atol=1e-2 rtol=1e-2","its per-launch cost measured in the same process and regime is strictly below the kernel[grid](...) baseline of 183.740 us"],"on_probe_failure":"the round terminates as capability-miss, triton_mm_encoder_attention_e2_003.py remains canonical, and the probe evidence is retained under log/probes/","profile_amendment":"none; the frozen snapshot stays Unknown and round-local evidence does not license any later round"}}
```

## Unified Sketch

```json
{"artifact":"rounds/sketch_004.json","sha256":"d3e52f6af032014381908e03e87a6b1c3f5694090686df2af3bfe3a6d9474dbf","rendering":"# D Declarations\ntensor query shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor key shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor value shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor out shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\nscalar out_shape shape=[3] dtype=int64 layout=scalar memory=host\ntile q_tile shape=[BLOCK_M,HEAD_DIM] dtype=fp16 layout=blocked memory=register\ntile k_tile shape=[BLOCK_N,HEAD_DIM] dtype=fp16 layout=blocked memory=register\ntile v_tile shape=[BLOCK_N,HEAD_DIM] dtype=fp16 layout=blocked memory=register\nscalar scale shape=[1] dtype=fp32 layout=scalar memory=register\n\n# O Operations\nalloc out <- out_shape on the ModelNew instance; a cache hit performs no allocation\nload q_tile <- query[b, 0:BLOCK_M, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask row_idx < S\nload k_tile <- key[b, 0:BLOCK_N, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask offs_n < S\nload v_tile <- value[b, 0:BLOCK_N, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask offs_n < S\ncompute qk = dot(q_tile, trans(k_tile)) * scale  # fp32 accumulate, conversion declared\ncompute p = masked_softmax(qk)\ncompute acc = dot(p.to(fp16), v_tile)\ncompute acc_norm = acc / rowsum(p)\nstore out[b, 0:BLOCK_M, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] <- acc_norm mask row_idx < S\n\n# C Control\nparallel bh over B*NH\nguard row_idx < S\nguard offs_n < S\n\n# H Target Hints\ntarget=triton_ascend\nBLOCK_M=128\nBLOCK_N=128\nHEAD_DIM=64\naccumulator_dtype=fp32\nnum_warps=4\nnum_stages=1\n"}
```

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward launch invocation site","ModelNew.__init__ launcher cache state","lazily cached compiled-kernel handle"],"state_owner":"ModelNew instance; the handle is an ordinary attribute and is not module state, so it is never serialized into state_dict","lifetime":"model lifetime; the handle is resolved on the first successful launch after construction and reused until a cache-key component changes","allocation_reuse":"the resolved launcher handle is reused across calls; no per-call allocation is added, and the round-003 output-buffer cache is retained unchanged with its own key","cache_key":["kernel specialization key","grid tuple","BLOCK_M BLOCK_N HEAD_DIM","num_warps and num_stages","output shape tuple dtype device and query stride tuple"],"invalidation":"any change to a cache-key component discards the handle and falls back to the proven kernel[grid](...) launch; a failed, unproven, or incorrect resolution is a miss, never a silent reinterpretation","concurrency":"one ModelNew instance is not shared across concurrent forwards; the benchmark drives a single sequential stream from one thread, and no lock or thread-local is introduced","device_stream_behavior":"the same device and the caller's current stream are used; no stream is created, captured, or switched, and the harness's per-call torch.npu.synchronize boundary is unchanged","unchanged_behavior":["returned shape","returned dtype","returned device","returned contiguity","numerical semantics within atol=1e-2 rtol=1e-2","no aliasing of query key or value","kernel launch count stays at one","public constructor and forward signature"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-004","intervention":"replace the per-call Triton JITFunction dispatch in ModelNew.forward with a cheaper launch path established by a Decision-scoped Ascend launch-ABI probe, so the steady-state forward no longer pays per-call argument specialization, cache-key construction, and dispatch for the same compiled kernel","expected_causal_chain":["a Decision-scoped probe establishes a cheaper launch path for the same compiled kernel","the steady-state forward stops paying per-call argument specialization, cache-key construction, and JITFunction dispatch","launch_path_us_per_call falls below the 183.740 us baseline","per-call host time inside ModelNew.forward decreases below the 206.375 us baseline","device time and kernel count stay fixed so the wall delta is attributable to host","synchronized wall median decreases by at least five percent"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"launch_path_us_per_call","expectation":"decrease below the 183.740 us bare-launch baseline"},{"name":"host_us_per_call","expectation":"decrease below the 206.375 us forward-alone baseline"},{"name":"device_us_per_call","expectation":"unchanged at approximately 13.4224"},{"name":"kernel_count_per_call","expectation":"unchanged at 1.00"}],"guardrails":["correctness:pass","output shape dtype device and contiguity unchanged","numerical tolerance atol=1e-2 rtol=1e-2","output bit-identical to the accepted kernel","device kernel name stays _fused_attention_kernel","BLOCK_M BLOCK_N HEAD_DIM num_warps and num_stages unchanged","kernel launch count stays at one","no aliasing of query key or value","cached buffer fully overwritten every call","public constructor and forward signature unchanged","base.py bytes unchanged","probe evidence retained under log/probes/ on both outcomes"],"profiling_level":"targeted","causal_graph":{"nodes":["n_launch_dispatch","n_host_work","n_device_unchanged","n_wall"],"edges":[["n_launch_dispatch","n_host_work"],["n_host_work","n_wall"],["n_device_unchanged","n_wall"]]}}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`: entries 011 through 016 are grouped-top-k selection on MLU590-H8 under a different compiler lowering. None matches this operator, backend, or host-side launch mechanism, so no recorded failure invalidates this path.
- Consulted `references/bottleneck-judgment.md`. `device_ratio` is `0.0445`, so the classification is `host-bound` and the launcher row of the compressible-versus-fixed table applies: the launcher path is "potentially compressible", but the required check is "target profile and same-regime wall evidence". That check is exactly the gate in the Optimization Intent, and it is why the probe must measure in the same process and regime rather than in isolation.
- **The Unknown capability is not being declared normative.** `lifecycle.fast-launcher` is `unknown` in the frozen snapshot and `failure_classification.unprovable_required_use` maps to `capability-miss`. This decision does not assert the path exists. It asserts that a probe can determine whether it exists, and it binds the intervention to that determination. Per the Designer contract, the expected lowering is declared conditionally, never as a guarantee, because the profile supports a probe rather than an inference.
- `fallback_provenance` is deliberately absent. The vNext contract restricts it to an `algorithm-substitution` fallback with a `qualification_disposition` embedded in the project claim. `state/project_capability_claim.json` carries `qualification_dispositions: []`, so no such disposition exists and claiming one would fail validation. This round is not an algorithm substitution in any case: the arithmetic and dataflow are untouched.
- The probe is **round-local evidence only**. It does not amend the frozen profile snapshot (which is hash-pinned and stays `Unknown`), does not set `uses_algorithm_substitution`, and does not create a campaign-level qualification disposition. A later round must re-establish legality on its own evidence.
- If the probe succeeds, the fallback on any subsequent cache-key change or resolution failure is the proven `kernel[grid](...)` launch, so the worst case at runtime is the accepted behaviour, not a wrong answer.
- **Measurement trap carried from round 003.** A `TorchDispatchMode` allocation count will still report one `aten.empty.memory_format` per call, because the Triton launch path allocates on its own independently of the cached output buffer. That is expected and is not a regression. `output_allocations_per_call` must be read at the Python level, where it stays `0.00`; the launch-path allocation is folded into `launch_path_us_per_call` and is part of what this round targets.
- Device time is not a fourth independent slice. Per `report_003.md`, the `13.4224 us` of device work sits inside the `51.815 us` synchronize term, which sits inside the harness-fixed `91.035 us`. Declaring `device_us_per_call` and `kernel_count_per_call` as unchanged control observables therefore keeps the device story out of the attribution entirely, which is required given the `4.0902%` device ceiling.

## Rationale and Evidence

All runtime facts come from `rounds/report_003.md`, including its Level 2 host decomposition. No value here is a Designer measurement.

### 1. Why this target and not the wrapper

The Level 2 decomposition, measured against a `297.410 us` wall:

| Slice | us/call | Share of wall | Reachable by a host round? |
|---|---:|---:|---|
| harness-fixed (outside `ModelNew.forward`) | 91.035 | 30.61% | **no** |
| Triton launch path (bare launch) | 183.740 | 61.78% | only by `launch-path-reduction`, capability Unknown |
| residual `forward` wrapper | 22.635 | 7.61% | yes, no new capability needed |
| device kernel time | 13.4224 | — | sub-component of the sync term; bounded at 4.09% |

The adoption budget is `0.05 * 297.410 = 14.871 us`. That is:

```text
14.871 / 183.740 =  8.09% of the launch path
14.871 /  22.635 = 65.71% of the residual wrapper
```

The launch path needs one part in twelve to be captured. The wrapper needs two parts in three. The launch path is also roughly eight times larger. This is the reason to spend the round on the launch path despite its Unknown capability, and the reason not to spend it on a wrapper squeeze that has to be near-total just to reach the threshold.

### 2. Why interventional-with-gate rather than probe-only

I chose the gated interventional form for four reasons.

1. **The Decision contract has no probe-only form.** `decision` is `proceed` or `abort`; `proceed` requires `change_scope` in `kernel|host|mixed` and a falsifiable intervention. A probe is not an intervention, so a probe-only round would have to be written as `abort` with `change_scope: none`.
2. **That would force schema_version 1.** `_validate_metadata_v2` requires `decision == "proceed"`, so an abort is only expressible at v1. Writing a v1 abort would abandon the v2 artifact chain for no contract benefit.
3. **It would mis-signal.** An abort asserts that no hypothesis worth a round exists. That is false: a `183.740 us/call` target was just measured at `61.78%` of wall.
4. **It would cost a second round.** A probe-only round produces evidence but no code path to an improvement. If the probe succeeds, the Coder has to be re-dispatched in a fresh round to exploit it. Putting the probe inside the round as a gate reaches the improvement in the same round, which is what "Coder runs Decision-scoped capability/compile probes before `candidate-ready`" describes.

**Terminal classification in each case:**

| Probe outcome | Wall outcome | Terminal classification | Canonical after |
|---|---|---|---|
| path proven | improves ≥5% | `accepted` | new round-004 candidate |
| path proven | improves <5% | `no-improvement` | `triton_mm_encoder_attention_e2_003.py` |
| path absent, incorrect, or slower | not measured | `capability-miss` | `triton_mm_encoder_attention_e2_003.py` |

In all three rows the probe evidence is retained under `log/probes/`, so the capability question is answered once and does not have to be re-litigated.

### 3. Why the kernel definition stays byte-identical

The `_fused_attention_kernel` `@triton.jit` definition is unchanged from `e2_001` and `e2_003`, and this round keeps it that way. Only the invocation site in `ModelNew.forward` and a lazily cached launcher handle change. Three reasons: the device side is bounded at `4.0902%` and cannot carry the attribution; keeping the kernel fixed preserves the `device_us_per_call` and `kernel_count_per_call` control observables that made round 003 attributable; and it keeps the change in one family. If the probe shows the only viable fast path requires a kernel-signature change, the round terminates `capability-miss` and a `mixed`-scope round would be needed to authorize it.

### 4. What the probe must answer

The gate is narrow. The probe must establish, on Ascend910B4 with triton 3.2.0 / torch_npu 2.7.1.post4 / CANN 9.0.0:

1. the alternative launch path exists;
2. it launches **the same compiled kernel** with the same grid and the same `BLOCK_M` / `BLOCK_N` / `HEAD_DIM` / `num_warps` / `num_stages`;
3. its output is bit-identical to the accepted kernel at `atol=1e-2`, `rtol=1e-2`;
4. its per-launch cost, measured in the same process and regime, is strictly below the `kernel[grid](...)` baseline of `183.740 us`.

Item 4 is the decisive one and is why the probe is Decision-scoped rather than a general capability survey: legality here is a *performance* claim about this runtime, not merely a compile claim.

Candidate mechanisms, in the order Coder should try: the `fast_libentry` fast launcher named by `lifecycle.fast-launcher`; a cached `CompiledKernel` direct invocation that bypasses `JITFunction.run`; any vendor precompiled launch entry point.

### 5. Expected gain is conditional, not promised

`expected_wall_improvement_pct` is `15.0`. That number is a judgment about capturing a modest fraction of a `183.740 us` term, and it is conditional on the probe: if the probe finds no path the realized improvement is zero and the round is a `capability-miss`, not a failed bet. The Designer contract forbids presenting such a number as guaranteed when the profile supports only a probe, and this section should be read as the conditional statement it is. The only adoption test remains the interleaved paired wall median against the re-measured reference, and the round is `no-improvement` unless wall time improves by at least 5% even if every mechanism observable moves as predicted.

### 6. What this round deliberately leaves alone

The `91.035 us` harness-fixed term is unreachable and is not targeted. Its composition is now understood — `51.815 us` of synchronize, `39.220 us` of seed drain plus `sync_devices()` accelerator probing, where `sync_devices()` costs `11.96 us/call` more than a bare `torch.npu.synchronize()` because `_iter_accelerators()` calls `torch.npu.is_available()` every time — but that is harness code and `bottleneck-judgment.md` forbids altering the harness to manufacture a speedup. The `22.635 us` residual wrapper is left intact as the fallback family: if this round is a `capability-miss`, that wrapper plus a smaller `query.device` construction is what remains inside the current family.

### 7. Scope discipline

One attributable cause: the per-call Triton launch dispatch. Nothing here touches the kernel definition, the launch configuration, the block sizes, the warp or stage counts, the output-buffer cache, or the public interface. `device_us_per_call` and `kernel_count_per_call` are unchanged control observables so that any wall movement is attributable to the host launch link rather than absorbed into a device story.
