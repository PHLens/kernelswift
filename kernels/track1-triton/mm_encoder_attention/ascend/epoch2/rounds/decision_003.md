# Decision 003

## Metadata

```json
{"schema_version":2,"decision":"proceed","decision_kind":"optimization","round":"003","reference_implementation":"triton_mm_encoder_attention_e2_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"host","change_family":"allocation-reuse","sketch_ref":"rounds/sketch_003.json","sketch_sha256":"51ebe3a735c7659309e781fd2f35286fd4e67acc86b5d0a9f6676f08f08af69c","implementation_profile_snapshot_ref":"state/implementation_profile_snapshot/profile.yaml","implementation_profile_snapshot_sha256":"a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321","project_capability_claim_ref":"state/project_capability_claim.json","project_capability_claim_sha256":"a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"eliminate the per-call output allocation in ModelNew.forward by caching the output buffer on the ModelNew instance under an explicit cache key, so the steady-state forward performs no allocation and no per-call launch-constant reconstruction while the kernel body and launch configuration stay byte-identical","allowed_changes":["ModelNew.forward host code","ModelNew.__init__ host code","output allocation ownership and lifetime"],"invariants":["fused attention kernel body unchanged","kernel launch count stays at one","BLOCK_M BLOCK_N HEAD_DIM accumulator_dtype num_warps and num_stages unchanged","ModelNew public contract","output shape dtype device and contiguity","numerical tolerance atol=1e-2 rtol=1e-2","no aliasing of query key or value","base.py bytes unchanged"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

```json
{"artifact":"rounds/sketch_003.json","sha256":"51ebe3a735c7659309e781fd2f35286fd4e67acc86b5d0a9f6676f08f08af69c","rendering":"# D Declarations\ntensor query shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor key shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor value shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor out shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\nscalar out_shape shape=[3] dtype=int64 layout=scalar memory=host\ntile q_tile shape=[BLOCK_M,HEAD_DIM] dtype=fp16 layout=blocked memory=register\ntile k_tile shape=[BLOCK_N,HEAD_DIM] dtype=fp16 layout=blocked memory=register\ntile v_tile shape=[BLOCK_N,HEAD_DIM] dtype=fp16 layout=blocked memory=register\nscalar scale shape=[1] dtype=fp32 layout=scalar memory=register\n\n# O Operations\nalloc out <- out_shape on the ModelNew instance; a cache hit performs no allocation\nload q_tile <- query[b, 0:BLOCK_M, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask row_idx < S\nload k_tile <- key[b, 0:BLOCK_N, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask offs_n < S\nload v_tile <- value[b, 0:BLOCK_N, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask offs_n < S\ncompute qk = dot(q_tile, trans(k_tile)) * scale  # fp32 accumulate, conversion declared\ncompute p = masked_softmax(qk)\ncompute acc = dot(p.to(fp16), v_tile)\ncompute acc_norm = acc / rowsum(p)\nstore out[b, 0:BLOCK_M, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] <- acc_norm mask row_idx < S\n\n# C Control\nparallel bh over B*NH\nguard row_idx < S\nguard offs_n < S\n\n# H Target Hints\ntarget=triton_ascend\nBLOCK_M=128\nBLOCK_N=128\nHEAD_DIM=64\naccumulator_dtype=fp32\nnum_warps=4\nnum_stages=1\n"}
```

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward","ModelNew.__init__","output allocation"],"state_owner":"ModelNew instance; the buffer is an ordinary attribute and is not module state, so it is never serialized into state_dict","lifetime":"model lifetime; created on the first forward after construction or after a cache-key change, held until the instance is released","allocation_reuse":"allocate once with torch.empty using the cached shape dtype and device, then return the same tensor on every subsequent forward whose cache key matches; the kernel store fully overwrites every element of the buffer before it is returned, so no stale value can escape","cache_key":["output shape tuple","output dtype","output device","query stride tuple"],"invalidation":"compare the cache key on every call and discard and reallocate the cached buffer whenever any component differs from the cached values; a shape dtype device or stride change is a miss, never a silent reinterpretation","concurrency":"one ModelNew instance is not shared across concurrent forwards; the benchmark drives a single sequential call stream from one thread, and no lock, thread-local, or per-call state is introduced","device_stream_behavior":"the buffer is allocated on query.device and the kernel store executes on the caller's current stream; no stream is created, captured, or switched, and the harness's existing per-call torch.npu.synchronize boundary is unchanged","unchanged_behavior":["returned shape","returned dtype","returned device","returned contiguity","numerical semantics within atol=1e-2 rtol=1e-2","no aliasing of query key or value","kernel launch count stays at one","public constructor and forward signature"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-003","intervention":"eliminate the per-call output allocation in ModelNew.forward by caching the output buffer on the ModelNew instance under an explicit cache key, so the steady-state forward performs no allocation and no per-call launch-constant reconstruction while the kernel body and launch configuration stay byte-identical","expected_causal_chain":["the steady-state forward performs zero output allocations instead of one","per-call host work inside ModelNew.forward decreases","device time and kernel count stay fixed so the wall delta is attributable to host","synchronized wall median decreases by at least five percent"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"output_allocations_per_call","expectation":"decrease from 1 to 0 on a cache hit"},{"name":"host_us_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"unchanged at approximately 13.4064"},{"name":"kernel_count_per_call","expectation":"unchanged at 1.00"}],"guardrails":["correctness:pass","output shape dtype device and contiguity unchanged","numerical tolerance atol=1e-2 rtol=1e-2","returned tensor is not an alias of query key or value","cached buffer is fully overwritten by the kernel store on every call","public constructor and forward signature unchanged","base.py bytes unchanged"],"profiling_level":"targeted","causal_graph":{"nodes":["n_output_alloc","n_host_work","n_device_unchanged","n_wall"],"edges":[["n_output_alloc","n_host_work"],["n_host_work","n_wall"],["n_device_unchanged","n_wall"]]}}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`: entries 011 through 016 are grouped-top-k selection on MLU590-H8 under a different compiler lowering. None matches this operator, backend, or host-side mechanism, so no recorded failure invalidates this path.
- Consulted `references/bottleneck-judgment.md`. `device_ratio` is `0.0407`, so the classification is `host-bound` and §4 applies: allocation work was observed, so the Host Plan must specify state owner, lifetime, cache key, invalidation, concurrency, device, and stream behavior before the change is made. All eight fields are filled above.
- The same reference permits Level 2 `targeted` evidence only when the hypothesis concerns host, launcher, allocation, context, stream, or harness-fixed time. This hypothesis is about allocation, so `targeted` is the correct level and the Level 2 host decomposition requested below is justified rather than routine.
- `device-win-wall-loss` is not the governing risk this round, because this round makes no device claim at all. It is still the reason `device_us_per_call` and `kernel_count_per_call` are declared as `unchanged` control observables: if the wall moves while device time also moves, the attribution is broken and the round must not be adopted on a device story.
- Buffer reuse is a real semantic change, not a free optimization. The two hazards are stale data and aliasing. Stale data is closed by the kernel's store coverage: the grid is `(B*NH,)` = 16 programs and together they write every row `0..S-1` of every head slice, so the whole `[2, 83, 512]` buffer is overwritten on every call. Aliasing is impossible because the buffer is created with `torch.empty`, never as a view of `query`, `key`, or `value`.
- Caching across calls is only safe because the returned tensor is fully rewritten before it escapes. If a future round ever introduces a masked or partial store, this Host Plan must be revisited, because the reuse invariant would no longer hold.
- The `torch.empty_like` internal-format warning recorded in `rounds/coder_result_001.md` is part of the per-call allocation path this round removes. Coder asserted it does not affect the measured path; that assertion was never measured. Removing the call settles the question either way rather than relying on the assertion.
- No capability in the frozen profile is newly required. `make_block_ptr`, `async_copy`, `vectorize`, and `lifecycle.fast-launcher` all remain Unknown and none is declared normative here.

## Rationale and Evidence

All runtime facts come from `rounds/report_001.md`. No value in this decision is a Designer measurement.

### 1. Why host, and why now

Round 002 established a hard bound on the device side:

```text
5% adoption budget = 0.05 * 327.770          = 16.3885 us/call
complete device budget                       = 13.4064 us/call
best possible device-only wall improvement   =  4.0902%
```

The maintainer then authorized host-side code (`team-state.md` Policy Revisions, commit `de1b9b7`), naming `launch-path-reduction` and `allocation-reuse` as permitted Host Plan rounds. The device side stays closed by the bound above; nothing in this round revisits it.

### 2. Where the timed cost actually sits

Read from `auto_bench.py` (`time_forward`, harness sha256 `71fb3ad0…`), the per-sample timed region is:

```text
set_seed(seed)          <- outside the timed region
start = time.perf_counter()
  torch.no_grad(): model.forward(*inputs)
  sync_devices()        <- torch.npu.synchronize()
elapsed = time.perf_counter() - start
```

Two consequences. First, seed setup is not timed, so the measured wall is `forward` plus a full device synchronize. Second, and decisively, **every microsecond of Python and allocator work inside `ModelNew.forward` is inside the timed region**, because the synchronize waits for the device after the host has already returned. Host work in `forward` is directly billable, which is exactly why a host round can move wall time at all.

### 3. The residual, stated honestly

The non-device residual is an arithmetic consequence of two Verifier numbers, not a measured host decomposition:

```text
candidate residual = 329.365 - 13.4064  = 315.9586 us/call
reference residual = 358.720 - 118.8920 = 239.8280 us/call
```

I do not know how that residual splits between the harness synchronize, the Triton launch path, and the output allocation, and this decision does not assume a split. The intervention is chosen because it removes work that is unambiguously present and unambiguously per-call, not because a decomposition located the residual there.

**Verifier evidence that would confirm the split:** a targeted Level 2 host decomposition in one process and regime, measuring (a) harness wall time, (b) `ModelNew.forward` alone, (c) `forward` plus the harness synchronize boundary, and (d) an allocation-free `forward` variant. The delta between (b) and (d) sizes this round's lever directly; the delta between (c) and (b) sizes the harness-fixed term that no host round can touch. Requesting that decomposition alongside the adoption measurement is the highest-value diagnostic in this epoch.

### 4. The intervention

`ModelNew.forward` currently performs, on every call, work that does not depend on the input values:

1. `torch.empty_like(query)` — a full allocation round-trip through the NPU caching allocator, plus the internal-format warning path that `rounds/coder_result_001.md` records as firing on this runtime;
2. `bsz, q_len, hidden = query.shape` unpacking and a `query.stride(0)` / `query.stride(1)` pair;
3. `self.num_heads`, `self.head_size`, and the `block` literal reconstruction of the grid.

This round moves (1) to an instance-owned buffer allocated once with `torch.empty` and returned on every cache hit, and hoists (3) to `__init__` as constructor constants. (2) is retained because the strides are launch arguments and are simultaneously the cache-key components that make reuse safe: a caller that passes a differently-strided tensor of the same shape is a cache miss, not a silent mislaunch.

The kernel body, the launch count, and every Sketch hint are unchanged. The Sketch is present because the v2 contract requires it, and it declares the computation boundary precisely so that "unchanged" is a normative statement rather than a claim in prose.

### 5. Why `allocation-reuse` before `launch-path-reduction`

Both families are authorized. `allocation-reuse` is taken first for a capability reason, not a magnitude reason. `launch-path-reduction` in its high-value form means bypassing the `JITFunction.run` dispatch and invoking the cached compiled kernel directly. The frozen profile records `launch_abi: "kernel[(grid)](args)"` and states that direct launch syntax is the **proven** launcher path, and `lifecycle.fast-launcher` is `Unknown` with no Ascend probe. Declaring an unproven launcher normative would convert a performance round into a `capability-miss` under `failure_classification.unprovable_required_use`. Allocation reuse requires no capability that is not already in use by the accepted kernel.

If this round returns `no-improvement`, `launch-path-reduction` is the next family, and it should be preceded by an Ascend probe of the launch ABI rather than attempted as a code experiment.

### 6. Expected gain is a judgment, not a measurement

`expected_wall_improvement_pct` is `8.0`. That number is a judgment about the cost of one allocator round-trip plus a per-call warning path against a `16.3885 us` budget, and it carries wide uncertainty: the true saving could plausibly land anywhere from a few microseconds to several tens. It is not a prediction Verifier is asked to confirm. The only adoption test is the interleaved paired wall median against the accepted reference, and the round is `no-improvement` unless wall time improves by at least 5% even if every mechanism observable moves in the predicted direction.

### 7. Round scope discipline

This decision changes one attributable cause: redundant per-call host work that is invariant across calls at fixed shape. It does not touch the kernel, the launch ABI, the block sizes, the warp or stage configuration, or the public interface. `device_us_per_call` and `kernel_count_per_call` are declared as unchanged control observables so that any wall movement can be attributed to the host link rather than absorbed into a device story.
