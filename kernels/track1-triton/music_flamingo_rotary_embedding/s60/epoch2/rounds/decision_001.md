# Decision 001

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "optimization",
  "round": "001",
  "reference_implementation": "baseline_adapter.py",
  "reference_report": "rounds/report_000.md",
  "language": "triton",
  "backend": "gcu",
  "target_profile": "triton_gcu",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "mixed",
  "change_family": "triton-launch-fusion",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "15c2055ed921227a35490a3d010e2ba730f4254bd76918ab50564908f6336827",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_gcu.yaml",
  "implementation_profile_snapshot_sha256": "7cd0cdf4b01b064b91f2b8f199cff6d12b175903a2c8d24ba7153f4d6a6aa6a0",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "2428781222c3a29927618669d692497ba0f8c0c22e37bd80553ed13c7c9ef809"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "mixed",
  "intervention": "PARTIAL fusion: replace the base eager freqs elementwise chain (batch_positions/max_seq_len div, *inv_freq mul, repeat_interleave(2), [:,None,:] unsqueeze, broadcast_tensors, cat dim=-1, *angle.unsqueeze(-1)) with ONE direct-launched Triton kernel (grid=(B,seq_len)=(4,32)=128 programs, one per (b,t), num_warps=1, HALF=32 power-of-2 tl.arange) that computes freqs[b,t,:128] into a single intermediate [4,32,128] fp32 buffer; cos/sin are RETAINED as VENDOR torch.cos/torch.sin on the host side against that freqs buffer (the epoch-1 full-fusion lesson: GCU math-dialect tl.cos/tl.sin is ~44% slower than the vendor trig library, which caused the -13%). Kernel body per (b,t): load inv_freq[i] for i in [0,32); batch_freq=(b/max_seq_len)*inv_freq[i]; load time_freq=position_angles[t,2i] (even column, since position_angles is already repeat_interleave(2) so adjacent columns duplicate); angle=-timestamps[b,t]*2*pi; f_bf=batch_freq*angle; f_tf=time_freq*angle; store freqs[b,t,2i]=f_bf, freqs[b,t,2i+1]=f_bf, freqs[b,t,64+2i]=f_tf, freqs[b,t,64+2i+1]=f_tf. Host: return (freqs.cos(), freqs.sin()).",
  "allowed_changes": [
    "kernel: one stateless @triton.jit elementwise kernel computing the freqs chain (mul/div/load/store only, NO tl.cos/tl.sin) into an intermediate [4,32,128] fp32 buffer",
    "host: forward collapses the eager elementwise chain to torch.empty([4,32,128]) + one kernel launch + freqs.cos() + freqs.sin() (vendor) — cos/sin are the ONLY host-side math ops and MUST remain vendor torch.cos/torch.sin",
    "indexing: batch_freq uses inv_freq[i] with repeat_interleave emulated by writing f_bf to BOTH columns 2i and 2i+1; time_freq reads position_angles[t, 2i] (even column, adjacent duplicate) and writes f_tf to BOTH columns 64+2i and 64+2i+1",
    "dtype: fp32 throughout (loads, registers, intermediate freqs buffer, vendor cos/sin)",
    "strictly NOT: no tl.cos / no tl.sin anywhere in the kernel (cos/sin delegated to vendor host ops)",
    "strictly NOT: no torch.compile / no caching launchers / no runtime codegen strings of any kind",
    "strictly NOT: no tl.dot (operator is pure elementwise; matrix.dot is irrelevant), no reduction, no algorithm substitution",
    "strictly NOT: no tl.arange extent that is not a power of 2 (HALF=32); no .contiguous(), no output copies beyond the single intermediate freqs buffer, no extra kernels"
  ],
  "invariants": [
    "correctness:pass under the unchanged comparator (exact-match; preflight measured diff=0.0 against base, deterministic elementwise + vendor trig)",
    "output is the tuple (cos, sin), two tensors each [4,32,128] fp32, returned fresh from forward",
    "public API: ModelNew(dim=64, max_seq_len=256, base=10000.0); forward(timestamps, seq_len) — no run_out surface (base returns a fresh tuple)",
    "register buffers inv_freq [32] and position_angles [256,64] computed host-side in __init__ and NOT fused; state_dict keys {inv_freq, position_angles} unchanged",
    "time_freq reads position_angles[t, 2i] (even column) because position_angles is already repeat_interleave(2) — adjacent columns are duplicates, so reading the even column and writing to both 2i/2i+1 reproduces the repeat_interleave exactly",
    "cos/sin MUST be vendor torch.cos/torch.sin (host side); the kernel MUST NOT call tl.cos/tl.sin (epoch-1 -13% root cause: GCU math-dialect trig ~44% slower than vendor)",
    "capability legality: tl.arange extent power-of-2 (HALF=32); num_warps=1; elementwise mul/div only; zero DANGER tokens (compile/capture) in candidate source",
    "stateless module: zero call-time instance state, zero caches, zero workspace; caller device and current stream preserved; AST-loader-safe module composition"
  ],
  "expected_wall_improvement_pct": 49.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_001.json",
  "sha256": "15c2055ed921227a35490a3d010e2ba730f4254bd76918ab50564908f6336827",
  "rendering": "one direct-launched Triton elementwise kernel as the freqs-chain computation boundary: grid=(B,T)=(4,32)=128 programs (one per (b,t)), num_warps=1, HALF=32 power-of-2 tl.arange; per program it loads inv_freq[i] (i in [0,32)) and position_angles[t,2i] (even column, repeat_interleave(2) adjacent-duplicate), computes angle=-timestamps[b,t]*2*pi, then f_bf=(b/max_seq_len)*inv_freq[i]*angle and f_tf=position_angles[t,2i]*angle, and stores into the intermediate [4,32,128] fp32 freqs buffer as freqs[b,t,2i]=f_bf, freqs[b,t,2i+1]=f_bf, freqs[b,t,64+2i]=f_tf, freqs[b,t,64+2i+1]=f_tf; the kernel contains NO tl.cos/tl.sin — the host returns (freqs.cos(), freqs.sin()) using vendor torch.cos/torch.sin, which is the round's key novelty over epoch-1 full fusion"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward host path: torch.empty([4,32,128], fp32) for the intermediate freqs buffer + one Triton kernel launch (grid=(4,32), num_warps=1) + freqs.cos() + freqs.sin() — replacing the base ~13 eager topsLaunchKernel launches (div/mul/repeat_interleave/broadcast/cat/mul-angle chain + vendor cos/sin)",
    "cos/sin vendor ops: freqs.cos() and freqs.sin() run on host via torch vendor trig library against the kernel-written freqs buffer (NOT re-launched inside the kernel)",
    "kernel launch configuration: grid=(B, T)=(4, 32), num_warps=1, constexpr HALF=32/D=64/MAX_SEQ_LEN=256 frozen at module-definition time as literals (AST-loader safe)",
    "per-call output allocation: forward allocates one fresh intermediate [4,32,128] fp32 freqs tensor; cos/sin are vendor views computed from it (two additional [4,32,128] outputs produced by torch.cos/torch.sin)"
  ],
  "state_owner": "NOBODY",
  "lifetime": "stateless module: the @triton.jit function object plus the framework-owned JIT specialization cache live for the module lifetime; per-call buffers (freqs, cos, sin) live for the call; no cross-call state is created or read",
  "allocation_reuse": "NONE",
  "cache_key": [
    "not-applicable: stateless module; Triton JIT specialization is keyed by the constexpr tuple (fixed literals) and the framework cache lifecycle, not by runtime cache_keys; no cache is owned by the module"
  ],
  "invalidation": "not-applicable (stateless); the kernel recompiles only if the constexpr tuple changed, which it cannot (literals frozen)",
  "concurrency": "stateless and side-effect-free host path; safe under concurrent callers with the same device/stream semantics as the base module",
  "device_stream_behavior": "all work launches onto the caller's current device and current stream via the standard Triton launcher and vendor torch trig ops; no side streams, no events, no capture, no synchronize, no memcpy",
  "unchanged_behavior": [
    "forward returns a fresh tuple (cos, sin), two fp32 [4,32,128] tensors with identical mathematical semantics to base (exact-match; preflight diff=0.0)",
    "register buffers inv_freq [32] and position_angles [256,64] are unchanged and unfused; state_dict keys preserved",
    "all host operations launch onto the caller's current device and current stream; no side streams, no events, no capture, no synchronization",
    "module remains JIT-warm after harness warmup 50 (first-call compile absorbed outside timed medians)",
    "cos/sin numerics are IDENTICAL to base (same vendor torch.cos/torch.sin applied to the numerically-equal freqs), which is why the round is exact-match rather than merely allclose"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-001",
  "intervention": "PARTIAL fusion: replace the base eager freqs elementwise chain (batch_positions/max_seq_len div, *inv_freq mul, repeat_interleave(2), [:,None,:] unsqueeze, broadcast_tensors, cat dim=-1, *angle.unsqueeze(-1)) with ONE direct-launched Triton kernel (grid=(B,seq_len)=(4,32)=128 programs, one per (b,t), num_warps=1, HALF=32 power-of-2 tl.arange) that computes freqs[b,t,:128] into a single intermediate [4,32,128] fp32 buffer; cos/sin are RETAINED as VENDOR torch.cos/torch.sin on the host side against that freqs buffer (the epoch-1 full-fusion lesson: GCU math-dialect tl.cos/tl.sin is ~44% slower than the vendor trig library, which caused the -13%). Kernel body per (b,t): load inv_freq[i] for i in [0,32); batch_freq=(b/max_seq_len)*inv_freq[i]; load time_freq=position_angles[t,2i] (even column, since position_angles is already repeat_interleave(2) so adjacent columns duplicate); angle=-timestamps[b,t]*2*pi; f_bf=batch_freq*angle; f_tf=time_freq*angle; store freqs[b,t,2i]=f_bf, freqs[b,t,2i+1]=f_bf, freqs[b,t,64+2i]=f_tf, freqs[b,t,64+2i+1]=f_tf. Host: return (freqs.cos(), freqs.sin()).",
  "expected_causal_chain": [
    "dispatch side: base ~13 eager topsLaunchKernel launches (elementwise div/mul/repeat_interleave/broadcast/cat/mul-angle chain + vendor cos/sin) collapse to 3 submissions (1 Triton launch + 2 vendor torch.cos/torch.sin); the ~10 elementwise launches are the collapse target",
    "device side: the kernel does ONLY fp32 mul/div/load/store (no trig), so its device time is a small multiple of the base's pure-elementwise slice, while the vendor cos/sin device time is unchanged — the epoch-1 -13% came from moving trig INTO the kernel via tl.cos/tl.sin (~44% slower than vendor), which this round avoids by retaining vendor trig",
    "net: preflight measured 367us -> 246us = 1.49x wall improvement, correctness exact-match (diff=0.0); the win is the launch collapse with the device penalty deliberately avoided by partial (not full) fusion"
  ],
  "primary_metric": {
    "name": "wall_time",
    "expected_improvement_pct": 5.0
  },
  "causal_graph": {
    "nodes": [
      "cn.dispatch-collapse",
      "cn.vendor-trig-retention",
      "cn.device-time-delta",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.dispatch-collapse", "cn.wall-time"],
      ["cn.vendor-trig-retention", "cn.wall-time"],
      ["cn.device-time-delta", "cn.wall-time"],
      ["cn.dispatch-collapse", "cn.device-time-delta"],
      ["cn.vendor-trig-retention", "cn.device-time-delta"]
    ]
  },
  "mechanism_observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "expectation": "preflight 1.49x (367us -> 246us); honest declaration with Verifier authoritative — the win branch holds if the launch collapse (13->3) dominates and the kernel's elementwise-only device floor stays below the base's elementwise-chain floor"
    },
    {
      "name": "runtime_launch_count_per_call",
      "expectation": "exactly 3.00 kernel/vendor submissions per call (1 Triton + 2 vendor cos/sin) vs base 13; ZERO extra submissions, ZERO graph launches, ZERO model-code synchronizations"
    },
    {
      "name": "cos_sin_vendor_retention_audit",
      "expectation": "zero tl.cos / tl.sin call sites in the kernel source; cos/sin realized exclusively by host torch.cos/torch.sin — the structural guarantee that separates this round from epoch-1's -13% full fusion"
    },
    {
      "name": "device_us_per_call",
      "expectation": "kernel device time is elementwise-only (mul/div/load/store, no trig); vendor cos/sin device time unchanged from base — GCU device-duration is unavailable (launch-only trace) so this is inferred from wall minus launch-API-time"
    },
    {
      "name": "correctness_exact_match",
      "expectation": "preflight diff=0.0 (exact-match) because freqs is numerically identical to base and cos/sin use the SAME vendor torch.cos/torch.sin; repeat_interleave emulation (write f to both 2i/2i+1) and time_freq even-column read must reproduce base bit-for-bit"
    },
    {
      "name": "power_of_2_arange_audit",
      "expectation": "every tl.arange extent is a power of 2 (HALF=32); num_warps=1; count of torch.compile/TORCHINDUCTOR/reduce-overhead strings = 0; zero .contiguous() calls in the forward host path; zero tl.dot / tl.cos / tl.sin in kernel"
    }
  ],
  "guardrails": [
    "correctness:pass under the unchanged comparator (exact-match; preflight diff=0.0)",
    "outputs remain the tuple (cos, sin), two fp32 [4,32,128] tensors with base-identical semantics",
    "register buffers inv_freq [32] and position_angles [256,64] unchanged and unfused; state_dict keys preserved",
    "cos/sin are vendor torch.cos/torch.sin (host side); the kernel contains NO tl.cos/tl.sin",
    "stateless module: no instance attributes written at call time, no caches, no workspace; caller device and current stream preserved; no added synchronization",
    "cold first-call JIT compile stays outside timed medians (harness warmup 50 absorbs it)",
    "AST-loader-safe module composition (safe-literal module constants; retained defs)",
    "capability legality: tl.arange power-of-2 only (HALF=32); num_warps=1; elementwise mul/div only; zero DANGER tokens (compile/capture) in candidate source"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- anti-patterns.md consulted: no cataloged failure matches — the MLU selection-network / reduction / dynamic-gather failures are all selection or reduction workloads; this operator is dense, selection-free, reduction-free, tie-free elementwise + trig.
- epoch-1 root cause named and excluded BY DESIGN: the epoch-1 `triton_rotary_001.py` full-fusion moved cos/sin INTO the kernel via `tl.cos`/`tl.sin`, which the epoch-1 final_summary quantified as a 157us / ~44% device penalty — GCU's MLIR math-dialect trig is far slower than the vendor trig library. This round deliberately PARTIAL-fuses: the kernel stops at the freqs buffer and cos/sin stay vendor. The preflight confirms this flips -13% to 1.49x with exact-match correctness (diff=0.0).
- epoch-1 structural mistakes named and excluded: (i) the Round-1 single-program `grid=(1,)` serialization (5.16ms, -1011%) is avoided by grid=(4,32)=128 programs; (ii) the flattened-index int div/mod index de-resolution (b = offs//(seq_len*2D), etc.) is avoided by a 2-D program-id decomposition where each program owns exactly one (b,t) and a HALF=32 power-of-2 tl.arange for the freq axis.
- S60-SPECIFIC capability constraints (probe-backed, MUST be honored by Coder):
  1. `tl.arange` requires POWER-OF-2 extent → HALF=32 (never 24 or any non-power-of-2). The capability_claim.json notes the CORRECTED constraint: "M/N/K must all be POWERS OF TWO, not merely multiples of 16" — this applies to tl.arange extents as well.
  2. `num_warps=1` is the proven launch configuration for this elementwise shape (profile proves num_warps 1/2/4/8 compile; 1 is the canonical choice here; higher warp counts do not add independent parallelism on the 2-MP device for a gather-style elementwise kernel).
  3. NO `tl.cos` / `tl.sin` — the math-dialect trig is the epoch-1 penalty; cos/sin are vendor host ops. This is the single most binding constraint of the round.
  4. The kernel is elementwise mul/div/load/store only: no `tl.dot` (irrelevant — no GEMM), no reduction, no algorithm substitution (the capability_claim fallback_contract is null for this project).
- numerics: fp32 throughout; the freqs value is computed identically to base (the repeat_interleave is emulated by writing the same f_bf/f_tf to both adjacent columns, and time_freq reads position_angles[t,2i] which equals [t,2i+1] because position_angles is pre-repeat_interleave'd); cos/sin are the SAME vendor functions on the SAME freqs values → exact-match (diff=0.0), not merely allclose.
- cold JIT compile is absorbed by harness warmup 50; no runtime codegen strings anywhere in the candidate.
- DANGER-token binding notes for Coder: zero compile/capture strings; no tl.cos/tl.sin; num_warps=1; tl.arange power-of-2 (HALF=32); zero .contiguous()/copy_ in host paths; stateless audit; forward returns a fresh (cos, sin) tuple.

## Rationale and Evidence

**Reference and canonical anchors.** Accepted pair: baseline_adapter.py @ 9fc87abbe0e6268f06c969e94f5400abea51cdf315276a4ac5cef5bd0ad8a26f and rounds/report_000.md (baseline median ≈ 0.447 ms). The base forward issues ~13 eager launches for a pure elementwise + trig chain: batch_positions/max_seq_len (div), *inv_freq (mul), repeat_interleave(2), [:,None,:] unsqueeze, broadcast_tensors, cat(dim=-1), *angle.unsqueeze(-1) (mul), then vendor freqs.cos() / freqs.sin(). The computation is a flat elementwise map with no GEMM and no data-axis reduction.

**Preflight evidence (orchestrator-scoped, measured).** base 13 launches, wall ~447us (harness) / 367us (bare probe). epoch-1 FULL fusion (cos/sin via tl.cos/tl.sin) was -13% — the GCU math-dialect tl.cos/tl.sin is ~44% slower than the vendor trig library. The NEW direction (epoch-1 did NOT try) is PARTIAL fusion: fuse only the freqs elementwise chain (div/mul/repeat_interleave/broadcast/cat/mul-angle) into one kernel, keep cos/sin as vendor torch.cos/torch.sin. Preflight reached 1.49x (367us -> 246us), correctness exact-match (diff=0.0).

**Why partial (not full) fusion is the right boundary.** Full fusion amortizes the launch tax but moves trig into the kernel, where GCU's math dialect is ~44% slower than vendor — a device penalty that ate the launch savings (net -13%). Partial fusion captures the launch-collapse win (13 -> 3 submissions) while keeping the trig exactly where it is fast (vendor). The kernel's device work is a small, pure-elementwise fp32 slice (mul/div/load/store over HALF=32 per program), so its added device cost is far below the ~44% trig penalty it avoids.

**Break-even arithmetic (honest, Verifier authoritative).** Adoption bar 5% of ~447us ≈ 22.3us (harness median), or of 367us ≈ 18.4us (probe). Preflight 1.49x clears both comfortably (246us vs 367us = 121us saved). The mechanism is attributable: ~10 elementwise launches collapse to 1, while the 2 vendor trig launches remain. The residual risk is the kernel's device floor for 128 programs each doing a HALF=32 gather-style load of position_angles[t,2i] (strided-by-2 across a [256,64] buffer) — but that is a read of 32 contiguous-adjacent even columns, far cheaper than the trig it avoids.

**Change-scope justification.** change_scope is `mixed`: the Triton kernel rewrite (computing freqs) and the host-side vendor cos/sin retention are ONE inseparable mechanism — the kernel produces the freqs buffer that the host vendor trig then consumes. They are separately observable (runtime_launch_count_per_call vs cos_sin_vendor_retention_audit vs wall_time), satisfying the observability requirement. change_family `triton-launch-fusion` names the partial-fusion-with-vendor-trig-retention family.

**Why PROCEED with expected_wall_improvement_pct 49.0.** The preflight is a real, measured 1.49x with exact-match correctness — unlike epoch-1's -13%, this direction has no known device penalty. Even under regression risk, the round banks the campaign's PRIMARY contractual product (a correctness-PASS Triton submission) and the structural lesson (partial fusion boundary). The Verifier's wall measurement is authoritative; the 49.0% is an honest pre-adoption estimate, not a guarantee.

**Artifacts consulted.** project.md (identity, DELIVERABLE RULE, runtime fingerprint, Key Prior preflight); rounds/report_000.md (canonical baseline + 13-launch census); baseline_adapter.py (immutable reference semantics via ../../base.py); profile_snapshot/triton_gcu.yaml @7cd0cdf4… and capability_claim.json @24287812… (frozen envelope: tl.arange power-of-2, num_warps 1/2/4/8, elementwise primary_contract, null fallback); epoch-1 archive ../final_summary.md + ../triton_rotary_001.py + ../rounds/decision_001.md (the -13% full-fusion prior and its 157us/44% trig penalty, excluded by design); sibling campaign mm_encoder_attention/s60/epoch2 rounds/decision_001.md + sketch_001.json (schema-v2 typed-sketch format reference); skills references invariants.md, anti-patterns.md, bottleneck-judgment.md, decision-template.md; auto_bench.py (harness/AST-loader behavior); state/team-state.md.
