# Decision 003

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "optimization",
  "round": "003",
  "reference_implementation": "baseline_adapter.py",
  "reference_report": "rounds/report_000.md",
  "language": "triton",
  "backend": "cuda",
  "target_profile": "triton_cuda",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "mixed",
  "change_family": "graph-replayed-triton-direct-address",
  "sketch_ref": "rounds/sketch_003.json",
  "sketch_sha256": "bdf423556e7c80369ae38d4980529a739a52a3d18033e572927354b23e0a4e64",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_cuda.yaml",
  "implementation_profile_snapshot_sha256": "dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "aeba3a87f0494c2bb349b92fe668370c70d77fdebea29eac52824c3556b0d4d8"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "host-bound",
  "intervention": "compose the two PROVEN mechanisms from rounds 001-002 into one candidate: the r002 Triton attention kernel (authoritative D_cand = 19.5550 us/call attributed, host-invariant, outputs bitwise-stable and bitwise-equal to r001) is captured ONCE PER POINTER-SET as a manual torch.cuda.CUDAGraph whose single kernel launch reads the CALLER'S OWN input addresses directly (harness auto_bench.time_forward reuses one cloned inputs list across all warmup+repeat calls — address stability source-verified) and writes a static out_ws [2,83,512] fp16 workspace; per timed call the model executes ONLY a three-way data_ptr guard, ONE cudaGraphLaunch-class replay submission, and one ~166 KB DtoD copy-out into a fresh invocation-owned buffer (forward) or the caller-provided buffer (run_out) — the r002 per-call python launcher (T_launcher = +84.5712 us/call measured, the entire r001/r002 failure mode) never executes on the replayed route; any pointer mismatch serves THAT call via the copy-in replay tier (static q_in/k_in/v_in workspaces + 3 copy-ins + replay + copy-out, bitwise-identical results) with a bounded recapture (max 4 lifetime, first-seen pointer sets only) so the harness's stable-address timed regime rides the lean tier after at most one warmup-time recapture; three-tier permanent chain: direct-address replay -> workspace copy-in replay -> r002-style direct Triton launch (eager, nw2 config); the kernel itself is BYTE-IDENTICAL in intent to r002's (same mathematics, same (32,32) fp32 widened dots, same num_warps=2, same 48-program grid) — round 003 changes ONLY the execution boundary",
  "allowed_changes": [
    "manual CUDA-graph capture of the SINGLE r002 @triton.jit kernel launch, bound to the caller's own input pointers (direct-address tier): static out_ws [2,83,512] fp16 workspace written inside the graph; captured on a side capture stream per torch.cuda.graph pattern, once per first-seen pointer set, recapture budget 4 lifetime",
    "per-call tier-1 host path reduced to: 3x data_ptr guard comparisons, ONE graph replay submission, ONE small copy-out (DtoD) into an invocation-owned fresh buffer (forward) / caller buffer (run_out); zero copy-ins; zero model-code synchronization; ~5-6 python-visible ops",
    "tier-2 copy-in replay: static-workspace machinery (3 fp16 copy-ins [2,83,512] + replay + copy-out) serving any call whose input pointers mismatch the tier-1 anchors, bitwise-identical results, zero artifacts on non-target regimes",
    "tier-3 eager: the r002 direct-launch path (kernel + torch.empty, nw2 config) for non-target regimes and any replay failure; permanent downward tier binding on capture/replay exception",
    "NO change to: kernel mathematics (scale=0.125, bidirectional full attention with -inf only on S=83 padding, online fp32 softmax), capability legality (tl.dot strictly at proven (32,32)@(32,32) fp32 on widened operands, num_warps=2 r002-probe-qualified, num_stages unset), output contract (single fp16 [2,83,512]), public signatures",
    "strictly NO: torch.compile / reduce-overhead / inductor machinery, fp16-operand dots (capability-NEGATIVE per r002 sweep), non-32 dots, num_warps != 2, algorithm-substitution fallback (reduction.sum BLOCKED, waiver NOT granted), result-return from graph-resident memory, model-code synchronization"
  ],
  "invariants": [
    "correctness:pass under the unchanged harness comparator (allclose atol=1e-2 rtol=1e-2 equal_nan=True, seed 42) THROUGH EVERY TIER, including the correctness phase's per-call cloned inputs (pointer-varying) served by tier-2",
    "outputs remain single fp16 tensors [2,83,512]; run_out fills the caller buffer before returning None; forward returns a freshly-written invocation-owned buffer every call — NEVER a graph-resident reference",
    "cross-tier bitwise retention: tier-1, tier-2, and tier-3 outputs are BITWISE-EQUAL for identical input bits (same kernel, same bits, deterministic configuration; copy boundaries preserve bits); any deviation is a capture defect and fails immediately",
    "stale-address impossibility: tier-1 replay only fires when the three data_ptr guards match the captured anchors exactly; mismatched calls are NEVER served from stale addresses (tier-2 copy-in recomputes from live bits)",
    "public ModelNew constructor and forward(query, key, value) signatures unchanged; run_out(query,key,value,out) unchanged 4-arg contract",
    "capability legality: every tl.dot call site compiles to (32,32)@(32,32) fp32 operands/accumulator (widened operands); num_warps=2 per the r002 probe qualification; binding ledger audits all dot sites",
    "caller-selected device preserved; captures execute ONCE per pointer-set on a dedicated side stream; replays and copy-outs run on the CALLER'S current stream; model code performs no device-context creation/removal and adds no synchronization",
    "bounded state: the instance owns at most 2 graph handles (tier-1 ptr-bound + tier-2 workspace), the static workspaces (out_ws, q_in/k_in/v_in), pointer-anchor constants, a recapture counter (<=4), and monotone tier flags; NOTHING else persists; results are never stored across calls",
    "zero torch.compile / TORCHINDUCTOR / reduce-overhead strings; AST-loader-safe module composition (safe-literal constants, retained defs)"
  ],
  "expected_wall_improvement_pct": 0.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_003.json",
  "sha256": "bdf423556e7c80369ae38d4980529a739a52a3d18033e572927354b23e0a4e64",
  "rendering": "normative contract is rounds/sketch_003.json; the captured computation is EXACTLY the r002 kernel dataflow (six widened 32-chunk loads -> two proven-shape fp32 dots per key-tile scaled by 0.125 -> S=83 boundary -inf mask only (bidirectional, no causal skip) -> online running-max softmax -> PV dots -> normalize) with its store retargeted to the static out_ws [2,83,512] workspace; round 003 changes ONLY the execution boundary: the single kernel launch is captured once per pointer-set into a manual CUDA graph bound to the caller's own input addresses, each call pays a data_ptr guard + ONE replay submission + one ~166 KB copy-out, the r002 python launcher never runs on this route, and pointer mismatches fall to the copy-in replay tier with bitwise-identical results"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward routing across the three-tier chain (direct-address replay -> copy-in replay -> eager direct launch)",
    "ModelNew.run_out routing across the three-tier chain (copy-out to the CALLER-provided buffer on tier-1/2; direct kernel write on tier-3)",
    "one-time tier-1 capture per first-seen input pointer set (bounded recapture counter <= 4 lifetime), captured against the caller's live q/k/v addresses + static out_ws",
    "one-time tier-2 capture (static workspace placeholders q_in/k_in/v_in + 3 copy-ins + replay + copy-out), built lazily on first pointer mismatch or first capture failure",
    "static state set: out_ws[2,83,512] fp16, q_in/k_in/v_in[2,83,512] fp16, pointer-anchor triple, recapture counter, tier flags (direct_replay_failed, copyin_replay_failed)",
    "per-call forward copy-out target: one invocation-owned fresh [2,83,512] fp16 buffer allocated OUTSIDE the graph each call"
  ],
  "state_owner": "the ModelNew instance owns the two graph handles, the static workspace tensors, the pointer-anchor constants, the recapture counter, and monotone tier flags; framework-owned graph-private pools back allocations performed DURING capture; every user-visible result originates as an invocation-owned buffer (forward) or the caller's buffer (run_out) filled by per-call copy-out — workspace and graph-resident contents are transient computation state fully rewritten each call and NEVER returned directly or read across calls; the pointer anchors are CACHE-KEY STATE (data_ptr of q/k/v) whose every use is guarded by per-call comparison",
  "lifetime": "graph handles and workspaces persist for the instance lifetime after successful capture; tier flags transition monotonically downward at most once each (direct-address -> copy-in -> eager); the recapture counter decrements irreversibly; all state becomes garbage with module destruction; per-call forward output buffers live exactly one call",
  "allocation_reuse": "MODEL CODE performs zero per-call allocations INSIDE any replayed region; the only per-call allocations are the forward-path invocation-owned result buffer (fresh torch.empty each call, outside the boundary) — run_out performs zero allocations; lower tiers allocate exactly as their proven precedents (tier-3 = the r002 direct path: torch.empty + one launch)",
  "cache_key": ["data_ptr(q)", "data_ptr(k)", "data_ptr(v)", "shape", "dtype", "device"],
  "invalidation": "tier-1 is VALID if and only if the three per-call data_ptr values equal the captured anchors (strongest possible cache key — a pointer match guarantees the bytes read are the live caller bytes); any mismatch routes THAT call to tier-2 copy-in replay (never stale-address service); recapture is attempted only when the counter budget (>0) remains AND the pointer set was never bound before (same-set revisits go to tier-2, preventing alternation cost); recapture is never triggered after budget exhaustion; any capture/replay exception binds the offending tier permanently downward",
  "concurrency": "one model instance is not shared across concurrent forwards; sequential per-call guard + replay + copy-out satisfies graph output-lifetime rules; pointer-anchor binding is single-set by design — alternating caller pointer sets beyond the recapture budget simply ride tier-2 correctly",
  "device_stream_behavior": "caller-selected device preserved; tier-1 and tier-2 captures execute ONCE each per binding on a dedicated side capture stream per the torch.cuda.graph recommended pattern; afterwards every call runs on the CALLER'S current stream via stream-safe replay + copy-out; model code contains ZERO synchronization, zero device queries beyond data_ptr reads, zero context operations; harness seed/sync behavior untouched",
  "unchanged_behavior": [
    "forward(query, key, value) public signature and return structure: single fp16 [2,83,512] tensor",
    "numerical semantics of the r002 kernel (scale=0.125, bidirectional full attention, per-(batch,head) online fp32 softmax, widened fp32 dots, num_warps=2)",
    "run_out(query,key,value,out) fills the caller buffer bitwise-equal to forward for identical inputs and returns None",
    "non-target regimes (any shape/dtype/device mismatch) route to tier-3 eager with correct outputs and zero graph artifacts",
    "zero torch.compile usage; the only device kernel is the candidate's own Triton kernel (sanctioned r002 change, unchanged this round)"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-003",
  "intervention": "compose the two PROVEN mechanisms from rounds 001-002 into one candidate: the r002 Triton attention kernel (authoritative D_cand = 19.5550 us/call attributed, host-invariant, outputs bitwise-stable and bitwise-equal to r001) is captured ONCE PER POINTER-SET as a manual torch.cuda.CUDAGraph whose single kernel launch reads the CALLER'S OWN input addresses directly (harness auto_bench.time_forward reuses one cloned inputs list across all warmup+repeat calls — address stability source-verified) and writes a static out_ws [2,83,512] fp16 workspace; per timed call the model executes ONLY a three-way data_ptr guard, ONE cudaGraphLaunch-class replay submission, and one ~166 KB DtoD copy-out into a fresh invocation-owned buffer (forward) or the caller-provided buffer (run_out) — the r002 per-call python launcher (T_launcher = +84.5712 us/call measured, the entire r001/r002 failure mode) never executes on the replayed route; any pointer mismatch serves THAT call via the copy-in replay tier (static q_in/k_in/v_in workspaces + 3 copy-ins + replay + copy-out, bitwise-identical results) with a bounded recapture (max 4 lifetime, first-seen pointer sets only) so the harness's stable-address timed regime rides the lean tier after at most one warmup-time recapture; three-tier permanent chain: direct-address replay -> workspace copy-in replay -> r002-style direct Triton launch (eager, nw2 config); the kernel itself is BYTE-IDENTICAL in intent to r002's (same mathematics, same (32,32) fp32 widened dots, same num_warps=2, same 48-program grid) — round 003 changes ONLY the execution boundary",
  "expected_causal_chain": [
    "r001 measured the failure mode precisely: T_launcher = +84.7651 us/call net (r002 re-measured +84.5712 — invariance band PASS) — the Triton python launcher path costs more than the entire 33-op aten stack it replaces; r002 cut the device term to 19.5550 us/call attributed with bitwise-equal outputs and a fully host-invariant census (1.00 aten op, 1.00 cuLaunchKernel, zero memcpys/graphs/syncs); the direct family's wall arithmetic is closed (win needs D_cand <= -75.7 us)",
    "the composition neutralizes the launcher (replayed launches execute no python) behind the lean boundary proven in the sibling campaign's r003: per timed call = 3x data_ptr guard + ONE cudaGraphLaunch + ONE ~166 KB copy-out (2 GPU submissions, zero copy-ins on the stable-address timed regime — harness time_forward reuses one cloned inputs list, source-verified; the correctness phase's per-call cloned inputs ride tier-2), zero model-code synchronization",
    "priced identity with canonical numbers (same-session paired basis): net = -T_launcher(84.571) + R-term(69.02, sibling-measured build-intrinsic replay-sync — transfer to bsz=2 is itself a named observable) + boundary(~13: graph launch ~5.5 + memcpy ~5.5 + ~2 aten) + (D_kernel_in_graph + copy-out 3.7 - vendor 17.42); at the attributed D_kernel = 19.555: net = +3.28 us/call WORSE => composed wall ~148.3 us => ~0.98x; at the graph-assisted regime pace 15.317 (r002 p13): net = -0.95 us => ~1.007x; report_002's conservative class folds boundary/R-term uncertainty into 0.94-0.96x — the honest composed band is 0.94-1.01x, and the kernel-in-graph device regime (attributed 19.555 vs graph-assisted 15.317) is adjudicated BY this round's census",
    "the >=5% adoption bar needs net <= -7.27 us => D_kernel_in_graph <= 9.2 us — 10.36 us below the measured authoritative floor: UNREACHABLE, declared honestly (expected_wall_improvement_pct 0.0); the round's expected verdict is no-improvement #3 => campaign auto-termination — spent deliberately, per the DELIVERABLE RULE, to bank the composed submission",
    "regardless of wall outcome the round banks the campaign's PRIMARY contractual product per project.md DELIVERABLE RULE: the composed correctness-PASS Triton submission at the 0.94-1.01x class (vs the banked direct 0.6258x — the best Triton submission this lineage can produce, the exact terminal move of the sibling flexattention campaign), plus the graph-family physics closure at bsz=2: R-term transfer, boundary terms, and the kernel-in-graph device regime"
  ],
  "primary_metric": {
    "name": "wall_time",
    "expected_improvement_pct": 5.0
  },
  "causal_graph": {
    "nodes": [
      "cn.graph-replay-direct-address",
      "cn.launcher-python-time",
      "cn.boundary-cost-delta",
      "cn.device-time-delta",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.graph-replay-direct-address", "cn.launcher-python-time"],
      ["cn.graph-replay-direct-address", "cn.boundary-cost-delta"],
      ["cn.launcher-python-time", "cn.wall-time"],
      ["cn.boundary-cost-delta", "cn.wall-time"],
      ["cn.device-time-delta", "cn.wall-time"],
      ["cn.graph-replay-direct-address", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "expectation": "at least 5% below the same-session accepted reference median across interleaved pairs at warmup 50 / repeat 100 — honestly declared two-sided with the composed band 0.94-1.01x (net -1.0 to +3.3 us): the win branch requires the kernel-in-graph regime at <= 9.2 us, 10.36 us below the authoritative floor, and is expected NOT to fire; the expected reading is sub-parity-to-parity, i.e., no-improvement #3 with the composed deliverable banked"
    },
    {
      "name": "tier1_hit_rate_in_timed_regime",
      "expectation": "all timed calls ride the direct-address tier: census shows ZERO copy-in DtoD memcpys and exactly 1.00 cudaGraphLaunch + 1.00 copy-out memcpy per call; recapture count during timed samples = 0 (at most one during warmup); hit-rate 0 under stable harness addresses would falsify the design premise (pre-declared reading d)"
    },
    {
      "name": "aten_cpu_ops_per_call",
      "expectation": "<=6/call on the timed path (empty + copy-out copy_ + guard/replay-visible ops), consistent with guard + replay + copy-out and nothing else"
    },
    {
      "name": "submission_and_sync_census",
      "expectation": "exactly 2.00 GPU submissions/call (1 graph launch + 1 memcpyAsync) and ZERO cuLaunchKernel on the replayed route; model-code sync/driver-query count = 0; any per-call cudaDeviceSynchronize/cudaDriverGetVersion OBSERVED in the candidate scope is recorded as the build-intrinsic replay cost (the R-term at bsz=2 — pre-declared pessimistic branch c), not silently absorbed"
    },
    {
      "name": "rterm_transfer_at_bsz2",
      "expectation": "the canonical bsz=2 R-term measurement: the per-call replay-sync-class cost implied by paired wall minus census-attributed terms, compared against the sibling campaign's 69.02 us/call (bsz=1) — closes the last unmeasured line of the graph-family physics map this lineage has; material deviation (> +/-5 us) re-prices every future graph-composition decision in this operator family"
    },
    {
      "name": "device_us_per_call",
      "expectation": "composed attributed band: kernel-in-graph ~15.3-19.6 us (the graph-assisted vs attributed regime adjudication — a named product of the round; attribution may coarsen for graph-replayed kernels per the sibling branch-B precedent, census substitutes) + copy-out ~3.7; two-sided: materially higher readings trigger the pessimistic branch and attribute cleanly"
    },
    {
      "name": "cross_tier_bitwise_retention",
      "expectation": "tier-1, tier-2, tier-3 outputs bitwise-equal for identical input bits through BOTH entry surfaces; stale-trap: calls with changed input pointers return CORRECT fresh results (never stale bytes); run_out poisoned-buffer writes x2 bitwise with data_ptr preserved; composed outputs bitwise-equal to the r002 direct kernel on identical bits (same kernel, copies preserve bits)"
    },
    {
      "name": "proven_envelope_binding_audit",
      "expectation": "every tl.dot call site (32,32)@(32,32) fp32 widened operands; num_warps=2 (r002 probe-qualified) at the single launch site; zero torch.compile/TORCHINDUCTOR/reduce-overhead strings; bounded-state audit (<=2 graph handles, workspaces, anchors, counter <=4, monotone flags); zero model-code synchronization calls"
    }
  ],
  "guardrails": [
    "correctness:pass through every tier under the unchanged comparator",
    "outputs remain single fp16 [2,83,512] tensors; results never served from graph-resident memory",
    "stale-address impossibility via per-call data_ptr guards; mismatched calls recompute from live bits (tier-2)",
    "state bounded to the declared set (2 graph handles, workspaces, anchors, counter, flags); caller device and current stream preserved; no model-code synchronization",
    "run_out bitwise==forward for identical inputs; caller buffers never aliased to graph memory",
    "cold capture/JIT cost stays outside timed medians (harness warmup 50 absorbs one recapture)",
    "capability legality: kernel byte-identical in intent to r002's — (32,32) fp32 widened dots, num_warps=2, num_stages unset; fp16-operand dots stay excluded (capability-NEGATIVE per the r002 sweep)",
    "no algorithm substitution (reduction.sum BLOCKED, waiver NOT granted); zero DANGER tokens (compile/capture-strings) in candidate source"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`: no cataloged failure matches; the MLU selection-network entries concern tie-surface selection workloads (absent here — dense softmax, tie-free); the scalar-FMA lesson is respected (all dots stay tl.dot at the proven envelope); the mutation-skip lesson is respected by MANUAL capture (no inductor consult) and by the captured region mutating ONLY graph-private/static workspace state (out_ws).
- Why this is a DISTINCT legal family, not a re-run: round 002 changed the kernel's execution configuration (device term; causal nodes cn.register-occupancy/cn.mma-path); round 003 changes the EXECUTION BOUNDARY around a byte-identical kernel (causal nodes cn.launcher-python-time/cn.boundary-cost-delta) — different mechanism layer, different failure surface, and the two predecessor rounds measured exactly the two terms this composition multiplies (r001/r002: T_launcher +84.77/+84.57 and the host-invariant census; r002: D_cand 19.555 authoritative; sibling campaign: R-term 69.02 and the lean-boundary price).
- Capture hazards for Coder: (i) capture ONCE per pointer-set on a side stream (torch.cuda.graph pattern); the captured region is EXACTLY one kernel launch — no branches, no prints, no .item(), no data_ptr reads INSIDE the region; (ii) tier-1 anchors must be captured from the LIVE caller tensors (kernel arguments are the caller pointers themselves, not workspace); (iii) recapture only on first-seen pointer set with budget >0 — never inside the timed loop more than once (harness warmup absorbs it); (iv) the copy-out runs OUTSIDE the graph on the caller's current stream into a fresh/invocation-owned buffer (forward) or the caller buffer (run_out); (v) if torch.cuda.graph capture of a Triton launch fails on this build, bind permanently to tier-2 (copy-in replay — the p13 probe already graph-assisted this exact kernel successfully, so capturability is probe-supported, not speculative; pre-declared reading (e) still covers total failure); (vi) NO synchronization anywhere in model code — if the build's replay path intrinsically syncs, the census will show it and the R-term-at-bsz2 observable records it (pessimistic reading c).
- Numerics: unchanged from r002 (fp32 online softmax, -inf exact-zero masking on S=83 padding only, widened operands, 1e-2 tolerance dominates; no index-carrying reductions => no tie surface); cross-tier bitwise identity is structural (same kernel, same bits, deterministic config; copies preserve bits) — and the r002 kernel's outputs are themselves bitwise-equal to r001 (verified), so the composed deliverable inherits the full correctness pedigree including the extreme-suite reading (3.052e-05 vs fp32 ground truth while the vendor diverges 1457).
- DANGER notes for Coder binding statement: torch.compile / TORCHINDUCTOR / reduce-overhead / tf32 strings count 0 REQUIRED; tl.dot audit vs proven envelope REQUIRED; num_warps != 2 FAILS; model-code cudaDeviceSynchronize/stream-query/driver-query calls FAIL; returning graph-resident or workspace tensors from forward/run_out FAILS; unbounded recapture (no counter) FAILS; the captured region must contain exactly ONE kernel launch.

## Rationale and Evidence

**Canonical anchors.** Reference pair unchanged through r001/r002 (both no-improvement): baseline_adapter.py @c3980a2c… + rounds/report_000.md @20b21646…. Round-002 history (rounds/report_002.md @bb46dee7…, verdict_002.json @a86c2e8c…): wall −59.8032% (0.231689 vs 0.144984 ms paired median, inside the declared band); AUTHORITATIVE D_cand(nw2) = 19.5550 µs/call attributed (probe-method 15.317, replay-regime bias +4.2–4.7 µs measured twice); T_launcher +84.5712 µs/call net (invariance band PASS); host census fully unchanged; correctness PASS on all five suites with outputs bitwise-equal to r001; capability matrix closed: fp16-operand dots compile but FAIL exactness (max_abs 1459, vendor-saturation signature) at every warp count — capability-NEGATIVE; nw4 no gain over nw2; deliverable banked @cc98318b at 0.6258x. Streak 2/3; round budget 2/20; the p13 probe already graph-assisted this exact kernel successfully (capturability probe-supported).

**The three-way calculus (dispatch's framing), decided.**
- (γ) any third family: EMPTY by measurement. The ≥5% bar needs D_kernel_in_graph ≤ 9.2 µs — 10.36 µs below the authoritative floor; fp16 dots are capability-NEGATIVE (canonical); num_warps is exhausted (nw2 optimal, nw4 no-gain); in-envelope restructuring is exhausted (r002's own no-headroom analysis); the host floor is build-intrinsic (T_launcher invariant across two rounds); larger tiles would need another probe round and land at best a ~0.64x direct candidate that can never be composed afterward (streak exhausted) — strictly dominated by composing now. No family clears 5%; none is expected to.
- (β) honest close-out now: leaves the composed 0.94–1.01x-class submission UNBUILT — a strictly worse competition deliverable — and per the dispatch, an abort does not even end the campaign mechanically (dispatch_next_round stays true; a close-out would then need user stop). The ledger-cleanliness difference (miss #3 vs failed-attempt +1) only matters if this lineage is reopened later, and a reopened lineage starts a fresh campaign regardless.
- (α) F2 graph composition: CHOSEN. The project.md DELIVERABLE RULE (binding, user-corrected precedent) makes the best correctness-PASS Triton submission the campaign's PRIMARY contractual product; the composed candidate improves that product from 0.6258x to the 0.94–1.01x class in one round — the exact terminal move of the sibling flexattention campaign (whose final submission WAS the composed 1.00x variant after 3/3 no-improvement). The expected verdict is no-improvement #3 => clean mechanical auto-termination with the best deliverable banked; a capture-failure REJECTED round does not consume the streak and permits a replacement round (budget 2/20 — rounds are not scarce).

**Priced identity (why the expectation is honestly 0.0).** Same-session paired: net = −T_launcher(84.571) + R(69.02 sibling, transfer measured by the rterm_transfer_at_bsz2 observable) + boundary(~13) + (D_kernel_in_graph + 3.7 − 17.42). At the attributed D_kernel: net = +3.28 µs (−2.3%, ~0.98x); at the graph-assisted pace: net = −0.95 µs (+0.7%, ~1.007x); report_002's conservative class 0.94–0.96x. All branches sit below the +5% bar (needs −7.27 µs). The round is expected to FAIL the wall criterion and TERMINATE the campaign — spent deliberately, with full pre-declared attribution, to convert the last bullet into the best submission this operator lineage can produce plus the final physics closure (R-term at bsz=2, boundary terms, kernel-in-graph regime adjudication between 19.555 attributed and 15.317 graph-assisted).

**One-attributable-change compliance.** change_scope mixed under the inseparability clause: the graph wrapper and the captured kernel are ONE composed mechanism whose host and device effects are separately observable (census vs device time); the kernel itself is byte-identical in intent to r002's (same mathematics, same legality envelope, same configuration — the r002-vs-r001 diff discipline applies); no other mechanism is touched.

**Pre-declared failure readings (two-sided honesty).** (a) wall ≥ +5% with tier-1 hit-rate 100 ⇒ win (not expected — needs device ≤ 9.2); (b) wall < +5% with hit-rate 100 AND lean census ⇒ sub-parity composition confirmed: no-improvement #3, campaign terminates with the composed deliverable banked — THE EXPECTED READING; (c) wall < +5% WITH observed per-call sync/driverGet ⇒ build-intrinsic replay floor at bsz=2 named via the R-term observable, deliverable still banked, still no-improvement #3; (d) hit-rate 0 under stable harness addresses ⇒ design premise falsified (harness-behavior root cause); (e) total capture failure ⇒ tier-2-only wall ≈ +7 µs worse ⇒ no-improvement with Triton-capturability root cause (probe-supported against, but covered); tier-2 also failing ⇒ tier-3 eager = the r002 wall (−59.8%) with root cause; (f) any correctness/bitwise deviation ⇒ candidate-failed channel, never slack reinterpretation.

**Artifacts consulted.** rounds/report_002.md @bb46dee7… (canonical r002 evidence: authoritative D_cand 19.5550, T_launcher +84.5712, capability matrix, F2 projection +3.10 µs); rounds/verdict_002.json @a86c2e8c…; rounds/report_001.md @13adafe9… (T_launcher +84.7651, D_cand 28.2030); rounds/report_000.md @20b21646… + baseline_adapter.py @c3980a2c… (reference pair); triton_mm_encoder_attention_e2_002.py @cc98318b… (the byte-identical kernel being composed; p13 graph-assisted probe evidence); rounds/decision_002.md @20b360ac… + rounds/sketch_002.json @c16b1528…; ../../base.py @86ac5703…; profile_snapshot/triton_cuda.yaml @dc8fa4c0… + capability_claim.json @aeba3a87… (frozen pins; reduction.sum waiver NOT granted); auto_bench.py @71fb3ad0… (time_forward inputs-list reuse source-verified; run_forward per-call clones drive the tier-2 design); team-state.md (streak 2/3, budget 2/20, miss-3 auto-termination policy); references/invariants.md, anti-patterns.md, bottleneck-judgment.md, decision-template.md; sibling flexattention/bi150/epoch2 decision_003.md + sketch_003.json @4ef267b9… (the proven composition architecture and boundary pricing this round replicates at bsz=2); state/designer_context.md (the campaign model this decision concludes).
