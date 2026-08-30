# Round Status 003 — Verification of triton_flexattention_e2_003 (graph-replayed-triton-direct-address)

## Status: END — verification complete, classification `no-improvement` #3 → campaign auto-termination

- role: verifier; phase verifying → terminal evidence emitted (Orchestrator owns the stop transition)
- measurement_exclusive honored; default stream everywhere; forward-mode only (D2-r001 standing)
- terminal classification: `no-improvement` (+0.2186% paired — a wash, below the +5% bar); canonical stays `baseline_adapter.py` @`b8ec3458…` as FINAL DELIVERABLE; miss streak 3/3 per Orchestrator accounting

## Identity checks (all re-verified live)

candidate `6ffb0c94…bf1e` (15316 B) ✓ | decision `d4f7203e…b203` ✓ | sketch `4ef267b9…c6f0` ✓ | report_000 `a90df70d…2e7c` ✓ | report_001 `8c93d473…336` ✓ | report_002 `2b93a9ed…5ec8` ✓ | verdict_002 `a3b7f117…1985` ✓ | accepted `b8ec3458…45d1` ✓ | base `dd1359ad…a6d0` ✓ | harness `71fb3ad0…fe29` ✓ | binding LIVE `f8be3a6b…a7c1` ✓
- Kernel block BYTE-IDENTICAL to r002 (machine extraction-diff `@triton.jit..class ModelNew`) ✓
- AST gate OK; 14 DANGER tokens zero (incl. sync/query/driverGet/return-workspace); 4 dot sites (32,32)@(32,32) fp32; num_warps=1 single site; state audit matches declared set

## Completed commands

1. Hash ledger + kernel byte-identity + audits — PASS.
2. Verifier correctness probe (`log/probes/verifier_correctness_003.py`) — all_checks_pass TRUE, first run: harness-phase replication with behavioral counters (initial binding budget-free → EXACTLY ONE warmup recapture → ZERO timed recaptures, 100/100 timed bitwise==twin); 5-way bitwise seed42 ALL TRUE; stale-trap pass; extreme suite max_abs 7.8125e-03; T=41 tier-3 zero artifacts.
3. Authoritative pairs ×3 — PASS accuracy ×3; v0 [0.151042, 0.149147, 0.147234] → 0.149147; v1 [0.148821, 0.150517, 0.143639] → 0.148821; improvement +0.2186% (wash; identity-control pair Δ −0.021 µs confirms noise floor).
4. Dual-scope forward profile pw=20/pi=100 — exit 0, PASS accuracy; trace @`c8182c25…5c5f3`. Reference scope canonical (0.98 kernels/call @ 14.99 µs); candidate scope: canonical summarizer reports "scope has no kernel events" (in-graph attribution coarsening, r001 branch-B pattern) → Verifier census substitutes per the decision's pre-declared rule.
5. Census (`diagnostic_scope_census_round003.json`): 3.00 aten ops/call; 2.00 GPU submissions/call (1.00 cudaGraphLaunch @5.46 µs + 1.00 cudaMemcpyAsync @5.51 µs); ZERO cudaLaunchKernel; ZERO cuLaunchKernel (launcher executes ZERO times in serving — r002 tax neutralized); **R-TERM FOUND: 1.00 cudaDeviceSynchronize @69.02 µs/call + 1.00 cudaDriverGetVersion @0.18 µs — build-intrinsic replay floor (branch (c) triggered)**; 0.99 copy-out DtoD @3.70 µs/call device, zero copy-ins.
6. report_003.md emitted (five-number decomposition included); fact pack extracted post-write → pin `a7b0f319b57f8557124e147cd573c12f1f653a3993da7d1fee5e38b8b719240c` CONFIRMED.
7. verdict_003.json emitted + validated → `{"valid": true, "terminal_result": "no-improvement"}`; file sha256 `e92f076c2cffdce4b26b988e7b707e3212aad3c80851787b953757a6a482ebcc`.
8. state/verifier_context.md updated (context_epoch 4, terminal state).

## Raw samples

- reference_raw_samples_ms: [0.151042, 0.149147, 0.147234] → median 0.149147
- candidate_raw_samples_ms: [0.148821, 0.150517, 0.143639] → median 0.148821
- improvement_pct: +0.2186 (wash; bar ≥+5.0 FAILED)
- Anchors: paired v0 basis +0.2186 (headline; == accepted-reference basis); cross-anchor report_000 0.151107 → +1.5128% (below bar); manifest anchor identical to report_000 anchor — stated in report.

## Key findings (closing physics — all measured)

- Design premise VERIFIED: tier-1 hit-rate 100% in the timed regime (zero timed recaptures), lean 2-submission boundary, launcher fully neutralized.
- Wall: WASH (+0.2186%) — the build-intrinsic replay-sync floor (69.02 µs/call cudaDeviceSynchronize on a sync-free candidate route) absorbs the priced ~17–28 µs python savings; pre-declared readings (b)+(c) apply; (d)(e)(f) never triggered.
- Hypothesis verdict: partially-confirmed (launcher neutralization + lean boundary + device-in-band + hit-rate premise all CONFIRMED; wall outcome falsified with R-term attributed).
- FIVE-NUMBER CAMPAIGN DECOMPOSITION (full physics, all census-grade): (1) base host path ≈134 µs/call; (2) r001 wrapper net +2.6 µs/call; (3) r002 launcher tax ≈+86–89 µs/call; (4) r003 composed net −0.3 µs/call (R-term 69.02 µs/call absorbs the prize); (5) device floors Ixmma 13.61–15.0 vs Triton 16.51 µs/call.

## Next safe action

Orchestrator validates report_003.md + verdict_003.json, applies termination (no-improvement 3/3), finalizes with baseline_adapter.py @`b8ec3458…` as the final deliverable (manifest anchor 0.151107 ms). Verifier idle — no next dispatch expected.
