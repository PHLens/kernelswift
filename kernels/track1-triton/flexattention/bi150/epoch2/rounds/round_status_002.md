# Round Status 002 — Verification of triton_flexattention_e2_002 (triton-attention-dispatch-collapse)

## Status: END — verification complete, classification `no-improvement` delivered

- role: verifier; phase verifying → terminal evidence emitted (Orchestrator owns transitions/counters)
- measurement_exclusive honored; default stream everywhere; D2 kernel-mode blockage stands (forward-mode only)
- terminal classification: `no-improvement` (regression −60.3422% paired; canonical remains baseline_adapter.py @`b8ec3458…`; miss streak → 2/3 per Orchestrator accounting)

## Identity checks (all re-verified live)

candidate `570bc2be…12b1` (6445 B) ✓ | decision `459e8d37…730` ✓ | sketch `fb5bec0b…87c` ✓ | report_000 `a90df70d…2e7c` ✓ | report_001 `8c93d473…336` ✓ | verdict_001 `c804df77…362` ✓ | accepted `b8ec3458…45d1` ✓ | base `dd1359ad…a6d0` ✓ | harness `71fb3ad0…fe29` ✓
- Binding statement LIVE `331859628ad6e891a835be7fa71baa99b69999f744c947293d8463e8b02c3278` — coder ledger's `ad4d4ba7…` is `p12_r002_sweep_result.json`'s hash (bookkeeping defect recorded; binding content independently verified: 14 DANGER tokens zero, 4 envelope-legal dot sites, num_warps=1, stateless).
- AST gate OK; independent DANGER rescan all-zero.

## Completed commands

1. Hash ledger + AST gate + DANGER rescan + dot-site/num_warps/stateless audit — PASS.
2. Verifier correctness probe (`log/probes/verifier_correctness_002.py`) — all_checks_pass TRUE: seed42 max_abs 9.766e-04; fp16-extreme tie-free suite max_abs 7.8125e-03 (within 1e-2); T=41 spot 9.766e-04; run_out poisoned ×2 bitwise==forward + data_ptr preserved; repeat bitwise stability; stateless audit clean. (One probe-side aggregation fix during the round: my attr-allowlist was too strict vs nn.Module boilerplate keys; underlying checks true from first run; candidate untouched.)
3. Authoritative pairs ×3 — PASS accuracy ×3; v0 [0.200049, 0.147338, 0.143139] → 0.147338; v1 [0.238962, 0.236245, 0.234321] → 0.236245; improvement −60.3422%.
4. ABAB supplement — A1 control −0.265 µs, B1 candidate +92.400 µs, A2 control −0.257 µs, B2 candidate +92.173 µs (drift-corrected ≈ +92.5 µs/call). Adapter-as-v0 direct pair attempted and BLOCKED by harness v0-contract format (v0 must define `Model`) — named deviation with exit-1 evidence.
5. Dual-scope forward profile pw=20/pi=100 — exit 0, PASS accuracy; trace @`d1e91bbd…9586`. Canonical summarizer blocked on candidate scope by kineto gpu_user_annotation duplicate-span artifact (named tool limitation); Verifier census scoped on host user_annotation span.
6. Census (`diagnostic_scope_census_round002.json`): candidate 1.00 aten op/call + 1.00 cuLaunchKernel/call (3.53 µs host) + 0 memcpys/graphs/syncs; `_causal_attn_fwd` 16.510 µs/call device vs base Ixmma 13.614; base 38 aten ops + 1 cudaLaunchKernel.
7. report_002.md emitted (improvement_pct corrected to exact −60.3422 during pinning); fact pack re-extracted post-write → pin `1ca59fa75293930df86ece437c9cefc9fe31266b5f4cb436c1d0ab3c5e8d3aa7` CONFIRMED.
8. verdict_002.json emitted + validated → `{"valid": true, "terminal_result": "no-improvement"}`; file sha256 `a3b7f117567fbd756356c9b10df58965665b8cd481513f47855da52db1c11985`.
9. state/verifier_context.md updated.

## Raw samples

- reference_raw_samples_ms: [0.200049, 0.147338, 0.143139] → median 0.147338
- candidate_raw_samples_ms: [0.238962, 0.236245, 0.234321] → median 0.236245
- improvement_pct: −60.3422 (regression; bar ≥+5.0 FAILED)
- Anchors: paired v0 basis −60.3422 (== accepted-reference basis since adapter is byte-equivalent to base pipeline; adapter-as-v0 structurally blocked, stated); cross-anchor report_000 0.151107 → +56.34% slower (session v0 drift −2.49% this round); manifest anchor identical to report_000 anchor.

## Key findings (D1 adjudication CONFIRMED)

- Mechanism ENGAGED: aten 38→1/call, single driver-API launch, stateless, zero copies/syncs.
- Device HEALTHY: T_triton 16.510 µs/call vs Ixmma 13.614 (+2.9 µs; feared ≥60 µs band did not materialize).
- Wall FAILED on host: Triton python launcher path costs ~+86–89 µs/call NET more than the entire replaced base host path (~134 µs), of which only 3.53 µs is the driver submission — pure python launcher overhead ~82–86 µs/call.
- Hypothesis verdict: partially-confirmed (collapse + device-band edges confirmed; wall falsified with census-grade root cause).

## Next safe action

Orchestrator validates report_002.md + verdict_002.json gate; applies counters (no-improvement streak 2/3, total_rounds 2) and keeps last_accepted = baseline_adapter.py @`b8ec3458…`. Per campaign rules a third no-improvement anywhere terminates. Verifier idle until next dispatch.
