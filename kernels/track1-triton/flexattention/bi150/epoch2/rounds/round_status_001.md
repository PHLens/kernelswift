# Round Status 001 — Verification of triton_flexattention_e2_001 (manual-cuda-graph-workspace-replay)

## Status: END — verification complete, classification `no-improvement` delivered

- role: verifier; phase verifying → terminal evidence emitted (Orchestrator owns transitions/counters)
- measurement_exclusive honored; D1 default-stream discipline honored everywhere (no stream tricks)
- terminal classification: `no-improvement` (regression −1.6873% paired vs +5.0% bar; canonical remains baseline_adapter.py @`b8ec3458…`)

## Identity checks (all re-verified live this turn)

candidate `b490acc6…cadec` ✓ | decision `fa11b115…7b1d` ✓ | sketch `199275b8…1fce` ✓ | anchor report_000 `a90df70d…2e7c` ✓ | accepted reference `b8ec3458…45d1` ✓ | base `dd1359ad…a6d0` ✓ | harness `71fb3ad0…fe29` ✓ | binding statement `916058cb…8dc5b` ✓ (DANGER rescan independently re-run: all-zero)

## Completed commands

1. Hash ledger + AST gate + DANGER rescan — PASS.
2. Verifier correctness/active-tier probe (`log/probes/verifier_tier_check_001.py`) — exit 0, `all_checks_pass: true` (14/14 booleans). First run exposed an aggregation bug in MY probe (misread `flags_after=false` = "not failed" as failure); probe script fixed, re-run clean. All underlying checks were true from the first run; no candidate change.
3. Authoritative pairs ×3 (`--warmup 50 --repeat 100`, v0=base.py vs v1=candidate) — PASS accuracy ×3; samples below.
4. Kernel-mode attempt on THIS candidate — failed verbatim: `TypeError: ModelNew.run_out() missing 2 required positional arguments: 'value' and 'out'`, exit 1 (D2 arity deviation recorded; harness passes `run_out(gating_output, *output_args)` only).
5. Forward dual-scope profile pw=20/pi=100 — exit 0, PASS accuracy; trace `log/r001_forward_100iter.pt.trace.json` @`b6732432…3271`.
6. Per-scope summaries (`r001_summary_reference.json`, `r001_summary_candidate.json`) + host census (`diagnostic_scope_census_round001.json`).
7. `rounds/report_001.md` emitted; fact pack extracted post-write → pin `f4c72154…0ff7d` CONFIRMED.
8. `rounds/verdict_001.json` emitted and validated via `validate_verdict` API → `{"valid": true, "classification": "none", "terminal_result": "no-improvement"}`; verdict file sha256 `c804df77d0d9ad6cff85c4cfd0b587da76b7ceb06a4843f352eee59ac9e6e362`.
9. `state/verifier_context.md` ledger updated.

## Raw samples

- reference_raw_samples_ms: [0.155043, 0.153720, 0.156726] → median 0.155043
- candidate_raw_samples_ms: [0.161593, 0.154599, 0.157659] → median 0.157659
- improvement_pct: −1.6873 (regression; bar ≥+5.0 FAILED)
- Anchors: paired v0 basis == direct same-session r000-v0 pair (identical comparison, last_accepted IS base adapter) −1.6873; cross-anchor report_000 0.151107 → −4.3355 (session drift +2.604% on v0 side); manifest anchor identical to report_000 anchor this early.

## Key findings

- ACTIVE TIER = manual-replay (graph handle + workspace + flag-false; 150/150 at-scale bitwise; 1.00 cudaGraphLaunch/call in trace).
- Mechanism ENGAGED: aten dispatch 34→6/call; trip structure exactly 3 copy-ins + 1 replay + 1 copy-out; attributed kernels 0.14/call vs base 0.86/call (branch A).
- Wall outcome: regression −1.6873% — decision reading (a): host floor is per-submission/sync-bound, not dispatch-count-bound; candidate route pays 5 GPU submissions/call vs eager 1 (+ per-call cudaDeviceSynchronize/cudaDriverGetVersion observed).
- Hypothesis verdict: partially-confirmed (mechanism links confirmed; expected wall outcome falsified).

## Next safe action

Orchestrator validates report_001.md + verdict_001.json gate; applies counters (valid no-improvement streak 1/3, total_rounds 1) and keeps last_accepted = baseline_adapter.py @`b8ec3458…`. Verifier idle until next dispatch.
