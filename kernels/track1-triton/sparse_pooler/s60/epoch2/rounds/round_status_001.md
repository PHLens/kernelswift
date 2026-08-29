# Round Status 001 — Candidate Verification (sparse_pooler s60 epoch2)

## Status: END — verification complete, classification `no-improvement` delivered

- role: verifier
- phase: verifying (Round 001) → complete (Orchestrator owns transitions)
- measurement_exclusive: honored throughout; no other commands ran on the device during pairs
- candidate: `triton_sparse_pooler_e2_001.py` @`f99538b13f7768297d7aa95a25e4c33231eb12321575bdb80ede401b226d81fa` (unchanged before AND after all runs; zero repairs consumed)
- interpreter: `/usr/bin/python3`; device: `gcu` (Enflame S60); CoreX bootstrap in every shell

## Identity checks (all confirmed live)

| Artifact | Declared | Live | Verdict |
|---|---|---|---|
| candidate sha256 | `f99538b1…d81fa` | same, re-verified post-run | match |
| decision_001.md | `264c7be4…2a4ab` | same | match |
| sketch_001.json | `a92ec784…a295` | same | match |
| baseline_adapter.py | `359f4c80…bde8` | same | match |
| base.py | `46106baa…27d58` | same, re-verified | match |
| auto_bench.py | `71fb3ad0…fe29` | same, re-verified | match |
| profile_snapshot/triton_gcu.yaml | `7cd0cdf4…aa6a0` | same | match |
| measurement fingerprint | `sp-s60-e2` | team-state.md canonical = match | match |
| trace report_001_forward.pt.trace.json | `2b7db3cf…88b7b` | computed live | match |

## Completed commands

1. Hash ledger (9 artifacts) + measurement-fingerprint reference — PASS.
2. Authoritative timing pair 1/3 — PASS accuracy; v0=0.922848 ms, v1=3.496313 ms (0.264x).
3. Authoritative timing pair 2/3 — PASS accuracy; v0=0.844259 ms, v1=3.447952 ms (0.245x).
4. Authoritative timing pair 3/3 — PASS accuracy; v0=0.857295 ms, v1=3.430022 ms (0.250x).
5. Dual-scope forward-mode profile (pw=20/pi=100) — runtime_launch_count_per_call: base 11.0 (topsLaunchKernel @132.42us) vs candidate 8.0 (7 topsLaunchKernel @65.08us + 1 topsModuleLaunchKernel @9.20us); device_time_available = false; aten census 83 → 59/call (trace log/report_001_forward.pt.trace.json sha256 `2b7db3cf…88b7b`).
6. Post-measurement hash re-verification — all frozen artifacts unchanged.
7. rounds/report_001.md written (with vNext Fact Pack).

## Raw samples

- reference_raw_samples_ms: [0.922848, 0.844259, 0.857295] → median 0.857295
- candidate_raw_samples_ms: [3.496313, 3.447952, 3.430022] → median 3.447952
- improvement_pct: −302.189678 (decisively below the +5.0% bar; candidate ~0.249x)
- session drift: paired same-session basis absorbs; delta ~60× the bar in the negative direction — no plausible drift affects classification

## Terminal classification

`no-improvement` (streak 1/3; canonical unchanged) — report at rounds/report_001.md.

Dual-gate highlights: dispatch collapse ENGAGED (launch count 11 → 8/call; aten tail `max`/`log1p`/`relu` collapsed out); D2H sync ELIMINATED device-side (`.tolist()` count 0, `cumsum`/`sub` offsets — beyond decision's conservative "tolist retained" note); but the fused-tail Triton kernel is ~4x slower on device than base's PyTorch tail (~2.59ms inferred device deficit ≈ 3× entire reference wall), so wall regressed −302.2%. GEMMs (481us / 61%) remain vendor-bound and structurally untouchable (768/30522 not powers of two). First Triton deliverable banked for sparse_pooler per DELIVERABLE RULE.

## Next safe action

Orchestrator validates report_001.md gates, applies no-improvement transitions (performance_miss_streak 1/3, canonical unchanged), and dispatches the next round. Verifier idle until next dispatch.
