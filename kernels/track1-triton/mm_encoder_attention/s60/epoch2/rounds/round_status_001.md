# Round Status 001 — Candidate Verification (mm_encoder_attention s60 epoch2)

## Status: END — verification complete, classification `no-improvement` delivered

- role: verifier
- phase: verifying (Round 001) → complete (Orchestrator owns transitions)
- measurement_exclusive: honored throughout; no other commands ran on the device during pairs
- candidate: `triton_mm_encoder_attention_e2_001.py` @`f2f8b9b6c6f6a16cfbf162cf3f9b115461fc7a5716601eb8e3723961a8536ead` (unchanged before AND after all runs; zero repairs consumed)
- interpreter: `/usr/bin/python3`; device: `gcu` (Enflame S60); CoreX bootstrap in every shell

## Identity checks (all confirmed live)

| Artifact | Declared | Live | Verdict |
|---|---|---|---|
| candidate sha256 | `f2f8b9b6…6ead` | same, re-verified post-run | match |
| decision_001.md | `4ae2b613…8771` | same | match |
| sketch_001.json | `ef71920a…8f70` | same | match |
| baseline_adapter.py | `1127e8d9…7c8e` | same | match |
| base.py | `86ac5703…6ed2` | same, re-verified | match |
| auto_bench.py | `71fb3ad0…fe29` | same, re-verified | match |
| profile_snapshot/triton_gcu.yaml | `8dfabd0a…2b70` | same | match |
| measurement fingerprint | `c335b39c…ad61f9` | project.md canonical = match | match |
| trace report_001_forward.pt.trace.json | `597ddb35…5562` | computed live | match |

## Completed commands

1. Hash ledger (9 artifacts) + measurement-fingerprint reference — PASS.
2. Authoritative timing pair 1/3 — PASS accuracy; v0=0.276584 ms, v1=0.303552 ms (0.911x).
3. Authoritative timing pair 2/3 — PASS accuracy; v0=0.276234 ms, v1=0.305640 ms (0.904x).
4. Authoritative timing pair 3/3 — PASS accuracy; v0=0.276579 ms, v1=0.306290 ms (0.903x).
5. Dual-scope forward-mode profile (pw=20/pi=100) — runtime_launch_count_per_call = 1.0 per scope (base `topsLaunchKernel` @10.16us; candidate `topsModuleLaunchKernel` @10.14us); device_time_available = false; aten census 28 → 2/call (trace log/report_001_forward.pt.trace.json sha256 `597ddb35…5562`).
6. Post-measurement hash re-verification — all frozen artifacts unchanged.
7. rounds/report_001.md written (with vNext Fact Pack); rounds/verdict_001.json already present (Orchestrator product, unchanged).

## Raw samples

- reference_raw_samples_ms: [0.276584, 0.276234, 0.276579] → median 0.276584
- candidate_raw_samples_ms: [0.303552, 0.305640, 0.306290] → median 0.305640
- improvement_pct: −10.504681 (decisively below the +5.0% bar; candidate ~0.906x)
- session drift: paired same-session basis absorbs; delta ~2× the bar in the negative direction — no plausible drift affects classification

## Terminal classification

`no-improvement` (streak 1/3; canonical unchanged) — report at rounds/report_001.md; verdict at rounds/verdict_001.json (Orchestrator product, terminal_result no-improvement, consistent with this report).

Dual-gate highlights: S60 launcher tax ~17.4us/call (5x smaller than BI150's 84.77us — graph-replay has NO material prize here); device-bound diagnosis confirmed (hand-written tl.dot ~166us vs CNNL SDPA ~158us floor; TP=128 padding forces 58% FLOP waste); host chain ~11us + launcher ~17us = ~28us compressible total < device deficit; num_warps sweep nw=2 optimal (166.2us). Power-of-2 capability constraint (not mult-of-16) must propagate back to the triton_gcu profile.

## Next safe action

Orchestrator validates report_001.md + verdict_001.json gates, applies no-improvement transitions (performance_miss_streak 1/3, canonical unchanged), and dispatches the next round. Verifier idle until next dispatch.
