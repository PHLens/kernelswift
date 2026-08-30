# Round Status 001 — Candidate Verification (centre_random_augmentation s60 epoch2)

## Status: END — verification complete, classification `accepted` delivered

- role: verifier
- phase: verifying (Round 001) → complete (Orchestrator owns transitions)
- measurement_exclusive: honored throughout; no other commands ran on the device during pairs
- candidate: `triton_centre_random_augmentation_e2_001.py` @`542293c0ed3488b4f30c6c3758780115325593a592b11bc656cfa605f9d79522` (unchanged before AND after all runs; zero repairs consumed)
- interpreter: `/usr/bin/python3`; device: `gcu` (Enflame S60); CoreX bootstrap in every shell

## Identity checks (all confirmed live)

| Artifact | Declared | Live | Verdict |
|---|---|---|---|
| candidate sha256 | `542293c0…9522` | same, re-verified post-run | match |
| decision_001.md | `459a1f9b…8c40` | same | match |
| sketch_001.json | `017b423b…429f` | same | match |
| baseline_adapter.py | `7d4a79ae…061b` | same | match |
| base.py | `02e7020f…8553` | same, re-verified | match |
| auto_bench.py | `71fb3ad0…fe29` | same, re-verified | match |
| profile_snapshot/triton_gcu.yaml | `8dfabd0a…2b70` | same | match |
| measurement fingerprint | `cra-s60-e2` | project.md canonical = match | match |
| trace report_001_forward.pt.trace.json | `63c94e1a…c622` | computed live | match |

## Completed commands

1. Hash ledger (7 artifacts) + measurement-fingerprint reference — PASS.
2. Authoritative timing pair 1/3 — PASS accuracy; v0=3.025109 ms, v1=1.585115 ms (1.908x).
3. Authoritative timing pair 2/3 — PASS accuracy; v0=3.192140 ms, v1=1.679165 ms (1.901x).
4. Authoritative timing pair 3/3 — PASS accuracy; v0=2.316304 ms, v1=1.287518 ms (1.799x).
5. Dual-scope forward-mode profile (pw=50/pi=100) — per-scope census: base `topsLaunchKernel` 96/call @921.87us + cooperative 1/call; candidate `topsModuleLaunchKernel` 1/call @11.06us + `topsLaunchKernel` 10/call @96.43us + cooperative 1/call; aten+GCU cpu_ops 534 → 62/call; device_time_available = false (trace log/report_001_forward.pt.trace.json sha256 `63c94e1a…c622`).
6. Post-measurement hash re-verification — all frozen artifacts unchanged.
7. rounds/report_001.md written (with vNext Fact Pack).

## Raw samples

- reference_raw_samples_ms: [3.025109, 3.192140, 2.316304] → median 3.025109
- candidate_raw_samples_ms: [1.585115, 1.679165, 1.287518] → median 1.585115
- improvement_pct: +47.601761 (decisively above the +5.0% bar; candidate ~1.91x)
- session drift: paired same-session basis absorbs; reference-side variance 2.316–3.192 ms (GCU launch-tax tail) does not affect classification — worst-case pairing still +27.5%

## Terminal classification

`accepted` — report at rounds/report_001.md; canonical pointer should move to `triton_centre_random_augmentation_e2_001.py` on adoption (Orchestrator owns the verdict/transition).

Dual-gate highlights: launch collapse DECISIVE — `topsLaunchKernel` 96 → 10/call, candidate's own Triton kernel = 1 `topsModuleLaunchKernel` @11.06us; launch-API time ~922us → ~118us/call (net ~814us saved ≈ wall delta, confirming launch-bound base — opposite of the mm_encoder_attention device-bound sibling). aten+GCU cpu_ops 534 → 62/call. First S60 fusion-class operator to beat base.

## Next safe action

Orchestrator validates report_001.md, applies accepted transitions (canonical pointer → candidate, performance_miss_streak reset), and dispatches the next round. Verifier idle until next dispatch.
