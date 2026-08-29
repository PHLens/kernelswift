# Round Status 000 — Phase 0 Baseline Verification (mm_encoder_attention s60 epoch2)

## Status: END — verification complete, classification `baseline` delivered

- role: verifier
- phase: 0-initializing → round "000" baseline verification complete (Orchestrator owns the transition)
- measurement_exclusive: honored throughout; no other commands ran on the device during pairs
- interpreter: `/usr/bin/python3`
- device: `gcu` (Enflame GCU)

## Identity checks (all confirmed live, 2026-08-28)

| Artifact | Declared | Live | Verdict |
|---|---|---|---|
| base.py sha256 | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (2284 B) | same, re-verified after all runs | match |
| auto_bench.py sha256 | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 B) | same, re-verified after all runs | match |
| baseline_adapter.py sha256 | n/a (new Phase-0 candidate) | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` (2331 B) | recorded |
| measurement fingerprint | `c335b39cbf2eaa15e1a358be90d0aab85d0fd7e8ffd4b7b4e825df0901ad61f9` | recomputed `sha256(base‖NUL‖harness‖NUL‖settings)` = `c335b39c…1ad61f9` | match |
| runtime fingerprint | project.md#runtime-fingerprint | triton 3.6.0, triton_gcu 3.6.0+1.0.20260722, torch 2.10.0+cpu, torch_gcu 2.10.0+3.8.0.2, GCU major=3 minor=0, 2 SM, 43878764544 B, gcu_available=True | match |

## Completed commands

1. Hash ledger + measurement-fingerprint recompute — PASS.
2. Runtime probe — PASS.
3. Correctness + timing pair 1/3 — PASS accuracy; v0=0.227986 ms, v1=0.230700 ms.
4. Correctness + timing pair 2/3 — PASS accuracy; v0=0.230975 ms, v1=0.200134 ms (pair-2 candidate host transient on byte-identical code; documented in report_000.md).
5. Correctness + timing pair 3/3 — PASS accuracy; v0=0.230378 ms, v1=0.229836 ms.
6. Dual-scope forward-mode profile pw=20/pi=100 — harness exit 0; trace log/report_000_forward.pt.trace.json sha256 `f7a6a51075246b13bddd33ce6058efb88c705aa2a2083d4cd9acbc31e23cfc49`.
7. summarize_trace.py — device_time unavailable (GCU launch-only trace); runtime_launch_count_per_call=2.0, topsLaunchKernel 21.99 µs/call.
8. aten cpu-op census: 28.00 aten ops/call (8 transpose + 8 as_strided + 4 view + 3 empty + SDPA chain + empty_like + empty_strided + reshape); 2 topsLaunchKernel/call.
9. Post-measurement hash re-verification — all frozen artifacts unchanged.
10. rounds/report_000.md written; state/verifier_context.md updated.

## Raw samples

- reference_raw_samples_ms: [0.227986, 0.230975, 0.230378] → median 0.230378
- candidate_raw_samples_ms: [0.230700, 0.200134, 0.229836] → median 0.229836
- identity-level speedup ≈ 1.00x (adapter-of-base)

## Terminal classification

`baseline` — report at rounds/report_000.md; correctness PASS in all runs; wall medians v0/v1 ≈ 0.230/0.230 ms; BASE SDPA = vendor `_scaled_dot_product_flash_attention` with 2 `topsLaunchKernel` launches/call; device-duration unavailable on GCU (launch-only trace); measurement fingerprint `c335b39c…1ad61f9` confirmed live.

## Next safe action

Orchestrator validates report_000.md gate and applies Phase-0 completion transitions (`last_accepted_*`, phase → ready). Verifier idle until next dispatch.
