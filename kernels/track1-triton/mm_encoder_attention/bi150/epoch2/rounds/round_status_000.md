# Round Status 000 — Phase 0 Baseline Verification (mm_encoder_attention bi150 epoch2)

## Status: END — verification complete, classification `baseline` delivered

- role: verifier
- phase: 0-initializing → round "000" baseline verification complete (Orchestrator owns the transition)
- measurement_exclusive: honored throughout; no other commands ran on the device during pairs
- interpreter: `/usr/local/bin/python3`
- device: `cuda:0` (Iluvatar BI-V150)
- corex bootstrap: `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` in every shell

## Identity checks (all confirmed live, 2026-08-28)

| Artifact | Declared | Live | Verdict |
|---|---|---|---|
| base.py sha256 | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (2284 B) | same, re-verified after all runs | match |
| auto_bench.py sha256 | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 B) | same, re-verified after all runs | match |
| baseline_adapter.py sha256 | n/a (new Phase-0 candidate) | `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f` (1832 B) | recorded |
| measurement fingerprint | `0c4c7d664c85e65d0580091ca5e3a77ff769a0d28f7e679f5bdf78fe5d0d966e` | recomputed `sha256(base‖NUL‖harness‖NUL‖settings)` = `0c4c7d66…966e`; positive control reproduced sibling flexattention fingerprint `6dc07009…2af4` exactly | match |
| runtime fingerprint | project.md#runtime-fingerprint | python 3.10.18, torch 2.7.1, triton 3.1.0 (`/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton`), CoreX 4.4.0 nvcc V10.2.89, BI-V150 capability 7/1, 16 SM, 17179869184 B, cuda_available=True | match |

## Completed commands

1. Hash ledger + measurement-fingerprint recompute (with sibling positive control) — PASS.
2. Runtime probe — PASS.
3. Correctness + authoritative timing pair 1/3 — PASS accuracy; v0=0.150149 ms, v1=0.150147 ms (log/pair_001_timing.txt).
4. Correctness + authoritative timing pair 2/3 — PASS accuracy; v0=0.147639 ms, v1=0.148039 ms (log/pair_002_timing.txt).
5. Correctness + authoritative timing pair 3/3 — PASS accuracy; v0=0.153581 ms, v1=0.204876 ms (log/pair_003_timing.txt). Pair-3 candidate window carried a sustained host-side transient on byte-identical code (same pair's v0 normal, pairs 1–2 within 0.3%); median-of-pairs unaffected — documented in report_000.md.
6. Kernel-mode profile attempt — failed as structurally predicted: `KsCompareError: candidate_baseline_adapter: kernel profiling requires a callable ModelNew.run_out`, harness exit 1 (log/kernel_mode_attempt.txt). Named deviation recorded in report_000.md.
7. Dual-scope forward-mode profile pw=20/pi=100 — harness exit 0; extra correctness PASS (v0=0.149579, v1=0.148119); trace log/mmenc_baseline_forward_100iter.pt.trace.json sha256 `661b8b78d037e7c1285db419b37949b16dbea78cc19dd482f0eb8aeecdbeabdb` (log/profile_forward_run.txt).
8. summarize_trace.py per scope — log/summary_reference_baseline_adapter.json (device 16.537197265625 µs/call, 95 kernels, device_ratio 0.11014) and log/summary_candidate_baseline_adapter.json (17.559228515625 µs/call, 101 kernels, device_ratio 0.11695).
9. aten cpu-op census + launch census (log/aten_census.txt): 33.00 aten ops/call (10 distinct), 1.00 cudaLaunchKernel/call, SDPA→flash dispatch chain confirmed.
10. Post-measurement hash re-verification — all frozen artifacts unchanged.
11. rounds/report_000.md written; state/verifier_context.md updated.

## Raw samples

- reference_raw_samples_ms: [0.150149, 0.147639, 0.153581] → median 0.150149
- candidate_raw_samples_ms: [0.150147, 0.148039, 0.204876] → median 0.150147
- improvement_pct: +0.001332 (identity-level; adapter-of-base)

## Terminal classification

`baseline` — report at rounds/report_000.md; correctness PASS in all four runs; wall medians v0/v1 = 0.150149/0.150147 ms; BASE launch census = single fused `FlashAttnFwdF16Ixmma<128,128,16,(64,64),Causal=0,Alibi=0,f16>` kernel per call (one cudaLaunchKernel/call), 16.54 (reference scope) / 17.56 (candidate scope) µs/call, host share ≈ 89.0%/88.3% of wall; 33 aten ops/call; measurement fingerprint `0c4c7d66…966e` confirmed live.

Designer transfer-model check (dispatched validation targets): single-kernel structure CONFIRMED; host-dominance CONFIRMED (~89% vs predicted ~91%); device µs measured 16.54/17.56 vs predicted 14.9 (+11–18%, session-band 13.6–17.6 across campaigns; 20 µs re-model trigger NOT hit); aten count 33 vs ~38 predicted; bidirectional (Causal=0) visible in template args CONFIRMED, bsz not a template arg (runtime FlashAttnFwdParams, single launch covers bsz=2). Neither re-model trigger (wall > 0.16 ms, device > 20 µs) fired.

## Next safe action

Orchestrator validates report_000.md gate and applies Phase-0 completion transitions (`last_accepted_*`, phase → ready). Verifier idle until next dispatch.
