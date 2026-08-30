# Round Status 000 — Phase 0 Baseline Verification (flexattention bi150 epoch2)

## Status: END — verification complete, classification `baseline` delivered

- role: verifier
- phase: 0-initializing → round "000" baseline verification complete (Orchestrator owns the transition)
- measurement_exclusive: honored throughout; no other commands ran on the device during pairs
- interpreter: `/usr/local/bin/python3`
- device: `cuda:0` (Iluvatar BI-V150)
- corex bootstrap: `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` in every shell

## Identity checks (all confirmed live)

| Artifact | Declared | Live | Verdict |
|---|---|---|---|
| base.py sha256 | `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` (2479 B) | same | match |
| auto_bench.py sha256 | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 B) | same | match |
| baseline_adapter.py sha256 | n/a (new Phase-0 candidate) | `b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1` (2090 B) | recorded |
| measurement fingerprint | `6dc07009177b649f7c2cad8f7be5e9aad74235bd9f50abfebc88bdb273e32af4` | recomputed `sha256(base‖NUL‖harness‖NUL‖settings)` = `6dc07009…2af4` | match |
| runtime fingerprint | project.md#runtime-fingerprint | torch 2.7.1, triton 3.1.0 (corex dist), CoreX 4.4.0 nvcc V10.2.89, BI-V150 sm71 16-SM 17179869184 B | match |

Note on first recompute attempt: an initial shell one-liner produced a mismatching hash due to bash quoting mangling the settings string; the corrected invocation reproduced both this project's fingerprint AND the groupedtopk-r2 positive control (`8deb1b01…431`) exactly. No artifact discrepancy exists.

## Completed commands

1. Hash ledger + measurement-fingerprint recompute — PASS (see above).
2. Runtime probe — PASS.
3. Correctness + timing pair 1/3 — PASS accuracy; v0=0.151079 ms, v1=0.151440 ms (log/pair_001_timing.txt).
4. Correctness + timing pair 2/3 — PASS accuracy; v0=0.151107 ms, v1=0.150994 ms (log/pair_002_timing.txt).
5. Correctness + timing pair 3/3 — PASS accuracy; v0=0.151336 ms, v1=0.150791 ms (log/pair_003_timing.txt).
6. Kernel-mode profile attempt — failed as structurally predicted: `KsCompareError: candidate_baseline_adapter: kernel profiling requires a callable ModelNew.run_out`, exit 1 (log/kernel_mode_attempt.txt). Named deviation recorded in report_000.md.
7. Dual-scope forward-mode profile pw=20/pi=100 — exit 0; trace log/flexattention_baseline_forward_100iter.pt.trace.json sha256 `1185fa8de04fb094d7a099f6bd002d843a90d7532a539a7f33501d35d66828a5`.
8. summarize_trace.py per scope — summary_reference_baseline_adapter.json (device 13.56029296875 µs/call, 88 kernels) and summary_candidate_baseline_adapter.json (14.96966796875 µs/call, 98 kernels).

## Raw samples

- reference_raw_samples_ms: [0.151079, 0.151107, 0.151336] → median 0.151107
- candidate_raw_samples_ms: [0.151440, 0.150994, 0.150791] → median 0.150994
- improvement_pct: +0.074778 (identity-level; adapter-of-base)

## Terminal classification

`baseline` — report at rounds/report_000.md; correctness PASS; wall medians v0/v1 = 0.151107/0.150994 ms; BASE launch census = single fused `FlashAttnFwdF16Ixmma<128,128,16,(64,64),Causal2,f16>` kernel, 13.56–14.97 µs/call, host share ≈ 90–91% of wall.

## Next safe action

Orchestrator validates report_000.md gate and applies Phase-0 completion transitions (`last_accepted_*`, phase → ready). Verifier idle until next dispatch.
