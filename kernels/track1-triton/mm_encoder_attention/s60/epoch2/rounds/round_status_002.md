# Round Status 002 — Candidate Verification (mm_encoder_attention s60 epoch2)

## Status: END — verification complete, classification `no-improvement` delivered

- role: verifier
- phase: verifying (Round 002) → complete (Orchestrator owns transitions)
- measurement_exclusive: honored throughout; no other commands ran on the device during pairs
- candidate: `triton_mm_encoder_attention_e2_002.py` @`7b411daf3903c88ebcaa9426a628f6fe76638fd7be635c0563ee4f63fc1be818` (unchanged before AND after all runs; zero repairs consumed)
- interpreter: `/usr/bin/python3`; device: `gcu` (Enflame S60); CoreX bootstrap in every shell

## Identity checks (all confirmed live)

| Artifact | Declared | Live | Verdict |
|---|---|---|---|
| candidate sha256 | `7b411daf…be818` | same, re-verified post-run | match |
| decision_002.md | `04f6dc0b…031e2f` | same | match |
| sketch_002.json | `c3c585d1…97695dd` | same | match |
| baseline_adapter.py | `1127e8d9…7c8e` | same | match |
| base.py | `86ac5703…6ed2` | same, re-verified | match |
| auto_bench.py | `71fb3ad0…fe29` | same, re-verified | match |
| profile_snapshot/triton_gcu.yaml | `8dfabd0a…2b70` | same | match |
| measurement fingerprint | `c335b39c…ad61f9` | project.md canonical = match | match |
| trace report_002_forward.pt.trace.json | `90266df5…f739` | computed live | match |

## Completed commands

1. Hash ledger (9 artifacts) + measurement-fingerprint reference — PASS.
2. Authoritative timing pair 1/3 — PASS accuracy; v0=0.255767 ms, v1=0.275038 ms (0.930x).
3. Authoritative timing pair 2/3 — PASS accuracy; v0=0.230800 ms, v1=0.242818 ms (0.951x).
4. Authoritative timing pair 3/3 — PASS accuracy; v0=0.251601 ms, v1=0.277810 ms (0.906x).
5. Dual-scope forward-mode profile (pw=20/pi=100) — runtime_launch_count_per_call = 2.0 total (per scope 1.0: base `topsLaunchKernel` @11.41us; candidate `topsModuleLaunchKernel` @11.38us); device_time_available = false; trace log/report_002_forward.pt.trace.json sha256 `90266df5…f739`.
6. Post-measurement hash re-verification — all frozen artifacts unchanged.
7. rounds/report_002.md written (with vNext Fact Pack).

## Raw samples

- reference_raw_samples_ms: [0.255767, 0.230800, 0.251601] → median 0.251601
- candidate_raw_samples_ms: [0.275038, 0.242818, 0.277810] → median 0.275038
- improvement_pct: −9.315146 (decisively below the +5.0% bar; candidate ~0.915x)
- S60 wall noise confirmed: base v0 fluctuated 0.230800–0.276798 ms across the four harness invocations — authoritative conclusion rests on 3-pair median + paired improvement, not single shots

## Terminal classification

`no-improvement` (streak 2/3; canonical unchanged) — report at rounds/report_002.md.

Highlights: fp16 QK^T dot is a REAL device direction — paired wall moved r001 −10.5% → r002 −9.3% (right direction, ~30.6us cross-session wall cut), host path fully invariant (launch census structurally identical to r001). But S60 remains DEVICE-BOUND: TP=128 power-of-2 padding forces 58% FLOP waste, so D_cand still sits above the CNNL SDPA ~158us floor; the +5% wall bar is not cleared.

## Next safe action

Orchestrator validates report_002.md, applies no-improvement transitions (performance_miss_streak 2/3, canonical unchanged), and dispatches the next round. Verifier idle until next dispatch.
