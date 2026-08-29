# Round Status 001

- round: `001`
- phase: `verifying` → `verified`
- verification_tier: `authoritative`
- terminal_classification: `no-improvement`
- result: candidate slower than accepted reference (paired −6.4%, ~0.940x), below the +5.0% adoption bar; still a 2.2x-over-epoch-1 correctness-PASS Triton deliverable

## Measurement Summary (authoritative, Verifier)

- warmup 50 / repeat 100 / seed 42 / interleaved ordered pairs
- reference_median_ms: `0.250796` (5 raw samples: 0.225535, 0.250796, 0.250807, 0.272492, 0.244433)
- candidate_median_ms: `0.266835` (5 raw samples: 0.231918, 0.266835, 0.267612, 0.282353, 0.258608)
- improvement_pct: `-6.395387` (negative; all five pairs negative-sign)
- correctness: `PASS accuracy` 6/6 invocations (5 timing pairs + profile run)

## Profiler Census (dual-scope forward-mode, 100 iterations)

- baseline (base.py): 1 `topsLaunchKernel`/call @ 13.31 us/call
- candidate (e2_001): 1 `topsModuleLaunchKernel`/call @ 13.35 us/call
- device_time_available: `false` (GCU launch-only trace)
- trace: `log/report_001_forward.pt.trace.json` @`7ec0189d0c98f61395e9949d9a039d340c9bbecec4dd0a9406a4dd1c4312a10f`

## Key Hashes (re-verified live)

- candidate: `6a62042904bd774006154ba75d8bbcc8212449438d2cd8b4aaa02a5415eed0e9`
- decision_001: `8a2bb5a7a6bcd2ccb8ecb704c30c5edbb540fb5c52fc4cae34f2afeef57c5d86`
- sketch_001: `aad322a8b806d9f97bc9c5056c8ae1ea62c5bd8ecc8bb502fb6fc72399a61247`
- base.py: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0`
- harness: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`

## Verdict Owning

Verdict, team-state, and canonical pointer updates are owned by the Orchestrator. Verifier only wrote `report_001.md` and `round_status_001.md`.

## Stop Recommendation

- `continue` (no-improvement #1/3; round budget 1/20; Triton deliverable + canonical physics numbers banked)
