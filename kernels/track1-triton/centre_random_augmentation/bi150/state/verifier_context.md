# Verifier Context State

- role_contract_sha256: `<verifier.md — not yet hashed at write time>`
- context_epoch: `3`
- last_completed_round: `002`
- accepted_kernel: `triton_centre_random_augmentation_002.py` (pending Orchestrator canonical update)
- accepted_report: `rounds/report_002.md`
- recent_three_round_evidence: `round 002 = accepted; wall 0.711623→0.239284 ms (+66.37%); device 238.19→29.24 us/call; kernels 54.92→5.52; full deterministic transform fused into single _centre_aug_kernel (6.81 us/call)`
- open_hypotheses: `remaining cost is irreducible host-side RNG draws (3x torch.rand + 1x torch.randn) + s_trans no-op mul; device_ratio≈0.12 (host-bound floor); diminishing returns`
- artifact_read_hashes: `candidate002 efac6ee7..., decision002 2290e37b..., canonical001 4e33276e..., base 02e7020f..., harness 3d4fa4ee...`

## Current Bottleneck

- Host-bound floor: after full fusion (Round 002), `kernel_count_per_call ≈ 5.52`, `device_us_per_call ≈ 29.24`, `device_ratio ≈ 0.122`. The remaining wall time is dominated by the irreducible host-side RNG draws (3×`torch.rand` for u1/u2/u3 + 1×`torch.randn` for T, which the decision mandates stay host-side in exact order) plus the single fused-kernel launch and a numerical no-op `s_trans * randn` multiply.

## Recent Three-round Evidence

- round 000, `baseline`, report `rounds/report_000.md`, change family `not-applicable: Phase 0` (top-level class rename).
- round 001, `accepted`, report `rounds/report_001.md`, change family `kernel-fusion` (fused centering+rot_vec_mul+translation+mask; +30.35%).
- round 002, `accepted`, report `rounds/report_002.md`, change family `kernel-fusion` (fused quaternion→matrix transcendental chain; +66.37%, single-digit kernels).

## Open Hypotheses or Checks

- Further wall-time gains are unlikely: remaining kernels are the mandatory host-side RNG draws and a no-op `s_trans` multiply; `device_ratio` is ~0.12.
- `tl.sqrt`/`tl.sin`/`tl.cos` are now locally proven to lower on the CoreX Triton 3.1.0 BI150 backend (bit-compatible with torch, max abs diff 0.0 per Coder probe; 4.77e-07 output-level per Verifier probe) — candidate for a profile-update note, not an optimization target.
- Profiler note: both scopes now contain Triton kernels, so `summarize_trace.py` reports "overlapping scope events" on BOTH scopes (device-side `record_function` projection, pid=0/tid=1). Use the CPU-side scope interval (pid==tid, non-zero pid) filter to compute kernel totals.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `kernels/track1-triton/centre_random_augmentation/base.py` | `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553` | `002` |
| `kernels/track1-triton/centre_random_augmentation/bi150/baseline_adapter.py` | `012754740961f6ec10d515563e51cd07eeaf35caefe33731d5c1e9a88387fe9b` | `001` |
| `kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py` | `4e33276ec28f3695aa08462aa6cb796a160aca47dad889168a7cdd8aa8e16036` | `002` |
| `kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py` | `efac6ee782e859701bb14aca04b7f56516a575a5f74507958e1930a95005a530` | `002` |
| `kernels/track1-triton/centre_random_augmentation/bi150/rounds/decision_002.md` | `2290e37b81072b794ca5735dddba52ed19805c943a8e7109b598e5fd1f65af8e` | `002` |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | `002` |
