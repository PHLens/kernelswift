# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `1`
- last_completed_round: `001`
- accepted_kernel: `null` (canonical pointer not yet advanced; last_accepted_kernel is still `baseline_adapter.py`)
- accepted_report: `null` (last_completed_report is still `rounds/report_000.md`)
- recent_three_round_evidence: `round 001 (kernel-fusion, tl.range dynamic loop), candidate-ready, speedup 7.910x vs baseline`
- open_hypotheses: `Verifier must confirm kernel_count_per_call and device_us_per_call dropped (fusion mechanism); tl.static_range at 19 iters is a proven compile-time blow-up on BI150, avoid in future rounds`
- artifact_read_hashes: see table

## Current Bottleneck

- Verifier/Phase 0 fact: device-bound, device_ratio ~0.61, 132.88 kernels/call,
  dominated by Sinkhorn-loop tiny kernels (reduce_kernel sum 442.5 us/call,
  DivFunctor 254.2 us/call, CUDAFunctorOnSelf_add 161.4 us/call).

## Recent Three-round Evidence

- round 001, candidate-ready, kernel-fusion (single fused Triton kernel, one
  program per (b,s) position, 16-program grid), change_family kernel-fusion.
  tl.static_range(19) compile-time unroll is not feasible on BI150 (compiles >300s
  vs ~instant at 4 iters); fell back to tl.range dynamic loop preserving semantics.

## Open Hypotheses or Checks

- Verifier Level 1: confirm kernel_count_per_call ~1-2 and device_us_per_call
  decreased (fusion mechanism observables).
- Future capability check: whether a partial static_range unroll (e.g. 2-4 iters)
  plus dynamic remainder can reduce loop overhead without compile blow-up.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---:|---:|
| `skills/kernel-opt-loop/prompts/coder.md` | `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196` | 001 |
| `rounds/decision_001.md` | `81a84adb36438b572455d0e28277bd7e9c2f6921a60998fdc7579628f30fa34f` | 001 |
| `baseline_adapter.py` (canonical) | `ceebdc6185de4c980156a7833073678a0964fb7ccb5edf74b42be6156652eaed` | 001 |
| `../base.py` | `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5` | 001 |
| `triton_mhc_head_compute_mix_001.py` (candidate) | `a98b1b12593d858ca29c787afa939a3ae0061df4ec6b51aa9a0fe7fa43c6b473` | 001 |
