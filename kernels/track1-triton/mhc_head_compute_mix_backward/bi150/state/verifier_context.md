# Verifier Context

Task 10 `mhc_head_compute_mix_backward` (BI150 backend).

## Round 000 (Phase 0) — baseline

- Result: `baseline`
- Wall median (reference/base): `0.351449 ms`
- Device: `185.599 us/call`, `9.74 kernels/call`, device_ratio `0.528`

## Round 001 — accepted (kernel-fusion, H-001 confirmed)

- Candidate: `triton_mhc_head_compute_mix_backward_001.py`
  - SHA256 `5d419f5d2e920abf3cf583a22f155e76047f9e5bc3a5cc36baca5477fae94349`
- Result: `accepted`
- Correctness: PASS (speedup 1.737x); independent probe allclose (max abs diff `1.14e-5`)
- Authoritative timing:
  - reference_raw `[0.350071, 0.345161, 0.349112]`, median `0.349112 ms`
  - candidate_raw `[0.199972, 0.198597, 0.196444]`, median `0.198597 ms`
  - improvement_pct `43.11%`
- Profiler:
  - reference: `186.057 us/call`, `9.74 kernels/call`
  - candidate: `14.692 us/call`, `~2.96 kernels/call`
  - the two standalone `sum` reductions eliminated; fused into `_mhc_head_compute_mix_backward_kernel` (7.416 us/call)
- Hypothesis verdict: `confirmed`

## Key Notes

- Profiler nested-scope artifact: the fused Triton launch makes the candidate
  scope emit two overlapping `record_function` events; `summarize_trace.py
  --scope` rejects it. Candidate device totals recovered by manual `cat=kernel`
  filtering inside the inner scope.
- Next-round bottleneck: candidate `device_ratio ≈ 0.074` → strongly host-bound.
  Remaining device time is dominated by `torch.zeros` accumulator init
  (`FillFunctor`, ~7.276 us/call ≈ half of device time) plus host/launch overhead.

## Frozen Artifact Hashes

| File | SHA256 |
|---|---|
| base.py | `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc` |
| baseline_adapter.py | `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d` |
| triton_..._001.py | `5d419f5d2e920abf3cf583a22f155e76047f9e5bc3a5cc36baca5477fae94349` |
| decision_001.md | `dc0a4837cc8a5aeb867e9d71f8c1e4bc1930ee57d431a279f761329271e5371a` |
| auto_bench.py | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` |

## Deliverables

- `rounds/report_001.md`
- `rounds/round_status_001.md`
- `state/verifier_context.md` (this file)
- trace `log/round_001_forward_50iter.pt.trace.json`
  (SHA256 `c13dbb5389a99f17cbdefb45955a21769c92da5220dff44942638e2e87e5d976`)
