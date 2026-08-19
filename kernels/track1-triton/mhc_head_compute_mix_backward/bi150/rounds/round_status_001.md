# Round Status 001

- phase: `verifying` (Round 001 authoritative verification)
- round: `001`
- result: `accepted`
- started_at: `2026-08-19T16:20:00Z`

## Frozen Artifact Hashes (verified before measurement)

| File | SHA256 | Match |
|---|---|---|
| `base.py` | `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc` | pass |
| `baseline_adapter.py` | `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d` | pass |
| `triton_mhc_head_compute_mix_backward_001.py` | `5d419f5d2e920abf3cf583a22f155e76047f9e5bc3a5cc36baca5477fae94349` | pass |
| `decision_001.md` | `dc0a4837cc8a5aeb867e9d71f8c1e4bc1930ee57d431a279f761329271e5371a` | pass |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | pass |

## Command Status

| Step | Command | Return code | Outcome |
|---|---|---|---|
| frozen SHA256 verify | `sha256sum ...` | `0` | all match |
| correctness 50/100 (base vs candidate) | `--full-traceback` | `0` | `PASS accuracy; v0=0.344201 ms, v1=0.198203 ms, speedup=1.737x` |
| independent numerical probe | `/tmp/probe_mhcbwd_verify.py` | `0` | all 3 outputs allclose; reduction semantics confirmed |
| authoritative pair 1 | 50/100 | `0` | ref=0.350071, cand=0.199972 |
| authoritative pair 2 | 50/100 | `0` | ref=0.345161, cand=0.198597 |
| authoritative pair 3 | 50/100 | `0` | ref=0.349112, cand=0.196444 |
| targeted profiler 20/50 | `--profile-reference-file baseline_adapter.py` | `0` | trace written |
| summarize `reference_baseline_adapter` | `summarize_trace.py` | `0` | 186.057 us/call, 9.74 kernels/call |
| summarize candidate (manual, nested-scope) | python inline | `0` | 14.692 us/call, ~2.96 kernels/call |

## Raw Samples

- reference_raw_samples_ms: `[0.350071, 0.345161, 0.349112]`
- reference_median_ms: `0.349112`
- candidate_raw_samples_ms: `[0.199972, 0.198597, 0.196444]`
- candidate_median_ms: `0.198597`
- improvement_pct: `43.11367125736153`

## Trace

- path: `log/round_001_forward_50iter.pt.trace.json`
- SHA256: `c13dbb5389a99f17cbdefb45955a21769c92da5220dff44942638e2e87e5d976`

## Note on candidate profiler scope

The candidate scope `candidate_triton_mhc_head_compute_mix_backward_001` produced two
nested `record_function` X-events (a PyTorch profiler artifact from the fused
Triton launch), which makes `summarize_trace.py --scope` reject it as
"overlapping scope events". The candidate device time was therefore summarized by
manually filtering `cat=kernel` events inside the inner (clean) candidate scope,
yielding `_mhc_head_compute_mix_backward_kernel` ×50 (one per forward) plus
`FillFunctor` (the `torch.zeros` accumulator init) ×~98.

## Next Safe Action

Verification complete. Await Orchestrator transition.
