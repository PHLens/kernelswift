# Round 001 Status

Phase: `verifying` (authoritative timing)
Result: `accepted`

## Progress

| Step | Status | Return code | Evidence |
|---|---|---|---|
| frozen-file SHA256 verification | done | `0` | candidate `f3fd85a2...`, adapter `359f4c80...`, decision `0fbbdb69...` all match |
| correctness (v0=base, v1=candidate) 50/100 | done | `0` | `PASS accuracy; v0=1.057676 ms, v1=0.881433 ms, speedup=1.200x` |
| independent numerical probe | done | `0` | 4× allclose True, max_abs ~1.19e-07, structure/dtype matched |
| baseline wrapper creation | done | `0` | `/tmp/sp_baseline_model_001.py` SHA `1edaf2ad...` |
| wall sample 1, 50/100 | done | `0` | ref=1.055067, cand=0.880377 |
| wall sample 2, 50/100 | done | `0` | ref=1.060573, cand=0.879838 |
| wall sample 3, 50/100 | done | `0` | ref=1.060911, cand=0.885816 |
| targeted profiler 20/50 | done | `0` | `log/round_001_forward_50iter.pt.trace.json` |
| summarize reference scope | done | `0` | 743.80 us/call, 11.92 kernels/call |
| summarize candidate scope | done | `0` | 609.40 us/call, 6.88 kernels/call (manual outer-interval, see note) |

## Raw Samples

- reference_raw_samples_ms: `[1.055067, 1.060573, 1.060911]`
- candidate_raw_samples_ms: `[0.880377, 0.879838, 0.885816]`
- reference_median_ms: `1.060573`
- candidate_median_ms: `0.880377`
- improvement_pct: `16.990`

## Artifact Hashes

- candidate_sha256: `f3fd85a2c913d477e2cac7f65ed1f79dd5e1b9a3a60481782dbb4acaa43d2d98`
- decision_sha256: `0fbbdb6929e1b75f939fc2d513c28878b7a53587f33e8fcaf66401f1269256f1`
- baseline wrapper sha256: `1edaf2addf3d6b84a648ccefb46600281226a4270f624f598333c9c870bad054`
- trace SHA256: `fda3dc194770f2439988967bc58edcea9b9bb8eaa235e6d14e07e76933f99754`

## Profiler Scope Overlap Note

The candidate scope in the raw trace contains two overlapping `X` events with the
name `candidate_triton_sparse_pooler_001` (a nested `record_function` emitted by
Triton's `cuLaunchKernel` instrumentation reusing the same external id). The
stock `summarize_trace.py` rejects overlapping scope events. The candidate scope
was summarized against the outer enclosing interval (the event without the
`finished: True` marker, dur=31194.912 us), which correctly contains all 50
forward calls' kernel events. Values recorded here are that outer-interval
summary; they are semantically correct.

## Next Safe Action

Report complete; awaiting Orchestrator to record terminal `accepted` transition
and advance canonical pointers.
