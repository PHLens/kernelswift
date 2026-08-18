# Coder Result 008

Result: candidate-ready

## Identity

- Round: `008`
- Decision: `rounds/decision_008.md`
- Decision SHA-256: `bec59b81693001fd27302a610ab48123e38a4a81c44b65cedfff9530b059e5d1`
- Canonical source: `triton_grouped_topk_004.py`
- Canonical source SHA-256: `881a549cf95746dda93ee4c898e7ab0e67e3133a526088553091f8b8d7431d83`
- Candidate: `triton_grouped_topk_008.py`
- Candidate SHA-256: `d1fb6b03d3be92cdd6423f1f44f33ea81d13f0e4df18227fe2d5f7dceb582535`
- Language/backend/profile: `triton` / `cuda` / `triton_cuda`
- Runtime fingerprint: `project.md#runtime-fingerprint`

## Implementation

The candidate retains the accepted two direct Triton stages and both exact
library `torch.topk` boundaries. `ModelNew.__init__` creates a
constructor-owned `torch.compile` callable around the target-shape forward;
`forward` dispatches it only for the accepted configuration and falls back to
the accepted eager two-stage path after compilation failure or for non-target
inputs. Temporary tensors remain per-forward and no stream/context/cache of
outputs is introduced.

## Gate Evidence

| Gate | Observation | Verdict |
|---|---|---|
| AST | `python3 -m py_compile bi150/groupedtopk/triton_grouped_topk_008.py` | pass |
| Harness compile smoke | BI150 `auto_bench.py --warmup 5 --repeat 10 --full-traceback`: `PASS accuracy`; candidate `0.353165 ms` | pass |
| All-equal ties | IDs `[7,6,4,5,1,0,2,3]`, weights allclose | pass |
| Two-expert tie | IDs `[1,0,2,3,4,5,6,7]`, weights allclose | pass |
| Structured group tie | IDs `[32,0,64,96,4,3,1,2]`, weights allclose | pass |

## Conformance

- Uses no new custom selector or direct selection primitive.
- Retains caller device/current-stream behavior and accepted per-forward buffer ownership.
- Uses constrained `torch.compile` evidence from `scripts/bi150_torch_compile_probe.py`; the candidate itself validates the target-shape graph path on BI150.
- Does not use `tl.dot`, block pointers, `fast_libentry`, explicit launch hints, or mixed precision.

## Handoff

Remote smoke is a Coder gate only. Route this hash to Verifier for independent
correctness, paired timing, and targeted profiling before adoption.
