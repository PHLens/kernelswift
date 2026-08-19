# Coder Result 002

- round: `002`
- result: `candidate-ready`
- change_family: `kernel-tuning`
- bottleneck_class: `device-bound`

## Source and Decision

| Artifact | Path | SHA-256 |
|---|---|---|
| canonical source (last accepted) | `candidate_001.py` | `b74e407348d424c9265ddf831b245cda90297a48bdbaa576fa7e6b57b5d121f9` |
| immutable base (v0 reference) | `../base.py` | `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3` |
| decision | `rounds/decision_002.md` | `0539d245c659369917660581165e8a332e00a65ca9d56128f7a0fe4fbf4d2a21` |

## Candidate

| Field | Value |
|---|---|
| path | `candidate_002.py` |
| SHA-256 | `6a66f302b3cbf2316b99c9d207e32161cb2bc05e4ea327279ce7be3d8955357c` |
| target_profile | `triton_ascend` |
| config change | `BLOCK_C`: 256 -> 1280; `num_warps`: 4 -> 2 |

## Implementation Summary

`candidate_002.py` is a minimal edit of the accepted `candidate_001.py`:
identical kernel structure (single `(8192,)` grid, weights `comb_res_mix[4,4]`
and `post_layer_mix[4]` loaded once per program, explicit 4-way fp32 reduction,
single fp32->bf16 cast). Only the two tuning knobs changed:

- `BLOCK_C`: `256 -> 1280` (the c-loop `tl.static_range(0, C // BLOCK_C)`
  collapses to a single iteration, eliminating the five serialized c-block
  iterations that dominate the latency-bound profile).
- `num_warps`: `4 -> 2` (best median in the warp sweep).

No 2D-grid flattening (the R001 0.926x regression is avoided); no `tl.dot`
(`m=4` below the probed `(16,16)@(16,16)` shape, rejected by decision).

## Tuning Sweep Evidence

Interleaved round-robin device-time measurement (60 rounds, torch.npu.synchronize
per sample) over BLOCK_C x num_warps, correctness checked against a torch
`einsum` reference at atol/rtol=1e-2:

| BLOCK_C | num_warps | correct | median ms | min ms |
|---|---:|---:|---:|---:|
| 256 (accepted) | 4 | PASS | 0.7937 | 0.7304 |
| 256 | 8 | PASS | 0.7908 | 0.7278 |
| 320 | 2 | PASS | 0.7898 | 0.7196 |
| 320 | 4 | PASS | 0.7925 | 0.7250 |
| 512 | 2/4/8 | PASS | ~0.925 | — |
| 640 | 8 | PASS | 0.7883 | 0.6895 |
| 1280 | 2 | PASS | 0.7787 | 0.6827 |
| 1280 | 4 | PASS | 0.7815 | 0.6837 |
| 1280 | 8 | PASS | 0.7833 | 0.6809 |

Observations:
- `BLOCK_C=512` is pathological (~0.92 ms): `C=1280` is not divisible by 512, so
  it emits a masked 3-iteration tail with misaligned loads.
- `BLOCK_C=1280` (single pass, no loop) is the consistent winner, ~1.5-2% better
  median device time than the accepted `256/4`, and ~7% better on the `min`
  statistic (0.68 vs 0.73 ms). This matches the latency-bound hypothesis: fewer,
  larger contiguous c-iterations raise memory-level parallelism.
- `num_warps` in {2,4,8} has negligible effect at BLOCK_C=1280 (within ~0.005 ms);
  `2` chosen (best median, matches decision sketch hint).

## Primitive / Hint Conformance

| Primitive / hint | Decision requirement | Used | Status |
|---|---|---|---|
| grid / c-loop | keep `(8192,)` grid + load-weights-once c-loop | preserved exactly | conform |
| matmul lowering | explicit fp32 reduction (no `tl.dot`) | unchanged | conform |
| `tl.static_range` | compile-time loop | 1 iteration (BLOCK_C=1280) | conform |
| `tl.arange` | extents 64/128/256 proven | extent 1280 (unproven in profile, verified locally by sweep compile+correctness) | conform note |
| `num_warps` | proven {1,2,4}; 8 requested | 2 | conform |
| `num_stages` / `vectorize` / `make_block_ptr` / `async_copy` | Unknown, not required | not used | conform |

Conformance note: `tl.arange(0, 1280)` is beyond the profile's proven extent set
{64,128,256}. It compiled and produced correct results on this runtime in the
sweep (all BLOCK_C values 320/512/640/1280 compiled and passed correctness), so
it is locally established, not a capability-miss.

## Numerical Guardrail

- fp32 accumulation before a single bf16 cast (unchanged from accepted candidate).
- Harness correctness `PASS` at atol=1e-2 / rtol=1e-2 across all sweep configs
  and the final smoke.

## Local Gate

| Gate | Command | Status |
|---|---|---|
| ast.parse | `ast.parse(candidate_002.py)` | PASS |
| harness loader | `auto_bench.load_ks_module(candidate_002.py)` | PASS |
| warmup/compile smoke | `auto_bench.py --v0_file base.py --v1_file candidate_002.py` | PASS accuracy |

## Attempt Ledger

| # | Change | Defect | Result |
|---|---|---|---|
| 1 | sweep BLOCK_C {256,320,512,640,1280} x warps {1,2,4,8} (scratch script) | none | all compile+correct; 512 pathological; 1280 best |
| 2 | interleaved focused re-measure | none | 1280/2 best median 0.7787 ms |
| 3 | write candidate_002.py (256/4 -> 1280/2) | none | gate + smoke PASS |

## Smoke vs Accepted (informational; authoritative timing is Verifier's)

Head-to-head harness runs (warmup 50 / repeat 100), 3 interleaved pairs:

| Pair | candidate_001 (256/4) v1 ms | candidate_002 (1280/2) v1 ms |
|---:|---:|---:|
| 1 | 0.8788 | 0.8765 |
| 2 | 0.8807 | 0.8695 |
| 3 | 0.8886 | 0.8762 |

candidate_002 is consistently ~0.5-1% faster (median ~0.876 vs ~0.881 ms). This
is a real but small wall-time gain, well below the decision's 15% expectation and
below the 5% adoption threshold. The kernel remains ~19% of HBM peak, with the
remaining wall time dominated by the harness-fixed ~0.26 ms host/sync gap
(`device_ratio` ~0.70), which this kernel-only change cannot address.

## Reason Code

`candidate-ready` — candidate conforms to the immutable design (kernel-tuning only:
BLOCK_C and num_warps; all invariants preserved, single kernel launch, weights
loaded once, explicit fp32 reduction, no `tl.dot`, no 2D-grid regression).
The measured gain (~0.5-1%) is directionally consistent with the latency-bound
hypothesis but below the adoption threshold; Verifier's authoritative timing will
determine `accepted` vs `no-improvement`.
