# Pre-G2 Measurement Note

Scope: post-round-002, pre-G2 measurement census (read-only device-timing).
`measurement_exclusive=false`. This is a measurement NOTE, not a verdict. It
does not modify `verdict_00{1,2}.json`, `team-state.md`, `project.md`, or any
candidate/decision/sketch source.

## Question 1 — the four ATEN routing-prelude device numbers

Two measurement conventions coexist in the round-001 evidence and must not be
conflated:

1. **Eager device numbers (launch-inclusive, kineto cat=kernel)** — this is the
   convention the round-001 report cited, and the one that matters for the
   "inside the graph" G2 question because the prelude currently runs INSIDE the
   captured region where each aten op is a launch inside the graph interior.
2. **Clean device-only numbers (graph-assisted isolated CUDA-event timing)** —
   this is the pure kernel cost with the python launch tax amortized away.

### Eager numbers (confirming the round-001 figures)

Fresh eager profiler over the prelude only, 100 calls, seed 42, default stream
(same method as `diagnostic_scope_census_001.json` eager control):

| op | fresh eager µs/call | report-001 cited | verdict |
|---|---:|---:|---|
| `aten::topk` (gatherTopK + bitonicSort) | **41.472** (22.675 + 18.797) | ~39.936 | confirmed (noise band) |
| `aten::sum` (reduce_kernel) | **14.714** | ~13.551 | confirmed (noise band) |
| `aten::div` | **6.821** | ~6.711 | confirmed |
| `aten::_softmax` | **5.194** | ~4.957 | confirmed (noise band) |
| `aten::copy_` (fp16 cast) | **3.790** | — (folded into casts) | confirmed |

The four numbers the team-lead carries (topk ~39.936, sum ~13.551, div ~6.711,
softmax ~4.957) are **CONFIRMED** as the eager (launch-inclusive) device figures.
The small deltas are run-to-run noise, not a correction. topk dominates the
prelude and is itself two kernels (`sbtopk::gatherTopK` ~22.7 + `bitonicSortKVInPlace`
~18.8).

### Clean device-only numbers (graph-assisted isolated, this census)

`log/pre_g2_prelude_timing.py`, graph-assisted CUDA-event timing, N_CAP=200,
R_REP=20, SEGMENTS=5, median, each op timed in isolation on precomputed inputs:

| op | isolated µs/call |
|---|---:|
| `softmax` | 1.925 |
| `topk` (gatherTopK+bitonicSort together) | 22.268 |
| `sum` (renorm denominator) | 3.750 |
| `div` (renorm divide) | 2.459 |
| `cast` (fp16) | 1.597 |
| **full prelude** (softmax→topk→sum→div→cast) | **32.865** |

Sum of isolated parts (31.999 µs) reconciles with the full-prelude measurement
(32.865 µs) to < 1 µs, and the captured full prelude is bitwise-equal to the
verbatim aten sequence (weights and ids both `torch.equal`). The routing prelude
as a whole is **~33 µs/call of device work**; its isolated parts are all small
except topk (~22 µs).

## Question 2 — single net-µs/call number for the G2 fold

**Net reclaim ≈ 9–11 µs/call of device time, and ≈ 0 µs/call of wall time.**

Reasoning (2 lines):

1. **Launch count:** the prelude currently launches INSIDE the captured graph,
   so its aten ops already cost ~0 host time per call — they are replayed as
   part of the single `cudaGraphLaunch`. Folding them into the Triton kernels
   does NOT remove any submission (submission count stays 2.0: 1 graph launch +
   1 copy-out), it only moves ~33 µs of aten kernel work onto the Triton kernels'
   device time. The reclaimable device total is the non-topk part: softmax 1.9 +
   sum 3.8 + div 2.5 + cast 1.6 ≈ **9.8 µs/call**, because topk (~22–41 µs) is
   frozen by the tie-semantics invariant and cannot be folded.

2. **Host cost:** because the prelude is already inside the graph, removing it
   saves ~0 host time — the graph's front-end/replay cost is unchanged and the
   launch count is unchanged. The only lever is device time (~10 µs of non-topk
   aten math absorbed into the Triton kernels), against a 10.99 µs adoption gate
   measured on WALL time. Since the device time sits largely off the critical
   path under the harness's ~122 µs `cudaDeviceSynchronize` floor, a ~10 µs
   device saving is unlikely to convert to wall, exactly as round 001's device
   lever (58.231 vs 55.954 µs) failed to convert. **G2's honest net is ~10 µs
   device, ~0 µs wall.**

## Question 3 — the GATE FLAG (does softmax-fold trip the reduction.sum waiver?)

**YES — folding softmax into Triton TRIPS the NOT-granted `reduction.sum`
waiver, and G2 in that form is DEAD.**

- `profile_snapshot/capability_claim.json` declares `fallback_contract:
  "reduction.sum"` with `fallback_signature: {"axis": "k", "dtype": "fp32"}`.
- `decision_001.md` line 414 states verbatim: "`reduction.sum` waiver not granted
  — softmax, renormalize, and casts stay aten." The invariant list (decision_001
  line 47) also forbids "no reduction.sum and no reduction.argmax".
- Softmax over `router_logits [83,8] fp32` requires a `tl.sum` over the k axis of
  **fp32 activation data** — precisely the `{axis: k, dtype: fp32}` signature the
  waiver gates. Reimplementing `torch.softmax` in Triton therefore exercises the
  NOT-granted `reduction.sum` waiver and is out of contract.

**The surviving form of G2 (if chartered) is the one that keeps softmax ATEN and
folds only topk/renorm/cast — but topk is frozen by the tie-semantics invariant
and renorm's `sum` is ALSO a fp32 reduction over k (the renormalize denominator
`topk_weights.sum(-1)`), so even "renorm/cast fold" trips `reduction.sum`.** The
only waiver-clean fold is the **fp16 cast** (elementwise, no reduction), worth
~1.6 µs device. Practically: **there is no waiver-clean G2 fold of meaningful
size** — softmax is dead (reduction.sum over fp32 k), renormalize-sum is dead
(same reason), topk is frozen (tie semantics), leaving only the cast (~1.6 µs).

## Summary for the Orchestrator

- Four prelude numbers: **CONFIRMED** (topk ~41.5, sum ~14.7, div ~6.8, softmax
  ~5.2 µs/call eager; the team-lead's ~39.9/13.6/6.7/5.0 are the same numbers
  within noise).
- G2 fold net: **~10 µs device / ~0 µs wall**; prelude already inside the graph
  so no launch/submission is removed.
- GATE FLAG: **softmax-fold trips the NOT-granted `reduction.sum` waiver
  (`{axis:k, dtype:fp32}`) ⇒ G2 dead in that form**; renorm-sum fold trips the
  same waiver; topk is frozen; only the fp16 cast (~1.6 µs) is waiver-clean.

Artifacts: `log/pre_g2_prelude_timing.py` + `log/pre_g2_prelude_timing.json`.
