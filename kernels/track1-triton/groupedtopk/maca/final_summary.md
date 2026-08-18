# Grouped TopK C500 Campaign Final Summary

- stop_reason: `valid-no-improvement-limit`
- stopped_at: `2026-08-18T10:12:07Z`
- run_branch: `kernel-opt/grouptopk-c500-20260818`
- base_commit: `6a970c9`
- total_terminal_rounds: `4`
- accepted_round: `001`
- canonical: `triton_grouped_topk_001.py`
- accepted_report: `rounds/report_001.md`
- measurement_fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`

## Accepted Progress

- Round 001: single per-token Triton-MACA kernel fusing the fixed benchmark
  softmax, group-max/group-top4, masked expert-top8, and renormalization chain
  (grid `(83,)`, `BLOCK_E=256`, direct launch, `num_warps=1`),
  `69.59021613749428%` wall improvement (`0.2245 -> 0.0683 ms`); canonical
  advanced to `triton_grouped_topk_001.py`.

## Continued Evidence

- Round 002: host output-allocation coalescing (one fresh int32 backing with
  disjoint fp32/int32 views), `-13.71%` regression, `no-improvement`; canonical
  unchanged.
- Round 003: expert rank selection switched from separate full-width
  `tl.argmax` plus `tl.sum(tl.where(...))` extraction to one masked
  `tl.max(return_indices=True, return_indices_tie_break_left=True)`,
  `+0.049%`, `no-improvement`; canonical unchanged.
- Round 004: host fast-path predicate specialization (remove hidden_states
  metadata eligibility checks, compare `gating_output.shape` directly, read
  `gating_output.device` once), `+2.686%`, `no-improvement`; standard
  correctness, semantic guard (`18` cases), and group/expert tie parity all
  passed; only the authoritative wall threshold failed. Canonical unchanged.
- Third consecutive valid `no-improvement` reaches `valid_no_improvement_limit`
  (`3`); the campaign stops with `stop_reason=valid-no-improvement-limit`.

## Final State

No candidate source was promoted after Round 001; the accepted canonical
remains `triton_grouped_topk_001.py`. Round 004 standard correctness passed
with reference/candidate raw samples `0.072364/0.070434 ms` (smoke timing);
formal wall samples were `[0.067650, 0.072364, 0.068439]` /
`[0.065375, 0.070434, 0.066601]` ms. The targeted profiler was contractually
gated on the wall threshold and not run. Local and remote SHA256 matched for
base, harness, canonical, one-line reference adapter, and candidate before
Round 004 execution. Raw profiler logs remain gitignored; reports retain their
hashes and reproduction commands.

## Reconsideration Conditions

A future run should first obtain a same-runtime microbenchmark or matched
evidence identifying a candidate-owned bottleneck with a defensible `>=5%`
path under the MACA runtime, e.g. proving the Round 003 value/index reduction
primitive or the Round 002 allocation/view path beats the accepted Round 001
kernel. A new host/launcher experiment requires a distinct candidate-owned
mechanism from Round 004. No changes were made to shared anti-pattern
references.
