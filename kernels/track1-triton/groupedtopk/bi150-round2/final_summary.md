# GroupedTopK @ BI150 — Round-2 Campaign Final Summary (Epoch 2, contract v3)

Campaign: `kernel-opt/round2-bi150-20260827` · run_epoch 2 · skill kernel-opt-loop v3.0.0
(contract_version 3 / typed-sketch-v1 / verdict-v1) · terminal `user-intervention` stop at
round 006 after designer FORMAL ABORT (natural decision-space exhaustion).

## Final Deliverable

`triton_grouped_topk_r2_004.py` @ sha256 `c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb`
— manual-cuda-graph-workspace-replay canonical (accepted round 004), report_004.md
@c79cc018…, verdict_004.json @13340553….

## Result

| basis | value |
|---|---|
| wall paired median (regime `--warmup 50 --repeat 100`) | 0.483530 → **0.196909 ms** |
| vs manifest anchor | **+59.28% (2.41x)** |
| correctness | PASS everywhere incl. four manufactured tie suites (ids exact); outputs bitwise-stable per accepted round |

## Round Trajectory

| round | family | outcome | headline |
|---|---|---|---|
| 000 | baseline adapter | baseline | v0 0.483530 ms · 14.94 kernels/call · device_ratio 0.372 |
| 001 | preprocess-fusion-triton-stages | ACCEPTED +11.41% | three Triton stages around both retained torch.topk; first on-device Triton kernels this runtime; `run_out` surface established |
| 002 | compile-graph-default | ACCEPTED (+18.22% direct) | shared compiled callable; host-dispatch compression proven by flat device band |
| 003 | compile-graph-replay-reduce-overhead | NO-IMPROVEMENT | Inductor mutation-skip demoted capture every invocation; wrapper-only overhead −8.09% same-session |
| 004 | manual-cuda-graph-workspace-replay | ACCEPTED +42.54% direct | static workspace capture outside Inductor heuristics; launches collapsed to ~3 boundary copies/call |
| 005 | boundary-dispatch-coalescing | NO-IMPROVEMENT | `_foreach_copy_` coalesces python dispatch only; GPU submission counts unchanged on this build |
| 006 | formal ABORT | aborted | decision space exhausted with documented reopening conditions |

## Key Learnings (BI150 / CoreX 4.4.0 specific)

1. **Inductor cudagraph-trees skips on mutated inputs** in this pattern — manual
   `torch.cuda.CUDAGraph` with static workspace + boundary copy-in/out is the working
   substitute (r004: single graph launch replaces ~6.9 kernels/call of dispatch).
2. **Per-call device time is barrier-locked** around the two retained vendor top-k
   kernels (~48.6 µs gatherTopK + ~37.3 µs bitonicSortKVInPlace at r002 scale);
   replacements are tie-certifiability-blocked (`sbtopk/bitonic` permutations are
   implementation-emergent; cross-implementation score-bit ambiguity breaks exact-ID).
3. **torch._foreach_copy_ merges dispatch, not submissions** on this build.
4. Harness host floor dominates small-shape walls; after r004 the wall is ~½ device /
   ~½ residual host.

## Reopening Conditions (of record, decision_006 Rationale)

a. maintainer-authorized on-device tie-rule derivation (replaces uncertain emulation);
b. harness/workspace infrastructure restoring inductor-level capture compatibility;
c. revised frozen profile promoting argmax-family selection beyond unique-maximum constraint.

## Campaign Governance

Phase-0 triton_cuda machine-readable partial profile promoted this campaign
(`skills/kernel-opt-loop/profiles/triton_cuda/`, snapshot pinned dc8fa4c0…) — future
triton_cuda campaigns inherit a validated load_profile target. All six decisions,
sketches, coder results, reports, verdicts validated by their deterministic gates;
zero verifier-routed repairs consumed; measurement fingerprint `8deb1b01…` unchanged
start-to-stop.
