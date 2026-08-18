# Designer Context

## Context Epoch

- epoch: 4
- current_round: 003
- last_completed_round: 002
- last_accepted_round: 001
- status: Round 003 decision validated and ready for Coder

## Canonical Pointer

- accepted implementation: `triton_grouped_topk_001.py`
- accepted report: `rounds/report_001.md`
- current decision: `rounds/decision_003.md`
- canonical implementation SHA256: `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384`
- canonical report SHA256: `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a`
- current decision SHA256: `cfcee8a61b91536da0aa302504b8bc4119c9c2deac5150878b6371870791f6b7`
- measurement fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`

## Durable Evidence

1. Round 000 baseline: canonical wall was `0.231739 ms`; the separately
   scoped device trace reported `147.75267 us/call`, `15 kernels/call`,
   and `89.674 us/call` across the four gatherTopK/bitonicSort launches.
2. Round 001 accepted `kernel-fusion`: the fixed benchmark became one
   Triton program per token with canonical wall `68.280 us/call`; the
   separately scoped device result is `10.7442822265625 us/call` and
   exactly one kernel/launch per call. The supplemental CPU scope is
   `41.58952 us/call`; two inclusive `aten::empty` events total
   `10.03988 us/call` and the inclusive launch event is
   `4.88562 us/call`. Inclusive events are not additive and do not
   reconstruct wall time.
3. Round 002 `fresh-allocation-coalescing` is a durable no-improvement.
   Its capability, single-backing, disjoint-view, lifetime, alias,
   concurrency, fallback, kernel-equivalence, and correctness checks passed,
   but formal wall medians were `0.071684 ms` reference versus
   `0.081513 ms` candidate, an improvement of
   `-13.711567434852972%`. Profiling was skipped after the wall gate failed,
   so the regression has no durable causal attribution. Canonical remains
   Round 001 and this allocation family must not be carried into Round 003.

## Current Bottleneck Judgment

The accepted implementation is host-bound overall, but the only remaining
device work is still a measurable `10.7442822265625 us` single kernel.
Clearing the 5% wall gate at `68.280 us` requires `3.414 us/call`; a
kernel-only intervention therefore needs roughly a 31.8% device-time
reduction if host work is unchanged. Round 003 targets a countable redundancy
inside that kernel: eight expert ranks each perform a 256-lane argmax and then
a second 256-lane sum solely to recover the selected value.

## Ranked Backlog

| Rank | Hypothesis / family | Bottleneck | Expected wall gain | Risk | Evidence pointer | Validation cost |
| --- | --- | --- | ---: | --- | --- | --- |
| 1 | Selected Round 003: fuse each expert rank's separate argmax and selected-value sum into one standard Triton value-plus-index max reduction with explicit left tie break (`value-index-reduction-fusion`) | redundant full-width device reductions | about 6%; adoption still requires at least 5% | high: pinned MACA support and exact tie IDs are unproven capability gates | `triton_grouped_topk_001.py`, `rounds/report_001.md`, `rounds/decision_003.md` | high: compile/runtime gate, targeted ties, correctness, wall, then device profile |
| 2 | Reduce launch-wrapper overhead only if a standard, target-profile-supported direct-launch mechanism is first proven (`launcher-overhead-reduction`) | host launch path | conditional 5-7% | high: `fast_libentry` is unsupported and current direct launch is the only proven path | `state/verifier_context.md`, target profile | high: capability probe plus full wall/profile and stream/device checks |
| 3 | Specialize fixed-shape Python dispatch/guard work only after targeted host attribution identifies at least `3.414 us/call` of removable exclusive work (`fast-path-dispatch-specialization`) | unattributed host overhead | conditional 5-7% | medium-high: current CPU durations are inclusive and cannot establish the saving | `state/verifier_context.md`, `rounds/report_001.md` | medium: targeted CPU attribution, contract/fallback audit, then wall gate |

## Closed Direction

- Do not retry, cache, pool, or extend Round 002's one-backing/two-view fresh
  allocation coalescing without genuinely new causal evidence. Its requested
  mechanism was realized safely but failed the primary wall metric.

## Artifact Hash Ledger

- `team-state.md`: `d60f670f8e383ca8a269f72093732b00047121ba7cef049634fa7a1fd659faaf`
- `project.md`: `41f73ad526412fe37a41116701a3257cb7f90bffbae88611f69a99a4e2bb7750`
- `base.py`: `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb`
- `triton_grouped_topk_001.py`: `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384`
- `reference_triton_grouped_topk_001.py`: `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9`
- `rounds/report_001.md`: `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a`
- `rounds/decision_002.md`: `96b175002ab35ebbdeab2e647e1f0acfb150d08ca30792db1c6657a3afea7c55`
- `rounds/report_002.md`: `a5ad9cfe8ead4e1e3cf06ef990ea0817537af4c088219f1eed9a551055426365`
- `rounds/decision_003.md`: `cfcee8a61b91536da0aa302504b8bc4119c9c2deac5150878b6371870791f6b7`
- `state/verifier_context.md`: `c270638dc54852a64e6d931ac625940d4e019422a5cf4d924728a79f4f1f6c75`
- `skills/kernel-opt-loop/prompts/coder_targets/triton_maca.md`: `2cfa08c2664f01e70bb43eec7bb998be836a6a719b17535268a8d6ca18c85540`
