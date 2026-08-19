# Coder Result 002

Result: implementation-failed

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Decision SHA-256: `d3c0f316945706acaad5c6f68ae0d93e9bbf3c848ca735b20c2356b304107d37`
- Canonical source: `baseline_adapter.py`
- Canonical source SHA-256: `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016`
- Candidate: `triton_grouped_topk_002.py`
- Candidate SHA-256: `3cda42c8aee3b35bb44e1ec4e7231101765800be7602191ac8a5019a70925e87`
- Language: `triton`
- Backend: `cuda`
- Target profile: `triton_cuda`
- Runtime fingerprint: `project.md#runtime-fingerprint`

## Result Classification

`implementation-failed`: two bounded implementation attempts produced a
fixed-shape direct Triton candidate that compiles and passes the seeded harness
smoke, but neither implementation preserves the required PyTorch top-k tie
ordering across the required grouped selection cases. A further repair would
need a new dataflow that reproduces the vendor library's active-set-dependent
selection network, which is a major deviation from the immutable Round 002
sketch.

## Profile Conformance

- Used direct Triton launch only.
- Used matched grouped-profile primitives: contiguous fp32 load/store, arange,
  program_id, reshape, max/sum reductions, exp, argmax, zeros/full, where,
  broadcast_to, and static_range.
- Did not use `tl.dot`, block pointers, `fast_libentry`, `num_warps`,
  `num_stages`, or mixed precision.
- The first tie repair encoded deterministic priorities with supported vector
  operations, but the target's repeated selection ordering remained more
  complex than the decision's lower-ID rule.

## Attempt Ledger

| Attempt | Candidate SHA-256 | Commands and gate | Observation | Result |
|---|---|---|---|---|
| 1 | `8356acbbab5b21244ae2411546455cacc72fe4b40e233074bd920f6ebc984704` | `python3 -m py_compile bi150/groupedtopk/triton_grouped_topk_002.py`; remote `auto_bench.py --warmup 2 --repeat 3 --full-traceback`; remote all-zero tie check through `auto_bench.load_ks_module` | AST and real harness smoke passed (`PASS accuracy`, v1 `0.170447 ms`). All-zero input failed integer IDs: reference `[7,6,4,5,1,0,2,3]`; candidate `[0,1,2,3,4,5,6,7]`. | repair required |
| 2 | `3cda42c8aee3b35bb44e1ec4e7231101765800be7602191ac8a5019a70925e87` | `python3 -m py_compile bi150/groupedtopk/triton_grouped_topk_002.py`; remote harness smoke; remote all-zero and structured-tie checks through `auto_bench.load_ks_module` | AST and harness smoke passed (`PASS accuracy`, v1 `0.165995 ms`). All-zero IDs matched. A structured group-tie input failed: reference `[32,0,64,96,4,3,1,2]`; candidate `[0,32,64,96,7,6,4,5]`. | implementation failed |

## Gate Evidence

- Local AST gate: pass for both attempts.
- Real harness AST loader and remote compile smoke: pass for both attempts.
- Seeded harness correctness: pass for both attempts.
- Required exact-ID tie correctness: fail after the final allowed local repair.
- Floating weights matched in the all-zero and structured-tie checks with
  `atol=1e-2, rtol=1e-2`; the integer ordering mismatch is disqualifying.

## Stable Reason Code

`topk-tie-ordering-active-set-mismatch`

The BI150 reference returns a selection ordering that depends on the active
candidate set. Evidence includes all-zero IDs `[7,6,4,5,1,0,2,3]`, a two-expert
tie `[1,0,2,3,4,5,6,7]`, and the structured group-tie mismatch above. The
Round 002 kernel's static priority repair cannot preserve these cases without a
new decision that specifies a proven compatible tie-ordering mechanism.

## Handoff

No candidate is eligible for Verifier timing or profiling. Keep
`baseline_adapter.py` as canonical and complete the round as `candidate-failed`.
