# Coder Context

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba`
- context_epoch: `1`
- last_completed_round: `002`
- accepted_kernel: `triton_fused_moe_e2_001.py` *(round-001 ACCEPTED by Verifier: wall 3.193262 → 0.219792 ms, +93.248% vs canon 3.255288 (14.81x), correctness 12/12 incl. all five expert-activation variants)*
- binding_ledger: `rounds/binding_001.json` sha256 `2ea87f554a8fa8dc7d30bb411e9ae849e2d05a6059e04271787d4504b930c506` — **NOW SUPPLIED** (was missing at Verifier dispatch; Coder-owned per prompts/coder.md:96). `validate_binding.py` → **VALID**, 21/21 statements covered (17 implemented, 4 accommodated). Spans are AST-derived, not hand-written; regenerate/re-validate with `log/probes/p04_r001_binding_ledger_build.py`. Audits over final bytes in `log/probes/p04_r001_binding_audits.json`: DANGER scan all-zero, 3 tl.dot sites (fp16→fp32, 2.441e-04 max_abs, exactness-POSITIVE), num_warps single pinned site = 1, static grid (8,11) with on-device empty-tile exit.
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `R001 (this round, noncanon): counting-sort grouped-GEMM restructure 1→2 Triton launches + three-tier manual CUDA-graph replay. GEMM device 261.7→27.4 µs (9.5x, grid 8→88 programs on 16 SMs, 12.34x replicated GEMM work eliminated). Pipeline 92.9 µs/call (probe, graph-assisted). Intact harness: PASS accuracy, 3.204→0.219 ms, 14.62x. best_num_warps=1, BLOCK_M=16 (FR-4: sibling nw2 prior does NOT transfer). Matrix prior (noncanon): epoch-1 6.60x (wall 3.259→0.493 ms, 123.9→9.82 kernels/call, device 968→141 µs) via per-expert dispatch fusion + tl.dot GEMM; preserved archive at ../.`
- open_hypotheses: `sort is now ~28% of pipeline (26.2 µs of 92.9) after the [BLOCK,E] rank-matrix fix; a future round could fold the rank into the expert kernel's tile loop. Frozen aten routing prelude (~64 µs) is now the largest single component and was declared out of scope by decision-001.`
- artifact_read_hashes: `decision_001.md=62820af457c7b0b84232dc28bffd07009b5bc1ee482059728da06761381fd1d5 (re-verified unchanged post-work); sketch_001.json=6a46d4fd67b0cbce7a34ce41eac0c2b4cc19f00dd6e6098cf91a60e879634cb4 (re-verified unchanged post-work); profile_snapshot/triton_cuda.yaml=dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae (read-only)`

## Current Bottleneck

- `Post-R001: device time is now dominated by the FROZEN aten routing prelude (~64 µs of the 92.9 µs pipeline), which decision-001 declared out of scope. The grouped GEMM itself is only 27.4 µs and the counting sort ~26.2 µs. Both remaining Triton costs are small; further gains need either the routing prelude (blocked by the refused fp32-reduce waiver) or a different decomposition.`
- `Caveat: the epoch-1 control arm measures 261.7 µs under graph-assisted probe timing vs the Verifier's 55.80 µs kineto census for the same kernel — probe numbers are internally consistent but on a different measurement path, so they must not be quoted as authoritative device time.`

## Round-002 Evidence (host/boundary only — REGRESSION, premise falsified)

- **FINAL (C3, shipped by Orchestrator ruling)** candidate
  `triton_fused_moe_e2_002.py` sha256
  `781d341cae2236917da988988fbe2754fc808ea0f016d7dff82fd142822d1b2d`;
  binding `rounds/binding_002.json` sha256
  `8be91ccae9c3887c480451698d6bd02f1d1eb2b5c8c0d8ea08c55570f6b4e876`.
  SUPERSEDED two-hop variant `ffd4dac3...` / ledger `35a4500e...` — DO NOT
  MEASURE (it is +5.11 us vs r001).
- Classification **candidate-ready — correctness-hardening**. NOT expected to
  clear the 5% wall gate (10.99 us); shipped for the retention guarantee.
  Harness parity confirmed: 14.57x-14.82x vs r001 14.56x-14.72x (two-hop was
  14.17x-14.21x).
- validate_binding VALID, 22/23 statements; `op_alloc_dest` is kind `alloc`,
  not required, and is unbindable — no allocation contract exists in the
  profile — recorded under `unbound_statements` in p18 audits).
- **CORRECTNESS IS CLEAN.** Retention test PASS: a tensor returned by forward
  and retained across 50 further forwards is BYTE-IDENTICAL
  (`retained_output_unchanged`). All 7 suites pass vs base.py (8/7/2/2/1
  active + fp16-extreme + non-target T=128/E=16). Bitwise-equal to round 001.
  Tier equality, determinism, run_out poisoned x2, intact harness PASS
  (14.17-14.21x vs round-001 14.56-14.72x).
- **PERFORMANCE PREMISE FALSIFIED.** `torch.empty_like` measures **0.005 us**
  (graph-pool: 0.0049 — the decision's 16.219 us is NOT this allocation; most
  likely the harness's own output handling inside the timed region). The
  allocation **cannot** be removed: round-001's single empty_like played BOTH
  the copy-target and the returned-tensor role, so persisting the target
  leaves the return allocation. The only allocation-free shape (rotating pool)
  FAILS retention exactly at the pool size (n=8 for pool=8), and auto_bench
  runs 150 forwards, so it is unsafe below ~150.
- **ORCHESTRATOR RULING: ship C3.** Two-hop (out_ws -> out_dest -> fresh) was
  +5.113 us/call; C3 (out_ws -> fresh, single copy, out_dest NOT written) is
  -0.022 us. The literal `op_copy_out` `writes: [out_dest]` is OVERRIDDEN: it
  rested on the now-falsified premise that persisting the destination pays for
  itself, and mandating +5.11 us to satisfy a literal span on a disproven
  premise is "formalism over evidence". Recorded in binding_002.json as
  `status: accommodated` (op_copy_out) + `rulings[0] = RULING-002-C3` in
  p18 audits.
- **Orchestrator corrected my allocation magnitude**: they measured
  `torch.empty_like(83,128,fp16)` = ~4.13 us (empty 5.59, empty_strided 5.56;
  graph.replay() ~34.13 us). My p14 said 0.005 us (UNDER-measured — the
  graph-replayed method did not capture the full allocation path); the designer
  said 16.219 us (OVER-attributed). Conclusion unchanged either way: G1's
  ceiling ~4 us is BELOW the 10.99 us gate regardless.
- **Retention is the round's real product**: r001 had non-aliasing only
  implicitly; r002 makes it explicit and tested. Protocol is p12 (CHANGING data
  each call) — p11's constant-data version was a FALSE PASS.
- Sweep: **best_BLOCK_M = 16** (margin 9.829 us, outside tie band).
  **num_stages NOT recorded** (argmin `default`, margin 0.031 us, inside the
  0.5 us tie band). num_warps pinned at 1, NOT re-swept (FR-4 settled).
- FR-4 PASS: device +0.058 us (threshold ±15), control pre-bound, and the two
  @triton.jit bodies are byte-identical to round 001 (digest 12d5e7eb...).
  NOTE: an earlier digest check was silently VACUOUS (hashed an empty kernel
  set to e3b0c442...); fixed and made non-vacuous.

## Recent Three-round Evidence

- **R001** (current): family `manual-graph-replay-fused`. Candidate
  `triton_fused_moe_e2_001.py` sha256
  `da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7`.
  Classification SUCCESS-FINAL; 21/21 sketch statements bound; gate suite
  G1–G13 all PASS.
  - Dot re-qualification: fp16-operand dots at M16/M32 × K128/K64 are
    **exactness-POSITIVE** on this rig (max_abs ≤ 2.441e-04 vs 1e-2 tol,
    0 out-of-tol) — mm_encoder's fp16 negative does **not** transfer here.
    `int64→int32` kernel-side narrowing qualified (was `unknown`).
  - Sweep: `best_num_warps=1`, `BLOCK_M=16`, won outright (92.855 vs 122.253
    µs at nw2). FR-4 confirmed: sibling nw2 prior does not transfer.
  - Two runtime landmines found and avoided: (a) `[E]`-tile lane deposits via
    `tl.where`/int32-mask **compile but silently return zeros**; only
    reduce-based deposits work. (b) The naive `[BLOCK,BLOCK]` scatter rank
    matrix costs 553.9 µs (95% of pipeline); the shipped `[BLOCK,E]` form
    costs 26.2 µs and is bitwise-equal to a host stable counting sort.
    An atomic one-pass cursor is fast (6.6 µs) but **WRONG** — rejected.
  - Intact harness: `PASS accuracy; v0=3.204104 ms, v1=0.219205 ms,
    speedup=14.617x` (seed 42, atol=rtol=1e-2, warmup 20, repeat 50).
  - **Deviation D1 — ORCHESTRATOR RULING: ACCEPTED, deviation does NOT fire.**
    The counting sort's `tl.sum` uses (lines 128, 134, 144, 149, 150) operate
    on **int32 index/bucket-count tiles**; the waiver gate at
    decision_001:275/414 refuses specifically the **fp32 row-softmax reduction
    over activation data**, and softmax/topk/renorm/fp16 casts remain ATEN
    (lines 373-379, 382-383). **The reduction.sum waiver REMAINS NOT GRANTED
    and was NEVER exercised.** No waiver, no fallback provenance, no
    disposition entry required. Recorded verbatim in coder_result_001.md so a
    future reader does not mistake it for a waiver usage.
  - **D4 ACCEPTED**: all µs/call figures are probe-instrumentation kernel-only
    timings on a different measurement path than the Verifier's kineto census
    (control 261.7 vs their 55.80 for the same kernel). Do NOT quote as
    authoritative device time — the Verifier's census owns that number. The
    sweep RANKING remains valid because all 6 configs share the probe path.
  - **Rejected alternative (never a fallback)**: atomic one-pass cursor scatter
    (`tl.atomic_add(cursor_ptr + ids, 1)`) is fast (6.565 µs) but WRONG —
    vector-pointer atomics give colliding destinations, not deterministic, not
    allclose to the stable reference. Excluded on correctness, not performance.

## Open Hypotheses or Checks (round 002)

- `RESOLVED: C3 shipped per Orchestrator ruling (2026-08-28). No further action.`
- `LINE OF WORK TERMINATED (Orchestrator): no further allocation trickery (rotating pools etc.). "No per-call allocation" and "never aliases across calls" are mutually exclusive below ~150 forwards and auto_bench runs 150.`
- `Binding generator now addresses spans SYMBOLICALLY (occurrence index), not by hardcoded line number — an earlier docstring edit shifted every line by +7 and broke it. Re-run p18 after any candidate edit.`
- `Re-attribute the 16.219 us: if it is the harness's output handling (run_forward / clone_value / retained v1_output) landing inside the timed region, NO candidate change can reclaim it and G1 is dead as specified.`
- `G1 (boundary) is likely exhausted. Per decision-002's own budget note, if G1 cannot pay and G2 (routing prelude, ~20 us best case, waiver-gated, topk frozen) looks sub-gate, the honest recommendation is CONVERGENCE rather than a marginal round.`
- `"No per-call allocation" and "returned tensor never aliases across calls" are provably MUTUALLY EXCLUSIVE (p11/p12). Retention-safety requires a distinct buffer per call. Do not revisit without changing the harness contract.`
- `Scanner-class bug to watch: two DANGER false positives (docstring, private-helper return) and one VACUOUS kernel-digest check (empty set -> e3b0c442...) were caught in this round. Always assert non-emptiness before comparing digests.`

## Open Hypotheses or Checks (round 001, retained)

- `D1: RESOLVED (ruled accepted, deviation does not fire). No further action.`
- `FR-2 device delta: NOT a candidate defect. Verifier attributed it to their eager control re-allocating sort buffers every call. I concur — the candidate's eager (tier-3) path allocates shape-correct buffers per call DELIBERATELY, to serve non-target shapes (p03 G3 proves T=128/E=16 correct); the target regime (tiers 1-2) uses static buffers with no per-call allocation. No candidate change warranted.`
- `Profile capability matrix is coarse for fused Triton: no reduction.sum contract, arithmetic.elementwise maps to tl.exp only (not tl.sigmoid), and tl.atomic_add is undeclared. Four statements are therefore bound as "accommodated" rather than claiming contracts the profile never granted. Worth flagging if the profile is ever extended.`
- `Binding generator is idempotent: re-run log/probes/p04_r001_binding_ledger_build.py after any future candidate edit to regenerate + re-validate the ledger (hand-written spans would drift).`
- `Fold the counting-sort rank into the expert kernel's tile loop (~26 µs of 92.9).`
- `The frozen aten routing prelude (~64 µs) is now the largest component; only addressable if the routing-fusion waiver is revisited.`
- `Probe figures are graph-assisted kernel-only; re-derive device time under the Verifier's own census before quoting.`
