# CODE RESULT — round-001 (coding)

**Project**: `/root/CodeBuddy/20260818191200/kernelswift/kernels/track1-triton/fused_moe/bi150/epoch2`
**Family**: `manual-graph-replay-fused`
**Classification**: **SUCCESS-FINAL**

## Classification

`SUCCESS-FINAL`

Decision-001 is implemented in full: a counting-sort grouped-GEMM restructure
that raises Triton launches from 1 to 2, wrapped in a three-tier manual
CUDA-graph replay chain. All 21 sketch statements are bound, all binding
invariants hold, the pre-adoption sweep is folded in and completed, and every
decision-scoped gate passes. The implementation is ready for Verifier
adoption; no further round is needed from the coder.

The one substantive deviation, **D1** (counting sort uses reduce-based tile
deposits and a `[BLOCK_ROWS, E]` rank matrix rather than the literal "masked
adds only" wording), was **RULED ACCEPTED by the Orchestrator — the deviation
does NOT fire**. The reduction.sum waiver at decision_001:275/414 was never
exercised: all six `tl.sum` sites operate on int32 index tiles, while
softmax / topk / renormalize / fp16 casts remain ATEN. Full ruling and source
citations recorded under "Deviations → D1".

**Status: final. No code changes pending; the candidate hash is frozen for
Verifier measurement.**

## Artifacts

| path | sha256 |
| --- | --- |
| `triton_fused_moe_e2_001.py` (candidate) | `da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7` |

### Immutable inputs re-verified unchanged (post-work)

| path | sha256 | expected |
| --- | --- | --- |
| `rounds/decision_001.md` | `62820af457c7b0b84232dc28bffd07009b5bc1ee482059728da06761381fd1d5` | match |
| `rounds/sketch_001.json` | `6a46d4fd67b0cbce7a34ce41eac0c2b4cc19f00dd6e6098cf91a60e879634cb4` | match |
| `profile_snapshot/triton_cuda.yaml` | `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae` | read-only, untouched |

Canonical source `../triton_fused_moe_002.py` (PRESERVED ARCHIVE) was read and
its routing semantics and fp16-cast placement were carried forward verbatim;
it was not modified.

### Decision-scoped probes (all under `log/probes/` — no other path touched)

| probe | purpose | verdict |
| --- | --- | --- |
| `p01_r001_capability_probe.py` | fp16 dot re-qualification at the NEW tiles + counting-sort capability | PASS |
| `p01b_r001_countingsort_isolate.py` | isolate the counting-sort compile error (full traceback) | PASS (diagnostic) |
| `p01c_r001_prefixsum_isolate.py` | 5 prefix-sum formulations, lane-deposit vs reduce | PASS (diagnostic) |
| `p01d_r001_p5_robustness.py` | reduce-form robustness R1–R4 | PASS |
| `p02_r001_config_sweep.py` | **pre-adoption sweep** (num_warps × BLOCK_M) | PASS |
| `p02b_r001_perkernel_attribution.py` | per-launch time attribution + control arm | PASS (diagnostic) |
| `p02c_r001_sort_alternatives.py` | 5 scatter formulations | PASS (diagnostic) |
| `p02d_r001_s2_determinism.py` | atomic-scatter determinism/correctness | S2 **REJECTED** |
| `p02e_r001_scatter_correct.py` | correct+deterministic scatter search | PASS, U5 selected |
| `p02f_r001_u5_robustness.py` | U5 robustness V1–V5 | PASS |
| `p03_r001_candidate_gates.py` | candidate gate suite G1–G13 | PASS |

Result JSONs sit beside each script. **No wall benchmarks and no profilers were
run**; every timing figure below is graph-assisted kernel-only CUDA-event
timing (probe instrumentation only), which the frozen profile does not gate.

---

## Dot-shape audit (re-qualification evidence)

The decision requires re-qualifying `fp16-operand` dots at the NEW tiles,
because the frozen capability matrix records only
`matrix.dot.fp32-fp32-fp32.small-blocked-tiles` at (32,32) as `constrained`
and `cast.narrow.int64-to-int32-kernel-side` as `unknown`.

Probe `p01` re-qualified each shape on the matched runtime against an **fp32
ground truth** on fp16-extreme operands (tiers 4096/2048/256/1/2⁻²⁴/0, mixed
signs, clamped to ±1024 so the test measures exactness and not overflow) *and*
on a plain-randn arm, at the UNCHANGED comparator `atol=rtol=1e-2`.

| shape | operands → acc | fp16-extreme max_abs | n_out_of_tol | verdict |
| --- | --- | --- | --- | --- |
| M16 K128 N64 (`op_dot_gate`/`op_dot_up` @ BLOCK_M=16) | fp16 → fp32 | **2.441e-04** | 0 | **exactness-POSITIVE** |
| M32 K128 N64 (BLOCK_M=32 arm) | fp16 → fp32 | 2.441e-04 | 0 | exactness-POSITIVE |
| M16 K64 N128 (`op_dot_down` @ BLOCK_M=16) | fp16 → fp32 | ≤2.441e-04 | 0 | exactness-POSITIVE |
| M32 K64 N128 (BLOCK_M=32 arm) | fp16 → fp32 | ≤2.441e-04 | 0 | exactness-POSITIVE |

**This is exactness-POSITIVE, not the mm_encoder negative.** mm_encoder found
fp16 dots on the attention contraction pattern landing 2 ULPs outside tolerance
(1459 vs vendor 1457). At *these* contractions (128 and 64, M ∈ {16,32}) the
same rig holds ~2.4e-04 against a 1e-2 tolerance — a ~40× margin. The
mm_encoder result is labelled evidence about a different contraction pattern
and does not transfer; that is exactly why this probe exists rather than an
assumption either way.

End-to-end confirmation: the full candidate against `base.py` reaches
`max_abs` 1.526e-05 on seed-42 and 9.766e-04 on the fp16-extreme suite — both
deep inside tolerance, and consistent with the per-shape audit.

`int64 → int32` kernel-side narrowing (`cast.narrow.int64-to-int32-kernel-side`,
previously `unknown`): **qualified**. `flat_ids` arrives as int64 from
`torch.topk` and is narrowed in-kernel with `.to(tl.int32)`; verified by the
counting sort reproducing host ground truth exactly on 18 id vectors.

---

## Pre-adoption sweep (folded into binding — required observable)

`p02` swept the full space `num_warps ∈ {1,2,4} × BLOCK_M ∈ {16,32}` (9
configs) **before** finalizing. Selection rule applied verbatim from
decision-001: fastest config with **bitwise-identical** outputs; ties
(≤0.5 µs) toward lower `num_warps`; residual ties toward smaller `BLOCK_M`.

| config | grid | bitwise vs nw1/BM16 | allclose vs base | probe device time |
| --- | --- | --- | --- | --- |
| **BM16_nw1** | (8, 11) | ref | PASS | **92.855 µs** ← selected |
| BM32_nw1 | (8, 6) | True | PASS | 102.685 µs |
| BM16_nw2 | (8, 11) | True | PASS | 122.253 µs |
| BM32_nw2 | (8, 6) | True | PASS | 125.758 µs |
| BM16_nw4 | (8, 11) | True | PASS | 127.913 µs |
| BM32_nw4 | (8, 6) | True | PASS | 130.327 µs |

All 6 configs compiled, all bitwise-identical to the reference, all
allclose-passing on all 3 suites (seed42 / fp16-extreme / seed7).

### Observable

```json
{"best_num_warps": 1, "best_BLOCK_M": 16, "config": "BM16_nw1"}
```

`best_num_warps = 1` won **outright** (92.855 vs 122.253 for nw2, a 30.0 µs /
24.4% margin — far outside the 0.5 µs tie band), so the tie-break was not
invoked.

**FR-4 reads: the sibling `nw2` prior does NOT transfer to this kernel.** This
is recorded as a legitimate result; `nw2` was **not** forced.

This is a genuine finding, not an artifact of the methodology: the *first* run
of the sweep selected `nw4`, and that was wrong. `p02b` attribution showed the
pipeline was then 95% counting-sort (553.5 µs of 580.9), so `num_warps` was
merely spreading a pathological 256×256 rank matrix across more warps. After
`p02c`/`p02e` replaced that matrix, the sort dropped to ~26–37 µs and the
ranking inverted to `nw1`. A sweep run on an unattributed pipeline would have
shipped `nw4` and silently paid +38 µs/call.

### Control arm and per-launch attribution (`p02b`)

| component | µs/call | note |
| --- | --- | --- |
| epoch-1 ungrouped kernel (grid (8,), BLOCK_M=256) | 261.7 | control; Verifier census says 55.80 |
| **grouped expert kernel (grid (8,11), nw1)** | **27.4** | **9.5× faster than control** |
| counting sort, `[BLOCK,BLOCK]` rank form | 553.5 | the bottleneck — replaced |
| counting sort, `[BLOCK,E]` rank form (shipped) | 26.2–36.8 | **15–21× faster than the naive form** |
| `out.zero_()` | ~1.4 | negligible |
| routing prelude (aten softmax+topk+renorm+casts) | ~64 | frozen, matches report_000 |

The grouped-expert restructure delivers a **9.5× device-time reduction** on the
GEMM itself (261.7 → 27.4 µs), which is the restructure decision-001 predicted
(via eliminating the 12.34× replicated GEMM work and lifting grid 8 → 88
programs on 16 SMs). Note the control reads 261.7 µs here vs the Verifier's
55.80 µs census: these are graph-replayed kernel-only numbers on a
different measurement path, so they are comparable **within** this table only
and must not be quoted as the Verifier's device time.

---

## Binding statement

### Invariant: no host data-dependent branch and no host data-dependent grid

**Holds.** Both launches use constexpr grids computed from shapes alone:
counting sort `(1,)`, expert kernel `(E, num_tiles)` with
`num_tiles = ceil(T*K / BLOCK_M)`. The per-expert row count is read
**device→register** (`tl.load` into `tile_n`) and empty tiles exit through an
on-device `if tile_n > 0` guard. No `.item()`, no D2H read, no print inside the
captured region.

Verified by **G12**: forcing all rows to expert 0 leaves 7 of 8 experts empty,
and the static `(8,11)` grid still produces a correct result
(`allclose=True`, `max_abs=1.526e-05`). Verified additionally by AST scan of
`_pipeline` for `item/tolist/cpu/numpy` calls: **zero hits** (an initial
source-grep version of this check gave a false positive on the words appearing
in docstrings; the AST scan is the correct instrument).

### Invariant: never return graph-pool memory

**Holds.** `out_ws` is graph-pool-backed and is never returned. `forward`
returns a fresh `torch.empty_like` filled by `copy_`; `run_out` fills the
**caller's own** buffer by `copy_` and returns `None`.

Verified by **G11**: across all three tiers, no returned tensor aliases any
workspace pointer (`out_ws`, `x_in`, `rl_in`, `sorted_rows`, `sorted_w`,
`expert_counts`, `expert_offsets`) — `alias_tiers == []`; and `run_out`
preserves the caller's buffer identity (`data_ptr` unchanged before/after).

### Invariant: three-tier chain, permanent-on-failure, bounded recapture

**Holds.** Verified by **G5** (all three tiers fired and produced correct
results), **G6** (budget trail `4,3,2,1,0,0,0`, monotone, never negative, all
outputs correct), **G7** (budget exhausted → new pointer set rides tier-2 and
is correct), **G8** (both flags permanent-once, both handles cleared, no
resurrection, still correct through tier-3).

### Invariant: `run_out(hidden_states, router_logits, out)` 3-arg surface

**Holds.** Kernel-mode profiling remains **UNAVAILABLE** for this surface:
`auto_bench.make_profile_call` passes `run_out(gating_input, *reference_outputs)`
= 2 args, while the signature takes 3. Per the project contract, forward-mode
profiling is canonical and **no synchronization or accommodation was added**.
Verified by **G9**: two successive `run_out` calls into the same buffer with
different inputs each leave that call's own output, with no stale carry-over.

### DANGER token audit

| DANGER token | status |
| --- | --- |
| `cuda_graph` | **handled** — manual `torch.cuda.CUDAGraph`, 3 warmups on a side stream before capture, both graphs captured once per binding, no capture during timing |
| `host_sync` | **absent** — no `.item()`, `.cpu()`, `.tolist()`, or `synchronize()` inside the captured region or any hot path |
| `implicit_sync_pattern` | **absent** — the copy-out is a device `copy_` outside the replay boundary; no D2H transfer occurs |
| `autocast` | **absent** — fp16 casts are explicit and inside the captured region, exactly as in the epoch-1 canonical source |
| `fallback_suspicion` | **none** — no silent `except: pass`; both tier demotions are recorded in monotone flags and reported as deviations |

---

## Attempt ledger

| # | attempt | outcome |
| --- | --- | --- |
| 1 | Dot-shape re-qualification (`p01`) — 4 fp16 shapes vs fp32 ground truth | All 4 **exactness-POSITIVE**, max_abs ≤ 2.441e-04, 0 out-of-tol |
| 2 | Counting sort, `tl.static_range` masked-add lane deposit (`counts = counts + one_if`) | **REJECTED** — hard compile error, shapes 8 vs 256 |
| 3 | Prefix sum via `tl.where(be == b, x, 0)` lane deposit (P1–P4) | **REJECTED** — compiles but **silently yields zeros** |
| 4 | Prefix sum via `(be == b).to(tl.int32) * x` | **REJECTED** — same silent-zero failure |
| 5 | Reduce-based deposit `tl.sum(tl.where(be < b, counts, 0))` (P5) | **ADOPTED** — correct, verified R1–R4 |
| 6 | Scatter via `[BLOCK,BLOCK]` rank matrix (S1/U1) | Correct + deterministic but **553.9 µs** — 95% of pipeline; superseded |
| 7 | Scatter via atomic one-pass cursor (S2) | Fast (6.565 µs) but **WRONG** — not deterministic, not allclose to the stable reference. **REJECTED** |
| 8 | Scatter via `[BLOCK,E]` rank matrix (U5) | **ADOPTED** — 26.2 µs, bitwise-correct on 18 vectors, deterministic, nw-invariant |
| 9 | Sweep run 1 (pipeline with the S1 sort) | Selected `nw4` — **superseded**, artifact of the sort dominating |
| 10 | Sweep run 2 (pipeline with the U5 sort) | Selected **`nw1` / BLOCK_M=16** — shipped |
| 11 | Hardcoded `_NUM_TILES=11` + 83-row `out_ws` on the eager path | **BUG FOUND + FIXED** — T=128/E=16 returned an [83,128] tensor; eager path made shape-generic |
| 12 | Tier demotion via bare attribute write | **BUG FOUND + FIXED** — flag alone did not clear the handle; routed through `_bind_*_failed()` and made the flag authoritative |

---

## Correctness sweep (G3, vs `base.py` at atol=rtol=1e-2)

| suite | active experts | max_abs | verdict |
| --- | --- | --- | --- |
| seed-42 `[83,128]` | 8 | 1.526e-05 | PASS |
| fp16-extreme | 8 | 9.766e-04 | PASS |
| expert 7 forced out of top-2 | 7 | 1.526e-05 | PASS |
| only experts 0,1 active | 2 | 1.526e-05 | PASS |
| all-tie (zero logits) | 2 | 7.629e-06 | PASS |
| all rows to expert 0 (7 empty experts) | 1 | 1.526e-05 | PASS |
| **non-target shape T=128, E=16** (tier-3 eager) | — | 1.526e-05 | PASS |

Additional gates: **G4** tier outputs bitwise-identical across all three tiers;
**G10** cross-instance alternation bitwise-consistent; **G13** 20 consecutive
forward calls bitwise-identical.

**Intact-harness smoke** (`auto_bench.py`, untouched, seed 42,
atol=rtol=1e-2, warmup 20, repeat 50):

```
PASS accuracy; v0=3.204104 ms, v1=0.219205 ms, speedup=14.617x
```

(An earlier 6.48 ms reading was transient GPU contention; two consecutive
re-runs returned 14.56× and 14.72×. An initial 42 ms reading was cold-start
graph capture under `--warmup 0`.)

---

## Deviations

### D1 — Counting sort uses reduce-based deposits and a `[BLOCK_ROWS, E]` rank matrix

> **ORCHESTRATOR RULING (final): ACCEPTED — the deviation does NOT fire. This
> is COMPLIANT with decision-001's intent, not a waiver usage.**
>
> The invariant line reads *"no reduction.sum ... counting sort uses
> `tl.static_range` masked adds only"*, and the waiver gate at
> **decision_001:275/414** refuses specifically **the fp32 row-softmax
> reduction over activation data** (softmax / renormalize / casts stay aten).
>
> Verified against the candidate source: all six `tl.sum` sites —
> **lines 128, 134, 144, 149, 150** — operate on **int32 index / bucket-count
> tiles**: bucket counting (`cnt_b`), exclusive prefix (`acc`, `earlier`), rank
> and offset selection. Meanwhile **lines 373-379** confirm `torch.softmax`,
> `torch.topk` (tie semantics verbatim), the renormalize divide and the fp16
> casts of `w1`/`w2` all remain **ATEN inside the captured region**,
> byte-for-byte with the epoch-1 canonical source.
>
> Therefore: **the reduction.sum WAIVER REMAINS NOT GRANTED and was NEVER
> exercised.** `tl.sum` was used on small integer index tiles, which is not the
> gated capability. **No waiver, no fallback provenance, no disposition entry
> required.**
>
> Recording note for future readers: the presence of `tl.sum` in this file must
> **not** be mistaken for an invocation of the refused fp32-activation-reduce
> waiver. The gated operation is an fp32 reduction over activation data; this
> code performs int32 reductions over index tiles only, and every activation
> reduction (softmax, renormalize) stays aten.

### D1 background — capability evidence for this rig (not a waiver claim)

The decision invariant reads: *"no reduction.sum and no reduction.argmax;
counting sort uses `tl.static_range` masked adds only."* The shipped kernel
departs from that **literal** wording in two places, and both departures are
forced by measured behavior on this rig:

1. **Tile deposits use `tl.sum`.** The masked-add-only lane deposit
   `counts = counts + one_if` is a **hard compile error** (shapes 8 vs 256),
   and both `tl.where(be == b, x, 0)` and `(be == b).to(tl.int32) * x` for the
   `[E]`-tile prefix sum **compile but silently return zeros** (`p01c` P1–P4
   all returned `[0]*8`). Reduce-based deposits are the only correct form
   found. The invariant as literally written is **not implementable
   correctly** on this runtime.

2. **The scatter rank uses a `[BLOCK_ROWS, E]` matrix, not masked adds.** The
   masked-add-equivalent formulation builds a `[256, 256]` matrix per bucket
   and costs **553.9 µs/call** — 95% of the whole pipeline, i.e. it would have
   turned the round's device win into a large regression. The `[256, 8]` form
   costs **26.2 µs** (21× less) and is bitwise-equal to a host stable counting
   sort.

**Scope, and why it does not breach the refused waiver.** `decision_001`
refused a waiver to move the fp32 row-softmax reduce into Triton. That refusal
is **fully respected**: softmax, top-k, the renormalize divide, tie order, and
the fp16 casts of `w1`/`w2` all remain **aten** ops inside the captured region
(candidate lines 373-379, 382-383), exactly as in the epoch-1 canonical source.
The `tl.sum` calls here (lines 128, 134, 144, 149, 150) are **int32 reductions
over small index tiles** — bucket counting and rank computation — a different
operation, on a different dtype, in a different role. No fp32 reduction over
activation data enters Triton.

Evidence: `p01b`, `p01c`, `p01d`, `p02c`, `p02d`, `p02e`, `p02f`.

**Capability finding retained in the ledger** (per Orchestrator direction):
the masked-add-compiles-but-returns-zeros behavior (`p01c`) and the 553.9 µs
`[256,256]` rank matrix (`p02b`/`p02c`) are genuine properties of this rig and
are recorded here so a future round does not rediscover them.

### D1-R — Rejected alternative: atomic one-pass cursor scatter (6.565 µs)

**REJECTED ALTERNATIVE — this was never adopted and must never be treated as a
fallback.** `p02c` S2 replaced the rank computation with a per-lane
`tl.atomic_add(cursor_ptr + ids, 1)` cursor bump, costing only **6.565 µs/call**
(84× faster than the `[256,256]` form). It is **wrong**, and `p02d` proved it:
the vector-pointer atomic gives colliding destinations for lanes sharing a
bucket, so the result is **not deterministic** across 20 runs and **not even
allclose** to the stable reference. It is excluded on correctness grounds, not
on performance, and any future round considering it must re-prove
determinism before use.

### D2 — Sort-free variant measured, not adopted

`p02c` S5 (drop the sort launch; have the expert kernel scan the 166 flat ids
per program) was measured at 96.9 µs/call — correct in structure but slower
than the shipped two-kernel split, because each of the 88 programs redundantly
scans all 166 rows. Rejected on measurement; launch count stays 2 as decided.

### D3 — Number of `tl.sum` calls is a lower bound on int32 reduction cost

The counting sort performs `O(E)` tile reductions for counts, `O(E)` for the
prefix sum, and one `[BLOCK_ROWS, E]` matrix reduce for the rank. This is
~2 048 lane-ops for the rank versus 524 288 for the naive form. The absolute
sort cost (26.2 µs at nw1) is now ~28% of the shipped pipeline's 92.9 µs, so
the sort is no longer the bottleneck, but it is not free either — a future
round could consider folding the rank into the expert kernel's tile loop.

### D4 — Probe timing is graph-assisted and not the Verifier's census — ACCEPTED

**Orchestrator ruling: ACCEPTED.** All µs/call figures above come from
graph-assisted kernel-only CUDA-event timing (N_CAP=100 pipelines captured,
R_REP=10 replays per segment, median of 3 segments). This was required because a
python launch loop is host-bound on this rig (~85 µs/call Triton launcher floor)
and would mask every candidate kernel.

These numbers are **probe-instrumentation kernel-only timings on a different
measurement path than the Verifier's kineto census** — the control arm reads
261.7 µs here against the Verifier's 55.80 µs for the same kernel. **They must
not be quoted as authoritative device time; the Verifier's census owns that
number.**

The sweep **ranking remains valid**, because all 6 configs were measured on the
same probe path — the ordering is sound even though the absolute values are not
census-comparable. The candidate's own binding is unchanged by the probes.

---

## What the Verifier should focus on

1. **Adopt `best_num_warps = 1`** (`_BEST_NUM_WARPS = 1`, module-level literal
   so the harness AST loader retains it). FR-4 fires: the sibling `nw2` prior
   does **not** transfer.
2. **D1 is settled — no action needed.** Orchestrator ruled the implementation
   compliant; the reduction.sum waiver was never exercised, so there is no
   waiver or disposition entry to carry. The `tl.sum` sites are int32 index-tile
   reductions only.
3. **The device lever is `grid 8 → 88`**, delivering 261.7 → 27.4 µs on the
   GEMM under probe timing (9.5×). The remaining pipeline cost is dominated by
   the frozen aten routing prelude (~64 µs), which the decision declared out of
   scope.
4. **Re-derive device time under your own kineto census.** The probe numbers
   above are ranking-grade, not census-grade (see D4).
5. **Re-measure wall time under the intact harness with adequate warmup**;
   `--warmup 0` reports cold-start graph capture (42 ms) and is not
   representative. Steady state across repeated runs: 0.219–0.222 ms,
   14.56×–14.72×.

---

## Binding ledger (Coder-owned artifact — NOW SUPPLIED)

`rounds/binding_001.json` was missing at Verifier dispatch; the Verifier
correctly refused to fabricate it and left `binding_sha256` all-zero with
precondition `missing`. It is now supplied. It is a **Coder-owned** artifact
(prompts/coder.md:96), so the omission was mine to fix.

| field | value |
| --- | --- |
| path | `rounds/binding_001.json` |
| sha256 | `2ea87f554a8fa8dc7d30bb411e9ae849e2d05a6059e04271787d4504b930c506` |
| candidate sha256 | `da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7` (unchanged) |
| decision_sha256 | `62820af457c7b0b84232dc28bffd07009b5bc1ee482059728da06761381fd1d5` |
| sketch_sha256 | `6a46d4fd67b0cbce7a34ce41eac0c2b4cc19f00dd6e6098cf91a60e879634cb4` |
| conformance | **`validate_binding.py` → VALID**, 21/21 sketch statements covered |
| generator | `log/probes/p04_r001_binding_ledger_build.py` |

**Every source span in the ledger is DERIVED from the candidate's own AST**
using the validator's own `_analyze_python_ast`, not hand-written — so the
spans cannot drift from the bytes, and re-running the generator reproduces and
re-validates the ledger. The generator also computes the three audits below
over the FINAL bytes and writes them to
`log/probes/p04_r001_binding_audits.json`.

### Status breakdown (honest accounting)

21 bindings: **17 `implemented`**, **4 `accommodated`** —

| statement | why accommodated |
| --- | --- |
| `op_bucket_count` | Realized as `tl.sum(tl.where(...))`. The frozen profile declares **no `reduction.sum` contract at all**, so the reduce cannot bind to a declared capability; the binding records the declared primitive that builds the counted mask and declares the site accommodated rather than claiming a contract the profile never granted. |
| `op_bucket_scan` | Same: `tl.sum(tl.where(be < b, counts, 0))` for the exclusive prefix. |
| `op_act` | SiLU is `tl.sigmoid`, but the profile maps `arithmetic.elementwise` to **`tl.exp` only**, and `tl.exp` does not appear in this source. No contract_name could honestly be bound. |
| `op_accum` | Accumulation is `tl.atomic_add`, which is **not** a profile-declared implementation symbol; the binding records the store contract and cites the atomic site in its notes. |

This is the honest reading: the frozen capability matrix is coarse, and a
two-kernel fused Triton implementation uses primitives it never enumerated.
These are reported as accommodations, **not** as undeclared-capability claims
and **not** as waiver usage.

### Audits over the FINAL bytes

- **DANGER token scan — all four families ZERO as designed**: compile-family
  strings 0 (an initial raw-grep version false-positived on the word
  "benchmark" inside the `_is_target_regime` **docstring** at line 306; the
  shipped scan is AST-aware and strips docstrings/comments), `.contiguous()` 0,
  graph-pool return 0, host data-dependent grid 0, host D2H reads inside
  `_pipeline` 0.
- **Dot-shape audit**: exactly 3 `tl.dot` sites — `op_dot_gate` (M16 N64 K128),
  `op_dot_up` (M16 N64 K128), `op_dot_down` (M16 N128 K64) — all fp16 operands
  with fp32 accumulators. Re-qualification: **2.441e-04 max_abs** vs 1e-2
  tolerance, **0 out-of-tolerance**, verdict exactness-POSITIVE on fp16-extreme
  operands (mm_encoder's negative does not transfer to these contractions).
  `int64→int32` kernel-side narrowing (profile status was `unknown`) is
  **qualified** — verified by the counting sort reproducing host ground truth on
  18 id vectors across 6 activation patterns.
- **num_warps audit**: 2 launch sites, both bound to the single pinned constant
  `_BEST_NUM_WARPS = 1` (`best_num_warps=1` per the p02 sweep; nw1 92.855 vs
  nw2 122.253, a 24.4% margin, so it won outright and the tie-break was not
  invoked). FR-4 fires: the sibling nw2 prior does not transfer.
- **Static-grid audit**: sort grid `(1,)`, expert grid `(E, num_tiles)` = `(8,11)`
  at the target shape with `num_tiles` computed from SHAPES at line 370; no host
  data-dependent grid; on-device empty-tile exit via `if tile_n > 0` on a
  register loaded device-side. ATEN preserved at lines 373 (softmax),
  374 (topk), 376 (renormalize), 377/382/383 (fp16 casts).

### Reconciliation with the Verifier's FR-2 finding

The Verifier found the shipped code **correct** and attributed the FR-2 device
delta to a **control-methodology artifact** — their eager control re-allocates
sort buffers every call. I concur, and the ledger is consistent with that
reading: nothing in the FR-2 delta is a candidate defect, and no candidate
change is warranted.

Specifically, the candidate's eager (tier-3) path allocates shape-correct sort
buffers per call (see the `_pipeline` buffer-selection branch), which is
required for **shape-genericity** — p03 G3 proves a non-target shape
(T=128, E=16) is served correctly, which a target-shape-only static allocation
could not do. That allocation is therefore a deliberate correctness property,
not an oversight: on the target regime (tiers 1-2) the static buffers are used
and no per-call allocation occurs. Reconciling the two readings: the eager path
trades a per-call allocation for shape generality, which is exactly the right
trade for a tier that only serves non-target shapes.

## Status

No code changes were made after the Orchestrator rulings. The candidate hash
`da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7` is final and
is the artifact to measure. `rounds/binding_001.json` is supplied and passes
conformance.
