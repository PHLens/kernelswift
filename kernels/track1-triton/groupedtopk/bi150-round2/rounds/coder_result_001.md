# Coder Result 001

## Metadata

- Round: `001`
- Classification: **`candidate-ready`**
- Result: candidate conforming to immutable Decision-001; no major deviation, no capability miss.
- Selected target profile: `triton_cuda` (frozen snapshot `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae`)
- Runtime fingerprint: `project.md#runtime-fingerprint` (triton 3.1.0 / torch 2.7.1 / CoreX 4.4.0 / Iluvatar BI-V150 sm_71) — probed live during gates, match (`Iluvatar BI-V150`, capability `[7, 1]`)
- Measurement fingerprint: `8deb1b01…` — untouched; NO wall benchmarks or profiler sessions were run by Coder (not measurement-owner this phase).

## Artifact Hashes

| Role | Path (project-root-relative) | SHA-256 |
|---|---|---|
| Decision (input, immutable) | `rounds/decision_001.md` | `93783baafdc4c4c022773e30ca2d90f7bc94e954ae25cae057fe625b7c43532b` |
| Sketch (input, immutable) | `rounds/sketch_001.json` | `637917e07b4461258ea714d42021e2e5537e21d19765b57bc9cc1552ef6f6985` |
| Canonical source start (manifest `last_accepted_kernel`) | `baseline_adapter.py` | `ecce4dacee211a86ba38584b6b78fc2f575ba60cedccdc6f79ac4f6fb0139fa5` |
| Profile snapshot (input, frozen) | `profile_snapshot/triton_cuda.yaml` | `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae` |
| Base (immutable) | `../base.py` | `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58` |
| Harness | `../../../auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |
| **Candidate** | **`triton_grouped_topk_r2_001.py`** | **`4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3`** (11318 bytes) |

All input hashes were recomputed before coding (identical to pinned values above); the smoke-time embedded candidate hash equals the final candidate hash.

## Required Sequence Evidence

1. `validate_decision.py` with manifest selected target profile:
   `python3 skills/kernel-opt-loop/scripts/validate_decision.py rounds/decision_001.md --expected-implementation-profile triton_cuda --project-root .` → JSON `valid:true`, exit 0.
2. Environment/profile identity: CoreX bootstrap performed in every CUDA shell (`export COREX_VERSION=4.4.0 && . /usr/local/corex/enable`); device live-probed `Iluvatar BI-V150`, capability `(7,1)` → not environment-blocked.
3. Sketch primitives vs snapshot statuses: see Primitive Conformance below; `tl.argmax` NOT used anywhere (Constrained tie capability never exercised normatively).
4. No Unsupported/unprovable-Unknown requirement remains: the single declared UNKNOWN consumed by stage C (`cast.narrow.int64-to-int32-kernel-side`) was resolved by one bounded probe (below). `num_warps`/`num_stages` remain Unknown ⇒ left UNSET everywhere (direct default launches only).
5. Candidate built from `last_accepted_kernel` = `baseline_adapter.py` (decomposition preserved verbatim in `_eager_forward`; Triton fast path implements sketch flow).
6. Local gate passed: ast.parse + real harness loader (`auto_bench.load_ks_module`) + warm-up/compile-smoke execution + correctness spot-checks — details under Attempt Ledger; two repair attempts were NOT needed for defects (checker-script fixes occurred inside my own probe tooling before final artifacts froze; candidate source required zero semantic repairs, zero syntax repairs after first write).
7. This document is the structured result returned via Orchestrator.

## Bounded Probe (single, Decision-scoped)

- Purpose: resolve declared UNKNOWN capability `cast.narrow.int64-to-int32-kernel-side` on-device exactly as stage C needs it, plus the stage-B construct that depends on int64 scalar loads of the retained library group-id output (promote-compare against int32 lane-group tile, OR-chain membership, arithmetic where-mask). One file-backed script, one run, no timing/profiler calls (`benchmark_or_profiler_invoked:false` recorded).
- Command (exit 0):
  ```
  cd <root>/log/probes && export COREX_VERSION=4.4.0 && . /usr/local/corex/enable \
    && python3 probe_cast_narrow_int64_int32.py --out cast_narrow_probe_result.json   # EXIT:0
  ```
- Artifacts:
  - script `log/probes/probe_cast_narrow_int64_int32.py` @`e465c86aec5d33f0657a29efb123d44bcd16090b45c69e927cc33c789a8bfa2c`
  - result `log/probes/cast_narrow_probe_result.json` @`8f225a3eb8c881faee8bad5f2e6ad4232c31c2bfe48a694bfc136d332af07f50`
- Outcome: `summary=evidence-ready`
  - `cast.narrow.int64-to-int32-kernel-side`: level=`observed`, numerically_checked=true — int64 lanes from a real `torch.topk` indices tensor narrowed with `.to(tl.int32)` in-kernel and stored into an int32 tensor; exact match vs `.to(torch.int32)` reference.
  - `memory.load.int64-scalar-membermask`: level=`observed`, numerically_checked=true — shape-replica of stage-B over 6×256 with real `torch.topk(group_scores,k=4)[1]` int64 scalar loads; finite-value equality, identical `-inf` pattern, member score bits untouched.
- Consequence: stage-C implements kernel-side narrowing; NO host-side narrowing fallback was needed (permitted fallback unused).

## Compile-Smoke and Correctness Gate (warm-up/compile smoke, no timing)

- Script: `log/probes/coder_smoke_r2_001.py` @`4c7e7737f8740bd7ff0a7d465b35cf558e71dd6f057d7e3e091d403bc3c25523`
- Result: `log/probes/coder_smoke_result.json` @`09cd56ba499dbff02acd91687cf6cbf7f258b6657e71bd2807e731c2b5daf066`, `all_pass:true`
- Command lines + exit codes:
  ```
  cd <root>/log/probes && export COREX_VERSION=4.4.0 && . /usr/local/corex/enable \
    && python3 coder_smoke_r2_001.py --out coder_smoke_result.json                   # EXIT:0
  cd <root>/log/probes && python3 binding_statement_r2_001.py                        # EXIT:0
  ```
- Checks (all PASS): `ast-parse` (11318 bytes clean); `harness-loader` (candidate AND `../base.py` through the real AST-filtered loader); `warmup-compile-smoke` (first fast-path forward compiled+executed all three Triton stages on cuda:0 `Iluvatar BI-V150`, outputs match base); correctness spot-checks vs base.Model — seed42-regime random (ids_exact, weights allclose atol=rtol=1e-2, max_abs_w=5.96e-08), plus the three Decision-declared tie suites used as test-design guidance: all-equal (ids exact, max_abs_w=0.0), two-expert-tie (ids exact), structured-group-tie (ids exact, max_abs_w=0.0), duplicate-max-pairs (ids exact); outputs remain fp32 [83,8] weights + int32 [83,8] ids.

## Binding-Checker Statement

Artifact: `log/probes/binding_statement_report.json` @`5fbddd0d6f9f267783a4dc0e9b610082415d89bd64f9eb50b7bfad7d66d511d0` (generator `binding_statement_r2_001.py` @`9141979648cfdb1bdb788cfc7db29c6519e4b280c1ad5f3fdafb4b818bebaa07`; pure AST, offline). Machine verdict `all_checks_pass=true`. Statement:

1. **run_out surface confirmed**: `ModelNew.run_out(gating_output, topk_weights, topk_ids)` is callable with exactly those parameters; it completes its writes into the caller-provided buffers before returning (return value ignored, matching `auto_bench.make_profile_call` contract read at auto_bench.py lines 516–536; `model.run_kwargs` defaults `{}` via getattr when absent). Verified live: buffers written in place and bitwise-equal to forward outputs for identical inputs; a re-run over pre-poisoned buffers reproduces identical results (no cross-call caching of outputs or temporaries).
2. **Retained topk semantics confirmed**: `_triton_forward` contains EXACTLY two `torch.topk` call sites — site 1 `torch.topk(group_scores_out, k=self.topk_group, dim=-1)[1]` feeding stage-B, site 2 `torch.topk(masked_scores, k=self.topk, dim=-1)` returning (vals fp32[T,K], ids int64[T,K]) feeding stage-C; both preserve base argument values/shapes/dtypes/dim/ordering/tie behavior. Two additional sites exist ONLY inside `_eager_forward`, which mirrors `base.py` byte-for-byte for off-regime configs (scoring_func preservation invariant) and is outside the changed computation boundary.
3. **Per-statement source mapping**: every Triton-relevant Sketch load/compute/store statement has an exact implementation span recorded in the report (14 entries incl. narrowing method-call), with occurrence notes composing softmax rowmax/sum/divide and membership floor-divide/OR-chain constructs. Statements `op.store.lib_topk_group` and `op.store.lib_topk_expert` are RETAINED LIBRARY CALLS realized by the two preserved `torch.topk` sites (see report `library_statements`). Note on model limits: the frozen profile's `binding_model=primitive-call` expresses tl.* primitive spans only and carries no library-symbol mapping; therefore these two statements are bound by explicit call-site evidence here rather than by the deterministic ledger checker, consistent with `fallback_and_unknown_policy.waiver_requirements` ("retaining library torch.topk calls … requires no waiver"). `op.compute.narrow_ids` binds to the register method-call `ids_tile.to(tl.int32)`, evidenced by the bounded probe rather than a tl.* symbol entry.
4. **Forbidden/unknown construct absence**: zero `tl.argmax` calls; zero `num_warps`/`num_stages` keyword arguments module-wide; no `triton.autotune`; direct launch syntax `_kernel[(grid,)](...)` only.

## Primitive Conformance Notes (conformance, not design changes)

- Snapshot-supported constructs used within proven scope: contiguous fp32 load/store extent 256 (`E=256`, mask-free full-extent rows), extent-8 store family (K=8 outputs, G=8 group scores via `tl.arange(0,GROUPS)`), axis-0 float32 reductions over (256,), reshape `(256,)→(8,32)` with axis-1 max, elementwise exp/select (`tl.where`), axis-0 program_id, one-dimensional grid launches.
- `tl.sum` appears in the Markdown rendering as Supported but has no entry in the frozen machine-readable matrix (v1 `profile_status: partial`); it is used by stages A/C exactly as evidenced by epoch-1 accepted lineage candidates on this runtime, and validated numerically by this round's probes/smoke. Recorded here as conformance-note context, not silently omitted.
- `tl.static_range(KG=4-iter or fewer)` matches the proven four-iteration compile-time loop pattern from the grouped probe; OR-chained membership replicates the epoch-1 `_group_mask_kernel` pattern field-proven correct on this runtime/harness.
- Off-regime guard (`_fast_path_applies`) routes non-softmax/other-shape configs to the preserved base decomposition — this keeps invariant "scoring_func parameter semantics preserved" without touching the fast-path design; harness regime always satisfies the guard.
- `routed_scaling_factor` multiplies unconditionally with runtime value 1.0 in the regime config (bitwise identity x*1.0=x); when renormalize=False constexpr removes the divide entirely. Both behaviors equal the base conditional-multiply semantics.
- Outputs: `forward` returns freshly allocated fp32/int32 tensors on `gating_output.device`; temporaries (`scores_out`, `group_scores_out`, `masked_scores`) are per-call fresh allocations; instance state holds only config constants (no cross-instance/cross-call tensor caching) per Host Plan.

## Attempt Ledger

| # | Step / Command | Exit | Defect observed | Candidate SHA before → after |
|---|---|---|---|---|
| 1 | validate_decision.py gate (above) | 0 | none | not-yet-created |
| 2 | Narrowing/membership bounded probe run 1 | 0 | none (script warnings: unclosed-file ResourceWarnings only; result complete) | n/a (probe stage) |
| 3 | coder_smoke run 1 | 0 | none — all 11 checks PASS on first execution of the candidate | `4ae64cad…` (creation) → `4ae64cad…` (unchanged; zero repairs) |
| 4 | binding_statement generation run 1→3 | 1→2→0 | Defects confined to MY checker script (dropped base Name in dotted-symbol helper; annotation-exact param compare; `dim=-1` UnaryOp predicate; slice join bug). No candidate edit involved. | `4ae64cad…` → `4ae64cad…` |

Same-round Verifier repair budget: untouched (zero requests so far).

## Deviation Classification

- **Major deviation: NONE.** Implementation follows Optimization Intent, Unified Sketch, Host Plan, and Evaluation Contract of the immutable Decision-001 without algorithm/dataflow/lifecycle/Evaluation-Contract change.
- **Capability-miss: NONE.** The only consumed Unknown was proven locally by the bounded probe; Unset-status hints were honored by omission (no num_warps/num_stages anywhere).
- Fallback-per-decision use: permitted host-side narrowing fallback was NOT required (probe passed) — implemented nothing host-side beyond the sketch.
- The off-regime eager fallback and unconditional ×1.0 scaling multiply are conformance accommodations preserving all normative semantics (Small accommodation clause), detailed above.

## Return

Classification `candidate-ready`; candidate path `<root>/triton_grouped_topk_r2_001.py`, sha256 `4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3`. All local checks and probe artifacts live under `log/probes/` with hashes listed above. Awaiting Orchestrator routing to Verifier; measurement ownership remains with Verifier.
