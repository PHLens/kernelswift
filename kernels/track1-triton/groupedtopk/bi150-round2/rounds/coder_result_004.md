# Coder Result 004

## Metadata

- Round: `004`
- Classification: **`candidate-ready`**
- Result: candidate conforms to immutable Decision-004 (`manual-cuda-graph-workspace-replay`); no major deviation; no capability miss; retired tier absent by machine-audited DANGER rule.
- Canonical lineage: derived from ACCEPTED round-002 candidate `triton_grouped_topk_r2_002.py` @`ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12` (manifest `last_accepted_kernel`; r003 is RETIRED evidence — nothing was derived from it), hash re-pinned before coding.
- Runtime fingerprint: unchanged; device probed live (`Iluvatar BI-V150`, cap `(7,1)`).
- Measurement fingerprint: untouched — NO wall benchmarks or profiler sessions run by Coder. perf_counter reads exist ONLY as explicitly-permitted bounded capture/replay-completion sanity inside the smoke, labeled non-performance.

## Artifact Hashes

| Role | Path | SHA-256 |
|---|---|---|
| Decision (input, immutable) | `rounds/decision_004.md` | `e5465d7dfdbc35cdba8251b9d43a5d43eb05c64d63c57d89eb299723b0be3be1` |
| Sketch (input, immutable) | `rounds/sketch_004.json` | `ccf277f422ce254d09dc1402c997a6c311a1f63457423f23afd60a71b4d9ae59` |
| Canonical source start | `triton_grouped_topk_r2_002.py` | `ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12` |
| Profile snapshot (frozen) | `profile_snapshot/triton_cuda.yaml` | `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae` |
| **Candidate** | **`triton_grouped_topk_r2_004.py`** | **`c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb`** (19783 bytes) |

Smoke-time embedded candidate hash equals the final candidate hash.

## Required Sequence Evidence

1. `validate_decision.py` for decision_004 with manifest selected target profile → exit 0.
2. Environment identity: CoreX bootstrap in every CUDA shell; device live-probed.
3. Sketch primitives vs snapshot statuses: all computation primitives inherited byte-frozen; NEW capability consumed this round = manual capture/replay machinery (`torch.cuda.CUDAGraph`, side-stream warmup, static workspace) under the Decision's own Host Plan prescriptions; retired compile-replay tier ABSENT.
4. No Unknown beyond those resolved in rounds 001–003; fallback chain normative and proven permanent-once at BOTH edges plus a genuine construction-failure edge.
5. Candidate built from `last_accepted_kernel` with SEVEN verbatim segments + byte-frozen forward (see binding statement).
6. Local gate 21/21 PASS on finalized bytes; two probe-side setup artifacts were repaired inside MY tooling only (detailed in Attempt Ledger); candidate semantics never repaired after first write.
7. This document is the structured result returned via Orchestrator.

## Compile-Smoke and Correctness Gate

- Script: `log/probes/coder_smoke_r2_004.py` @`718cadd853a28a1a7e408269de75ee8f6b7cd7ecc8f1b46e77c88ed6ddf18925`
- Result: `log/probes/coder_smoke_result.json` @`54d14d896e7de0e6e7c6357a7d92ad724f91800324523aab9bc1db4b886e638f`, `all_pass:true`, **21/21 checks PASS**
- Command lines + exit codes:
  ```
  cd <root>/log/probes && export COREX_VERSION=4.4.0 && . /usr/local/corex/enable \
    && python3 coder_smoke_r2_004.py --out coder_smoke_result.json      # EXIT:0
  cd <root>/log/probes && python3 binding_statement_r2_004.py           # EXIT:0
  ```
- **Cold-capture smoke + warm-replay repeat** through the REAL harness loader (candidate + r002 anchor + base):
  - cold-capture-completion sanity **0.139 s** (workspace allocation → 3 warmup iterations on a side stream → single `torch.cuda.graph(...)` capture, all INSIDE one builder call at first target-regime use); ids exact vs base;
  - warm-replay sanity **0.001 s**, bit-identical repeat;
  - both numbers are bounded-completion SANITY ONLY — not performance evidence.

## Capture-Fired Proof (bootstrap demand: "document HOW you proved firing")

Proved WITHOUT profilers or timing, by intersecting four independent behavioral facts on the SAME instance:

1. **Handle alive & typed**: after serving correct results, `_manual_graph` is a real `torch.cuda.CUDAGraph` instance and workspace tensors exist (`_ws_gating/_ws_out_weights/_ws_out_ids`), with `_replay_failed=False`.
2. **Both lower-tier artifacts ABSENT**: `_compiled_staged is None` AND `_compile_failed is False` throughout the entire sweep — since tier-2 builds lazily ONLY when reached, correct answers CANNOT have come from any lower tier; tier 1 is the only surviving route that produced them.
3. **Recalculation stale-trap**: outputs differ across different inputs within the same instance and each matches its per-input r002 anchor bitwise — a dead/graphless path could not produce input-dependent bitwise-correct results twice while bypassing both lower tiers.
4. **Tier separation is detectable**: the two forced-edge injections below PROVE that whenever execution leaves tier 1, observable state changes (flags flip once, handles drop, lazily-built default handle appears). Therefore non-demotion during the sweep is itself evidence tier 1 served every call.

Result: `capture-fired-proof=true`, `replayed-tier-active-at-sweep-end=true`.

## Bitwise Sweep vs Accepted r002 THROUGH THE REPLAYED ROUTE

| Case | bitwise weights==r002 | bitwise ids==r002 | ids exact vs base |
|---|---|---|---|
| seed42-regime-random | true | true | true |
| all-equal | true | true | true |
| two-expert-tie-same-group | true | true | true |
| structured-group-tie-boundary | true | true | true |
| duplicate-max-pairs-cross-group | true | true | true |
| warm NEW-bytes (seed 31415) | true | true | — |

## Fallback-Edge Exercises

1. **Edge A — TRUE capture failure (construction raises)**: `m._build_manual_graph` replaced pre-first-call by a raiser. First target-regime call: construction exception caught ONCE, `_replay_failed=True` permanently, partial workspace refs dropped, and the compiled-default tier was LAZILY built INSIDE that same call — results bitwise==r002 served from it; second call stayed on default tier. ✔ bootstrap edge (capture failure → compiled-default permanent-once).
2. **Edge A2 — invocation failure on a LIVE handle** (realistic post-capture swap): genuine warmup+capture ran first; then `_manual_graph` swapped for a raiser exposing `.replay()`. The call performed copy-in over the intact workspace, hit the poisoned replay EXACTLY once, bound down permanently, dropped graph+workspace refs, and lazily built+served via compiled-default (bitwise==r002). ✔ same-call correctness preservation per Decision ("this call still returns correct results from the surviving tier").
3. **Edge B — stacked failure → staged cascade**: real capture, then BOTH upper handles poisoned. Single call cascaded tier-1(poison 1×)→tier-2(poison 1×)→unmodified staged returning bitwise==r002; both flags permanently true, handles cleared; next call went straight to staged. ✔ monotonic downward transitions, never recover upward.
4. **Selectivity / recovery**: T=41 off-regime produced ZERO artifacts (no graph, NO compilers, workspace never allocated — `hasattr(_ws_gating)` false) with staged results bitwise==r002 and base-consistent; the SAME instance then warmed+captured on [83,256] and served tier-1 anchors. ✔ mechanism_observable `fallback_tier_selectivity_and_recovery`.

## run_out == forward (poisoned buffers, both orderings)

- Ordering A — run_out-FIRST cold: warmup+capture fired while caller buffers were already supplied; copy-out wrote INTO them directly: poisoned (-9e30/-7) buffers rewritten IN PLACE, `data_ptr` preserved, bitwise==r002 anchors.
- Ordering B — forward-first warm: fresh forward then re-poisoned buffers rewritten in place, byte-exact vs anchors.
- Workspace-leak trap: a LATER forward on the run_out-using instance returned FRESH invocation-owned buffers (distinct data_ptr) carrying identical bits — workspace never leaks into user-visible results.
- Cross-path contract holds under the manual tier exactly as under prior accepted rounds (`run_out==forward bitwise for identical inputs`).

Declared-risk probes from the bootstrap, all exercised and green: stream discipline (side-stream capture ONCE at first use; subsequent calls submit replay on the caller's current stream context — legality documented and behaviorally stable across every case incl. alternation), first-call allocations strictly OUTSIDE the capture window (workspace before warmup; graph-pool intermediates arise only inside capture per supported pattern), warmup-before-capture requirement (3 iterations), and library-op capture safety (gatherTopK/bitonicSort pair captured successfully — had ANY component failed, the tier would have bound down permanently instead of partially capturing; Edge A/A2/B prove that path works).

## Binding-Checker Statement

Artifact: `log/probes/binding_statement_report.json` @`1e6b44a5d6db200d91a7686dea39069046e7e184c38de83eb54444a693ddf9bc` (generator `binding_statement_r2_004.py` @`a221f890797a553dc32107267e34122cd8309e7f6781ec0be47f5c5e7de8aafc`, pure AST/offline; verdict `all_checks_pass=true`):

1. **Frozen-segment inheritance**: SIX Decision-named segments BYTE+AST identical vs accepted r002 — three @triton.jit kernels, `_triton_forward` (both retained torch.topk sites with exact args), `_eager_forward`, `_fast_path_applies`. Additionally `forward` is ALSO byte-identical (stronger than required).
2. **run_out delta disclosure**: AST-identical to r002; sole byte delta is the updated THREE-LINE routing comment stating the manual-tier copy-out boundary (frozen set per Decision does not include run_out; recorded explicitly, not silently).
3. **DANGER rule honored**: token `reduce-overhead` appears **0 times** anywhere in the candidate source (retired tier fully absent); checker FAILS on counts>0 by construction of this audit.
4. **Compile-config inventory**: EXACTLY one torch.compile site with keywords exactly {mode:'default', dynamic:'False'} — the LAZY compiled-default fallback tier; zero other compile configurations exist.
5. **Carry-over allowlist scan all-zero**: reduced_precision_reduction, allow_tf32/tf32, torch.backends, TORCHINDUCTOR, TRITON_CACHE, backend=, max-autotune, num_warps, num_stages, tl.argmax, autotune, matmul_precision, set_float32, getenv, lowercase 'cudagraph' prose (the torch.cuda.CUDAGraph API name is capitalized and contains none of the banned substrings).
6. **Workspace/replay contract structure (machine-checked spans)**: builder allocates workspace BEFORE the capture window (empty_like precedes Stream()/warmup/capture in source order), performs 3 warmup iterations then ONE `torch.cuda.graph(graph_handle)` capture; `_manual_replay_call` ordering enforced copy-in(gating) → `.replay()` → copy-out(out_weights/out_ids) → return; tier flags present with lazy down-tier construction; shared route entered from BOTH public surfaces; constructor/forward/run_out signatures unchanged.

## Attempt Ledger

| # | Step / Command | Exit | Defect observed | Candidate SHA before → after |
|---|---|---|---|---|
| 1 | validate_decision gate + hash pinning | 0 | none | not-yet-created |
| 2 | Gate script authoring (single write) | — | none | n/a (probe stage) |
| 3 | coder_smoke run 1 | 2 EDGE FAILS | Both failures were PROBE SETUP ARTIFACTS: injected handles lacked the interfaces/realism the candidate contract expects (A2/B injected a bare callable where tier-1 invokes `.replay()` over a REAL workspace; my assertion counted poison calls that could never fire). Candidate behavior was CORRECT in every respect during these runs (correct results, correct flags, correct lazy down-tier construction). | creation @19755B → unchanged |
| 4 | Probe repair: realistic injections (genuine capture first, then swap handles); poison gains `.replay()` interface matching tier-1 calling convention | 2→2 | A2 still 0× — root cause isolated to missing `.replay()` attr on the poison object (AttributeError preceded poison) | candidate untouched @19755B |
| 5 | coder_smoke rerun (poison interface fix applied mid-run sequence) | 0 | NONE — 19 then final 21/21 PASS; the source-visible comment fix for run_out routing accuracy had been applied BEFORE freeze; final bytes fixed at sha above thereafter | `c02d956c…` (creation with comment fix) → `c02d956c…` |
| 6 | binding_statement runs 1→2 | 0→0 | none (first generator version PASSED outright) | `c02d956c…` → `c02d956c…` |

Candidate semantic repairs after first complete write: ZERO. Same-round Verifier repair budget: untouched (zero requests so far).

## Deviation Classification

- **Major deviation: NONE.** Manual workspace capture/replay implemented exactly per Host Plan (ownership supersession scoped precisely: transient full-overwrite workspace ≠ result caching; all user-visible results are invocation-owned buffers filled by per-call copy-out; zero result reuse or cross-call carryover — machine-proven by the leak-trap check).
- Frozen-segment guarantee exceeded: seven of eight inherited surfaces byte-identical (incl. forward); run_out carries only its disclosed three-line comment update (AST-identical).
- **Conformance notes (no deviation)**:
  - Tier-1 result path for forward allocates ONE fresh output pair per call INSIDE `_manual_replay_call` when out args are None — mandated by the copy-out-after-boundary design ("forward returns COPY-OUT results freshly written each call"); attributed kernel-count impact lands inside the decision's TWO-BRANCH pass rule (≈3 attributable launches/call expected: 2 copy-outs + copy-in memcpys, far below 6.90).
  - `torch.topk` executes eagerly on THIS call's copy-in bits via captured pool addresses (recorded during capture); tie behavior preserved because identical kernels consume identical bit-patterns — structural bitwise equality confirmed by the sweep table.
  - Capture happens on torch's internal side-stream per the `torch.cuda.graph` context manager; replays submitted against the caller's current stream context — verified behaviorally across 30+ invocations including cross-instance alternation without a single anomaly.
- No env overrides written; no flags flipped; recapture never triggered from model code.

## Return

Classification `candidate-ready`; candidate path `<root>/triton_grouped_topk_r2_004.py`, sha256 `c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb`. All check artifacts under `log/probes/` with hashes above. Awaiting Orchestrator routing to Verifier for authoritative H-004 wall measurement under fingerprint 8deb1b01…; measurement ownership remains with Verifier.
