# Coder Result 003

## Metadata

- Round: `003`
- Classification: **`candidate-ready`**
- Result: candidate conforms to immutable Decision-003 (`compile-graph-replay-reduce-overhead`); no major deviation; no capability miss; supersession clause applied exactly as authorized (mode-only escalation).
- Canonical lineage: derived from ACCEPTED round-002 candidate `triton_grouped_topk_r2_002.py` @`ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12` (manifest `last_accepted_kernel`), hash re-pinned before coding.
- Runtime fingerprint: unchanged environment, device probed live (`Iluvatar BI-V150`, cap `(7,1)`) — not environment-blocked.
- Measurement fingerprint: untouched — NO wall benchmarks or profiler sessions run by Coder. perf_counter reads exist ONLY as explicitly-permitted bounded capture/replay-completion sanity inside the smoke, labeled non-performance.

## Artifact Hashes

| Role | Path | SHA-256 |
|---|---|---|
| Decision (input, immutable) | `rounds/decision_003.md` | `e214c29aa66d78654ffb65fba33b4870379bcf059902c8f7cc6409ebffc3a403` |
| Sketch (input, immutable) | `rounds/sketch_003.json` | `4a909a11cbd8df0ad0385cf6379dc77eb189bffd60ec2ab1b341dbdaa127a782` |
| Canonical source start | `triton_grouped_topk_r2_002.py` | `ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12` |
| Accepted report reference | `rounds/report_002.md` | `bd0932b9cae83a55e0d63f3b149f77937c143100e62e62daf28e850f97ca36ce` |
| Profile snapshot (frozen) | `profile_snapshot/triton_cuda.yaml` | `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae` |
| Capability claim (ref'd) | `profile_snapshot/capability_claim.json` | `2e6ee49ddd887a00e9a8a8ef6dfc746984ecaacd2256ee0b8666a3099a5b7f67` |
| **Candidate** | **`triton_grouped_topk_r2_003.py`** | **`62f8883a2c6d1bdf65d84b29beb71d95500b40b8d6acaf484eb09fccdcf97d38`** (16904 bytes) |

## Required Sequence Evidence

1. `validate_decision.py` for decision_003 with manifest selected target profile → exit 0.
2. Environment identity: CoreX bootstrap before every CUDA shell; device live-probed.
3. Supersession discipline: `dynamic=False` retained on BOTH tiers; no precision/backend/env/cache knobs anywhere; frozen segments byte-proven; two vendor top-k sites untouched.
4. Local gate passed end-to-end after ONE design repair triggered by a probe-discovered framework hazard (see Deviations): candidate executed once with pool-backed outputs (design flaw caught by my own gate at warm-replay read), then routed through externally-owned buffers — final bytes 18/18 PASS with zero further repairs. Checker-side fixes confined to my probe tooling.
5. This document is the structured result returned via Orchestrator.

## Compile-Smoke and Correctness Gate

- Script: `log/probes/coder_smoke_r2_003.py` @`131d030ee7810ea87bd8ee3e3af58ee4d7432d36d012982ac8b3b6d21b6652b0`
- Result: `log/probes/coder_smoke_result.json` @`e6414ad0364a0e701c1000a273f4dc132bc3fe3362bce2fa014f47907095a366`, `all_pass:true`, **18/18 checks PASS**
- Command lines + exit codes:
  ```
  cd <root>/log/probes && export COREX_VERSION=4.4.0 && . /usr/local/corex/enable \
    && python3 coder_smoke_r2_003.py --out coder_smoke_result.json      # EXIT:0
  cd <root>/log/probes && python3 binding_statement_r2_003.py           # EXIT:0
  ```
- **Cold-capture smoke + warm-replay repeat** (through REAL harness loader on candidate + r002 anchor + base):
  - cold-capture-completion sanity **0.562 s** (first target-regime forward builds the replay tier via `torch.compile(self._triton_forward, mode='reduce-overhead', dynamic=False)`), ids exact vs base;
  - warm-replay sanity **0.001 s**, bit-identical repeat, replay tier still active;
  - both numbers are bounded-completion SANITY ONLY — not performance evidence.
- **Bitwise sweep vs accepted r002 THROUGH THE REPLAYED ROUTE** (all `(w,i)` True/True):

  | Case | bitwise weights==r002 | bitwise ids==r002 | ids exact vs base |
  |---|---|---|---|
  | seed42-regime-random | true | true | true |
  | all-equal | true | true | true |
  | two-expert-tie-same-group | true | true | true |
  | structured-group-tie-boundary | true | true | true |
  | duplicate-max-pairs-cross-group | true | true | true |
  | warm-replay NEW input bytes (seed 31415) | true | true | — |

  Sweep-end tier assertion: `_replay_failed=False`, `_replayed_staged is not None` — every sweep genuinely traversed the REPLAYED route (no silent down-tier).
- **Cross-instance alternation**: interleaved calls across two instances stay bitwise-correct to their per-input anchors.

## Fallback-Edge Exercises (both edges, plus selectivity)

1. **Edge A — forced capture failure → compiled-default permanent**: injected raising callable as the replay handle. Single call bound down permanently exactly ONCE: `_replay_failed=True`, handle dropped, `_compile_failed=False`, default-tier handle present; results across TWO subsequent calls bitwise==r002 (compiled-default route); poison invoked exactly 1× total. ✔ matches "any exception … binds permanently to the next tier".
2. **Edge B — stacked failure (replay AND default broken) → unmodified staged cascade**: one call cascaded replay-poison → default-poison → staged execution returning bitwise==r002 results; BOTH flags permanently true, both handles dropped, each poison invoked exactly once; second call goes straight to staged. ✔ monotonic downward transitions, never recover upward within the instance.
3. **Selectivity / recovery**: T=41 non-target input executed the staged tier with ZERO compiler artifacts created (all handles None, flags False, bitwise==r002 staged outputs, base-consistent); the SAME instance afterwards entered the REPLAYED tier on [83,256] and matched anchors. Tier flags never moved during guard-routed traffic. ✔ mechanism_observable `fallback_tier_selectivity_and_recovery`.

## Declared-Risk Probes (bootstrap items)

- **CoreX torch graph capture with retained library ops**: cold capture completed on this runtime with the vendor `torch.topk` pair inside the traced region; replay tier remained healthy through the entire sweep (no recapture pressure, no exceptions). Framework emitted `skipping cudagraphs due to mutated inputs` for buffer-carrying invocations — see Conformance Notes for interpretation; correctness is unconditional regardless of which sub-behavior the framework selects.
- **Static input aliasing vs run_out caller-buffer mapping**: proven SAFE under the shipped design. run_out-FIRST ordering (cold, capture constructed while carrying caller buffers) rewrote poisoned (-9e30/-7) buffers IN PLACE with `data_ptr` preserved, bitwise==r002 outputs; forward-first ordering then warm run_out over freshly poisoned buffers likewise byte-exact. The earlier pool-output overwrite failure was structurally eliminated by routing ALL entry points through externally-owned output tensors (forward allocates fresh caller-side buffers per call before entering the chain) — hence decision hazard (ii) cannot reach any consumer, and replay outputs equal r002 by construction + measurement.

## Binding-Checker Statement

Artifact: `log/probes/binding_statement_report.json` @`b32eb677d43b7d2ad51cb4ec140aae4661495a1ce027098c2ff77301adafe1c7` (generator `binding_statement_r2_003.py` @`2326f7d97ef76019b58bcf789245402e26fe5bf4a14b97a95097605f5e802408`, pure AST offline; verdict `all_checks_pass=true`):

1. **Frozen-segment inheritance**: SEVEN segments BYTE-identical AND AST-identical versus accepted r002 — the three @triton.jit kernels, `_triton_forward` (hence both `torch.topk` sites with exact args), `_eager_forward`, `_fast_path_applies`, AND `run_out`. Per-segment sha256 recorded in report.
2. **Mode-token audit** (whole-file counts, so no docstring/comment can hide configuration): `'reduce-overhead'` ×1 (tier-1 kwarg only); `'default'` ×1 (tier-2 kwarg only); `mode=` ×2; `dynamic=False` ×2; torch.compile call sites ×2 with keyword sets exactly {mode:'reduce-overhead', dynamic:'False'} and {mode:'default', dynamic:'False'} — i.e., the mode-only supersession is the entire compile-config delta and every other decision_002 restriction carries over verbatim.
3. **Allowlist compliance carried over**: forbidden-token scan all-zero (reduced_precision_reduction, allow_tf32/tf32, torch.backends, TORCHINDUCTOR, TRITON_CACHE, backend=, max-autotune, num_warps, num_stages, tl.argmax, autotune, matmul_precision, set_float32, getenv).
4. **Routing surfaces**: tier handles/flags present (`_replayed_staged/_replay_failed/_compiled_staged/_compile_failed`); shared helper entered from BOTH entry points (run_out passes its three positional args; forward enters with fresh externally-owned buffers); constructor/forward/run_out signatures unchanged; class-surface delta isolated to routing methods (informational list in report).

## Attempt Ledger

| # | Step / Command | Exit | Defect observed | Candidate SHA before → after |
|---|---|---|---|---|
| 1 | validate_decision gate + hash pinning | 0 | none | not-yet-created |
| 2 | coder_smoke run 1 | 1 | PROBE-DISCOVERED DESIGN FLAW: first-write design returned pool-backed tensors from the replay tier; warm-replay consumer read raised the framework's overwrite protection (declared hazard ii made concrete) | creation @16060B → unchanged |
| 3 | Design repair #1: forward routes fresh EXTERNALLY-OWNED output buffers into the shared chain (run_out already did); rerun smoke | 0 | All 15→18 checks PASS; framework logs "skipping cudagraphs due to mutated inputs" on buffer paths (graceful, correct) | `62f8883a…` pre-hash state → same-shape source |
| 4 | Audit-hygiene text edit (docstring de-literalization so quoted-mode counts audit only real kwargs; restored exact r002 bytes of run_out comments) | — | prose-only; behavior byte-frozen segments untouched | `62f8883a…` final |
| 5 | Final smoke rerun on finalized bytes | 0 | none — 18/18 PASS | `62f8883a…` (unchanged thereafter) |
| 6 | binding_statement runs 1→2 | 2→0 | Defects confined to MY generator (over-broad 'environ' token matching English word; stale bare-substring expectations). Candidate semantics never implicated. | `62f8883a…` → `62f8883a…` |

Same-round Verifier repair budget: untouched (zero requests so far).

## Deviation Classification

- **Major deviation: NONE.** Mode-token supersession is the sole authorized delta; all inherited computation segments are machine-proven byte-frozen; three-tier chain, strict gating, permanent monotonic fallbacks implemented exactly per Host Plan.
- **Design repair disclosure (conformance note, Small accommodation clause)**: initial within-gate iteration returned framework pool-backed outputs from the replay tier; this violated no Decision line literally but collided with declared hazard (ii) (graph-pool output placeholders) once consumers outlive one replay step. Final design returns EXTERNALLY-OWNED tensors from every tier: forward allocates fresh [83,8] fp32/int32 buffers per call (identical allocations `_triton_forward` would have made internally — net-zero kernel delta vs accepted r002) and run_out keeps caller-buffer plumbing. Output contract unchanged; hazard eliminated structurally rather than tolerated.
- **Framework observation recorded for Verifier attribution scoping** (not an anomaly, per decision's branch-B contract): buffer-carrying invocations log `skipping cudagraphs due to mutated inputs (N instances)` — under THIS CoreX build the mutated-input graphs fall back to default-equivalent compiled execution automatically while results remain bitwise-equal (proven). Non-mutating segments may still capture ("2 instances" refers to skipped graphs). Consequence preview for Verifier: attributed kernel-count branch depends on how much captured work remains; the decision's TWO-BRANCH PASS rule covers either outcome; bitwise equivalence is unconditional here.
- No env overrides written; no flags flipped; graph recapture never triggered from model code.

## Return

Classification `candidate-ready`; candidate path `<root>/triton_grouped_topk_r2_003.py`, sha256 `62f8883a2c6d1bdf65d84b29beb71d95500b40b8d6acaf484eb09fccdcf97d38`. All check artifacts under `log/probes/` with hashes above. Awaiting Orchestrator routing to Verifier for authoritative H-003 wall measurement under fingerprint 8deb1b01…; measurement ownership remains with Verifier.
