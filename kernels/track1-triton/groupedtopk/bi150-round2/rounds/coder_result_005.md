# Coder Result 005

## Metadata

- Round: `005`
- Classification: **`candidate-ready`**
- Result: candidate conforms to immutable Decision-005 (`boundary-dispatch-coalescing`) applied on the byte-frozen manual-replay architecture; no major deviation; no capability miss.
- Canonical lineage: derived from ACCEPTED round-004 candidate `triton_grouped_topk_r2_004.py` @`c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb` (manifest `last_accepted_kernel`), hash re-pinned before coding.
- Runtime fingerprint: unchanged; device probed live (`Iluvatar BI-V150`, cap `(7,1)`).
- Measurement fingerprint: untouched — NO wall benchmarks or profiler sessions run by Coder. perf_counter reads exist ONLY as explicitly-permitted bounded completion sanity inside the smoke, labeled non-performance.

## Artifact Hashes

| Role | Path | SHA-256 |
|---|---|---|
| Decision (input, immutable) | `rounds/decision_005.md` | `4a549653a939eafa2c36ade9b51e849633e702cdbd6d2f7463597f6257ed6021` |
| Sketch (input, immutable) | `rounds/sketch_005.json` | `21d13b983a4bf1ac1e6913bbaff635dd2932006bf9df04cd888406edcd6c92de` |
| Canonical source start | `triton_grouped_topk_r2_004.py` | `c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb` |
| **Candidate** | **`triton_grouped_topk_r2_005.py`** | **`cf68ed7713269416af5b49e901e040c7dcb97da9ec4f6eb4cc9bc5d70d288e9c`** (24214 bytes) |

Smoke/binding embedded candidate hashes equal the final file hash.

## Required Sequence Evidence

1. `validate_decision.py` for decision_005 with manifest selected target profile → exit 0.
2. Environment identity: CoreX bootstrap in every CUDA shell; device live-probed.
3. Named-delta discipline machine-audited: SEVEN frozen segments byte+AST identical vs r004 (three @triton.jit kernels, `_triton_forward` incl. both retained torch.topk sites, `_eager_forward`, `_fast_path_applies`, `_compile_regime_applies`); every other change confined to the Decision's affected_scope surfaces (guard micro-trim tuples, route dispatch in forward/run_out, strategy bind inside builder, non_blocking boundary copies, hot-callable machinery) plus three new helpers (`_route_target`, `_bind_strategy_and_hot`, `_invalidate_manual_tier`).
4. Local gate **24/24 PASS** on finalized bytes; one probe-side setup artifact repaired inside MY tooling (strategy-parity emulation initially re-probed the capability instead of forcing a genuine construction-time anomaly — corrected to a raiser-injected `_foreach_copy_`, which is also the exact semantics the Decision prescribes for branch B). Candidate source required ZERO repairs after first write.
5. This document is the structured result returned via Orchestrator.

## Compile-Smoke and Correctness Gate

- Script: `log/probes/coder_smoke_r2_005.py` @`ba0cf3c8b43f563f7f8e7eebec0efeb2d4efd0b2f1bd36eaa8106ac8b6d60ed3`
- Result: `log/probes/coder_smoke_result.json` @`1b7adf6d346622be58f6a2ba2e6823e25dd10f9d4dbcd55d48edf2575014537e`, `all_pass:true`
- Command lines + exit codes:
  ```
  cd <root>/log/probes && export COREX_VERSION=4.4.0 && . /usr/local/corex/enable \
    && python3 coder_smoke_r2_005.py --out coder_smoke_result.json       # EXIT:0
  cd <root>/log/probes && python3 binding_statement_r2_005.py            # EXIT:0
  ```
- **Cold-capture smoke + warm-replay repeat**: construction-time bind executed during build; sanity **0.118 s** / warm **0.001 s** (bounded-completion SANITY ONLY — not performance evidence); second invocation bit-identical through the SAME bound callable.

## Copy-Out Capability Branch Outcome (bootstrap ask)

- **RUNTIME BRANCH A — batched** (`copyout-strategy-branch-recorded`): `torch._foreach_copy_` bound at construction on this runtime; `_batched_copyout_ok=True`, error artifact `None`. The binder exercised BOTH demanded probes BEFORE binding: mixed int64→int32 cast parity against plain `copy_` AND batched-vs-legacy byte-parity over the REAL workspace output pair; any anomaly would have recorded an artifact and pinned legacy permanently (proven below).
- **Branch B path exercised genuinely end-to-end**: `torch._foreach_copy_` raised inside a dedicated instance's graph build → builder recorded `TypeError: probe-forced foreach capability anomaly`, pinned LEGACY two-copy non_blocking path permanently while **tier 1 SURVIVED** (graph alive, `_replay_failed=False`), and results stayed bitwise==r004 (`legacy-strategy-path-byte-parity`). ✔ "ANY anomaly binds the legacy two-copy path — never a partial mixture per call"; binding fixed at construction, never revisited at runtime.

## Guard Micro-Trim Selectivity Equivalence

`_route_target` returns chain/staged/eager with distribution IDENTICAL to the accepted `_fast_path_applies ∧ _compile_regime_applies` composition across ALL probe inputs (regime cases, T=41 staged case, fp16 eager case): precomputed constructor-derived tuples evaluated as single boolean expressions; no clause added or removed (`guard-selectivity-equivalence=true`).

## Bitwise Sweep vs Accepted r004 THROUGH THE REPLAYED ROUTE (branch A)

| Case | bitwise weights==r004 | bitwise ids==r004 | ids exact vs base |
|---|---|---|---|
| seed42-regime-random | true | true | true |
| all-equal | true | true | true |
| two-expert-tie-same-group | true | true | true |
| structured-group-tie-boundary | true | true | true |
| duplicate-max-pairs-cross-group | true | true | true |
| warm NEW-bytes (seed 31415) | true | true | — |

Plus stale-trap & fresh-buffer proof: NEW-bytes call returned results in buffers with DISTINCT data_ptr vs prior call (workspace never leaks); sweep-end state: tier flags clean, compiled-default NEVER constructed.

## Boundary Host Trip Count Observable (two-branch demonstration)

Artifact: `log/probes/boundary_trip_census.json` @`e289a5911011e33f32d8cd43631da6aceedf4315a469a1cdf1eb6be1d161e15c`.
- Runtime branch A bound ⇒ per-call boundary tensor-op trips = **2** (one non_blocking input copy-in + ONE batched output copy replacing two separate dispatches) — satisfies the branch-A clause "<=2 … replacing two".
- Branch B mapping recorded (total stays 3, observable downgrades to documentation-only per the Decision's own rule).
- Static counts backing the claim: `_foreach_copy_` site present in strategy binder; copy-in carries non_blocking=True inside the closure ordering copy-in → replay → copy-out.

## Fallback-Edge Exercises Through the NEW Boundary Code

1. **Edge A — TRUE capture failure**: `_build_manual_graph` shadowed by a raiser → caught once → SINGLE failure handler `_invalidate_manual_tier()` pinned `_replay_failed`, dropped graph+ALL workspace refs AND cleared `_hot_call`; compiled-default built LAZILY inside the same failing call; results bitwise==r004; second call stayed default-tier.
2. **Edge A2 — hot-callable swap post-capture (stale-bound invalidation edge)**: genuine capture first; then `_hot_call` replaced by a raiser. One call: poisoned callable invoked EXACTLY once → SAME handler fully invalidated hot/graph/workspace (DANGER rule proven verbatim) → lazily-built compiled-default served bitwise==r004.
3. **Edge B — stacked failures**: poisoned hot + poisoned default (post-genuine-capture) → cascade to UNMODIFIED STAGED returning bitwise==r004; both flags permanently true; each poison exactly once; handles cleared.
4. **Selectivity/recovery**: T=41 produced NO artifacts whatsoever (no graph/compiler/hot/workspace attr) with staged outputs bitwise==r004; same instance then warmed+captured and served tier-1 anchors through the coalesced boundary.

## run_out == forward (poisoned buffers, both orderings) + non_blocking hazard confirmation

- Ordering A run_out-FIRST cold: capture fired with caller buffers supplied; copy-out wrote INTO them directly under non_blocking stream-ordering → poisoned (-9e30/-7) rewritten IN PLACE, `data_ptr` preserved, bitwise==r004.
- Ordering B forward-first then warm run_out over re-poisoned buffers: byte-exact vs anchors.
- Fresh-copyout leak-trap: later forward returned FRESH buffers (distinct data_ptr) carrying identical bits.
- **Non-blocking read-before-write hazard confirmation (deviation note requested by bootstrap)**: CONFIRMED SAFE without adding any synchronization — all three boundary copies enqueue on the CALLER's current stream context in program order ahead of any consumer read enqueued by the comparison logic itself; bit-correct consumption verified under BOTH entry orders plus all sweeps (`non_blocking_read_before_write_confirmation.confirmed=true`). No events/streams were added beyond r004 behavior, per the Decision's explicit prohibition.

## Binding-Checker Statement

Artifact: `log/probes/binding_statement_report.json` @`b28abf7200c1a904fb0bf56233e1b4ba2f4a1c315e1369ab8d43c9b624f0535e` (generator `binding_statement_r2_005.py` @`e23da0820db08b54ac1bc0ca8ecdd37cca86fcd0b875c2d08467a9d1ad318641`; verdict `all_checks_pass=true`):

1. **Frozen inheritance (byte+AST)**: the seven segments listed above are untouched versus r004; named deltas isolated to the six authorized routing/boundary surfaces + three new helpers (per-method spans recorded).
2. **Strategy-bind contract (machine-checked spans)**: mixed-cast exercise precedes its parity assertion; real-pair byte-parity probe present; anomaly path writes `_batched_copyout_bind_error` and pins `_batched_copyout_ok=False` WITHOUT propagating; success sets True; `_batched_copyout_ok = True` reached only post-parity; closure ordering enforced copy-in(non_blocking) < `.replay()` < `_foreach_copy_([out_weights,out_ids],src_pair)`; `self._hot_call = _hot_entry` assigned after resolution.
3. **Stale-invalidation rule**: `_invalidate_manual_tier` clears flag/graph/workspace/hot together; both exception sites in the shared invocation helper delegate to it (exactly 2 references); DANGER trap satisfied.
4. **Boundary flags**: three `non_blocking=True` occurrences in the canonical reference method (copy-in + two legacy outs) + closure copies; replay submission unchanged.
5. **Token audits**: retired tier token `reduce-overhead` ×0; full carry-over allowlist scan clean (reduced_precision_reduction, allow_tf32/tf32, torch.backends, TORCHINDUCTOR, TRITON_CACHE, backend=, max-autotune, num_warps, num_stages, tl.argmax, autotune, matmul_precision, set_float32, getenv, lowercase 'cudagraph'); EXACTLY one torch.compile site {mode:'default', dynamic:'False'} (lazy fallback tier).
6. **Routing/signatures**: forward/run_out dispatch via `_route_target` with unchanged public signatures and parameter contracts; constructor params unchanged (six).

## Attempt Ledger

| # | Step / Command | Exit | Defect observed | Candidate SHA before → after |
|---|---|---|---|---|
| 1 | validate_decision gate + hash pinning | 0 | none | not-yet-created |
| 2 | coder_smoke run 1 | 2 | ONE FAIL was PROBE-SETUP ARTIFACT: my strategy-parity emulation flipped the flag then re-called `_bind_strategy_and_hot()`, which by contract RE-PROBED the capability and re-bound branch A (binding must never be revisited at runtime — candidate behaved correctly). All other 20 checks passed including both edges through the new boundary code. | creation @24214B → unchanged |
| 3 | Probe repair: genuine construction-time anomaly injection (raisers patched onto `torch._foreach_copy_` for a dedicated instance, restored after) exercising branch B exactly as the Decision defines it | 0 | NONE — 24/24 PASS on finalized bytes | `cf68ed77…` → `cf68ed77…` |
| 4 | binding_statement runs 1→1 | 0→0 | none (first generator version PASSED outright) | `cf68ed77…` → `cf68ed77…` |

Candidate semantic repairs after first complete write: ZERO. Same-round Verifier repair budget: untouched (zero requests so far).

## Deviation Classification

- **Major deviation: NONE.** Coalescing implemented exactly as authorized; frozen architecture preserved byte-level except the Decision-named boundary/hot-path deltas; three-tier chain, monotone flags, gating selectivity, workspace discipline, cold-cost placement, and compile-config discipline identical to accepted r004 semantics (machine-audited).
- Conformance notes (Small accommodation clause):
  - New instance attrs extend state_owner exactly as the Host Plan enumerates (`_batched_copyout_ok`, optional prebound handles inside the closure — captured locals, NOT new tensor state — and one string error artifact when branch B binds).
  - `_manual_replay_call` retained as the canonical audit-reference sequence (non_blocking upgraded per delta b); the hot path supersedes it after successful binding, which the docstring states explicitly to keep runtime/audit coherence.
  - Forward-path fresh result allocations moved INSIDE the hot closure (identical shapes/dtypes/device to r004's `_manual_replay_call` behavior); zero net allocation-policy change ("apart from the fresh result pair each forward call").
  - Binding may ONLY engage before first successful capture completes its finalization (construction-time window); runtime calls never consult or mutate it — "fixed at construction, never revisited" holds with machine evidence.
- No env overrides written; no flags flipped; no synchronization primitives added; recapture never triggered from model code.

## Return

Classification `candidate-ready`; candidate path `<root>/triton_grouped_topk_r2_005.py`, sha256 `cf68ed7713269416af5b49e901e040c7dcb97da9ec4f6eb4cc9bc5d70d288e9c`. All check artifacts under `log/probes/` with hashes above. Awaiting Orchestrator routing to Verifier for authoritative H-005 wall measurement under fingerprint 8deb1b01…; measurement ownership remains with Verifier.
