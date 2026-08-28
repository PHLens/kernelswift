# Coder Result 002

## Metadata

- Round: `002`
- Classification: **`candidate-ready`**
- Result: candidate conforms to immutable Decision-002 (`compile-graph-default`); no major deviation; no forbidden configuration path taken.
- Canonical lineage: derived from ACCEPTED round-001 candidate `triton_grouped_topk_r2_001.py` @`4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3` (manifest `last_accepted_kernel`), verified by recomputed hash before coding.
- Runtime fingerprint: unchanged round-001 environment (`triton 3.1.0 / torch 2.7.1 / CoreX 4.4.0 / Iluvatar BI-V150 sm_71`, capability `(7,1)` probed live) — not environment-blocked.
- Measurement fingerprint: untouched — NO timing benchmark or profiler sessions were run by Coder. The ONLY perf_counter reads are the explicitly-permitted bounded compile-completion sanity inside the smoke, labeled and reported as sanity, not as performance evidence.

## Artifact Hashes

| Role | Path (project-root-relative) | SHA-256 |
|---|---|---|
| Decision (input, immutable) | `rounds/decision_002.md` | `31c972fb31d9760acf4bb271bbff9d919c910cf0231b5b9215f9c871af82ff37` |
| Sketch (input, immutable) | `rounds/sketch_002.json` | `0ccbec4756d447d1365d0cae81ff2f8e3a020ecc3b99d84bbe2d4d7ce5d84cf3` |
| Canonical source start (`last_accepted_kernel`) | `triton_grouped_topk_r2_001.py` | `4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3` |
| Accepted report reference | `rounds/report_001.md` | `f9fbb9bf38f8d63ff9eeeed39bbd2e823ed6a34784f5121901a86e279c7a4fcc` |
| Profile snapshot (frozen) | `profile_snapshot/triton_cuda.yaml` | `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae` |
| **Candidate** | **`triton_grouped_topk_r2_002.py`** | **`ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12`** (14348 bytes) |

All input hashes recomputed before coding — identical to pinned values. Smoke-time embedded candidate hash equals final candidate hash.

## Required Sequence Evidence

1. `validate_decision.py` for decision_002 with manifest selected target profile → exit 0 (`valid:true`, rerun stderr-suppressed exit captured).
2. Environment identity: CoreX bootstrap before every CUDA command; device live-probed `Iluvatar BI-V150`.
3. Sketch/config primitives vs frozen snapshot statuses: the Constrained family consumed here is compilation-graph machinery — constrained `mode='default' / dynamic=False` used EXACTLY at its provenance posture: exactly ONE `torch.compile` call site with exactly those two keyword arguments; no precision/back-end/cache knobs anywhere (machine scan below). All Triton-stage and library-call primitives inherited byte-identically from the accepted implementation.
4. No Unknown capability consumed beyond round-001 resolved ones; fallback chain is NORMATIVE per decision and implemented permanently (never transitions back).
5. Candidate built from `last_accepted_kernel`; byte-identity of all inherited computation segments machine-proven.
6. Local gate passed on first execution of the candidate (zero candidate repairs; zero syntax repairs; checker-side cosmetic fixes only inside my own probe tooling).
7. This document is the structured result returned via Orchestrator.

## Compile-Smoke and Correctness Gate

- Script: `log/probes/coder_smoke_r2_002.py` @`0eb356b1ed259d0df1fc132b9f652a06217223ffdd91fbec0fe23970194befcf`
- Result: `log/probes/coder_smoke_result.json` @`031db1123cc563b91d8ff02bb9dbaa569601a0986f7a4eb21fc6e0f7288ecec5`, `all_pass:true`, 15/15 checks PASS
- Command lines + exit codes:
  ```
  cd <root>/log/probes && export COREX_VERSION=4.4.0 && . /usr/local/corex/enable \
    && python3 coder_smoke_r2_002.py --out coder_smoke_result.json     # EXIT:0
  cd <root>/log/probes && python3 binding_statement_r2_002.py          # EXIT:0
  ```
- **Compile-smoke through real harness loader** (`auto_bench.load_ks_module` on candidate + accepted r001 + base):
  - `warmup-compile-smoke-cold`: first target-regime forward constructed the shared callable via `torch.compile(self._triton_forward, mode='default', dynamic=False)` and matched base ids/weights; cold-completion sanity **3.544 s**;
  - `warmup-compile-smoke-warm`: second invocation bit-identical to cold result; warm-invocation sanity **0.001 s**;
  - both wall-clock numbers are BOUNDED-COMPLETION SANITY for this smoke only — they are NOT performance evidence and were not gathered under any measurement protocol.
- Correctness THROUGH the compiled route (seed42-shaped fp16[83,7168]/fp32[83,256], plus tie suites): seed42-regime-random, all-equal, two-expert-tie, structured-group-tie, duplicate-max-pairs — every case: ids exact vs base, weights within atol=rtol=1e-2, AND **bitwise equality against the accepted round-001 implementation outputs** (w,i) = True/True for all five cases.

## Fallback-Exercise Proof (Decision-normative)

- Strict regime selectivity (`fallback-non-target-regime-selective`): a T=41 [41,256] input executed the UNMODIFIED staged pipeline (bitwise-equal to accepted r001 output, base-consistent) while `_compiled_staged is None` and `_compile_failed is False` — i.e., the compiler was never even CONSTRUCTED for non-target regimes.
- Same-instance recovery of the compiled route (`same-instance-compiles-on-target-regime-after-non-target`): the identical instance then received [83,256] and entered the compiled route correctly (selectivity, not poison-by-contact).
- Forced dynamo/inductor failure (`fallback-forced-dynamo-failure-permanent`): an invokable object raising RuntimeError was injected as the compiled callable; invocation routed through the except branch, correct staged results returned bitwise-equal to accepted r001, `_compile_failed` transitioned True ONCE with handle dropped, and the poisoned callable was invoked EXACTLY once across two subsequent calls — permanent binding, never transitioning back.
- Non-target regime run_out stays in-place zero-copy (`run_out-non-target-regime-staged-inplace`) writing accepted-r001-bitwise results into caller buffers.

## run_out == forward Evidence

- `run_out-cold-cache-poisoned-buffers`: run_out-FIRST ordering (cold compiled buffer-signature variant) over buffers pre-filled with -9e30/-7 garbage rewrote them IN PLACE (data_ptr preserved) bitwise-equal to a fresh instance's forward output tuple.
- `run_out-warm-compiled-equality`: warm-compiled re-invocation remains byte-identical to warm forward and stable across repeats.
- Mechanism: the SAME shared compiled callable object serves forward (positional gating only; stage-C allocates fresh outputs internally) and run_out (positional `(gating_output, topk_weights, topk_ids)` mapping onto `_triton_forward`'s existing optional out arguments so stage-C stores land directly in caller buffers — zero added copy kernels in every regime).

## Binding-Checker Statement

Artifact: `log/probes/binding_statement_report.json` @`9315ba1b5f6b431713e7699f6ba89515d292e9bba56edd9d5cd4e18f5093a6b6` (generator `binding_statement_r2_002.py` @`c3ec32751145946105d143795a727bbb4b5c80021c226515d43191bc91a37508`, pure AST offline). Machine verdict `all_checks_pass=true`. Statement:

1. **Byte-for-byte inheritance proven**: segments `_softmax_group_scores_kernel`, `_group_mask_kernel`, `_renorm_scale_narrow_kernel`, `_triton_forward`, `_eager_forward`, `_fast_path_applies` are BYTE-identical (and AST-identical) versus accepted r001 — hence ALL THREE @triton.jit kernels and BOTH retained torch.topk call sites (inside `_triton_forward`, args `torch.topk(group_scores_out, k=self.topk_group, dim=-1)[1]` and `torch.topk(masked_scores, k=self.topk, dim=-1)`) are literally unchanged; per-segment sha256 recorded in the report.
2. **Compile-config allowlist compliance**: EXACTLY one `torch.compile` call site in the file with keywords exactly `{mode:'default', dynamic=False}`. Forbidden-token scan over the entire source: counts == 0 for reduced_precision_reduction, allow_tf32, tf32, torch.backends, TORCHINDUCTOR, TRITON_CACHE, backend=, 'reduce-overhead', "reduce-overhead", max-autotune, cudagraph, num_warps, num_stages, tl.argmax, autotune, matmul_precision, set_float32.
3. **Routing surfaces**: `run_out(gating_output, topk_weights, topk_ids)`, `forward(hidden_states, gating_output)`, constructor params unchanged; shared helper `_invoke_compiled_or_staged(gating_output[, out_weights, out_ids])` present and referenced from BOTH forward and run_out; strict guard `_compile_regime_applies` composes with the accepted `_fast_path_applies` so the compiled route engages only for contiguous fp32 cuda [83,256] + fixed config topk=8/renormalize=True/G=8/KG=4/'softmax'/scaling=1.0.
4. **Fallback chain semantics**: construction failure OR any invocation exception ⇒ `_compile_failed=True` (at-most-once transition, permanent) with identical-argument staged execution; live-proof as above.
5. **Forbidden constructs**: zero `tl.argmax`, zero `num_warps`/`num_stages`, no autotune (re-machine-checked this round).

## Attempt Ledger

| # | Step / Command | Exit | Defect observed | Candidate SHA before → after |
|---|---|---|---|---|
| 1 | validate_decision gate | 0 | none | not-yet-created |
| 2 | coder_smoke run 1 | 0 | NONE — all 15 checks PASS on first execution; warnings limited to unclosed-file ResourceWarnings inside MY probe script (cosmetic; hashes unaffected) | `ad703266…` (creation) → `ad703266…` (unchanged; zero repairs) |
| 3 | binding_statement runs 1→3 | 1→1→0 | Defects confined to MY generator script (class-method decorator_list probe; leftover variable; underscore typo in key name). No candidate edit involved. | `ad703266…` → `ad703266…` |

Same-round Verifier repair budget: untouched (zero requests so far).

## Deviation Classification

- **Major deviation: NONE.** Execution-strategy-only change per Optimization Intent: dataflow decomposition, library-selection boundaries, lifecycle ownership, and evaluation contract surfaces of the accepted pipeline are preserved byte-level where mandated.
- Unchanged-semantics guarantee carried by design + evidence: single shared callable compiled under default/dynamic=False; no precision/backend/cache configuration exists in the file (machine-scanned); partial-graph outcomes acceptable (no reordering claims made); fallback-to-staged is normative and proven, including permanent transition-once behavior.
- Conformance note (Small accommodation clause): routing adds two internal helper methods and lazily constructs the compiled callable at first target-regime use ("constructed once at first use" per allowed_changes; nothing built before it can be reached). `run_out` continues to write stage-C results directly into caller buffers (no added copy kernels in ANY regime), preserving the kernel-count surface measured at round-001 acceptance.
- host_us/cache_key risk postures from the decision recorded as acknowledged: no env overrides written; compiled wrapper engaged exclusively inside the fixed-shape/fixed-config target regime; cold/warm invocations produce bitwise-equal outputs.

## Return

Classification `candidate-ready`; candidate path `<root>/triton_grouped_topk_r2_002.py`, sha256 `ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12`. All check artifacts under `log/probes/` with hashes listed above. Awaiting Orchestrator routing to Verifier for authoritative measurement; measurement ownership remains with Verifier.
