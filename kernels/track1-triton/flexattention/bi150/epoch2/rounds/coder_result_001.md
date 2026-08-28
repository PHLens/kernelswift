# Coder Result 001

Result: candidate-ready

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`fa11b1152306e4cc4b33a02e31bc52d4c76de210c79385f41e02ee25c3bc7b1d` (validator-green re-confirmed locally before coding: `validate_decision.py … --expected-implementation-profile triton_cuda --project-root …` → `"valid":true`, exit 0)
- Sketch: `rounds/sketch_001.json` @`199275b85e831238c2f0c9c694d3c4c03550c6681bd7a8e87f3474642b3c1fce` (hash re-verified from file)
- Candidate: `triton_flexattention_e2_001.py` @`b490acc674ef5570900e8273bd6e3ab2a10102612b8c6fc6da63271a2dfcadec` (project root: `kernels/track1-triton/flexattention/bi150/epoch2/`)
- Canonical start (last_accepted_kernel): `baseline_adapter.py` @`b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1` (hash re-verified from file; candidate derives its eager tier byte-frozen from it)
- Base: `../../base.py` @`dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` (untouched)
- Harness: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py` @`71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (untouched, AST loader)
- Selected profile: `triton_cuda` via frozen snapshot `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae` (hash re-verified; migration-status header noted: explanatory rendering, machine profile pending promotion)
- Capability claim: `profile_snapshot/capability_claim.json` @`07aa5d489acb9c21717032087812d264dd5170fe79e7ea2326edb04cab657c1d` (hash re-verified)
- Runtime fingerprint: `project.md#runtime-fingerprint` re-probed live in p01 — torch 2.7.1, device `Iluvatar BI-V150` major=7 minor=1 multi_processor_count=16 total_memory=17179869184 on cuda:0, CoreX bootstrap `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` in every shell — match
- Measurement fingerprint: `6dc07009177b649f7c2cad8f7be5e9aad74235bd9f50abfebc88bdb273e32af4` (untouched; Coder ran NO timing/benchmarks/profilers)
- Binding statement artifact: `log/probes/binding_statement_report_001.json` @`916058cb682f65a65908fbe5bc3c0c8e4a397067eec7b4dfc7d6737b7cb8dc5b`

## Sketch Primitive and Hint Conformance

- Sketch declarations are tensors only; operations are boundary copy-ins, the retained fused compute op, and boundary copy-outs. Zero `triton_cuda` capability-matrix primitives are consumed (zero `tl.*` calls anywhere), so the constrained `matrix.dot` envelope CANNOT gate this round and the P1–P4 before-fallback ladder is untriggered by construction — exactly as decision_001 §Pitfalls records.
- Hints: none in sketch (`"hints": []`); `num_warps`/`num_stages`/block pointers/mixed precision remain untouched Unknowns.
- `torch.compile` is consumed nowhere (DANGER scan all-zero, below); the two-tier chain is manual-replay → framework-eager only.
- GQA broadcast omitted-by-construction inside the captured region; the framework-eager tier retains the `repeat_interleave` branch verbatim so its behavior stays byte-equivalent to `baseline_adapter.py` for ANY regime.
- Scale config: constructor `scale=None → 0.125` preserved; the guard pins `self.scale == 0.125` exactly (1/√64 double-exact), and the captured region passes `scale=self.scale` at the guarded constant.
- AST-loader retention strategy: every executable statement lives inside retained node types (imports, one ClassDef, two FunctionDefs); no top-level expressions; no `device='cuda'` string outside `get_inputs` (which the loader rewrites to the detected accelerator, a no-op here); guards use `tensor.is_cuda` instead of device strings.

## Attempt Ledger

| Attempt | Command (abridged) | Exit | Defect | Candidate SHA before → after |
|---|---|---|---|---|
| 1 | initial authoring | — | two DOCSTRING wordings contained the literal token sequence `tl.dot` (documentation only, zero code use); reworded pre-gate so the machine DANGER scan is unambiguously all-zero (non-semantic conformance edit) | (pre-gate wording edit) → `b490acc6…cadec` |
| 1 | `ast.parse` gate | 0 | none | `b490acc6…cadec` |
| 1 | DANGER token scan (9 tokens) | 0 | none — all counts 0 | `b490acc6…cadec` |
| 1 | real-harness smoke `auto_bench.py --v0 base.py --v1 candidate --warmup 5 --repeat 10 --full-traceback` | 0 | none — `PASS accuracy; v0=0.154739 ms, v1=0.154292 ms, speedup=1.003x` (first attempt; loader + comparator + capture route exercised) | `b490acc6…cadec` |
| 1 | probes p01–p08 (+p07b/p07c diagnostics) | 0 | probe-side counter-accounting and reference-construction defects only (p01 sdpa-window marks, p02 twin-sdpa window, p04 healthy-engine replay pollution → explicit windows, p05 6-head reference engine, p07 stream segment rescoped to canonical + documented-deviation characterization); CANDIDATE SOURCE UNTOUCHED across all probe iterations | `b490acc6…cadec` (unchanged end-to-end) |

No Verifier repair requests yet; zero same-round repairs consumed. Candidate hash never changed after the first gate.

## Decision-scoped Checks (log/probes/ only — non-authoritative, no timing/benchmarks/profilers)

| Probe | Verdict | Key evidence |
|---|---|---|
| p01 cold-capture smoke + warm-replay repeat | PASS exit 0 | vendor `ixattnbkd` SDPA pipeline CAPTURED (library-op capturability): 1 graph, construction window = exactly 4 sdpa invocations (warmup 3 + capture 1), cold call served via exactly 1 replay; bitwise tier-retention vs eager-pinned twin on seed42; 122 distinct-input replays with stable static workspace addresses; 100-call boundedness soak without flag flips |
| p02 warm-replay repeat deep | PASS exit 0 | 100 alternated-mutation calls bitwise==eager twin every call; window counters: model exactly 100 replays / zero sdpa; eager twin exactly 100 sdpa; flags unchanged |
| p03 capture-fired multi-fact intersection | PASS exit 0 | f1 live graph handle; f2 static addresses identical across distinct-input services; f3 stale-trap (distinct inputs → distinct outputs tracking their own bits; re-input deterministic); f4 lower-tier absence (5-serve window: replay+5 / sdpa+0, outputs bitwise-correct); f5 in-place mutation freshness — ALL True |
| p04 both fallback edges forced permanent-once | PASS exit 0 | vector A capture-denied (`CUDAGraph` instantiation raises): same-call output bitwise-correct via eager fallback, artifacts dropped, 5 follow-ups stay eager with zero replay attempts; vector B first-replay-denied on captured instance: exactly 1 replay attempt, same-call bitwise-correct, permanent down-tier; cross-instance isolation (healthy neighbor unaffected) |
| p05 non-target selectivity then recovery | PASS exit 0 | T=41 fp16 first call + fp32 + T=82 + config-divergent (heads=6 with matching inputs) all route eager with base-consistent outputs and construct NOTHING (cg=0/replay=0/workspace absent/flags intact); same instance then captures ONCE and serves via replay (recovery) |
| p06 run_out poisoned buffers ×2 + surface parity | PASS exit 0 | both orderings (forward→poisoned run_out; poisoned run_out→forward) bitwise-equal; caller data_ptr preserved in place; never aliased to workspace; copy-out isolation from later services; eager-tier surface parity ×2 orderings; cross-tier run_out bits equal |
| p07 cross-instance alternation + stream discipline | PASS exit 0 | 3 instances × 4 seeds interleaved: every service bitwise-matches its own engine/input pair; canonical default-stream route 5/5 bitwise-correct; non-default-stream segment reproduces the deterministic one-behind BUILD deviation signature exactly (documented, see Deviations D1) |
| p07b/p07c stream diagnostics | DIAGNOSTIC artifacts | input bits stream-neutral; cross-trial identical; divergent last-window output == previous input's reference; one-call-per-fresh-stream affected; repeated same-seed windows hide the signature; default-stream control always correct |
| p08 bitwise sweep (machine table) | PASS exit 0 | suites {seed42, causal-boundary, fp16-extreme} × surfaces {forward, run_out} × tiers {replay, eager}: 12/12 cells bitwise-equal (all four pairwise equalities per suite), allclose(atol=1e-2, rtol=1e-2, equal_nan) vs base everywhere, shape/dtype [83,512] fp16 everywhere; `p08_sweep_result.json` written |

All 8 gating probes PASS with exit 0 on the final candidate bytes. Two diagnostic artifacts (p07b/p07c) intentionally encode the deviating build behavior as characterization evidence; their gating FAIL lines describe the BUILD, not the candidate.

## Binding Statement

- **Segment freeze vs adapter**: `ModelNew._forward_eager` is a byte-frozen copy of `baseline_adapter.py::ModelNew.forward` (imports, statement sequence, GQA branch retained for any-regime equivalence). The captured region `_pipeline_body` is the same op sequence over static workspace placeholders with exactly the decision-authorized substitutions (live tensors → placeholder views fed by per-call copy-ins; GQA omitted-by-construction; epilogue writes `attn_flat_ws`; copy-out outside the boundary). `get_inputs`/`get_init_inputs` byte-equivalent. Machine details: `log/probes/binding_statement_report_001.json`.
- **DANGER token scan all-zero**: `torch.compile=0, reduce-overhead=0, TORCHINDUCTOR=0, allow_tf32=0, reduced_precision=0, float32_matmul_precision=0, tl.dot=0, @triton.jit=0, triton.language=0` over the final source (two initial docstring WORDING hits reworded pre-gate; recorded in the ledger).
- **run_out contract**: `run_out(query, key, value, out) -> None` exactly as the decision mandates; caller buffer filled via out-of-boundary copy-out each call, never aliased to workspace; bitwise-equal to forward on both tiers (p06).
- **Capture obligations satisfied by construction**: static addresses (workspace allocated BEFORE warmup/capture; address stability asserted), fixed shapes, no branches/prints/host reads inside the region, internal temporaries from the graph-private pool, warmup on a dedicated side stream then capture via `torch.cuda.graph` recommended pattern, per-call allocations limited to the forward result buffer OUTSIDE the boundary.
- **Primitive consumption**: zero; binding ledger per vNext `validate_binding.py` full-schema form is NOT producible this round without false claims — every sketch op maps to torch-level host calls and the frozen `triton_cuda` profile v1 capability matrix contains only Triton-kernel symbol pairs, so any `(contract_name, implementation_symbol)` entry would fail `binding-profile-mapping` or misattribute torch calls as Triton primitives. The coder-produced binding statement artifact above (sibling r004 precedent shape, consumed read-only by Verifier) is the honest equivalent; flagged for Orchestrator.

## Deviations

- **D1 (build fact, no code change)**: non-default-STREAM replay deviation on CoreX 4.4.0 / torch 2.7.1 / BI-V150 — the LAST serve issued inside a non-default stream window deterministically evaluates against the PREVIOUS call's copy-in bits (one-behind; p07b/p07c: inputs stream-neutral, cross-trial identical, divergent output == previous input's reference, one-call-per-fresh-stream affected, repeated same-seed windows hide it). Default-stream replay — the canonical harness and measurement route — is bitwise-safe everywhere (every probe). The candidate implements exactly the designed caller-stream replay; no semantic workaround was invented (a per-call device sync would violate the Host Plan's "no synchronization beyond base.py behavior plus one graph-launch submission" and collapse the mechanism). This is matched local evidence that profile-Unknown "stream and context semantics" is partially unsupported on this build; carried to Orchestrator/Designer for the record. Verifier can proceed entirely on the default-stream route.
- **D2 (evaluation-routing fact, no code change)**: unchanged-harness `--profile-mode kernel` cannot drive this candidate: `make_profile_call` calls `run_out(<inputs[-1]>, *outputs)` (harness lines 516–536), passing only the last input; this operator needs all three. The groupedtopk sibling satisfied the arity coincidentally. Candidate keeps the decision-mandated 4-arg signature verbatim; a cross-call input-smuggling "accommodation" would violate workspace-discipline invariants and was rejected. Resolution belongs to Orchestrator/Verifier (direct-call probe lambda invoking `run_out(q,k,v,out)`, or the canonical forward-mode dual-scope fallback already precedented in report_000). Kernel-count observable remains reachable through branch-B host-census evidence on forward-mode traces.
- **D3 (conformance note)**: workspace placeholders are zero-initialized at capture-time allocation (removes uninitialized-memory nondeterminism from the warmup window; contents are fully overwritten by copy-ins every call thereafter per the ownership supersession clause).
- **D4 (conformance note)**: `validate_binding.py` full ledger not emitted (see Binding Statement); binding evidence delivered as the coder-produced statement artifact per sibling precedent.

## Evidence for Verifier

- Candidate: `triton_flexattention_e2_001.py` @`b490acc674ef5570900e8273bd6e3ab2a10102612b8c6fc6da63271a2dfcadec`.
- Probe logs under `log/probes/` (hashes below). Canonical measurement route: unchanged harness, default stream, seed 42.
- Suggested canonical kernel-mode-style invocation if pursued: `model.run_out(query, key, value, out_buffer)` direct-call lambda (see D2) — Verifier's choice; forward-mode fallback remains precedented.

### Artifact hash ledger

```text
b490acc674ef5570900e8273bd6e3ab2a10102612b8c6fc6da63271a2dfcadec  triton_flexattention_e2_001.py
fa11b1152306e4cc4b33a02e31bc52d4c76de210c79385f41e02ee25c3bc7b1d  rounds/decision_001.md
199275b85e831238c2f0c9c694d3c4c03550c6681bd7a8e87f3474642b3c1fce  rounds/sketch_001.json
b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1  baseline_adapter.py
dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  /root/CodeBuddy/20260818191200/kernelswift/auto_bench.py
dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae  profile_snapshot/triton_cuda.yaml
07aa5d489acb9c21717032087812d264dd5170fe79e7ea2326edb04cab657c1d  profile_snapshot/capability_claim.json
916058cb682f65a65908fbe5bc3c0c8e4a397067eec7b4dfc7d6737b7cb8dc5b  log/probes/binding_statement_report_001.json
81769a9f3028adfeb89c2df94bbd7d9c2b7798efa75fc5113b721682fcadd529  log/probes/p01_cold_capture_smoke.py
079b648deccb117f9b4efd9e4af667aef4da36f75fa0cf529792fa268677a4e5  log/probes/p01_cold_capture_smoke.log
693464ada278af36bbf5feb819daf6f5e45d9c79cfe05a69a12179cbf5f6518d  log/probes/p02_warm_replay_repeat.py
cd78adb3cb08539c20baadd40d0f3d660115d1a674f4315005c97445ebafa572  log/probes/p02_warm_replay_repeat.log
cfccfb97a8adfd52983569ca0fc8644c42d5cdb918bf1f43f026527446da1245  log/probes/p03_capture_fired_multifact.py
f852b25c73622363b777dd55948c37f1b402ebe29b0d4b67735402d9c3e72c33  log/probes/p03_capture_fired_multifact.log
7a73ed3c524174d7c1852720e1a0daeb6990dd5a469f6f5c06b1f23cda1c8935  log/probes/p04_fallback_edges.py
bf6fc9cfe806c4da8733bcadf2506ff83731b268548d3d3f264de3ca9423a209  log/probes/p04_fallback_edges.log
ac03e03221756c9208c069cc6c930589f43eb95db63df76c4e1d8ca7cc2554b0  log/probes/p05_selectivity_offregime.py
f3d10fca8ca6d1ad557ee62e7ed91380444a73c99ed8a1ee7c19e2dd6ee5bb2b  log/probes/p05_selectivity_offregime.log
2d135e7c371db6402455d4125d397208464ddcba2d494b70565532b2c0f6020b  log/probes/p06_run_out_poison_and_altsurface.py
19cb42d6930d9c58f169ed468f17abb1fa9e441d753c375cc04fb9de381405bf  log/probes/p06_run_out_poison_and_altsurface.log
4dce2cb3c150ec81f0d9379121fb493a8ee652a1077fe264cc91960edb2e6e73  log/probes/p07_cross_instance_alternation_stream.py
fd7ce71fda0dd7cc77a2df9ddc65680e52cf7db69e711bba0d2dcfe326d6861c  log/probes/p07_cross_instance_alternation_stream.log
d8940348fe210abf7ef975fa09e294acc0eee26c2bba66a14482184ec61401dc  log/probes/p07b_stream_diagnostic.py
c4f7fd84931fa16e4cee0977a4c65aff10b12c01b16158587f905c224320b468  log/probes/p07b_stream_diagnostic.log
409b1fedda577c9613ece28ede3d2601d6e5612f15295471a8856f7919c55418  log/probes/p07c_stream_isolate.py
b826e144b577a006cc2df796b697da308fef890a00d89a5d1817885cd7eab642  log/probes/p07c_stream_isolate.log
0c6bb2be703b174eb3b7f81b154c17544e40be066efbe497adbc1859ce5727e6  log/probes/p08_bitwise_sweep.py
e24a209a2b77b758a2becfb6dfee440880b104c875342e933363fdc8efeb5d85  log/probes/p08_bitwise_sweep.log
1af0ff8ce6a63d9455025bcfbfbbb4d0a4e554eaa707bc635ea0565f9c3ea744  log/probes/p08_sweep_result.json
```

## Exact Commands (all with `cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable;` prefix; device cuda:0)

```bash
# decision re-validation (exit 0)
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/validate_decision.py \
  kernels/track1-triton/flexattention/bi150/epoch2/rounds/decision_001.md \
  --expected-implementation-profile triton_cuda \
  --project-root kernels/track1-triton/flexattention/bi150/epoch2

# ast gate (exit 0) + DANGER scan (all-zero)
/usr/local/bin/python3 -c "import ast; ast.parse(open('kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_001.py').read()); print('AST_PARSE_OK')"

# real-harness smoke (exit 0, PASS accuracy)
/usr/local/bin/python3 auto_bench.py \
  --v0_file kernels/track1-triton/flexattention/base.py \
  --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_001.py \
  --warmup 5 --repeat 10 --full-traceback

# probes p01..p08 (+p07b/p07c), each exit 0 except labeled diagnostics
/usr/local/bin/python3 kernels/track1-triton/flexattention/bi150/epoch2/log/probes/<probe>.py
```

Coder claims no measurement and no verdict; classification is candidate-ready. Orchestrator owns the verification dispatch.
