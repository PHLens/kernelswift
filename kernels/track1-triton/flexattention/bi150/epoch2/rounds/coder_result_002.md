# Coder Result 002

Result: candidate-ready

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md` @`459e8d37219b5534103a82a7a342c61ef04e147158a6851d794b73e2a44f8730` (hash re-verified from file; Orchestrator-validated)
- Sketch: `rounds/sketch_002.json` @`fb5bec0b957a04ffa19d20edb2f0fdb92de156c0aea6429b1c796a86b89bd87c` (hash re-verified)
- Candidate: `triton_flexattention_e2_002.py` @`570bc2be2cb8e79a06ebb32e5e8bf4f79aa62a38d5382b9a1a5f12426f3512b1` (project root: `kernels/track1-triton/flexattention/bi150/epoch2/`)
- Canonical start (last_accepted_kernel): `baseline_adapter.py` @`b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1` — semantics derived from it; **r001 candidate is retired evidence and zero r001 workspace/graph machinery was reused** (DANGER scan includes r001 machinery tokens, all-zero)
- Harness: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py` @`71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (untouched)
- Runtime fingerprint: torch 2.7.1, `Iluvatar BI-V150` sm71 mp16, CoreX bootstrap `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` in every shell, cuda:0
- Binding statement artifact: `log/probes/binding_statement_report_002.json` @`ad4d4ba779d73baf9ef1eb7bfc555bab234b93d071dbd838c00e7771336d5ac8`

## Implementation Summary (decision-002 exact)

- ONE `@triton.jit` kernel `_causal_attn_fwd`, direct launch `_causal_attn_fwd[(H*nt_m,)](...)`, `num_warps=1`, `num_stages` unset (token count 0). Grid at T=83 = 8 heads × 3 mtiles = **24 programs**; each program owns one (head, 32-row query tile) and loops NT=3 key tiles via `tl.static_range` with the online fp32 running-max softmax (sketch ctrl.parallel.head/mtile parallel, ntile as the loop domain).
- `BM=BN=32`, `BD=32` (D=64 split into two 32-chunks); T/H/D/NT passed as constexpr (frozen specialization; shape changes recompile naturally — off-regime spot-checks exercised this path).
- **Legality binding**: every fp16 tile load is WIDENED via `.to(tl.float32)` BEFORE its first dot use; all 4 dot call sites are (32,32)@(32,32) fp32/fp32→fp32. fp16-operand dots: 0. Non-32 dot shapes: 0. Keys are loaded directly in transposed layout so NO trans op exists.
- Causal mask: −inf PRE-softmax (`tl.where(causal & valid, s, -inf)`); exp(−inf)=0 exactly → exact-zero post-softmax contributions; invalid-key and out-of-range rows masked at load (other=0.0) and at store (mask), respectively.
- Results stored DIRECTLY into the final `[T, H*D]` fp16 token-major layout (`o_ptr + row*(H*D) + head*D + d`) — zero view/copy/relayout ops.
- `forward` = TWO python-visible ops (`torch.empty` + ONE kernel launch); `run_out(query,key,value,out)` 4-arg retained (ONE launch, zero allocations, returns None, bitwise-equal to forward).
- STATELESS: instance attrs exactly the 4 constructor-config attrs (+nn.Module-standard `training` from `.eval()`); no caches, no graphs, no workspace, no cross-call state; Triton JIT compile cache is framework-owned, one-time, absorbed by harness warmup.

## Sketch Primitive and Hint Conformance

- Kernel uses exactly the proven dot envelope ((32,32)@(32,32) fp32) and `num_warps=1` (frozen snapshot Constrained: proven-stable).
- Non-dot shape variants beyond the frozen proven list — axis-1 `tl.max`/`tl.sum` over (32,32) fp32, `tl.full([32], -inf)`, 2D fp16 loads (row-strided, widened), transposed-access load, fp16 2D direct-layout store, `tl.static_range(3)` — were established by the Decision-scoped file-backed capability probe p10 (results under `log/probes/`), per the coder contract's probe allowance; no Unsupported/Unknown requirement remains unprobed for the constructs actually used.
- `tl.argmax` not used anywhere (tie-order non-certifiability noted in the frozen snapshot); no atomics, no fp64, no trans op.
- DANGER tokens (compiler family + r001 machinery + num_stages): all-zero; `num_warps` appears at exactly one site with value 1.

## Attempt Ledger

| Attempt | Command (abridged) | Exit | Defect | Candidate SHA |
|---|---|---|---|---|
| 1 | authoring + docstring wording pre-gate edit (num_warps/num_stages mentions) so machine audits are unambiguous | — | none (candidate) | `570bc2be…12b1` (final; unchanged after first gate) |
| 1 | `ast.parse` gate | 0 | none | `570bc2be…12b1` |
| 1 | DANGER token scan (14 tokens) + num_warps/stateless/dot-site counts | 0 | none — all-zero; 4 dot sites; 4 attr writes; num_warps=1 single site | `570bc2be…12b1` |
| 1 | real-harness smoke `auto_bench.py --v0 base.py --v1 candidate --warmup 5 --repeat 10 --full-traceback` | 0 | none — `PASS accuracy` (see smoke observation below) | `570bc2be…12b1` |
| 1–3 | probes p10/p11/p12 | 0 | PROBE-SIDE only: p10 launch-argument order swapped xr/rt (run-1 roundtrip "PASS" was a double-bug cancellation — wrong transposed addressing × swapped args; run-2 exposed it via the 992=1024−32 transposed-placement signature), p10 xr buffer undersized for static_range rows, p11 expected-attr list missed the nn.Module-standard `training` bit. CANDIDATE SOURCE UNTOUCHED across all probe iterations | `570bc2be…12b1` |

No Verifier repair requests yet; zero same-round candidate repairs consumed.

## Decision-scoped Checks (log/probes/ only — non-authoritative, no timing/benchmarks/profilers)

| Probe | Verdict | Key evidence |
|---|---|---|
| p10 capability probe (file-backed) | PASS exit 0 | (32,32) fp32 axis-1 tl.max/tl.sum; tl.full −inf; fp16 2D row-strided load widened fp32; transposed-access 2D load; fp16 2D direct-layout store roundtrip bitwise; tl.static_range(3). Isolation validated each non-dot primitive shape the candidate uses beyond the frozen proven list |
| p11 compile-smoke + stateless audit | PASS exit 0 | compile on cuda:0 via the real candidate path; seed42-regime allclose(atol=rtol=1e-2) with **max_abs=9.766e-04**; run_out returns None, poisoned-buffer fully overwritten, BITWISE-equal to forward; caller data_ptr preserved; repeat call bitwise-stable with FRESH allocation; instance attrs = 4 config attrs (+`training`), identical pre/post calls; zero r001-style attributes |
| p12 correctness sweep (machine table `p12_r002_sweep_result.json`) | PASS exit 0 | 6 suites (below), run_out bitwise==forward and shape/dtype ok on every suite |

### Sweep table (suites × max_abs vs base.py semantics; allclose atol=rtol=1e-2 equal_nan PASS everywhere)

| Suite | allclose | max_abs | run_out bitwise | shape |
|---|---|---|---|---|
| seed42_T83 | PASS | 9.766e-04 | True | ok |
| causal_T83 | PASS | 2.441e-04 | True | ok |
| extreme_T83 | PASS | 2.000e+00 (=1 fp16 ULP at magnitude 2048–4096; rtol regime) | True | ok |
| seed7_T41 | PASS | 9.766e-04 | True | ok |
| seed13_T82 | PASS | 9.766e-04 | True | ok |
| seed19_T96 | PASS | 9.766e-04 | True | ok |

Numeric-risk characterization: online-softmax vs vendor SDPA accumulation order stays 1 order of magnitude under the 1e-2 tolerance on ordinary suites; the extreme suite diff is pure fp16 output quantization (1 ULP at large magnitudes), not softmax divergence.

## Binding Statement

- **Dot-shape audit**: 4 `tl.dot` call sites, each (32,32)@(32,32) with fp32 operands (widening casts between every fp16 load and its first dot use); 0 fp16-operand dots; 0 non-32 shapes — machine table in `binding_statement_report_002.json`.
- **num_warps**: exactly one launch site, value 1; `num_stages` absent (count 0). Kernel count: 1.
- **Stateless audit**: 4 instance-attr writes, all in `__init__` (constructor parity with baseline_adapter); post-call `__dict__` diff empty (p11); module level = 3 immutable int literals + kernel + class + input helpers; r001 machinery tokens all-zero.
- **Legality chain**: fp16 load → `.to(tl.float32)` → dot, at all four sites; keys loaded pre-transposed (no trans op); −inf pre-softmax masking with exact-zero post contributions; direct fp16 final-layout stores.
- **Off-regime policy**: stateless per-shape constexpr specialization (T=41/82/96 spot-checked correct); `num_kv_heads != num_heads` is outside decision-002 scope (GQA absent-by-construction, no fallback tier in this design) — documented, not silently mishandled.
- **AST-loader composition**: 4 imports + 3 literal assigns + `@triton.jit` FunctionDef + ClassDef + 2 helper FunctionDefs — all retained node types.

## Deviations

- **D1 (observation, no code change)**: harness smoke at `--warmup 5 --repeat 10` printed `PASS accuracy; v0=0.148187 ms, v1=0.243420 ms, speedup=0.609x` (exit 0). This is a harness-printed observation, NOT a coder measurement claim (Coder ran no timing/benchmarks/profilers). The smoke reading suggests the single-Triton-launch path is currently SLOWER than the whole vendor stack at this tiny repeat count; the plausible contributor on this build is the candidate Triton python-launcher host overhead (kernel work itself is trivially small). Flagged for Verifier's authoritative measurement design (e.g., higher repeat count; kernel-mode census will separate launch host cost from device time). No design change is legal from the coder seat: the direct `kernel[(grid,)]` launch with `num_warps=1` and no caching is exactly what decision-002 mandates.
- **D2 (conformance note)**: probe p10's run-1 "PASS" on the roundtrip check was a double-bug cancellation (probe-side transposed addressing error × swapped launch args); detected, root-caused (992=1024−32 transposed-placement signature), fixed probe-side, and re-run to genuine PASS. Candidate unaffected at every step (hash constant).
- **D3 (conformance note)**: GQA (`num_kv_heads != num_heads`) and non-fp16/non-cuda inputs are outside decision-002's declared scope; the stateless kernel assumes the decision-pinned contiguous fp16 [T,H,D] input contract and has no fallback tier by design.

## Evidence for Verifier

- Candidate: `triton_flexattention_e2_002.py` @`570bc2be2cb8e79a06ebb32e5e8bf4f79aa62a38d5382b9a1a5f12426f3512b1`.
- Probe logs under `log/probes/` (hashes below). Canonical measurement route: unchanged harness, default stream, seed 42.

### Artifact hash ledger

```text
570bc2be2cb8e79a06ebb32e5e8bf4f79aa62a38d5382b9a1a5f12426f3512b1  triton_flexattention_e2_002.py
459e8d37219b5534103a82a7a342c61ef04e147158a6851d794b73e2a44f8730  rounds/decision_002.md
fb5bec0b957a04ffa19d20edb2f0fdb92de156c0aea6429b1c796a86b89bd87c  rounds/sketch_002.json
b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1  baseline_adapter.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  /root/CodeBuddy/20260818191200/kernelswift/auto_bench.py
ad4d4ba779d73baf9ef1eb7bfc555bab234b93d071dbd838c00e7771336d5ac8  log/probes/binding_statement_report_002.json
41890aab21bcffc3bc442bbab3aee6478bf8bcd346febdb04978283f9af5cee9  log/probes/p10_r002_capability_probe.py
bb0b2d615977596263a2fa52fcd3f2e8fefb1db22e7999e2d9ce6ad8396e19c1  log/probes/p10_r002_capability_probe.log
51ef181f0bd147d0727f22935c611efe3ed0ee2f50b40e6b4965a805a8686f6e  log/probes/p11_r002_compile_smoke.py
9c4adba4fd995bdec2e8d31fa6b711b3b963ab09a2df36cac2c854574939b43a  log/probes/p11_r002_compile_smoke.log
c49d178a04ff5e72af0260246b14a67b56179a1e74eb68008bc5d2eb4a3dfeaf  log/probes/p12_r002_correctness_sweep.py
eccec064c562f0b1b370eb2be6dc9a516dada44f9f91e9f56055dbae914ec359  log/probes/p12_r002_correctness_sweep.log
ad4d4ba779d73baf9ef1eb7bfc555bab234b93d071dbd838c00e7771336d5ac8  log/probes/p12_r002_sweep_result.json
```

## Exact Commands (all with `cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable;` prefix; device cuda:0)

```bash
# gates
/usr/local/bin/python3 -c "import ast; ast.parse(open('kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_002.py').read()); print('AST_PARSE_OK')"
# DANGER scan + counts (see binding_statement_report_002.json for the recorded table)

# real-harness smoke (exit 0, PASS accuracy)
/usr/local/bin/python3 auto_bench.py \
  --v0_file kernels/track1-triton/flexattention/base.py \
  --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_002.py \
  --warmup 5 --repeat 10 --full-traceback

# probes (each exit 0)
/usr/local/bin/python3 kernels/track1-triton/flexattention/bi150/epoch2/log/probes/p10_r002_capability_probe.py
/usr/local/bin/python3 kernels/track1-triton/flexattention/bi150/epoch2/log/probes/p11_r002_compile_smoke.py
/usr/local/bin/python3 kernels/track1-triton/flexattention/bi150/epoch2/log/probes/p12_r002_correctness_sweep.py
```

Coder claims no measurement and no verdict; classification is candidate-ready. Orchestrator owns the verification dispatch.
