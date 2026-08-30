# Coder Result 001

Result: candidate-ready

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`67b96739c35adabb713081a1f3a50649193b28eed420dc32dd512572fab26c78` (hash re-verified from file; matches the dispatch-validated value; family "triton-attention-dispatch-collapse" F1 deliverable-grade, expected_wall_improvement_pct 0.0 honest deliverable round)
- Sketch: `rounds/sketch_001.json` @`a1c27dbae53b1c7a74681510a0d09ced6be58ed8501f86976ce55af1b4772363` (hash re-verified; matches decision sketch_sha256)
- Candidate: `triton_mm_encoder_attention_e2_001.py` @`4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2` (project root: `kernels/track1-triton/mm_encoder_attention/bi150/epoch2/`)
- Canonical start (last_accepted_kernel): `baseline_adapter.py` @`c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f` — semantics derived from it; **epoch-1 candidate `../triton_mm_encoder_attention_001.py` was read as prior evidence only: its layout-copy host path (`3x .contiguous() + reshape-copy`) and BLOCK_S=128 padding were NOT copied — direct strided addressing + direct-layout stores replace them**
- Base (immutable reference): `../../base.py` @`86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (re-verified unchanged; equals project.md declaration)
- Harness: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py` @`71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (untouched; AST loader)
- Runtime fingerprint: torch 2.7.1, triton 3.1.0 (`/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton`), `Iluvatar BI-V150` capability major=7 minor=1 multi_processor_count=16 total_memory=17179869184, CoreX bootstrap `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` in every shell, interpreter `/usr/local/bin/python3`, device `cuda:0` — matches `project.md#runtime-fingerprint` (live re-probed)
- Binding statement artifact: `log/probes/binding_statement_report_001.json` @`623783fd96ecfa90e77e88b59985d433ee8a31097f802d896bb82a9947630b2f`

## Implementation Summary (decision-001 exact)

- ONE stateless `@triton.jit` kernel `_mm_encoder_attn_fwd`; single shared launch site `ModelNew._launch` with direct launch `_mm_encoder_attn_fwd[(B*H*ceil(S/BM),)](...)`, `num_warps=1`, `num_stages` unset (count 0). Grid at the target regime = B*H*ceil(83/32) = 16*3 = **48 programs** ((batch,head) pairs x query tiles — sketch `ctrl.parallel.batch/head/mtile`); each program owns one (b,h) + 32-row query tile and loops NT=3 key tiles via `tl.static_range` **sequentially** with the fp32 online running-max softmax (sketch `ctrl.for.ntile` — the running-state dependency).
- `BM=BN=32`, `BD=32` (D=64 split into two 32-chunks — sketch q_lo/q_hi, k_lo/k_hi, v_lo/v_hi declarations); B/S/H/D/NM/NT passed as constexpr (frozen specialization; shape changes recompile naturally — live evidence: 4 distinct input shapes produced 4 distinct framework JIT specializations).
- **Direct strided addressing, ZERO layout-copy calls**: inputs are the fp16 `[B,S,H*D]` tensors addressed directly at offset `b*(S*H*D) + s*(H*D) + h*D + d` with all strides folded from constexpr (S,H,D); keys loaded directly in transposed layout (no trans op); results stored DIRECTLY into the final `[B,S,H*D]` fp16 token-major layout (forward's fresh buffer or run_out's caller buffer) — the epoch-1 `.contiguous()` mistake stays fixed (the word does not appear anywhere in the source; grep count 0).
- **Legality binding**: every fp16 tile load is WIDENED via `.to(tl.float32)` BEFORE its first dot use; all 4 dot call sites are (32,32)@(32,32) fp32/fp32→fp32. fp16-operand dots: 0. Non-32 dot shapes: 0.
- **-inf masking ONLY on S=83 tile-padding columns** (`tl.where(mask_n, s, -inf)` pre-softmax; exp(-inf)=0 exactly → padded keys contribute exactly zero); bidirectional — all 3 key tiles visited per query tile, NO causal mask; scale=0.125 applied post-dot as a runtime float arg (exact power of two).
- `forward` = TWO python-visible ops (`torch.empty` + ONE kernel launch); `run_out(query,key,value,out)` 4-arg surface (ONE launch, zero allocations, returns None, bitwise-equal to forward).
- STATELESS: instance attrs exactly the 4 constructor-config attrs (+nn.Module-standard `training` from `.eval()`); no caches, no workspace, no cross-call state; Triton JIT compile cache is framework-owned, one-time, absorbed by harness warmup 50.

## Sketch Primitive and Hint Conformance

- Kernel uses exactly the proven dot envelope ((32,32)@(32,32) fp32) and `num_warps=1` (frozen snapshot Constrained: proven-stable policy line); required hints `num_warps_1` and `proven_dot_envelope_32_fp32` both bound (machine table in `binding_statement_report_001.json`).
- Non-dot primitive shapes beyond the frozen proven list — strided 3D fp16 load (batch/token/head offsets, token-masked, other=0.0) widened fp32; direct-layout 3D fp16 store; `tl.full([32], -inf)`; `tl.maximum` fp32; axis-1 `tl.max`/`tl.sum` over (32,32) fp32; 2D elementwise `tl.exp` over (32,32) fp32; `tl.where` (32,32) with -inf branch; `tl.zeros` at [32]/(32,32); `tl.static_range(3)` — all established by the Decision-scoped file-backed capability probe p10 (results under `log/probes/`), per the coder contract's probe allowance; no Unsupported/Unknown requirement remains unprobed for the constructs actually used.
- `tl.argmax` not used anywhere (tie-order non-certifiability noted in the frozen snapshot); no atomics, no fp64, no trans op, no index-carrying reductions (tie-free by construction).
- DANGER tokens (compiler family + graph machinery + staging-count knob + layout-copy family incl. `.contiguous` and the bare word "graph"): all-zero; `num_warps` appears at exactly one site (the single shared launch site) with value 1.

## Attempt Ledger

| Attempt | Command (abridged) | Exit | Defect | Candidate SHA |
|---|---|---|---|---|
| 1 | authoring (initial inline launch duplicated in forward/run_out) | — | none | `4171de8d…fc2` after refactor |
| 1 | refactor to single shared `_launch` site (sibling r002 precedent: one launch site, one num_warps site, unambiguous machine audit) | — | none (pre-gate structural edit) | `4171de8d…fc2` (final; unchanged after) |
| 1 | `ast.parse` gate | 0 | none | `4171de8d…fc2` |
| 1 | DANGER token scan (16 tokens) + num_warps/dot-site/stateless counts | 0 | none — all-zero; 4 dot sites; 1 num_warps site value 1; 4 attr writes all in `__init__` | `4171de8d…fc2` |
| 1 | real-harness smoke `auto_bench.py --v0 base.py --v1 candidate --warmup 5 --repeat 10 --full-traceback` | 0 | none — `PASS accuracy` (see smoke observation below) | `4171de8d…fc2` |
| 1–2 | probe p10 | 0 (run 2) | PROBE-SIDE only: run-1 expected-value bugs (rowmax/rowsum compared the full 64-dim head against the kernel's lo 32-dim chunk; a 1-D/2-D shape mismatch in the static_range check crashed the DETAIL printer). Fixed probe-side, re-run to genuine PASS. CANDIDATE SOURCE UNTOUCHED (hash constant) | `4171de8d…fc2` |
| 1 | probes p11 / p12 | 0 / 0 | none — first-attempt PASS | `4171de8d…fc2` |

No Verifier repair requests yet; zero same-round candidate repairs consumed.

## Decision-scoped Checks (log/probes/ only — non-authoritative, no timing/benchmarks/profilers)

| Probe | Verdict | Key evidence |
|---|---|---|
| p10 capability probe (file-backed) | PASS exit 0 (run 2) | strided 3D fp16 load (batch/token/head offsets, token-masked, other=0.0) widened fp32 + direct-layout 3D fp16 store roundtrip BITWISE; axis-1 tl.max over (32,32) bitwise vs torch; masked-load padding exactly 0.0; axis-1 tl.sum (fma-order tolerance); 2D tl.exp (max_diff=1.907e-06); tl.where with -inf branch bitwise; tl.zeros [32]/(32,32) exact; tl.maximum; tl.full([32],-inf); tl.static_range(3) all-iterations. Isolation validated each non-dot primitive shape the candidate uses beyond the frozen proven list |
| p11 compile-smoke + stateless audit | PASS exit 0 | compile on cuda:0 via the real candidate path; seed42-regime allclose(atol=rtol=1e-2) with **max_abs=4.883e-04**; run_out returns None; run_out POISONED caller buffers **x2 orderings** (poison -777.0 with forward-first call ordering; poison +555.0 with run_out-first ordering) both BITWISE-equal to forward with data_ptr preserved; repeat forward bitwise-stable with FRESH allocation; instance attrs = 4 config attrs (+`training`), identical pre/post calls; zero workspace/graph/cache/plan attributes |
| p12 correctness sweep (machine table `p12_r001_sweep_result.json`) | PASS exit 0 | 6 suites (below), run_out bitwise==forward and shape/dtype ok on every suite |

### Sweep table (suites x max_abs vs base.py semantics; allclose atol=rtol=1e-2 equal_nan PASS everywhere)

| Suite | allclose | max_abs | run_out bitwise | shape |
|---|---|---|---|---|
| seed42_B2S83 (canonical regime) | PASS | 4.883e-04 | True | ok |
| boundary_B2S83 (padding-boundary: score spikes at tokens 0/32/64/82 across all 3 key tiles + value markers at every tile boundary 0/31/32/63/64/82) | PASS | 4.883e-04 | True | ok |
| extreme_B2S83 (fp16 extremes ±4096/±2048/±256/subnormals/zeros) | PASS | 2.000e+00 (=1 fp16 ULP at magnitude 2048–4096; rtol regime) | True | ok |
| seed7_B1S41 (non-target shape, stateless recompile) | PASS | 4.883e-04 | True | ok |
| seed13_B2S82 (S=82: 18-column padded tail) | PASS | 4.883e-04 | True | ok |
| seed19_B2S96 (S=96: exact 3 tiles, zero padding) | PASS | 4.883e-04 | True | ok |

Stateless-recompile routing evidence: after feeding the module 4 distinct shapes ([2,83,512], [1,41,512], [2,82,512], [2,96,512]) the framework JIT specialization cache held exactly 4 entries — per-shape constexpr specialization, zero module-owned state.

Numeric-risk characterization: online-softmax vs vendor SDPA accumulation order stays ~20x under the 1e-2 tolerance on ordinary suites; the extreme-suite diff is pure fp16 output quantization (1 ULP at large magnitudes), not softmax divergence.

## Binding Statement

- **Dot-shape audit**: 4 `tl.dot` call sites, each (32,32)@(32,32) with fp32 operands (widening casts between every fp16 load and its first dot use); 0 fp16-operand dots; 0 non-32 shapes — machine table in `binding_statement_report_001.json`.
- **num_warps**: exactly one launch site (shared `_launch`), value 1; `num_stages` absent (count 0). Kernel count: 1.
- **Addressing audit**: direct strided addressing of the `[B,S,H*D]` inputs (batch stride S*H*D, token stride H*D, head stride D — all constexpr-folded); transposed key loads (no trans op); direct fp16 final-layout stores; `.contiguous` count 0 and the bare word "contiguous" count 0 over the whole source; p10 roundtrip proves the load/store pair bitwise.
- **Stateless audit**: 4 instance-attr writes, all in `__init__` (constructor parity with baseline_adapter); post-call `__dict__` diff empty (p11); module level = 3 immutable int literals + kernel + class + input helpers.
- **AST-loader composition**: 4 imports + 3 literal assigns (_BM/_BN/_BD) + `@triton.jit` FunctionDef + ClassDef + 2 helper FunctionDefs — all retained node types; real-harness loader gate PASS.
- **Off-regime policy**: stateless per-shape constexpr specialization (4 shapes → 4 JIT specializations, live-verified); `num_kv_heads != num_heads` is outside decision-001 scope (GQA absent-by-construction, no fallback tier in this design) — documented, not silently mishandled.

## Deviations

- **D1 (observation, no code change)**: harness smoke at `--warmup 5 --repeat 10` printed `PASS accuracy; v0=0.148356 ms, v1=0.248290 ms, speedup=0.598x` (exit 0). This is a harness-printed observation, NOT a coder measurement claim (Coder ran no timing/benchmarks/profilers). The ~0.248 ms reading sits inside the decision's pre-declared honest band (0.235–0.29 ms) for the sibling-prior launcher-tax-transfers branch — exactly the two-sided falsification target this round was designed around (expected_wall_improvement_pct 0.0 declared honestly; the primary_metric 5.0% adoption bar is expected to fail honestly if the tax transfers). No design change is legal from the coder seat: the direct single-launch path with `num_warps=1` and no caching is exactly what decision-001 mandates.
- **D2 (conformance note)**: probe p10's run-1 FAILs were probe-side expected-value bugs (full-64-dim head vs the kernel's lo 32-dim chunk in the rowmax/rowsum references; a shape mismatch in the static_range DETAIL printer), NOT candidate defects — the roundtrip/exp/where/zeros/padding checks passed on run 1 already. Fixed probe-side and re-run to genuine PASS; candidate hash constant throughout (analogous to sibling r002's D2 probe-side repair note).
- **D3 (conformance note)**: GQA (`num_kv_heads != num_heads`), non-fp16 dtypes, and non-cuda devices are outside decision-001's declared scope; the stateless kernel assumes the decision-pinned fp16 `[B,S,H*D]` cuda input contract and has no fallback tier by design (sibling r002 D3 precedent).
- **D4 (conformance note, host-shape plumbing)**: block literals `_BM/_BN/_BD=32` are frozen module-level safe literals; the shape constexprs (B/S/H/D/NM/NT) derive from the live tensor shape and constructor config at call time (exactly the sibling r002 mechanism), so the frozen-at-module-definition requirement is satisfied at the block-geometry level while non-target shapes still route to fresh JIT specializations — required by the decision's own sweep mandate ([1,41,512]/[2,96,512] correctness).

## Evidence for Verifier

- Candidate: `triton_mm_encoder_attention_e2_001.py` @`4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2`.
- Probe logs under `log/probes/` (hashes below). Canonical measurement route: unchanged harness, default stream, seed 42, warmup 50 / repeat 100 interleaved pairs.

### Artifact hash ledger

```text
4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2  triton_mm_encoder_attention_e2_001.py
67b96739c35adabb713081a1f3a50649193b28eed420dc32dd512572fab26c78  rounds/decision_001.md
a1c27dbae53b1c7a74681510a0d09ced6be58ed8501f86976ce55af1b4772363  rounds/sketch_001.json
c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f  baseline_adapter.py
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  /root/CodeBuddy/20260818191200/kernelswift/auto_bench.py
0a14e062e12f2491048977dd5259b994c4dbfc344bcb4c6d873331430b3af41a  log/probes/p10_r001_capability_probe.py
73d1709a54e2d318681df0731f86b93290661f0c40777cf761ac0f2ef332c8af  log/probes/p10_r001_capability_probe.log
c6ab5fa1fcf9b0a0cffb34974c06dd3040096e4f685d7512748e7c858e98f83f  log/probes/p11_r001_compile_smoke.py
52bf81246c7715a6a04eabaf215cc4c42e37cb72da747f8327de41c86956cbc4  log/probes/p11_r001_compile_smoke.log
1ed1e2a6ab43c05226741fe464b513f8540f73fedc57d79acc32db9a69272030  log/probes/p12_r001_correctness_sweep.py
1a8853e90872f87d0da3ce77c72db78ddbdd177302b67551aca32ed3f25a296a  log/probes/p12_r001_correctness_sweep.log
d686ff938e4328ea617f4615e91588a904e3be5ca53edf1c80d256b1d1817b2c  log/probes/p12_r001_sweep_result.json
623783fd96ecfa90e77e88b59985d433ee8a31097f802d896bb82a9947630b2f  log/probes/binding_statement_report_001.json
```

## Exact Commands (all with `cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable;` prefix; interpreter `/usr/local/bin/python3`; device cuda:0)

```bash
# gates
/usr/local/bin/python3 -c "import ast; ast.parse(open('kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_001.py').read()); print('AST_PARSE_OK')"
# DANGER scan (16 tokens, all-zero) + counts: 4 tl.dot sites; 1 num_warps site value 1; 1 @triton.jit kernel; 4 attr writes all in __init__ (see binding_statement_report_001.json for the recorded table)

# real-harness smoke (exit 0, PASS accuracy)
/usr/local/bin/python3 auto_bench.py \
  --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
  --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_001.py \
  --warmup 5 --repeat 10 --full-traceback

# probes (each exit 0; p10 on run 2 after the probe-side fix)
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p10_r001_capability_probe.py
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p11_r001_compile_smoke.py
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/p12_r001_correctness_sweep.py
```

Coder claims no measurement and no verdict; classification is candidate-ready. Orchestrator owns the verification dispatch.
