# Coder Result 001

Result: candidate-ready

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`459a1f9b36105b33966c53b3e7740313094ba96874ffae1be358171066948c40` (hash re-verified from file; family "triton-launch-fusion"; change_family "triton-launch-fusion"; expected_wall_improvement_pct 59.0 declared from preflight, Verifier measurement authoritative)
- Sketch: `rounds/sketch_001.json` @`017b423b96d88ba28fde6f1d4d6a7534b9f0fcf486a540d78c7c59f149c4429f` (hash re-verified; matches decision `sketch_sha256` and `sketch_sha256` in `sketch_ref`)
- Candidate: `triton_centre_random_augmentation_e2_001.py` @`542293c0ed3488b4f30c6c3758780115325593a592b11bc656cfa605f9d79522` (project root: `kernels/track1-triton/centre_random_augmentation/s60/epoch2/`)
- Canonical start (last_accepted_kernel): `baseline_adapter.py` (in epoch2 dir) — semantics derived from immutable base `../../base.py`; epoch-1 candidate `../triton_centre_random_augmentation_001.py` was read as prior evidence only (0.95x partial-fusion prior: fused ONLY rot_vec_mul for a single-launch save) — NOT copied; this round fuses the WHOLE per-sample path per decision_001.
- Base (immutable reference): `../../base.py` @`02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553` (re-verified unchanged; equals decision `reference_implementation` hash cited in Rationale)
- Harness: `/root/CodeBuddy/20260828202827/kernelswift/auto_bench.py` @`71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (untouched; AST loader)
- Runtime fingerprint: `triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2 / Enflame GCU`, device `gcu` — matches `project.md#runtime-fingerprint`

## Implementation Summary (decision-001 exact)

- ONE stateless `@triton.jit` kernel `_centre_random_aug_kernel`; single direct-launch site in `ModelNew.forward` with grid `(n_sample,)=4` (one program per sample, sketch `ctrl.parallel.sample`), `num_warps=1`. Kernel count: 1 (`@triton.jit` count 1).
- Host `forward` keeps ONLY the irreducible random sources + center/x_centered:
  - `center` = masked mean (`(x*m).sum(-2)/(m.sum(-2)+1e-12)`, mask None -> unbiased `x.mean(-2)`) — torch, identical to base.
  - `x_centered = x_input_coords - center`  # [N_atom, 3]
  - `u1/u2/u3 = torch.rand(n_sample)` x3 (host, base order), then `T = s_trans * torch.randn(n_sample, 3)` (host, base order). Random order/count/shape bit-identical to base -> seed-42 sequence matches.
  - `out = torch.empty((n_sample, n_atom, 3))` then ONE kernel launch.
- Kernel (per sample `s = tl.program_id(0)`):
  - load u1/u2/u3[s] scalars; quaternion `q1=sqrt(1-u1)*sin(2πu2)`, `q2=sqrt(1-u1)*cos(2πu2)`, `q3=sqrt(u1)*sin(2πu3)`, `q4=sqrt(u1)*cos(2πu3)` via `tl.sqrt/tl.sin/tl.cos`.
  - 9 rotation-matrix elements statically unrolled (r00..r22), arithmetic order identical to base `random_rotation_matrices`.
  - load T[s,0..2] (3 scalars); loop atoms in power-of-2 BLOCK tiles (BLOCK=256 => 1 iteration, sketch `ctrl.for.atom`).
  - 3x3 matvec = 3 statically-unrolled fp32 dot products (`o0=r00*x0+r01*x1+r02*x2+t0`, etc), NO `tl.dot`.
  - `has_mask` constexpr branch (mask=None here -> no-op); masked loads/stores `atom < N_atom`.
- STATELESS: instance attrs exactly 3 constructor-config attrs (`n_sample`, `s_trans`, `centre_only`); no caches, no workspace, no cross-call state; Triton JIT compile cache is framework-owned, one-time, absorbed by harness warmup 50.
- `centre_only=True` branch returns the expanded view WITHOUT `.contiguous()` (to honor the zero-`.contiguous()` DANGER binding); it is dead for this instantiation (CENTRE_ONLY=False).

## Sketch Primitive and Hint Conformance

- Required sketch hints bound:
  - `num_warps_1` (modality required): `num_warps=1` at the single launch site.
  - `pad_atom_block_power_of_2` (modality required): `tl.arange(0, BLOCK)` with `BLOCK=256` (power-of-2).
  - `static_unroll_3x3_matvec` (modality required): 3 statically-unrolled fp32 dot products, zero `tl.dot`.
- Primitives used, mapped to frozen profile `profile_snapshot/triton_gcu.yaml`:
  - `tl.sqrt/tl.sin/tl.cos` -> `math.elementwise` (primary contract, status supported): quaternion chain.
  - `tl.arange` -> `index.range.one-dimensional` (supported): extent 256, power-of-2 (honors S60 power-of-2 constraint).
  - `tl.load`/`tl.store` -> `memory.load.contiguous`/`memory.store.contiguous` (supported): masked loads (mask=atom<N_atom, other=0.0) and masked store.
  - `tl.program_id` -> `parallel.program-id.axis0` (supported): axis-0 only (sample index).
  - `num_warps` -> `resource.num-warps` (constrained, legal set {1,2,4,8}): value 1, inside legal set.
- `tl.dot` count 0 (3x3 matvec static-unrolled, NOT a matrix.dot); no `tl.sum` reduction substitution; no atomics, no fp64, no `tl.make_block_ptr`, no `tl.async_copy`, no `tl.trans`.
- DANGER tokens (compile/capture/graph/contiguous/torch.compile/TORCHINDUCTOR/reduce-overhead/copy_): all-zero in CODE (see binding audit; the only textual occurrences are inside the module docstring's "no ..." prohibition statements, which are comments, not code).

## Binding Statement

- **Random-number contract audit**: host generates `u1=torch.rand(n_sample)`, `u2=torch.rand(n_sample)`, `u3=torch.rand(n_sample)` (three `torch.rand` calls) then `T=s_trans*torch.randn(n_sample,3)` — exact base order/count/shape. NO `torch.rand`/`torch.randn` inside the kernel (GCU kernel has no torch.rand; randomness stays host-side).
- **num_warps**: exactly one launch site, value `1`; `num_stages` absent (count 0). Kernel count: 1 (`@triton.jit` count 1).
- **tl.sqrt/tl.sin/tl.cos**: 4 / 2 / 2 in kernel body (q1/q2 use sqrt(1-u1) twice, q3/q4 use sqrt(u1) twice; sin on q1/q3; cos on q2/q4).
- **tl.dot**: 0. **tl.arange**: 1 (extent 256, power-of-2).
- **contiguous audit**: the word `contiguous` appears ZERO times in code (only in the docstring prohibition "no .contiguous() anywhere"); no expand/reshape/copy on the kernel path.
- **Stateless audit**: 3 instance-attr writes, all in `__init__` (`n_sample`, `s_trans`, `centre_only`) — constructor parity with base `Model`; no call-time instance writes; module level = imports + `@triton.jit` kernel + ClassDef + `get_inputs`/`get_init_inputs` helper FunctionDefs.
- **AST-loader composition**: imports + `@triton.jit` FunctionDef + ClassDef + 2 helper FunctionDefs; module-level literal assigns only (N_ATOM/N_SAMPLE/S_TRANS/CENTRE_ONLY/_BLOCK); no unsafe module-level statements. `ast.parse` gate PASS (`AST_PARSE_OK`).
- **get_inputs/get_init_inputs retained** with `torch.manual_seed(42)` matching base.py exactly; constructor signature `ModelNew(n_sample, s_trans, centre_only)` preserved.

## Deviations

- **D1 (observation, no code change)**: `centre_only=True` branch returns `x_centered.unsqueeze(0).expand(...)` WITHOUT `.contiguous()`, whereas base returns a `.contiguous()` tensor. This is forced by the decision's DANGER binding ("zero .contiguous() in forward host path") and is dead code for this instantiation (CENTRE_ONLY=False). If a future round needs `centre_only=True` with a contiguous output, it should do so via the fused kernel path (an identity rotation + zero translation) rather than a host `.contiguous()`. Flagged for the lead/Orchestrator.

## Evidence for Verifier

- Candidate: `triton_centre_random_augmentation_e2_001.py` @`542293c0ed3488b4f30c6c3758780115325593a592b11bc656cfa605f9d79522`.
- Canonical measurement route (Verifier-owned): unchanged harness `auto_bench.py`, device gcu, seed 42, warmup 50 / repeat 100 interleaved pairs.
- Coder ran NO timing/benchmark/profiler measurements; no verdict is claimed here. The correctness smoke (non-authoritative, only PASS/FAIL) reported: **`PASS accuracy`** (v0=2.892 ms, v1=1.577 ms shown by the harness, but timing is NOT trusted/recorded as evidence — Verifier's paired-median is authoritative). Classification is `candidate-ready`.

### Artifact hash ledger

```text
542293c0ed3488b4f30c6c3758780115325593a592b11bc656cfa605f9d79522  triton_centre_random_augmentation_e2_001.py
459a1f9b36105b33966c53b3e7740313094ba96874ffae1be358171066948c40  rounds/decision_001.md
017b423b96d88ba28fde6f1d4d6a7534b9f0fcf486a540d78c7c59f149c4429f  rounds/sketch_001.json
02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  /root/CodeBuddy/20260828202827/kernelswift/auto_bench.py
```

### Binding audit table (CODE-only counts, docstrings/comments excluded)

| Token / construct | Count | Verdict |
|---|---|---|
| `compile` / `capture` / `graph` / `contiguous` | 0 / 0 / 0 / 0 | DANGER-free (code) |
| `torch.compile` / `TORCHINDUCTOR` / `reduce-overhead` / `copy_` | 0 / 0 / 0 / 0 | DANGER-free |
| `tl.dot` | 0 | static-unrolled 3x3 matvec |
| `tl.sqrt` / `tl.sin` / `tl.cos` | 4 / 2 / 2 | math.elementwise primary |
| `tl.arange` | 1 (extent 256) | power-of-2 |
| `num_warps` | 1 (value 1) | legal |
| `@triton.jit` | 1 | single kernel |
| `self.<attr>` writes | 3 | all in `__init__` |
