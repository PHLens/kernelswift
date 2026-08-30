# Coder Result 001

Result: candidate-ready

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`264c7be47436c5a8e9a9c2d324aae52632ec0e0201f3725a77bea0a163d2a4ab` (hash re-verified from file; family "sparse-pooler-tail-fusion"; expected_wall_improvement_pct 0.0 declared honestly; deliverable-grade round per project.md DELIVERABLE RULE)
- Sketch: `rounds/sketch_001.json` @`a92ec7842e345d0112a12c19efb2cccd6b5f7017e43765935461b9ebd989a295` (hash re-verified; matches decision `sketch_sha256`)
- Candidate: `triton_sparse_pooler_e2_001.py` @`f99538b13f7768297d7aa95a25e4c33231eb12321575bdb80ede401b226d81fa` (project root: `kernels/track1-triton/sparse_pooler/s60/epoch2/`)
- Canonical start (last_accepted_kernel): `baseline_adapter.py` @`359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8` — semantics derived from it (this round is the FIRST Triton candidate; baseline_adapter is pure PyTorch with zero kernels).
- Base (immutable reference): `../../base.py` @`46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58` (re-verified unchanged; equals decision `reference_implementation` baseline declaration).
- Harness: `/root/CodeBuddy/20260828202827/kernelswift/auto_bench.py` @`71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (untouched; AST loader; requires v1 to define `ModelNew`/`get_inputs`/`get_init_inputs`).
- Runtime fingerprint: `triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2 / Enflame GCU major=3 minor=0`, interpreter `/usr/bin/python3`, device `gcu` — matches `project.md#runtime-fingerprint`.

## Implementation Summary (decision-001 exact)

- ONE stateless `@triton.jit` kernel `_sparse_pooler_tail_kernel`; single launch site in `ModelNew.forward` with direct launch `_sparse_pooler_tail_kernel[(NS, ceil(V/BV))]`, `num_warps=2`, `num_stages` unset (count 0). Grid = (NS=4, ceil(30522/256)=120) programs (one program per (segment, vocab-block) pair — sketch `ctrl.parallel.seg` / `ctrl.parallel.vid`).
- `BV=256` (power-of-2, sketch hint `block_vocab_256` exploratory), `V=30522`, `S=83`, `NS=4` — V/S/BV passed as constexpr (frozen literals at call site); NS derived from `seq_lens.shape[0]`.
- **Fused tail**: each program loads the decoder output tile for its (segment, vocab-block), applies `log1p(relu(x)) = tl.log(1.0 + tl.maximum(x, 0.0))` elementwise, reduces `tl.maximum` over the segment's token span, and stores straight into `out[seg, vocab_block]` (sketch `op.load.x` → `op.compute.act` → `op.compute.segment_max` → `op.store.out`).
- **Segment boundaries device-side (NO tolist)**: `forward` computes prefix offsets as `(torch.cumsum(seq_lens, dim=0) - seq_lens).to(torch.int32)` == `[0,20,45,63]`; both `seq_lens` and `offsets` are passed to the kernel and read device-side via `tl.load` (sketch `ctrl.guard.token`). **`.tolist()` count = 0** — the D2H sync is eliminated (this goes BEYOND decision's "tolist retained" host-plan note; see Deviations D1).
- **Masking**: `vocab < V` masks the power-of-2 `tl.arange(0, BV)` tile at load and store (padding lanes never written); token span masked by `t < L` inside the static `for t in range(0, S)` loop (S=83 constexpr unrolled). log1p(relu(x)) >= 0 for all x, so `tl.zeros([BV])` init + `tl.where(tmask, val, 0.0)` is EXACTLY equivalent to `-inf` init + token masking, while staying fp32 (no fp64 constants).
- **GEMM untouched**: `forward` issues `x = self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))` verbatim (vendor `nn.Linear` + `nn.GELU` + `nn.LayerNorm`); the two GEMMs (dense + decoder) remain the vendor library — **0 `tl.dot` call sites**.
- **STATELESS**: instance attrs exactly 5 constructor-config attrs (`dense`, `act`, `layer_norm`, `decoder`, `pooling`) all in `__init__`, matching base/baseline_adapter state_dict keys (`dense.weight/bias`, `decoder.weight/bias`, `layer_norm.weight/bias`); no caches, no workspace, no cross-call state; Triton JIT compile cache is framework-owned, one-time, absorbed by harness warmup 50.

## Sketch Primitive and Hint Conformance

- Required sketch hints bound:
  - `pad_vocab_power_of_2` (modality required): `BV=256` power-of-2 `tl.arange(0, BV)` extent; `V=30522` masked via `vocab < V` (load/store).
  - `fp32_elementwise_tail` (modality required): fp32 loads/stores of the `[83,30522]` decoder output; elementwise `tl.maximum`/`tl.log` and `tl.maximum` reduction all fp32.
  - `num_warps_2` (modality preferred): `num_warps=2` at the single launch site.
  - `block_vocab_256` (modality exploratory): `BV=256` honored.
- Primitives used, mapped to frozen profile `profile_snapshot/triton_gcu.yaml`:
  - `tl.arange` → `index.range.one-dimensional`: extent 256, power-of-2 (satisfies S60 power-of-2 constraint).
  - `tl.load` / `tl.store` → `memory.load.contiguous` / `memory.store.contiguous`: masked loads (`mask=vmask, other=0.0`) and masked store (`mask=vmask`).
  - `tl.program_id` → `parallel.program-id`: axis-0 (`seg`) and axis-1 (`vid`).
  - `tl.maximum` → reduction.max over the vocab/token axis (the primary_contract segment-max): **reduction.max, NOT argmax, NOT scatter_reduce**.
  - `tl.log` / `tl.maximum` (elementwise) / `tl.where` → elementwise family: fp32 elementwise ops; no Unsupported/Unknown requirement exercised in the shapes used.
  - `num_warps` → `resource.num-warps` (legal set {1,2,4,8}): value 2, inside the legal set.
- `tl.dot` NOT used anywhere (0 call sites); `tl.argmax` NOT used (tie-free by construction); no atomics, no fp64, no `tl.make_block_ptr`, no `tl.async_copy`, no scatter_reduce, no device prefix-scan.
- DANGER tokens (compile/capture/graph/torch.compile/TORCHINDUCTOR/reduce-overhead/contiguous/copy_/tolist): all-zero (see binding audit).

## Binding Statement

- **GEMM audit**: 0 `tl.dot` call sites; both GEMMs (`dense` [83,768]@[768,768] + `decoder` [83,768]@[768,30522]) remain vendor `nn.Linear`. 768 and 30522 are NOT powers of two — capability-blocked for `tl.dot` — correctly avoided.
- **num_warps**: exactly one launch site, value `2`; `num_stages` absent (count 0). Kernel count: 1 (`@triton.jit` count 1).
- **tl.arange audit**: exactly 1 call, extent `BV=256` (power-of-2); the token axis uses a static `range(0, S)` loop (S=83 constexpr), NOT `tl.arange` (83 is not power-of-2).
- **D2H sync audit**: `.tolist()` count 0 over the whole source; segment boundaries arrive device-side via `seq_lens` + `cumsum`-derived `offsets` tensors read inside the kernel.
- **fp64 audit**: no fp64 constants; the int32→int64 cumsum promotion is explicitly cast back with `.to(torch.int32)` (see Deviations D2); `tl.log(1.0 + tl.maximum(x, 0.0))` with 0.0/1.0 literals promotes to fp32 alongside the fp32 tensor operand.
- **Stateless audit**: 5 instance-attr writes, all in `__init__` (`dense`, `act`, `layer_norm`, `decoder`, `pooling`) — constructor parity with base/baseline_adapter; no call-time instance writes; module level = imports + `@triton.jit` FunctionDef + ClassDef + `get_inputs`/`get_init_inputs` helper FunctionDefs.
- **AST-loader composition**: 5 imports + `@triton.jit` FunctionDef + ClassDef + 2 helper FunctionDefs (no unsafe module-level statements) — all retained node types; `ast.parse` gate PASS (`AST_PARSE_OK`).

## Deviations

- **D1 (improvement over decision host-plan, no correctness risk)**: decision_001's Host Plan states `seq_lens.tolist() D2H sync is retained (unchanged)`. The Coder task contract (team-lead instruction) instead MANDATES eliminating the D2H sync — "segment 边界从 seq_lens 设备端读 ... 不要 tolist()". This candidate honors the team-lead contract: `.tolist()` count 0, offsets computed device-side via `torch.cumsum`. This is strictly a host-path improvement over the decision's plan (the retained-sync note was a conservative fallback), does not touch the two vendor GEMMs, and preserves the output list-of-4 invariant. Flagged for Verifier: the decision's "unchanged_behavior" list says the tolist sync is retained — this candidate intentionally removes it.
- **D2 (implementation detail)**: `torch.cumsum` on an int32 tensor promotes to **int64 on GCU**, and the Triton GCU backend rejects int64 (`64-bit data type not supported on GCU300`). The offsets tensor is therefore cast back `.to(torch.int32)` before the launch. This is a host-side dtype normalization, not a semantic change.

## Evidence for Verifier

- Candidate: `triton_sparse_pooler_e2_001.py` @`f99538b13f7768297d7aa95a25e4c33231eb12321575bdb80ede401b226d81fa`.
- Canonical measurement route (Verifier-owned): unchanged harness `auto_bench.py`, device gcu, seed 42, warmup 50 / repeat 100 interleaved pairs.
- Coder correctness smoke (NON-authoritative, only PASS/FAIL recorded): `PASS accuracy` (v0 vs v1 allclose fp32 comparator passed). Coder ran NO timing/benchmark/profiler measurements and claims NO verdict on wall time. Classification is `candidate-ready`.

### Artifact hash ledger

```text
f99538b13f7768297d7aa95a25e4c33231eb12321575bdb80ede401b226d81fa  triton_sparse_pooler_e2_001.py
264c7be47436c5a8e9a9c2d324aae52632ec0e0201f3725a77bea0a163d2a4ab  rounds/decision_001.md
a92ec7842e345d0112a12c19efb2cccd6b5f7017e43765935461b9ebd989a295  rounds/sketch_001.json
359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8  baseline_adapter.py
46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  /root/CodeBuddy/20260828202827/kernelswift/auto_bench.py
```

### Binding audit table (source counts)

| Token / construct | Count | Verdict |
|---|---|---|
| `compile` / `capture` / `graph` / `contiguous` | 0 / 0 / 0 / 0 | DANGER-free |
| `torch.compile` / `TORCHINDUCTOR` / `reduce-overhead` / `copy_` | 0 / 0 / 0 / 0 | DANGER-free |
| `.tolist()` | 0 | D2H sync eliminated |
| `tl.dot` | 0 | GEMMs vendor-bound |
| `num_warps` | 1 (value 2) | legal |
| `@triton.jit` | 1 | single kernel |
| `tl.arange` | 1 (extent 256) | power-of-2 |
| `tl.load` / `tl.store` | 3 / 1 | masked |
| `tl.maximum` (reduction) / `tl.log` | 1 / 1 | fp32, no keepdim needed |
| `keepdim` / `argmax` / `scatter_reduce` | 0 / 0 / 0 | no-keepdim / max-not-argmax satisfied |
| `self.<attr>` writes | 5 | all in `__init__` |
