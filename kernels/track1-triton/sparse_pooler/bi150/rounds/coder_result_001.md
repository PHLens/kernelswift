# Coder Result 001

Result: candidate-ready

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Decision SHA256: `0fbbdb6929e1b75f939fc2d513c28878b7a53587f33e8fcaf66401f1269256f1`
- Source canonical: `baseline_adapter.py`
- Source canonical SHA256: `359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8`
- Base SHA256: `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58`
- Candidate: `triton_sparse_pooler_001.py`
- Candidate SHA256: `f3fd85a2c913d477e2cac7f65ed1f79dd5e1b9a3a60481782dbb4acaa43d2d98`
- Selected target profile: `triton_cuda`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1), 16 SM, 16 GiB)
- Measurement fingerprint: `72be9562432197795bf6a24300483ccb2c3219b804b73258611048014cd804a9`
- Coder contract SHA256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`

## Decision Validation

`python3 skills/kernel-opt-loop/scripts/validate_decision.py
kernels/track1-triton/sparse_pooler/bi150/rounds/decision_001.md --expected-profile
triton_cuda` returned `"valid": true`. The decision's `language=triton`,
`backend=cuda`, `target_profile=triton_cuda` match the project identity and the
`triton_cuda` profile Identity/Match rules exactly.

## Implementation

### Fused kernel structure

A single `@triton.jit` kernel, `_sparse_pooler_fused_kernel`, replaces the six
post-decoder tail kernels (one `clamp_scalar` ReLU, one `log1p`, four
`reduce_kernel<MaxOps>` per-sequence max-pool). Grid is
`(num_seq, cdiv(vocab_size, BLOCK_V))` with `BLOCK_V = 1024`; each program
produces one `BLOCK_V`-wide tile of one sequence's pooled output.

Per program:
1. On-device `seq_len = seq_lens[pid_s]` and `seq_offset = sum(seq_lens[0:pid_s])`
   via a bounded prefix scan (at most 3 extra `tl.load`s; avoids the D2H sync of
   `seq_lens.tolist()` on the host).
2. Loop over the sequence rows, `tl.load` each row's vocab tile, apply the
   SPLADE activation `relu(x) = where(x > 0, x, 0)` then `log1p(relu(x)) =
   log(1 + x)`, and accumulate the column-wise max with `tl.maximum`.
3. `tl.store` the accumulated per-sequence max tile.

### Fusion guarantees

- The intermediate `[83, 30522]` activation tensor is never materialized: the
  activation is applied inline and reduced into the `acc` accumulator in the
  same kernel, eliminating the two ~10 MB write/read cycles.
- `dense` (768×768) and `decoder` (768×30522) GEMMs, GELU, and LayerNorm remain
  untouched torch library ops on the vendor TCU path.
- Output is a `list` of exactly 4 fp32 tensors each `[30522]`, in `seq_lens`
  order, matching the public contract and harness list comparator.

### log1p numerics

`tl.log1p` is not used as a primitive (unproven on this profile). Instead
`log1p(relu(x))` is expressed as `tl.log(1.0 + x)` after
`relu(x) = tl.where(x > 0.0, x, 0.0)`. Since `relu(x) >= 0`, `log(1 + x)` is
well-conditioned (no catastrophic cancellation), and the harness tolerance
`atol=1e-2, rtol=1e-2, equal_nan=True` absorbs any sub-ULP divergence from
`torch.log1p`. `relu(x)` for negative inputs yields `0.0`, matching
`F.relu`; `log1p(0) = 0` exactly in both paths.

### list output

`forward` allocates a single `(num_seq, vocab_size)` output tensor and returns
`[out[i] for i in range(num_seq)]`, a Python `list` of 4 views each `[30522]`.
The harness `compare_values` list branch recursively compares all 4 elements.

## Gate Evidence

| Gate | Command | Result |
|---|---|---|
| decision validation | `validate_decision.py ... --expected-profile triton_cuda` | `valid=true`, exit 0 |
| AST syntax | `python3 -m py_compile triton_sparse_pooler_001.py` | exit 0 |
| harness smoke | `auto_bench.py --v0_file base.py --v1_file triton_sparse_pooler_001.py --warmup 50 --repeat 100 --full-traceback` | `PASS accuracy; v0=1.058425 ms, v1=0.881216 ms, speedup=1.201x`, exit 0 |

## Conformance Notes

- The decision sketch declares a `parallel vocab over 30522` with a `guard
  vocab < 30522`; the candidate realizes this as a 2D grid over `(num_seq,
  cdiv(vocab_size, BLOCK_V))` with a `v_mask = v_offs < vocab_size` load/store
  mask. This is a syntactic realization of the same parallelization, not a
  semantic deviation.
- `num_warps`/`num_stages` are Unknown on this profile and are left unset
  (defaults), as the decision requires them to stay non-normative.
- `tl.log1p` is not asserted; `tl.log(1 + x)` is the numerically-equivalent
  elementwise lowering, consistent with the decision's own note that
  `log1p` may map to `log(1+x)`.

## Handoff

Candidate is ready for Verifier measurement. Result `candidate-ready`.

- Candidate: `kernels/track1-triton/sparse_pooler/bi150/triton_sparse_pooler_001.py`
- Candidate SHA256: `f3fd85a2c913d477e2cac7f65ed1f79dd5e1b9a3a60481782dbb4acaa43d2d98`
- Smoke: correctness PASS, `v0=1.058425 ms`, `v1=0.881216 ms`, `speedup=1.201x`
