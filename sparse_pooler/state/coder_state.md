# Coder State

Concise implementation facts and resume context for the current candidate.
No canonical-state claims.

## Current round

- Round: 002
- Phase: coding (complete from Coder's side; awaiting Orchestrator handoff to Verifier)
- Decision: `rounds/decision_002.md` (SHA-256 `0d39de9e280f6ffa2cc3d1d3322d393fa400eb8f405b7e7ee3ceb3ef845b3dd4`, `proceed`, change_scope=kernel)
- Source canonical: `triton_sparse_pooler_001.py` (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)
- Candidate: `triton_sparse_pooler_002.py` (SHA-256 `62dc853db5423cb5d99ad53433f3fb35919abe901a64d6e3acb3d815ac678248`)
- Result file: `rounds/coder_result_002.md`
- Classification: `candidate-ready`

## Candidate facts

- Byte-identical copy of `triton_sparse_pooler_001.py` with exactly one line changed: `BLOCK_V = 1024` -> `BLOCK_V = 2048` in `ModelNew.forward` (line 85).
- Kernel body (`_sparse_pooler_max_kernel`) unchanged; `BLOCK_V` is a `tl.constexpr` so the kernel recompiles automatically with the new value.
- Grid `(num_seq, cdiv(vocab_size, BLOCK_V))` = `(4, 15)` with `BLOCK_V=2048`, `num_warps=1` (down from `(4, 30)` = 120 programs to `(4, 15)` = 60 programs).
- Last vocab tile covers offsets 28672..30719 (1850 in-bounds, 198 masked); existing `v_mask = v_offs < vocab_size` handles the partial tile on both load (`other=-inf`) and store.
- On-device `seq_offset = sum(seq_lens[0:pid_s])` via a bounded `for i in range(pid_s)` loop; unchanged.
- Accumulator `tl.full((BLOCK_V,), -inf)`; per-row `tl.where(x>0, x, 0)` (relu) + `tl.log(1+x)` (log1p) + `tl.maximum(acc, x)`; unchanged.
- `dense/GELU/LayerNorm/decoder` left as PyTorch library ops; unchanged.
- `pooling == "sum"` keeps the Python fallback (off the measured hot path); unchanged.
- `ModelNew` constructor and `forward` signature preserved; four nn.Module attributes unchanged for `load_state_dict`.
- `get_inputs` / `get_init_inputs` preserved.
- `kernel_count_per_call` remains 5 by construction (no kernels added or removed).

## Probe outcomes (resume reference)

- `ast.parse` ok; harness `load_ks_module` exposes ModelNew/get_inputs/get_init_inputs.
- Correctness smoke vs `base.py`: 4/4 outputs `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`, max_abs_diff=1.79e-07.
- Harness end-to-end smoke (warmup=5, repeat=5): `PASS accuracy; v0=0.905974 ms, v1=0.615262 ms, speedup=1.473x` (smoke only; Verifier owns the 50/100 measurement).
- No fallback probes exercised: the normative `BLOCK_V=2048` compiled and ran correctly on the first attempt, so `BLOCK_V=4096` and other `num_warps` values were not probed.

## Repair budget

- Repairs used this round: 0 of 2. No semantic or syntax defects required repair.

## Ownership

- Coder owns: `triton_sparse_pooler_002.py`, `rounds/coder_result_002.md`, `state/coder_state.md`.
- Coder must not edit: decision, team-state, project.md, base.py, harness, target profile, Verifier-owned files.
