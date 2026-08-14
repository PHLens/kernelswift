# Coder State

Concise implementation facts and resume context for the current candidate.
No canonical-state claims.

## Current round

- Round: 001
- Phase: coding (complete from Coder's side; awaiting Orchestrator handoff to Verifier)
- Decision: `rounds/decision_001.md` (SHA-256 `0816c943...`, `proceed`, change_scope=mixed)
- Source canonical: `baseline_adapter.py` (SHA-256 `d7e69ed4...`)
- Candidate: `triton_sparse_pooler_001.py` (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)
- Result file: `rounds/coder_result_001.md`
- Classification: `candidate-ready`

## Candidate facts

- Fuses `relu + log1p + per-segment max pool` into one Triton kernel `_sparse_pooler_max_kernel` at module top level.
- Grid `(num_seq, cdiv(vocab_size, BLOCK_V))` = `(4, 30)` with `BLOCK_V=1024`, `num_warps=1`.
- On-device `seq_offset = sum(seq_lens[0:pid_s])` via a bounded `for i in range(pid_s)` loop; eliminates `seq_lens.tolist()` D2H sync.
- Accumulator `tl.full((BLOCK_V,), -inf)`; per-row `tl.where(x>0, x, 0)` (relu) + `tl.log(1+x)` (log1p) + `tl.maximum(acc, x)`.
- `dense/GELU/LayerNorm/decoder` left as PyTorch library ops.
- `pooling == "sum"` keeps the Python fallback (off the measured hot path).
- `ModelNew` constructor and `forward` signature preserved; four nn.Module attributes unchanged for `load_state_dict`.
- `get_inputs` / `get_init_inputs` preserved.

## Probe outcomes (resume reference)

- `tl.maximum`, `tl.log`, `tl.where` all compile and run on MLU590-H8 / triton 3.2.0 / torch_mlu 1.32.0 with `num_warps=1`.
- Both `from triton.runtime import fast_libentry` and `from triton.runtime.fast_libentry import fast_libentry` importable; not used (ordinary launch is the allowed fallback).
- `ast.parse` ok; harness `load_ks_module` exposes ModelNew/get_inputs/get_init_inputs/kernel.
- Correctness smoke vs `base.py`: 4/4 outputs `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`, max_abs_diff=1.79e-07.
- Harness end-to-end smoke (warmup=5, repeat=5): `PASS accuracy; speedup=1.513x` (smoke only; Verifier owns the 50/100 measurement).

## Repair budget

- Repairs used this round: 0 of 2. No semantic or syntax defects required repair.

## Ownership

- Coder owns: `triton_sparse_pooler_001.py`, `rounds/coder_result_001.md`, `state/coder_state.md`.
- Coder must not edit: decision, team-state, project.md, base.py, harness, target profile, Verifier-owned files.
