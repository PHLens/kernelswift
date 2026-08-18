# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: 3
- last_completed_round: 002
- accepted_kernel: `triton_rotary_001.py` (report_001 accepted, 46.33% improvement)
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `round 001 accepted (kernel-fusion, 46.33%) -> round 002 candidate-ready (row-parallel-vectorization), pending Verifier`
- open_hypotheses: `H-002 (row-per-program + eliminate redundant loads/div/mod/tl.where) -> candidate produced, awaiting Verifier`
- artifact_read_hashes: `see Artifact Read Hashes table below`

## Phase 0 Baseline Adapter Confirmation

- phase: Phase 0 (complete)
- baseline adapter: confirmed correct.

## Operator Observations

- No sibling-backend reference.
- `forward` is elementwise/reduction only (arange/repeat_interleave/broadcast/cat/cos/sin); NO matmul / `tl.dot`.
- `inv_freq [32]` and `position_angles [256,64]` are `register_buffer`s precomputed in `__init__`; Round 2 adds derived `batch_freq_base [64]` (interleaved inv_freq, non-persistent).
- Output tuple of two fp32 `[4,32,128]` tensors. `angle = -timestamps[b,t]*2π` depends only on `(b,t)`, not on `c` — enables row-per-program scalarization.

## Current Bottleneck

- `<Verifier-backed: report_001 shows single fused kernel ~48us/call device-bound (device_ratio ~14.5%), wall 0.334ms>`

## Recent Three-round Evidence

- round 000: baseline established (0.581820 ms reference, 14 kernels/call, device_ratio ~8%).
- round 001: `accepted` — `triton_rotary_001.py` (kernel-fusion, flat-index grid BLOCK=128, num_warps=1). Wall 0.622330 -> 0.333955 ms (46.33%); device_us unchanged ~48us (host-bound confirmed).
- round 002: `candidate-ready` — `triton_rotary_002.py` (row-per-program, scalar b/t/ts/angle, two 64-wide halves, batch_freq_base buffer, no tl.where). Awaiting Verifier.

## Open Hypotheses or Checks

- H-002: reduce device time via row-per-program + eliminating redundant dual load / per-lane div-mod / tl.where. Candidate produced; pending authoritative Verifier measurement.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` | `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341` | 000 |
| `baseline_adapter.py` | `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0` | 000 |
| `triton_rotary_001.py` | `51a9a33b82f550abfd80400bb0748b74fd181d0f3c4fd4b5d70b4ca1f5d6984e` | 001 |
| `triton_rotary_002.py` | `98e86e48f00a25254561ad5bf8ef91824c87e5bb9dfa0279f221633d910b07ab` | 002 |
| `rounds/design_001.md` | `fdf2f9f9a5660cea68d7546206f9767ddbbe94f61cba0ae34056b4c4b9825786` | 001 |
| `rounds/design_002.md` | `ee54366c2034343ad6206e02f9c8dd0d6340178d66ef6d70283e78932f61c4d0` | 002 |
| `rounds/report_001.md` | (not hashed) | 002 |
| `project.md` | `39fbcd74b2ab8fc513d1dfa48852a3074f0476c9a1d318f8a3d41c338ab5850c` | 000 |
| `team-state.md` | `fcb67e48b44238ee75a986476fded09742f283ced8a3d452a707f7f269597b53` | 000 |
| `../skills/kernel-opt-loop/prompts/coder.md` | `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196` | 000 |
| `../skills/kernel-opt-loop/prompts/coder_targets/triton_ascend.md` | `db54aa6269174f7f7d8707c6a084a36c4451b6826e8d5153003ce7e1b0523cc8` | 000 |
