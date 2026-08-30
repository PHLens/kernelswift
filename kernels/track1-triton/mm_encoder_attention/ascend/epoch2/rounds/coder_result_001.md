# Coder Result 001

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` (sha256 `3775c9548afc7070898ee73ead2e6ecad19225525b58052946f2ff5e3c4c0167`)
- Decision kind: `optimization`
- Sketch: `rounds/sketch_001.json` (sha256 `76818c21a7502a68b6ec5c6230607fa24bddf3e342e61d4d333990d16d639738`)
- Reference implementation: `baseline_adapter.py` (sha256 `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`)
- Candidate: `triton_mm_encoder_attention_e2_001.py`
- Candidate SHA256: `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124`
- Classification: `candidate-ready`

## Implementation Summary

One Triton kernel, one launch per forward call, grid `(B*NH,)` = `(16,)`.

- q/k/v are indexed in their native `[B, S, NH*HEAD_DIM]` layout with explicit
  strides, so nothing is transposed or copied.
- K is loaded transposed as `[HEAD_DIM, BLOCK_N]` through strides rather than
  with `tl.trans`. This is a deliberate capability decision: `tl.trans` is not in
  the reviewed capability set for `triton_ascend`, and the frozen profile only
  approves what has been probed.
- `tl.dot` does `(BLOCK_M, HEAD_DIM) @ (HEAD_DIM, BLOCK_N)` = `(128,64)@(64,128)`
  and then `(BLOCK_M, BLOCK_N) @ (BLOCK_N, HEAD_DIM)` = `(128,128)@(128,64)`, both
  fp16 inputs with fp32 accumulation. The first shape is exactly the probed
  `(128,128,64)` case with M/N swapped in naming terms; the second
  `(128,64,128)` case is inside the observed envelope but was not itself probed.
- Softmax is a plain row softmax with the invalid column range set to `-1e6`
  before the max, so `exp` cannot overflow and masked columns contribute zero.
- `S=83` fits inside `BLOCK=128`, so no KV-block loop is needed and the launch
  count stays at exactly one. The `BLOCK_M` and `BLOCK_N` hints in the Sketch
  (`128`, `128`, both `required`) are honored exactly.
- `num_warps=4` and `num_stages=1` are applied as the Sketch `preferred` hints.
  Both values are inside the profile-legal sets established by the onboarding
  probes (`num_warps` 1/2/4/8, `num_stages` 1/2/3/4).

## Sketch Conformance

| Sketch requirement | Status |
|---|---|
| `target=triton_ascend` | honored |
| `BLOCK_M=128`, `BLOCK_N=128`, `HEAD_DIM=64` (required) | honored |
| `accumulator_dtype=fp32` (required) | honored |
| `num_warps=4` (preferred) | honored |
| `num_stages=1` (preferred) | honored |
| `scope.entrypoints = [ModelNew.forward]` | only `forward` changed |
| `scope.unchanged_boundary` | `__init__`, `get_inputs`, `get_init_inputs`, output shape/dtype/device, tolerance contract all unchanged |

## Deviations

None. No algorithm substitution, dataflow change, precision change, or Host Plan
change was introduced. The Host Plan remains `not-applicable` because no host
state, allocation reuse, or cache was added; the host-side effect is strictly the
consequence of launching one kernel instead of seven.

## Compile and Smoke Evidence

- Command:
  ```bash
  cd /workspace/kernelswift-dev-4ff2094
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_001.py \
    --warmup 5 --repeat 10 --full-traceback
  ```
- Observation: `PASS accuracy; v0=0.364600 ms, v1=0.333250 ms, speedup=1.094x`
- Correctness passes at the harness default tolerance (`atol=1e-2`, `rtol=1e-2`).
- This is a smoke-tier number at `warmup 5 / repeat 10`. It is not the adoption
  measurement; Verifier owns the `warmup 50 / repeat 100` wall comparison.

## Risks Noted for Verifier

1. The second `tl.dot` shape `(128,64,128)` compiles and is numerically correct
   here, but it was not one of the eleven probed tiles. If Verifier observes any
   numerical instability at a different seed, this is the first place to look.
2. The kernel requires `S <= 128`. The campaign shape is `S=83`. A host-side
   guard raises rather than silently producing wrong output.
3. `torch.empty_like` emits an internal-format warning on this runtime. It is a
   warning only and does not affect the measured path.
