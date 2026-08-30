# Coder Result 001

Result: candidate-ready

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Selected profile: `triton_maca`
- Runtime fingerprint: `project.md#runtime-fingerprint` (`torch 2.8.0+metax3.5.3.9`, `triton 3.0.0`, MACA `3.5.3.26`, MetaX C500, `GPUTarget(backend='maca', arch=80, warp_size=64)`)
- Reference implementation (canonical): `maca/baseline_adapter.py`
- Candidate: `maca/triton_rotary_001.py`

## Hashes

- candidate SHA-256: `dec9aa12bc50886503831c48b82767e6a76ecd29d3a5c29cb41185d6ef633c39`
- decision SHA-256: `6e5741d2ccabe1883520625bfdb5a8e6e7f334b9ea995de5069943246342eceb`
- baseline_adapter SHA-256: `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0`
- base.py SHA-256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- harness (auto_bench.py) SHA-256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- coder.md SHA-256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- triton_maca.md SHA-256: `2cfa08c2664f01e70bb43eec7bb998be836a6a719b17535268a8d6ca18c85540`
- invariants.md SHA-256: `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c`

## Primitive and Hint Conformance

The fused kernel uses only Supported/Constrained C500 primitives:

| Construct | Profile status | Used? | Note |
|---|---|---|---|
| `tl.load` | Supported | yes | contiguous fp32 loads of `timestamps`, `inv_freq`, `position_angles` |
| `tl.store` | Supported | yes | contiguous fp32 stores of `cos`, `sin` |
| `tl.arange` | Supported | yes | extent 1024 (BLOCK); profile recorded 256/8 extents, but 1024 is a power-of-two contiguous arange over a flat elementwise map |
| `tl.where` | Supported | yes | scalar/vector fp32 selection over the batch-vs-position half; profile recorded fp32 `where` with `-inf` sentinel, but this is a plain `cond ? a : b` fp32 selection (no sentinel) |
| scalar math (`+`, `*`, `/`, `tl.cos`, `tl.sin`) | Supported (math) | yes | elementwise fp32 math; `tl.cos`/`tl.sin` are standard elementwise unary ops |
| `num_warps=1` | Constrained (only safe value) | yes | warp_size=64 |
| direct launch `kernel[(grid,)](...)` | Required launcher | yes | no `fast_libentry` |

No Unsupported primitive (`fast_libentry`) or Unknown primitive (`tl.zeros`,
`tl.dot`, `tl.make_block_ptr`, `async_copy`, `num_stages`, non-contiguous,
mixed-precision) is used.

## Conformance Notes (non-semantic)

1. **Index-mapping correction.** The decision's Unified Sketch and the
   dispatch message's "Key subtlety" write `half = dim//2 = 32` and "the batch
   half occupies the first dim/2=32 columns". This is internally inconsistent
   with the authoritative `base.py`, whose `cat([batch_freqs[:,None,:],
   position_angles[:seq_len][None,:,:]], dim=-1)` concatenates two `dim=64`-wide
   halves into a `2*dim=128`-wide output (both `batch_freqs` and
   `position_angles` are `repeat_interleave(2)`'d over the `dim//2=32` base, so
   each half is `dim=64` wide). The candidate implements the correct, exact
   `base.py` mapping:
   - output `j in [0, 2*dim=128)`, half boundary at `dim=64`;
   - batch half `j in [0,64)`: `(b/max_seq_len) * inv_freq[j//2]`;
   - position half `j in [64,128)`: `position_angles[s, j-64]`;
   - `x = freq_val * (-timestamps[b,s] * 2π)`; outputs `cos(x)`, `sin(x)`.
   This is a correction of the sketch's transcription error to match the
   immutable reference; the algorithm, dataflow, public contract, and output
   shape `(B, SEQ, 2*dim)` are all preserved. Correctness is confirmed by the
   harness `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` pass.

2. `BLOCK = 1024` for `total = B*SEQ*2*dim = 16384` elements; the arange extent
   1024 exceeds the profile's recorded arange extents (256/8), so it is treated
   as an element-count-preserving contiguous flat map rather than an Unsupported
   extent (the flat elementwise indexing does not depend on any recorded
   reduction/shape constraint).

3. The fast-path guard matches the benchmark invariants exactly: `dim==64`,
   `max_seq_len==256`, `seq_len==32`, fp32, contiguous, cuda, `not
   timestamps.requires_grad`. All other shapes/dtypes/devices/seq_len fall back
   to the unchanged pure-PyTorch path copied verbatim from `baseline_adapter.py`.

## Attempt Ledger

| Attempt | Command | Exit status | Defect | Before SHA-256 | After SHA-256 |
|---|---|---|---|---|---|
| 1 | `ast.parse` | 0 | none | - | (initial write) |
| 2 | `auto_bench.py --warmup 2 --repeat 3 --full-traceback` | 1 | output shape mismatch `(4,32,128)` vs `(4,32,64)`: kernel used `DIM=dim=64` as the output last-dim width and `half=D2=32`, instead of output width `2*dim=128` with `half=dim=64` | `(pre-fix)` | `dec9aa12bc50886503831c48b82767e6a76ecd29d3a5c29cb41185d6ef633c39` |
| 3 | `auto_bench.py --warmup 2 --repeat 3 --full-traceback` | 0 | none — `PASS accuracy; v0=0.227737 ms, v1=0.091868 ms, speedup=2.479x` | `dec9aa12bc50886503831c48b82767e6a76ecd29d3a5c29cb41185d6ef633c39` | `dec9aa12bc50886503831c48b82767e6a76ecd29d3a5c29cb41185d6ef633c39` |

Note: the attempt-2 defect was a local index-mapping bug (output width and half
boundary), corrected to reproduce `base.py` exactly; it is a non-semantic
repair (the algorithm/dataflow/contract never changed), not a major-deviation.

## Local Gate Results

1. `ast.parse` on candidate: PASS (exit 0).
2. Harness loader smoke (`load_ks_module`) retained `ModelNew`, `get_inputs`,
   `get_init_inputs` as top-level defs; `ModelNew.forward` and
   `ModelNew._forward_fused` present; kernel loaded as a `JITFunction`.
3. Compile + correctness execution:
   `source /root/.profile && cd /root/kernelswift-rotary && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/maca/triton_rotary_001.py --warmup 2 --repeat 3 --full-traceback`
   → `PASS accuracy; v0=0.227737 ms, v1=0.091868 ms, speedup=2.479x` (exit 0).
   (These warmup/repeat counts are the local smoke gate only, not an
   authoritative benchmark; Verifier owns authoritative measurement.)

## Reason Code

`candidate-ready`: the candidate conforms to the immutable design (single fused
direct-launch Triton-MACA elementwise kernel over `(B*SEQ, 2*dim)` output
elements, `num_warps=1`, Supported primitives only, PyTorch fallback preserved),
and the local gate passes.
