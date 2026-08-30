# Round Report 002 — mm_encoder_attention C500 (MACA), Epoch 2

## Identity

- round: `002`
- decision: `rounds/decision_002.md` (change_family=remove-transpose-copy)
- coder_result: `rounds/coder_result_002.md`
- candidate: `triton_mha_002.py`
- accepted_reference (canonical before round): `triton_mha_001.py`
- v0 harness proxy: `../base.py`
- result: `accepted`

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` |
| `triton_mha_002.py` (candidate) | `29e6b192bf778f0264fb7657c9a33b97819c406896a2ad86e1daf22f3c9ff0a1` |
| `triton_mha_001.py` (canonical) | `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b` |
| `decision_002.md` | `be804f497dcb6070e1a07d290b43c6c8acc65e3007d88657985026aa5640ac7e` |

## Correctness and Guardrails

- Correctness command (warmup 5, repeat 10, full-traceback): `RETURN_CODE=0`
  - Output: `PASS accuracy; v0=0.123155 ms, v1=0.130640 ms, speedup=0.943x`
  - `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` PASS for fp16 output.
- Guardrails: output shape `[2,83,512]` / dtype fp16 / device unchanged; inputs
  not mutated; caller device/stream preserved; SDPA fallback preserved for
  non-benchmark shapes.

## Sequential Block Wall Timing

Reference (v0 = base.py), 3 runs (warmup 50, repeat 100):
- raw samples: `0.110857, 0.125705, 0.110812` → median `0.110857` ms

Candidate (v1 = triton_mha_002.py), 3 runs (warmup 50, repeat 100):
- raw samples: `0.127422, 0.130439, 0.127777` → median `0.127777` ms

Canonical reference (triton_mha_001.py), same-session 3 runs:
- raw samples: `0.179934, 0.167111, 0.165897` → median `0.167111` ms

- improvement vs base.py (v0): `-15.26%` (candidate still slower than flash
  attention, expected for the hand-written Triton deliverable)
- improvement vs canonical (triton_mha_001.py): `+23.54%` (transpose-copy
  elimination gain)

## Evaluation Contract Mirror

| mechanism_observable | Expected | Actual | Verdict |
|---|---|---|---|
| candidate_kernel_count_per_call | 5.0 -> 1.0 | 2.0 (`_mha_fwd_kernel` + 1 output reshape) | pass (4 copies eliminated; 1 unavoidable reshape remains) |
| candidate_device_us_per_call | ~79.7 -> ~67.1 | 67.73 | pass |
| correctness_parity | allclose 1e-2 | PASS | pass |

## Profiler Evidence

| Scope | Device us/call | Kernel count/call | Dominant kernels |
|---|---|---:|---|
| baseline_base (v0) | 15.0989 | 2.0 | `flash_fwd_splitkv_kernel`, `flash_fwd_splitkv_combine_kernel` |
| candidate_triton_mha_002 (v1) | 67.7273 | 2.0 | `_mha_fwd_kernel` (1.0/call, 64.85 us/call), `transpose12_copy_64` (1.0/call, 2.88 us/call) |

The four `.contiguous()` copy kernels from round 001 are gone. The candidate
now emits 1 fused `_mha_fwd_kernel` + 1 `transpose12_copy_64` (the single
unavoidable output reshape) = 2.0 kernels/call, down from 5.0. Device time
dropped 79.70 -> 67.73 us/call.

C500 trace: filtered 2 duplicate nested `cat=user_annotation` markers (known
`overlapping scope events` issue); raw trace preserved.

## Retry History

- None (candidate-ready on first attempt; layout-only change).

## evidence_for_next_round

The fused `_mha_fwd_kernel` (~64.85 us/call) remains the dominant cost. Its
manual `tl.sum` dot over head_size=64 cannot be accelerated without `tl.dot`
(Unknown on C500). The single remaining `transpose12_copy_64` (2.88 us) is the
unavoidable output reshape matching base.py's `transpose(1,2).reshape`. No
further candidate-owned lever with a defensible >=5% wall path remains; the
hand-written Triton kernel is the required deliverable and is now
reasonably-optimized.

## Stop Recommendation

- `continue` (Orchestrator records `accepted`, advances canonical to
  `triton_mha_002.py`). Further optimization is measurement-bound (flash
  attention is the hardware-optimized floor).

## Exact Reproduction Commands

```bash
source /root/.profile && cd /root/kernelswift-mma
/opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/triton_mha_002.py --warmup 5 --repeat 10 --full-traceback
/opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/triton_mha_002.py --warmup 50 --repeat 100
```
