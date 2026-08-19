# CentreRandomAugmentation Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/workspace/kernelswift/.worktrees/centre-random-aug-ascend/kernels/track1-triton/centre_random_augmentation/ascend`
- base: `../base.py` (shared reference; no torch_mlu dependency)
- baseline_adapter: `baseline_adapter.py`
- harness: `/workspace/kernelswift/.worktrees/centre-random-aug-ascend/auto_bench.py`
- interpreter: `/usr/local/python3.11.15/bin/python3`
- device: `npu:0`
- implementation_language: `triton`
- implementation_backend: `ascend`
- target_profile: `triton_ascend`

## Semantics

- operator: CentreRandomAugmentation — centre coordinates, generate random rotation matrices (quaternion), apply rigid rotation + translation.
- inputs:
  - `x_input_coords`: `Tensor[256,3]` fp32 (N_ATOM=256)
  - `mask`: `Tensor[256]` fp32 (all ones)
- outputs: `Tensor[4,256,3]` fp32 (n_sample=4)
- mathematical_behavior:
  - centre: `center = mean(x_input_coords, dim=-2)` (mask all-ones so mask branch degenerates to mean)
  - `x = x_input_coords - center`; `x = x.unsqueeze(0).expand(4,-1,-1)`
  - `R = random_rotation_matrices(4)` from random quaternions (torch.rand u1,u2,u3)
  - `T = s_trans * randn(4,3)` (s_trans=1.0)
  - `x = rot_vec_mul(R[:,None], x) + T[:,None]`
  - `x = x * mask[None,:,None]`
- tolerance_and_tie_rules: `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` — loose tolerance, NOT bitwise equality. Comparison is value-based, so a candidate whose random draws differ from the reference would FAIL only if the difference exceeds 1e-2 (which a random rotation + translation almost certainly does). Correctness therefore requires reproducing the reference's random values, which the harness makes possible by resetting the RNG seed before every forward call.
- randomness_contract (KEY): The forward is NON-DETERMINISTIC in general — `random_rotation_matrices` draws `torch.rand(4)` (u1,u2,u3) and `T = s_trans * torch.randn(4,3)`, none seeded inside forward. HOWEVER the harness (`auto_bench.py`) makes the comparison deterministic:
  - `run_forward` (auto_bench.py:449) calls `set_seed(seed)` immediately before `model.forward(...)`, for BOTH v0 and v1.
  - `set_seed` (auto_bench.py:254) sets `torch.manual_seed(seed)` AND `mod.manual_seed_all(seed)` on every available accelerator (cuda/npu/mlu), so the device RNG stream is reset to an identical deterministic state before each forward.
  - `time_forward` (auto_bench.py:459) also calls `set_seed(seed)` before every timed call, so timing runs do not drift the RNG across iterations.
  - Net effect: v0 and v1 both consume the SAME deterministic random sequence (3× `rand(4)` for the quaternion, then `randn(4,3)` for translation) starting from the same seed. A candidate that issues the SAME RNG draw order (e.g. reuses torch's `rand`/`randn` for R and T, then rewrites only the pure rotation+translation math in Triton) will get bitwise-identical R and T, and the 1e-2 tolerance further absorbs any minor numerical reordering.
  - Implication for the rewrite: the candidate MUST NOT draw its own independent random numbers (e.g. `tl.rand`). It must either (a) reuse torch's seeded `rand`/`randn` for R and T, or (b) reproduce the exact same RNG consumption sequence. Drawing fresh/independent randomness will not match the reference and will fail the value comparison.
- public_contract: `ModelNew(n_sample=4, s_trans=1.0, centre_only=False)`, `forward(x_input_coords, mask) -> Tensor[4,256,3]`

## Invariants

- `base.py` is the immutable reference; bytes unchanged.
- Candidate output `Tensor[4,256,3]` fp32.
- Harness AST loader rewrites device strings.
- **Randomness invariant**: forward draws random rotation/translation via torch's device RNG (rand for quaternion u1,u2,u3; randn for T). The harness seeds per-forward-call (`set_seed` in run_forward and time_forward), so v0 and v1 draw the SAME deterministic random sequence. A candidate must NOT draw independent randomness; it must consume the SAME RNG stream (reuse torch rand/randn for R and T, or replicate the exact draw order) so R and T match bitwise, with 1e-2 allclose tolerance as slack.
- **Mask degeneracy invariant**: `get_inputs` returns `mask = torch.ones(256)` (all ones), so the mask branch degenerates: `center = mean(x, dim=-2)`, `x = x * 1.0`. `centre_only=False`, `s_trans=1.0`, `n_sample=4`.
- **Math-shape invariant**: pure elementwise 3x3 matvec over [256,3] + translation, broadcast from [4,3,3] and [4,3]; tiny tensors (256*3 = 768 elements, plus 4*3*3=36 rotation, 4*3=12 translation).
- **Host-bound hint invariant**: many small device kernel launches (rand, randn, stack, unbind, expand/contiguous, matvec) over ~768-element tensors; the dominant cost is likely host/launcher overhead, not device compute. Verifier must confirm device_ratio before any host-vs-kernel intervention.

## Runtime Fingerprint

```yaml
triton_distribution: triton
triton_version: 3.2.0
torch_version: 2.7.1+cpu
torch_npu_version: 2.7.1.post4
backend_target: triton_ascend
device_name: Ascend910B4
cube_core_num: 20
vector_core_num: 40
total_memory_bytes: 31662800896
L2_cache_size: 100663296
```

- target_profile_match: `pass`
- host: `ascend910b4`

## Measurement Regime

- harness_path: `/workspace/kernelswift/.worktrees/centre-random-aug-ascend/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `x_input_coords=[256,3] fp32; mask=[256] fp32`
- dtype: `fp32`
- device: `npu:0`
- warmup: `50`
- repeat: `100`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`

## Measurement Fingerprint

- measurement_fingerprint: `33ca785dc0b312e3d097f16bc2ea7de8f8d2dac779c04c2ac028a001f2b8aa4d`
- base_sha256: `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553`
- baseline_adapter_sha256: `7d4a79ae96328fc03a4489710f68b7f639ddea9cbd5c0f7bb45e1cec5472061b`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user

## Git Run Identity

- base_branch: `dev`
- base_commit: `d638dd3`
- run_branch: `kernel-opt/centre-random-aug-ascend`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 2.547680 | 291.9 | - | not-applicable | `baseline_adapter.py` |
| 001 | elementwise-launch-fusion | `triton_centre_random_aug_001.py` | accepted | `baseline_adapter.py` | 2.463270 | 216.06 | +17.84% | partially-confirmed | `triton_centre_random_aug_001.py` |
| 002 | no-change (abort) | - | aborted | `triton_centre_random_aug_001.py` | - | - | - | - | `triton_centre_random_aug_001.py` |

## Reproduction

```bash
cd /workspace/kernelswift/.worktrees/centre-random-aug-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/ascend/baseline_adapter.py --warmup 50 --repeat 100
```
