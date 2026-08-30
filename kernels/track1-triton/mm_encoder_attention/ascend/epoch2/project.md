# MM Encoder Attention Optimization Project (Epoch 2, Ascend)

## Project Identity

- schema_version: 2
- skill_version: 3.0.0
- contract_version: 3
- semantic_contract: typed-sketch-v1
- attribution_contract: verdict-v1
- project_root: `/workspace/kernelswift-dev-4ff2094/kernels/track1-triton/mm_encoder_attention/ascend/epoch2`
- base: `../../base.py` (shared device-neutral reference at the operator level)
- baseline_adapter: `baseline_adapter.py`
- harness: `/workspace/kernelswift-dev-4ff2094/auto_bench.py`
- interpreter: `/usr/local/python3.11.15/bin/python3`
- device: `npu:0`
- implementation_language: `triton`
- implementation_backend: `ascend`
- target_profile: `triton_ascend`
- target_id: `ascend910b`
- implementation_profile_id: `triton_ascend`
- implementation_profile_snapshot_ref: `state/implementation_profile_snapshot/profile.yaml`
- implementation_profile_snapshot_sha256: `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321`
- project_capability_claim_ref: `state/project_capability_claim.json`
- prior_lineage: `epoch-1 naive deliverable 0.92x (triton_attn_001.py) preserved intact at ../`
- base_branch: `trunk`
- base_commit: `db09613`
- run_branch: `kernel-opt/mmenc-attn-e2-ascend-20260830`

The concrete `target_id` (`ascend910b`) is distinct from the
`implementation_profile_id` (`triton_ascend`); API compatibility never transfers
capability evidence across vendors, devices, architectures, or toolchains.

## Semantics

- operator: MmEncoderAttention — `F.scaled_dot_product_attention` over query/key/value; non-GQA (`num_kv_heads == num_heads == 8`, `head_size == 64`). Self-attention (`q_len == kv_len == 83`).
- inputs: `query/key/value [2, 83, 512]` fp16 contiguous on `npu:0` = `[bsz, seq, num_heads*head_size]`
- outputs: `[2, 83, 512]` fp16 contiguous on `npu:0`
- mathematical_behavior: `scale = 1.0 / sqrt(64) = 0.125`; per (batch, head): `q = view/transpose -> [bsz, 8, 83, 64]`, `softmax((q @ k^T) * 0.125) @ v`, then `transpose(1,2).reshape(bsz, 83, 512)`. No dropout, no attention mask, no causal bias.
- tolerance_and_tie_rules: harness default `torch.allclose(atol=1e-2, rtol=1e-2)`; output compared only; harness clones v0 inputs into v1 so both see identical input bytes.
- public_contract: `ModelNew(num_heads=8, head_size=64, num_kv_heads=8).forward(query, key, value)` plus `get_inputs()` and `get_init_inputs()` returning `[8, 64, 8]`.
- DELIVERABLE RULE (binding): the competition submission is always the best correctness-PASS Triton candidate, even when it does not beat base.

## Invariants

- `../../base.py` is the immutable reference; its bytes are unchanged after baseline adapter generation (verified in Phase 0).
- Candidate output shapes, dtypes, device placement, numerical semantics, and the public constructor/forward contract remain compatible with `base.py`.
- The harness is loaded through its AST loader; direct import success is insufficient. The harness rewrites the `"cuda"` literal to `npu`, so candidate `get_inputs` must not hardcode `"cuda"`.
- `import torch_npu` must precede any NPU allocation or Triton launch; synchronization boundary is `torch.npu.synchronize()`.
- Wall time is measured by the unchanged harness with seed setup and NPU synchronization included; device kernel duration is diagnostic evidence only.
- Frozen profile snapshot governs capability legality: fp16 `tl.dot` is `constrained` (all 11 probed tiles numerically correct, including non-multiple-of-16 shapes), `num_warps` 1/2/4/8 and `num_stages` 1/2/3/4 are legal. `make_block_ptr`, `async_copy`, `vectorize`, and a fast launcher remain Unknown and may not be declared normative.

## Runtime Fingerprint

```yaml
triton_distribution: triton
triton_version: 3.2.0
torch_version: 2.7.1+cpu
torch_npu_version: 2.7.1.post4
backend_target: triton_ascend
backend_version: null
device_name: Ascend910B4
device_arch: ascend-910b4
cube_core_num: 20
vector_core_num: 40
cann_version: 9.0.0
```

- target_profile_match: `pass`
- discovery_commands: `python3 --version`; `python3 -c 'import torch, torch_npu, triton; ...'`; `python3 -c 'import triton.backends; print(triton.backends.backends.keys())'`
- discovered_at: `2026-08-30T00:00:00Z`
- host: `ascend910b4`

## Measurement Regime

- harness_path: `/workspace/kernelswift-dev-4ff2094/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `query/key/value=[2,83,512] fp16; num_heads=8; head_size=64; num_kv_heads=8`
- dtype: `fp16`
- device: `npu:0`
- warmup: `50`
- repeat: `100`
- timing_order: `ordered reference/candidate pairs; each pair uses the unchanged harness`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profile_mode: `forward`
- profile_warmup: `20`
- profiler_scopes: `reference_baseline_adapter,candidate_baseline_adapter`
- correctness_command: `cd /workspace/kernelswift-dev-4ff2094 && python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /workspace/kernelswift-dev-4ff2094 && python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/baseline_adapter.py --warmup 50 --repeat 100`
- profiler_command: `cd /workspace/kernelswift-dev-4ff2094 && python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/baseline_adapter.py --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_000_forward_50iter.pt.trace.json`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `1b1822d7b74a8cd41411a27fcbc18a89cb50b1cfefb9fdac2585cdd520e9a79a`
- base_sha256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- baseline_adapter_sha256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- measurement_settings_canonical_json: `{"device":"npu:0","dtype":"fp16","profile_iterations":50,"profile_mode":"forward","profile_warmup":20,"repeat":100,"shape":"query/key/value=[2,83,512] fp16; num_heads=8; head_size=64; num_kv_heads=8","warmup":50}`
- fingerprint_command: `sha256(base.py || NUL || auto_bench.py || NUL || canonical JSON settings with sort_keys=True and separators=(',',':'))`
- fingerprint_compatibility: identical to the epoch-1 baseline fingerprint, so epoch-1 and epoch-2 wall numbers are directly comparable.

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `trunk`
- base_commit: `db09613`
- run_branch: `kernel-opt/mmenc-attn-e2-ascend-20260830`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.347800 | 104.1264 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_mm_encoder_attention_e2_001.py` | accepted | `baseline_adapter.py` | 0.327770 | 13.4064 | +10.30% | confirmed on device and launch, partially-confirmed on host | `triton_mm_encoder_attention_e2_001.py` |
| 002 | `rounds/decision_002.md` | - | aborted | `triton_mm_encoder_attention_e2_001.py` | - | - | - | not-applicable: device-only ceiling is 4.09%, below the 5% threshold | `triton_mm_encoder_attention_e2_001.py` |
| 003 | `rounds/decision_003.md` | `triton_mm_encoder_attention_e2_003.py` | accepted | `triton_mm_encoder_attention_e2_001.py` | 0.298240 | 13.4224 | +17.40% | confirmed: device held at +0.095%, allocation 1 -> 0 | `triton_mm_encoder_attention_e2_003.py` |
| 004 | `rounds/decision_004.md` | `triton_mm_encoder_attention_e2_004.py` | no-improvement | `triton_mm_encoder_attention_e2_003.py` | 0.295850 | 13.3228 | +2.89% (bar +5%) | partially-confirmed: direction certain, magnitude short | `triton_mm_encoder_attention_e2_003.py` |

Round 004 measured the M1 `fast_libentry` launch path under strict pair-by-pair
alternation against the incumbent `e2_003` in one window: four pairs, every pair
favouring the candidate, best pair `+3.474%`, median `+2.8874%` against a `+5%` bar.
The capability is proven and the direction is not in doubt — 11 of 12 control blocks
and 4 of 4 strict pairs favour `e2_004` — but M1's entire launch saving is only
`19.305 us` on a `~300 us` wall, and roughly a quarter of it is lost in the
conversion to wall time. M1 is a real but sub-threshold improvement.

**Measurement methodology correction, established in round 004.** A speedup only
cancels drift when it is compared against another speedup drawn from the *same*
reference measurements. Within a single turn `base.py`'s median moved `-5.96%`
(`0.376040` → `0.353615`) while the candidate moved only `-2.26%`
(`0.295850` → `0.289150`), swinging `e2_004`'s speedup by `4.13%`
(`1.270171` → `1.217744`). Both candidates must therefore be measured in strict
pair-by-pair alternation inside one window; a cross-window or cross-turn gap below
about `0.1%` is two orders of magnitude under the method's own instability and must
not be defended.

Cumulative against the epoch-2 starting point (`baseline_adapter.py` at
`0.347800 ms`), the accepted candidate is `0.298240 ms`, a **14.25%** gain.

### Host-side decomposition (Verifier Level 2, round 003)

These are the numbers that bound the rest of the epoch. Measured in one process
and one regime against a wall median of `297.410 us`:

| Quantity | e2_001 | e2_003 |
|---|---:|---:|
| forward alone, no sync | 233.645 | 206.375 |
| forward + synchronize | 288.290 | 258.190 |
| harness-fixed (harness wall minus forward) | 93.890 | **91.035** |

- This round's lever: `27.270 us/call`.
- **Harness-fixed ceiling: `91.035 us/call` = 30.61% of wall.** No host round can touch it.
- Residual Python wrapper still in `forward`: `22.635 us/call` = 7.61% of wall.
- **The Triton launch path itself costs `183.740 us/call` = 61.78% of wall**, measured as a bare `kernel[grid](...)` with a preallocated output and no Python wrapper. It also issues exactly one `aten.empty.memory_format` per launch on its own. This is the concrete `launch-path-reduction` target and it is about seven times larger than everything left in `ModelNew.forward`.

Two harness facts that shape any future round: `set_seed` is untimed but its
device work is drained by the timed `sync_devices()`, so the seed op is billed
inside the timed region (`39.220 us`); and `sync_devices()` costs `11.96 us/call`
more than a bare synchronize because `_iter_accelerators()` calls
`torch.npu.is_available()` every time.

Round 002 abort rationale: the complete device budget (`13.4064 us/call`) is smaller
than the 5% adoption budget (`16.3885 us/call` of the `327.770 us` wall median), so
even eliminating device time entirely yields only 4.09%. Launch count is already
`1.00`, and the per-call host residual where all headroom lives is outside the
maintainer constraint for this epoch.

Reference (base.py) median was `0.349625` ms at `116.1696 us/call` device time;
the row records the candidate-side values per `rounds/report_000.md`. This
baseline drifted `+9.04%` versus the epoch-1 baseline (`0.320635` ms) under an
identical measurement fingerprint.

Orchestrator appends one row only after a terminal round transition is validated
and committed.

## Reproduction

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/baseline_adapter.py --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_000_forward_50iter.pt.trace.json
```
