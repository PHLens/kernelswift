# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: 1
- last_completed_round: `001`
- accepted_kernel: `triton_mhc_head_compute_mix_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `R000 baseline device-bound (926.395 us/call, 132.88 kernels/call, ratio 0.611); R001 kernel-fusion accepted (+87.17% wall, 1.433128→0.183889 ms, kernel 132.88→1.0, device 924.79→12.996 us/call, ratio 0.0755 host-bound); R002 abort (measurement-bound: remaining ~92.5% wall is harness-fixed seed+sync)`
- open_hypotheses: `<none — all device-side fusion exhausted; host side is harness-fixed; no falsifiable ≥5% intervention remains>`
- artifact_read_hashes: `base.py 4c5167f6..., baseline_adapter.py ceebdc61..., triton_mhc_head_compute_mix_001.py a98b1b12..., auto_bench.py 3d4fa4ee..., designer.md d32060e9... (see Artifact Read Hashes table)`

## Current Bottleneck

- Host-bound with no compressible device work: `device_ratio = 12.996 / 183.889 ≈ 0.0755` (report_001.md). Device is a single fused kernel at theoretical minimum; remaining ~92.5% wall is harness-fixed per-iteration `set_seed` + `sync_devices` overhead outside the candidate boundary.

## Recent Three-round Evidence

- R000 (Phase 0, `baseline_adapter.py`): device-bound, `926.395 us/call`, `132.88 kernels/call`, `device_ratio 0.611`; Sinkhorn loop dominated by ~120 tiny reduce/div/add kernels.
- R001 (`decision_001.md`, `kernel-fusion`): accepted, wall `1.433128 → 0.183889 ms` (+87.17%), kernel count `132.88 → 1.0`, device `924.79 → 12.996 us/call`. `tl.static_range(19)` confirmed infeasible (>300s compile); dynamic `tl.range(19)` used.
- R002 (`decision_002.md`, `no-change`): abort (measurement-bound). No falsifiable ≥5% intervention; device optimal, host harness-fixed.

## Open Hypotheses or Checks

- `<none — the operator is fused to a single kernel and the residual host time is harness-fixed; no stable ≥5% improvement can be justified without a user amendment to the harness/contract>`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | 002 |
| `skills/kernel-opt-loop/references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | 002 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_cuda.md` | `0fa3d4d7b9dba37536bc322a65318a9ff6455c70eb3511924af34883d052193d` | 002 |
| `kernels/track1-triton/mhc_head_compute_mix/base.py` | `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5` | 002 |
| `kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py` | `ceebdc6185de4c980156a7833073678a0964fb7ccb5edf74b42be6156652eaed` | 002 |
| `kernels/track1-triton/mhc_head_compute_mix/bi150/triton_mhc_head_compute_mix_001.py` | `a98b1b12593d858ca29c787afa939a3ae0061df4ec6b51aa9a0fe7fa43c6b473` | 002 |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | 002 |
