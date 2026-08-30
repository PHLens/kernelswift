# triton_gcu Approved Evidence

This directory holds approved, hash-verified probe-result payloads referenced by
the canonical `profile.yaml` capability matrix. Each file is the normalized
result JSON of one versioned probe run against the S60 (Enflame GCU) runtime.

## Records

| Evidence id | Capability | Probe | Result SHA-256 |
|---|---|---|---|
| `gcu-dot-mult16-001` | `matrix.dot.fp16-fp16-fp32.mult-of-16-tiles` | `triton-gcu-dot-mult16-001` | `1dfed09d…` |
| `gcu-num-warps-001` | `resource.num-warps` | `triton-gcu-num-warps-001` | `8209949d…` |

## Runtime scope

All records are scoped to one S60/GCU runtime:

- `target_id`: `s60`
- `device_arch`: `gcu-major3-minor0` (major=3, minor=0)
- `toolchain`: triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2

Evidence never transfers across vendors, devices, architectures, or toolchains.
