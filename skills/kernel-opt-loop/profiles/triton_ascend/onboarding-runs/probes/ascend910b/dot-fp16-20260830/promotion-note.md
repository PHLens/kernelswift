# Proposed profile promotion

- Review status: `proposed`
- Implementation profile: `triton_ascend`
- Probe: `triton-ascend-dot-tiles-001` (definition `9efc2c86c27a…`, result `c0670e9be7ec…`)
- Run: `dot-fp16-20260830`
- Onboarding disposition: `promotion-pending`

## Recommendations

- `matrix.dot.fp16-fp16-fp32.tile-coverage`: `unknown` -> `constrained`
  - Scope: `{"accumulator_dtype": "fp32", "device_arch": "ascend-910b4", "lhs_dtype": "fp16", "rhs_dtype": "fp16", "target_id": "ascend910b", "tile_groups": ["multiple-of-16", "non-multiple-of-16", "power-of-two"], "toolchain": "triton 3.2.0 / torch_npu 2.7.1.post4"}`
  - Rationale: Numerically checked observed success within the probe's exact scope; v1 renderer never recommends supported.

This note is a rendering of `promotion-candidate.json`, which remains authoritative. It never edits the canonical profile; promotion requires an explicit maintainer review commit.
