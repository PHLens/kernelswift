# Proposed profile promotion

- Review status: `proposed`
- Implementation profile: `triton_ascend`
- Probe: `triton-ascend-num-stages-001` (definition `fc4baa5a8109…`, result `3c4b32f7cbeb…`)
- Run: `ns-20260830`
- Onboarding disposition: `promotion-pending`

## Recommendations

- `resource.num-stages`: `unknown` -> `constrained`
  - Scope: `{"block": 64, "device_arch": "ascend-910b4", "dtype": "fp32", "target_id": "ascend910b", "toolchain": "triton 3.2.0 / torch_npu 2.7.1.post4", "values": [1, 2, 3, 4]}`
  - Rationale: Numerically checked observed success within the probe's exact scope; v1 renderer never recommends supported.

This note is a rendering of `promotion-candidate.json`, which remains authoritative. It never edits the canonical profile; promotion requires an explicit maintainer review commit.
