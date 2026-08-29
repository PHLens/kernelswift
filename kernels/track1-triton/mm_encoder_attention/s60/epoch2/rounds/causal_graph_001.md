# Causal Graph 001 — Direct single-kernel MHA via tl.dot

## Nodes

- `cn.dispatch-collapse`: mechanism — one direct-launched Triton kernel (grid=16) replaces the base 2-launch vendor-SDPA dispatch and 28 aten host ops; expected to reduce runtime_launch_count_per_call 2.0 → 1.0 and remove host overhead.
- `cn.device-time-delta`: mechanism — the kernel's device floor replaces base's ~118us device floor; probe measured candidate 148.6us vs base 139.9us (unfavorable -6.2% pre-adoption), the honest risk for round 1.

## Edges

- `op.store.out` → `cn.dispatch-collapse` (single kernel writes final layout, no host repack)
- `op.compute.qk` → `cn.device-time-delta` (tl.dot QK^T dominates kernel device time)
- `op.compute.pv` → `cn.device-time-delta` (tl.dot PV second device-time contributor)
- `op.compute.mask_softmax` → `cn.device-time-delta` (softmax within-program reduction)

## Measurable linkage

- dispatch collapse measured via `runtime_launch_count_per_call` in the summary trace (2.0 → 1.0).
- device-time delta measured via wall median delta against report_000 baseline under the same measurement regime (GCU device-duration unavailable; launch-API-time + wall delta are the Level 1 proxies).
