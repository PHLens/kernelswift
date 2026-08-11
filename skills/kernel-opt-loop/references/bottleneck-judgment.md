# Bottleneck Judgment

How to decide which of three classes — device-bound, host-bound, measurement-bound — is the current round's bottleneck, and which concrete optimization to pick within that class.

## The single most important ratio

```
device_ratio = sum(kernel dur) / wall_time
```

where `sum(kernel dur)` is the sum of `dur` for all `cat == "kernel"` events in the profiler JSON for ONE forward call, and `wall_time` is auto_bench's reported per-call wall time (microseconds, from `time_forward` median).

Compute this number FIRST every round. It maps to a bottleneck class:

| `device_ratio` | Class | What to optimize |
|---:|---|---|
| > 80% | **device-bound** | The kernel itself (fuse ops, change GEMM strategy, reduce compute) |
| 20%–80% | **mixed** | Both device and host have room. Pick device first if cheaper; otherwise pick host. |
| < 20% | **host-bound** | Host overhead (launcher, routing ops, allocator, context managers) |
| < 5% AND wall stuck | **measurement-bound** | Harness fixed costs (set_seed, sync_devices). Stop. |

Don't optimize across classes — if `device_ratio = 70%` and you optimize device, the wall drop is bounded by ~30%, and you can attribute the result cleanly. If you optimize host at the same time, you can't tell which helped.

## Procedure

### Step 1 — Get device time

Profile ONE forward call (or 50 iters, take total/50). Extract kernel time:

```bash
jq -r '.traceEvents[] | select(.cat == "kernel") | .dur' \
  <op>/log/<NNN>.pt.trace.json \
| awk '{s+=$1; c++} END {printf "total=%.2fus  count=%d  avg=%.2fus\n", s, c, s/c}'
```

If you only care about your own kernel (not all PyTorch ops it calls):

```bash
jq -r '.traceEvents[] | select(.cat == "kernel" and (.name | startswith("_fused_moe"))) | .dur' \
  <op>/log/<NNN>.pt.trace.json \
| awk '{s+=$1; c++} END {printf "kernel total=%.2fus  count=%d  avg=%.2fus\n", s, c, s/c}'
```

The first form is "all kernels launched during one forward" — use this for `device_ratio` because wall time includes ALL of them. The second is "just my Triton kernel" — use this to track your kernel's own progress across rounds.

### Step 2 — Compute device_ratio

```
device_ratio = (sum all kernel dur in one forward) / (auto_bench wall time per call)
```

Example (v5 fused_moe): all-kernel dur ≈ 21 us, auto_bench wall = 138 us. `device_ratio = 21/138 ≈ 15%` → **host-bound**.

### Step 3 — If host-bound, break down the host overhead

`host_overhead = wall - device_time`. To attribute it to sources, run three measurements in sequence on the same process:

1. **A. auto_bench `time_forward`** — the authoritative wall number.
2. **B. minimal loop**: `for _ in range(N): sync(); out = kernel(...); sync()` — no `set_seed`, no `build_case`, just the kernel call. The median × N approximates "kernel + launcher + bare sync" cost.
3. **C. B + `set_seed(seed)` before each call** — measures `set_seed` overhead = C - B.
4. **D. C + `sync_devices()` (sync both cuda and mlu if both visible) after each call** — measures `sync_devices` overhead = D - C.
5. **A - D** = remaining overhead from `build_case` + `load_state_dict` + accuracy run state.

Example (v5 fused_moe, 1 forward):
- A = 135 us (auto_bench)
- B = 82 us (kernel 21 us + launcher ~60 us)
- C = 94 us → `set_seed` = 12 us
- D = 134 us → `sync_devices` = 40 us (cuda+mlu double sync)
- A - D = 1 us → no build_case effect at this round (but can be ~24 us when build_case state differs)

Total breakdown: device 21 us (15%) + launcher 60 us (44%) + `set_seed` 12 us (9%) + `sync_devices` 40 us (29%) + other 2 us (3%) = 135 us.

### Step 4 — Decide the round's target

Based on the breakdown:

- **launcher is the biggest non-device chunk** → optimize launcher (`fast_libentry`, drop `torch.mlu.device()` context, cache output buffer). This was v3 + v5 of fused_moe.
- **routing PyTorch ops are visible in trace as separate kernels** → fuse them into the Triton kernel. This was v2 of fused_moe.
- **`sync_devices` is the biggest chunk AND it's harness overhead** → measurement-bound. Don't try to fix it in the kernel. Either declare done, or switch shape to make device time dominate again.
- **device_ratio > 50%** → optimize the kernel. Look at the trace to see which kernel dominates:

```bash
jq -r '.traceEvents[] | select(.cat == "kernel") | [.name, .dur] | @tsv' \
  <op>/log/<NNN>.pt.trace.json \
| awk -F'\t' '{a[$1]+=$2; c[$1]++} END {for (n in a) printf "%s\tcount=%d\ttotal=%.2fus\tavg=%.2fus\n", n, c[n], a[n], a[n]/c[n]}' \
| sort -t= -k3 -rn | head
```

The top row is the dominant kernel. If it's your own Triton kernel, look inside it for what's expensive (e.g. `tl.exp` in routing, elementwise outer product instead of `tl.dot`, redundant `tl.load`s). If it's a library kernel (CNNL topk, scatter), that's a candidate to fuse into your Triton kernel.

### Step 5 — Sanity-check the choice

Before committing to the round's target, answer:

- **Is the optimization expected to move the bottleneck class?** (e.g. host→device, or staying in class but reducing the dominant component.) If no — it's noise, don't do it.
- **Is the expected improvement > 5%?** The workflow spec says < 5% doesn't go into main. Re-run if needed to confirm.
- **Can I attribute the result cleanly?** (Only ONE thing changed.) If you also touched something else, you can't tell what helped.

## Worked example (fused_moe, all 5 rounds)

| Round | wall (us) | all-kernel dur (us) | device_ratio | Class | Target |
|---|---:|---:|---:|---|---|
| 0 (base) | 6940 | ~2700 | 39% | mixed | (write v1 — eliminate mask/scatter) |
| 1 (v1) | 564 | 21 | 4% | host-bound | fuse routing into kernel (v2) |
| 2 (v2) | 218 | 23 | 11% | host-bound | `fast_libentry` + cache output (v3) |
| 3 (v3) | 153 | 23 | 15% | host-bound | `tl.dot` for GEMM (v4) — picked device because host was close to floor |
| 4 (v4) | 164 | 21 | 13% | host-bound | drop `torch.mlu.device()` (v5) |
| 5 (v5) | 138 | 21 | 15% | host-bound | stop — remaining host is `set_seed` + `sync_devices` (harness) |

Note v3→v4: `device_ratio` was 15% (host-bound), but the host overhead was already close to the floor (`set_seed` 12 + `sync_devices` 40 + launcher 60 = 112 us, floor for this harness). Continuing to optimize host had diminishing returns, so even though it's host-bound by ratio, the round picked a device optimization (v4 `tl.dot`) — and indeed wall didn't drop (v4 164 > v3 153, within noise). v5 then got the last bit of host overhead (context manager) and dropped to 138.

The lesson: `device_ratio` tells you where the time is, but you also need to know whether the host overhead is **compressible** (launcher, your own code) or **fixed** (harness). Optimize compressible pieces; stop when only fixed remains.

## Compressible vs fixed host overhead (cheat sheet)

| Source | Compressible? | How |
|---|---|---|
| Triton launcher default path | yes | `fast_libentry()` |
| `torch.empty_like` per forward | yes | cache output on ModelNew instance |
| `with torch.mlu.device(...)` context | yes | drop it (caller sets device) |
| Routing PyTorch ops (softmax/topk/cast) | yes | fuse into kernel |
| `fast_libentry` residual path | partly | `num_warps`/`num_stages` tuning; or rewrite launcher in C++ |
| `set_seed` per forward | no (harness) | — |
| `sync_devices` syncing multiple accelerators | no (harness) | — |
| `build_case` + `load_state_dict` state diff | no (harness) | — |

If the breakdown shows the compressible items are gone, stop — further optimization requires either changing the harness or changing the shape.
