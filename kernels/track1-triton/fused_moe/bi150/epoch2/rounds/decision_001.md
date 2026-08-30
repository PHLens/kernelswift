# Decision 001

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "optimization",
  "round": "001",
  "reference_implementation": "baseline_adapter.py",
  "reference_report": "rounds/report_000.md",
  "language": "triton",
  "backend": "cuda",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "mixed",
  "change_family": "manual-graph-replay-fused",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "6a46d4fd67b0cbce7a34ce41eac0c2b4cc19f00dd6e6098cf91a60e879634cb4",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_cuda.yaml",
  "implementation_profile_snapshot_sha256": "dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "fcba080f084be2791c43bbe45baaaff695cb2b4a72cc4053a3e070ae6912cff5"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "kernel_launch_bound",
  "intervention": "Replace the one-Triton-launch ungrouped expert kernel with a two-Triton-launch grouped-dispatch pipeline (counting sort + per-expert grouped GEMM at BLOCK_M=16) replayed through a manual torch.cuda.CUDAGraph over static workspaces, so the per-call python launcher tax is paid once at capture instead of twice per call.",
  "allowed_changes": [
    "add a counting-sort Triton kernel that buckets the 166 (token, k) rows by expert into static index/count/offset buffers",
    "replace the single ungrouped expert kernel (grid (8,), BLOCK_M=256) with a grouped expert kernel on a static grid (8, 11) at BLOCK_M=16 with on-device empty-tile early exit",
    "add a manual torch.cuda.CUDAGraph capture over static fp16 workspaces with a direct-address guarded replay fast path and an eager fallback",
    "sweep num_warps over {1,2,4} as an in-round pre-adoption configuration sweep and adopt the argmin that stays bitwise-identical",
    "keep torch.softmax, torch.topk, the renormalize div, and the w1/w2 fp16 casts as aten ops inside the captured region"
  ],
  "invariants": [
    "public contract unchanged: Model(num_experts, top_k, hidden_size, intermediate_size, renormalize=True) and forward(hidden_states, router_logits)",
    "torch.topk tie semantics preserved verbatim; no Triton reimplementation of selection or sorting",
    "output shape [83,128], dtype fp16, all finite, within atol=rtol=1e-2 of base.py under seed 42",
    "base.py untouched (sha256 21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d, 3598 bytes)",
    "no host data-dependent control flow inside the captured region: static grid, no .item(), no D2H read",
    "no returned graph-pool or workspace memory",
    "no reduction.sum and no reduction.argmax; counting sort uses tl.static_range masked adds only",
    "w1/w2 remain fp32 nn.Parameter; cast happens inside the captured region"
  ],
  "expected_wall_improvement_pct": 31.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_001.json",
  "sha256": "6a46d4fd67b0cbce7a34ce41eac0c2b4cc19f00dd6e6098cf91a60e879634cb4"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "module-level fp16 workspaces allocated in __init__ before capture",
    "manual torch.cuda.CUDAGraph capture of the routing + two Triton launches",
    "per-call direct-address replay fast path guarded by data_ptr equality",
    "copy-out of the static out_ws into a fresh invocation-owned tensor"
  ],
  "state_owner": "Model owns out_ws, sorted_rows, sorted_w, expert_counts, expert_offsets, and the captured graph; all are created in __init__ on cuda:0 and live for the module lifetime",
  "lifetime": "graph and workspaces are created lazily on the first forward call after construction and persist for the module lifetime; they are never recreated per call",
  "allocation_reuse": "out_ws is resized only when num_tokens changes and is otherwise reused; sorted_rows/sorted_w are sized once for the worst case T*K=166 rows; expert_counts/expert_offsets are fixed at E=8; the returned tensor is always a fresh torch.empty_like(out_ws) filled by copy_, never the workspace itself",
  "cache_key": [
    "hidden_states.data_ptr()",
    "router_logits.data_ptr()",
    "hidden_states.shape",
    "hidden_states.dtype",
    "router_logits.shape",
    "router_logits.dtype"
  ],
  "invalidation": "any cache_key mismatch, any capture failure, or KS_E2_REPLAY=0 falls through to the eager path, which recomputes from scratch; correctness never depends on the replay path being taken",
  "concurrency": "single-device, single-stream, no cross-thread sharing; the graph is captured and replayed on the current stream and the module is not safe for concurrent multi-stream use",
  "device_stream_behavior": "capture and replay run on the current torch cuda stream; the eager fallback launches on the same stream; no stream is created, switched, or synchronized beyond the copy-out the harness already performs",
  "unchanged_behavior": [
    "routing arithmetic: torch.softmax -> torch.topk -> renormalize -> fp16 cast, all unchanged from the epoch-1 candidate",
    "output values: identical within atol=rtol=1e-2 on both the replay and eager paths",
    "output shape, dtype, and finiteness",
    "w1/w2 parameter dtype and state_dict() contents",
    "determinism: 20 consecutive forward calls under a fixed input produce byte-identical outputs"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-001",
  "intervention": "Replace the one-Triton-launch ungrouped expert kernel with a two-Triton-launch grouped-dispatch pipeline (counting sort + per-expert grouped GEMM at BLOCK_M=16) replayed through a manual torch.cuda.CUDAGraph over static workspaces, so the per-call python launcher tax is paid once at capture instead of twice per call.",
  "expected_causal_chain": [
    "the counting sort groups rows by expert",
    "BLOCK_M drops from 256 to 16, so GEMM work falls from 100.66 MFLOP to about 6.29 MFLOP and padding falls from 12.34x to about 1.3x",
    "active programs rise from 8 to about 16, filling all 16 SMs instead of half",
    "device time falls from 140.84 us/call to about 64 us/call",
    "N_triton rises from 1 to 2, giving the graph 170 us of launcher tax to remove against 112 us of R plus F",
    "the graph front-end is paid once at capture instead of twice per call",
    "host time falls by about 77 us/call",
    "wall time falls by about 153 us/call"
  ],
  "primary_metric": {
    "name": "wall_time",
    "expected_improvement_pct": 5.0
  },
  "causal_graph": {
    "nodes": [
      "counting_sort_groups_rows_by_expert",
      "block_m_drops_256_to_16",
      "gemm_work_falls_100p66_to_6p29_mflop",
      "active_programs_rise_8_to_16",
      "device_time_falls_140p84_to_64us",
      "n_triton_rises_1_to_2",
      "graph_removes_170us_launcher_tax",
      "host_time_falls_77us",
      "wall_time_falls_153us"
    ],
    "edges": [
      ["counting_sort_groups_rows_by_expert", "block_m_drops_256_to_16"],
      ["block_m_drops_256_to_16", "gemm_work_falls_100p66_to_6p29_mflop"],
      ["block_m_drops_256_to_16", "active_programs_rise_8_to_16"],
      ["gemm_work_falls_100p66_to_6p29_mflop", "device_time_falls_140p84_to_64us"],
      ["active_programs_rise_8_to_16", "device_time_falls_140p84_to_64us"],
      ["n_triton_rises_1_to_2", "graph_removes_170us_launcher_tax"],
      ["graph_removes_170us_launcher_tax", "host_time_falls_77us"],
      ["device_time_falls_140p84_to_64us", "wall_time_falls_153us"],
      ["host_time_falls_77us", "wall_time_falls_153us"]
    ]
  },
  "mechanism_observables": [
    {
      "name": "host_triton_launcher_tax_removed_us",
      "expectation": "decrease: the replay path removes about 170 us/call of python launcher tax across the two Triton launches, measured at least 40 us"
    },
    {
      "name": "host_submission_count_per_call",
      "expectation": "decrease: the 9.82 per-call launches collapse to a single graph submission plus the copy-out, ending below 4.0"
    },
    {
      "name": "device_us_per_call_nonreplay_tier",
      "expectation": "decrease: the eager tier drops from 140.84 us/call toward about 64 us/call, improving by at least 40 us"
    },
    {
      "name": "device_expert_kernel_us_per_call",
      "expectation": "decrease: _grouped_expert_kernel replaces the 55.80 us ungrouped kernel on the same 16x fewer GEMM work"
    },
    {
      "name": "device_aten_math_us_per_call",
      "expectation": "decrease: routing and cast aten items fall as the two Triton launches absorb work around the frozen 39.44 us topk"
    },
    {
      "name": "best_num_warps",
      "expectation": "directional: the num_warps sweep over {1,2,4} reports the argmin under a bitwise-identical-output tie-break"
    },
    {
      "name": "host_replay_sync_us",
      "expectation": "increase: the replay adds the R replay-sync term, about 66 us/call, which is the known cost the launcher tax must beat"
    },
    {
      "name": "graph_capture_stability",
      "expectation": "directional: 80 capture attempts and 80 replays with zero capture errors and zero diverged outputs"
    },
    {
      "name": "device_us_per_call",
      "expectation": "directional: the replay scope carries the about 46 us graph front-end, so this is read only as a cross-check against the non-replay tier, never as a falsification trigger"
    }
  ],
  "guardrails": [
    "correctness:pass",
    "output shape [83,128]",
    "output dtype fp16",
    "all finite",
    "base.py sha256 unchanged at 21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d",
    "manifest.json untouched",
    "seed 42 on every command",
    "determinism: 20 consecutive forward calls byte-identical"
  ],
  "profiling_level": "targeted",
  "falsification_rules": [
    {
      "id": "FR-1",
      "observable": "host_triton_launcher_tax_removed_us",
      "threshold": "improved by less than 40 us",
      "reads": "host half failed"
    },
    {
      "id": "FR-2",
      "observable": "device_us_per_call_nonreplay_tier",
      "threshold": "failed to improve by at least 40 us",
      "reads": "device half failed"
    },
    {
      "id": "FR-3",
      "observable": "host_submission_count_per_call",
      "threshold": "did not fall below 4.0",
      "reads": "graph mechanism absent on the timed path"
    },
    {
      "id": "FR-4",
      "observable": "best_num_warps",
      "threshold": "best_num_warps == 1",
      "reads": "the sibling nw2 prior does not transfer"
    },
    {
      "id": "FR-5",
      "observable": "mean_wall_ms",
      "threshold": "did not improve by at least 5 percent",
      "reads": "global conservative guard"
    }
  ],
  "observability_note": "A replayed CUDA graph emits zero cat=kernel events in its interior, so kernel_count_per_call and kernel_count-derived device_us_per_call are UNAVAILABLE, not zero, on the replay scope. Device observables are collected with KS_E2_REPLAY=0; the host API census in log/diagnostic_scope_census_001.json is the Level-2 substitute for launch-side attribution; adoption is decided on wall time with the census and the non-replay device tier as corroboration. No falsification rule references kernel_count_per_call. This is the groupedtopk report_004 branch-B pattern."
}
```

## Pitfalls and Anti-pattern Consultation

**Entry 018 — Triton/CUDA-graph manual replay (direct-address tier).** Applied as
designed. The per-call path is 2x `data_ptr` equality plus shape/dtype checks, one
graph replay, and one copy-out; capture is amortized at first call. Static fp16
workspaces are allocated before capture and the returned tensor is always a fresh
`copy_`, never workspace memory. Guard failure falls through to eager, so a graph
problem degrades latency, never correctness.

**Entry 003 — 2D-grid pointer arithmetic.** The gathered row list uses one flat
range `row = tl.arange(0, H)` and the token gather uses `token[:, None] * H +
r[None, :]`, matching the epoch-1 candidate's proven single-range form. No custom
2D pointer arithmetic is introduced.

**Entry 002 — 2D-grid semantic error.** Grid `(8,11)` maps `program_id(0)` to the
expert and `program_id(1)` to the row tile, exactly the semantic the kernel body
assumes (`base = expert_offsets[e]`, `start = tile * BLOCK_M`). The tile index is
never confused with the row index.

**Entry 013 — Triton gather.** Not used. No `tl.gather` appears; the row list is
loaded with plain masked pointer loads.

**Entry 016 — cumsum-based compaction.** The known regression (5%, MLU590, 8 lanes)
is recorded and consciously re-tested at a different scale: here compaction is
166 lanes into 8 buckets, done inside a **single program** (`grid (1,)`,
`BLOCK=256`) so it needs no cross-block synchronization at all, and it is cheap
precisely because the graph makes its extra launch free. If FR-2 fires, this is the
first thing to suspect.

**Entry 011 — launch-side "same kernel, different launches" bookkeeping.** The
per-call cost is not two Triton launches; it is one graph submission. The census
must report submissions, not raw kernel launches, or the host half will be
mis-attributed.

**Additional consultations.**
- *Do not let the grid depend on data.* Tempting: read `n_e` back and launch
  `ceil(n_e/BLOCK_M)`. That read is a D2H sync and breaks capture. `NUM_TILES=11` is
  `ceil(166/16)` from shapes, a constexpr, independent of any routing outcome; empty
  tiles exit on-device.
- *Do not return the workspace.* `out_ws` is graph-pool memory. Returning it can pass
  a single-call correctness check and then fail silently across the 100-iteration
  timed loop.
- *Do not judge the device lever by the replay trace.* The replay scope's device time
  includes the ~46 us front-end and will look worse even when the kernel got faster
  (mm_encoder measured 19.555 -> 64.467 with kernel math unchanged at 18.4). Device
  conclusions come from `KS_E2_REPLAY=0` only.
- *Do not reimplement top-k.* Tie ordering is the invariant; hand-rolled selection is
  the groupedtopk failure mode.
- *Do not use `tl.sum` / `tl.argmax`.* Both are waiver-gated or `constrained`. Eight
  `tl.static_range` masked adds over 256 lanes cost ~8 ALU ops and carry zero waiver
  exposure.
- *Do not assume `num_warps=4` helps.* The sibling measured nw2 optimal with no
  further gain at nw4; on a `[16,128]` tile nw4 may under-fill warps.
- *Do not treat +31% as a promise.* The device half depends on the restructure landing
  at full value; at a 2x instead of 16x arithmetic reduction the device half shrinks
  to ~25 us and the round lands near +16% on host compression alone. FR-2 thresholds
  at 40 us so a partial device landing can still be adopted on the host half.

## Rationale and Evidence

### Baseline

`report_000.md`, fingerprint `fe73bc58…`: wall **3.255288 ms** (v0 median, warmup 50
/ repeat 100), device **967.852 us/call**, **123.95** launches/call across 21
distinct kernels, `device_ratio` **0.297317**. Host is 70.3%: roughly 2.26 ms of
each call is launch and dispatch overhead against 968 us of device work.

The device-time target is **dispatch/indexing, not the GEMMs**:

| family | us/call | % device | launches |
|---|---:|---:|---:|
| dispatch/indexing (scatter-store, mask gather, `DeviceSelectSweep`, `mask.any()`, `DeviceReduce`, `DeviceCompactInit`, `flat_ids==e`) | **635.313** | **65.6%** | ~95 |
| the two GEMMs | **118.831** | **12.27%** | 16 |
| all remaining real math | < 320 | < 33% | ~13 |

No single kernel dominates: the largest is 127.402 us/call = 13.16% of device =
3.91% of wall. Base is a long tail of ~124 cheap launches.

### Why the round starts from the epoch-1 candidate, not from base

`report_000` measures base, but a fresh Triton file is built on the epoch-1 terminal
candidate (`triton_fused_moe_002.py`: **9.82** launches, **140.84 us** device):

| op | launches/call | device us/call | class |
|---|---:|---:|---|
| `_fused_moe_expert_kernel` (Triton, grid (8,), BLOCK_M=256) | **1.00** | **55.80** | **Triton — carries the 85 us tax** |
| `sbtopk::gatherTopK` + `bitonicSortKVInPlace` | 1.96 | 39.44 | aten — **invariant-frozen** |
| softmax + renorm sum + renorm div | 2.94 | 25.45 | aten |
| `float16_copy` (w1 + w2 + topk_w cast) | 2.94 | 15.56 | aten |
| `FillFunctor` out zero-init | 0.98 | 4.60 | aten |
| **TOTAL** | **9.82** | **140.84** | **N_triton = 1** |

### The headline rationale: the epoch-1 device number is not irreducible

Epoch-1 aborted with "BLOCK_M=256 is required by `tl.dot`'s `M>=16` power-of-two
constraint", treating 55.80 us as single-kernel optimal. **That inference is wrong.**
`M>=16` forces `BLOCK_M >= 16`, **not** `BLOCK_M >= 166`. 256 was chosen only because
one tile had to cover every row of an expert in a single ungrouped pass. Three
addressable consequences are all artifacts of that one choice:

1. **12.34x replicated GEMM arithmetic.** Useful work is `T*K = 166` rows x
   `(128*64 + 128*64 + 64*128)` MACs = **8.16 MFLOP**. The kernel does
   `8 programs x 256 rows x 24576` MACs = **100.66 MFLOP**. The `is_e` mask zeroes
   the loads and the atomic_add, but all three `tl.dot`s run over the full 256 rows.
2. **50% of the GPU idle.** `grid = (E,) = 8` programs on `multi_processor_count = 16`.
3. **Register-spill regime.** The `x` tile is `[256,128] fp16` = 32768 elements; at
   `num_warps=1` that is **512 x 4 B registers for `x` alone against a 255-register
   budget** — the class the sibling measured at ~288 regs/thread.

### Two-lever pricing and the break-even

Cross-campaign constants, all LABELED priors measured on this same BI-V150 rig:
`T_launcher = 85 us` (grid-independent; 84.77/84.57 mm_encoder, 86-89 flexattention);
`R = 66 us` (65.76 @bsz=2, 69.02 @bsz=1 — transfers); `F = 46 us` (build-intrinsic,
device-visible, kernel math unchanged); `R`/in-graph overlap `= -7 us`;
`d_aten = 0..5 us`; boundary `= 4..12 us` per copy.

```
Δ_wall = −(N_triton x 85) − (N_aten x d_aten) + (R + F) − overlap + boundary − device_savings
       = −(N_triton x 85) − (N_aten x 0..5) + 112 − 7 + ~10 − device_savings
```

| variant | host-only Δ us | with device savings | on 493 us |
|---|---:|---:|---:|
| **graph alone** (`N_triton=1`, candidate-002 as-is) | **+8** | — | **−1.6% — wash** |
| **device restructure alone** (eager, `N_triton=3`) | +170 … +255 | +94 … +179 | **−19% … −36% — LOSS** |
| **selected: graph x restructure** (`N_triton=2`) | **−77** | **−153** | **+31%** |
| extended (`N_triton=3`) | −162 | −240 | +49% |

**Break-even is explicit: the 5% bar needs `N_triton x 85 >= 103`, i.e. `N_triton >= 2`
whole kernels. The epoch-1 candidate has `N_triton = 1`.** Two negative controls
confirm the 1-launch case is marginal: mm_encoder scraped +5.08% (clearing by
0.077 pp) and flexattention was a wash at +0.22%. The positive control at
`N_triton=3` is groupedtopk: predicted `-3x85 + 112 - 7 - 3 = -153` vs measured
**-142** (8 us model error), with device flat.

**Therefore the two levers are multiplicative, not independent: the restructure
creates the launches the graph monetizes, and the graph is the only thing that makes
the restructure's launches affordable.** They cannot be split into two rounds — the
device restructure alone measures as a -19…-36% regression and is guaranteed
`no-improvement`. All three inseparability conditions hold: (i) B's launches are
unaffordable without A; (ii) A alone is a wash; (iii) both sub-effects stay
separately observable (host census and submission count vs the non-replay device
tier).

One mechanism, not two stacked tweaks: **"grouped dispatch + replayed launch".**

### Family pool

| # | family | verdict |
|---|---|---|
| **F1** | manual graph replay x counting-sort grouped GEMM | **selected** — only family net-positive on both levers |
| F3 | device restructure alone | **excluded** — falsified by arithmetic: +170 us tax vs −77 us device |
| F4 | graph replay over the base-shaped pipeline | **excluded** — direct negative control (flexattention −1.69%), and base's launch count is data-dependent (148/134/64), so the captured path is not input-stable |
| F5 | num_warps 1->2 alone | **folded in** as an in-round sweep; +2…+4% is below the 5% bar and does not deserve a round |
| F6 | weight pre-cast (host lifecycle) | **deferred** — changes `state_dict()` dtype with a silent-`try/except` `load_state_dict` hazard; not needed to clear 5% |
| F2 | cross-block cooperative persistent kernel | **excluded** — saves inter-wave scheduling, not the 85 us launcher tax that dominates; would also collide with the register pressure the sweep is meant to expose |

### Capture-correctness analysis

- **Base is not capturable.** `if not mask.any(): continue` is a live host-side branch
  driven by a D2H sync per expert; `report_000` measured the launch count varying
  148 / 134 / 64 with active-expert count (~14 launches per active expert). A capture
  over base would be valid only for a fixed active-expert set. **F4 is excluded for
  this reason.**
- **The capture target is capturable.** The new pipeline has zero host-side
  data-dependent flow: static grid `(8,11)`, `NUM_TILES = ceil(166/16)` from shapes,
  membership resolved on-device by masked loads, empty tiles exiting on-device.
- **Pointer stability is confirmed from harness source.** `auto_bench.py:459-475`
  binds `inputs` once and reuses the same tensor objects for all 50 warmup + 100
  timed iterations; `set_seed` calls only `torch.manual_seed` / `manual_seed_all` and
  never reallocates, clones, or moves tensors. So `data_ptr()` is constant across all
  150 calls — the condition under which mm_encoder's direct-address tier hit 100/100.
- **Activation patterns do not vary between calls** (router bytes are fixed), and
  would be harmless if they did, since the captured control flow is data-independent.

### Capability and ruling notes

- **`tl.dot` capability claim is stale (upheld, not fixed mid-campaign).** The frozen
  claim records only `(32,32)` fp32-fp32-fp32 as `constrained`; F3 needs fp16-operand
  dots at contraction `128/64` with `M=16`. Epoch-1 `report_002.md` measured this
  class at `1.53e-05` against `1e-2` tolerance — a **LABELED PRIOR, not profile
  evidence**. Before-fallback is satisfied by in-round re-qualification under
  `log/probes/`, never by editing the snapshot.
- **fp16-dot exactness is operator-dependent.** mm_encoder measured it
  exactness-negative on attention (1459 vs vendor 1457) while fused_moe passes at
  `1.5e-05`; neither result is imported, and the new tile is re-verified in-probe.
- **`reduction.sum` waiver not granted** — softmax, renormalize, and casts stay aten.
  This forgoes 17 us of device and holds `N_triton = 2`, which still clears
  break-even with margin.
- **`num_warps` is `constrained`** — swept in-round over {1,2,4}; the sibling
  −31%/bitwise-identical result is labeled, not profile evidence.
- **`qualification_dispositions` is empty in the frozen claim**, so no fallback
  provenance is declared. This is correct: F3 uses `tl.dot` within the proven
  capability (re-qualified at a new tile shape), which is not an algorithm
  substitution.

### Budget

Round 001 of 20, `no_improvement_streak = 0`, `budget_state = normal`. There is no
budget pressure. Spending the cheapest available round on the highest-ceiling family
is correct precisely because the budget is intact; a safe sub-5% first round would
bank +3% and leave the expensive composition for later, when the streak may be under
pressure. F5 is folded in as a sweep at zero additional round cost.

### Self-validation

- `schema_v2`: `valid`
- `self-validation`: `green`
- `validator`: `skills/kernel-opt-loop/scripts/validate_decision.py --expected-implementation-profile triton_cuda`
- `sketch_validator`: `skills/kernel-opt-loop/scripts/validate_sketch.py`
- `profile_validator`: `skills/kernel-opt-loop/scripts/validate_profile.py`

