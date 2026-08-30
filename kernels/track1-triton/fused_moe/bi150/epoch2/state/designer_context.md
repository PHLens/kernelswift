# Designer Context

- role_contract_sha256: `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5`
- context_epoch: `4`
- last_completed_round: `002` *(dispatch: decision_002 @dc782254... + sketch_002 @015da345..., schema-v2 self-validated green)*
- accepted_kernel: `triton_fused_moe_e2_001.py` @`da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7` *(accepted round 001)*
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `r001 ACCEPTED +93.248% (0.219792 ms vs canon 3.255288 ms, 14.81x), family manual-graph-replay-fused. Host half confirmed far beyond model (423 us replay-vs-eager delta vs 170 us modeled; 0 launcher executions/call; 2.0 submissions/call). Device half FALSIFIED as measured but attributable to control methodology: isolated Triton-only 58.231 us/call (sort 28.038 + expert 30.192) vs epoch-1's single 55.954 us kernel — DEVICE-NEUTRAL, not the predicted 2.4x. FR-2 and FR-4 fire; FR-5 adoption gate does not. r000 baseline 3.255288 ms / 967.852 us device / 123.95 kernels-per-call / device_ratio 0.297317.`
- open_hypotheses: `Round 002 DISPATCHED (G1 option ii): reuse a persistent out_dest as the fixed copy-out TARGET while keeping the copy_, removing the 16.219 us/call per-call alloc (7.4% of the 219.792 us wall, clears the 10.99 us gate alone). Orchestrator DENIED option (i) — compare_case retains v1_output at auto_bench.py:743 and passes it to make_profile_call, so 150 time_forward calls would overwrite it; that is a correctness-of-evidence defect. G3 (BLOCK_M {16,32}, num_stages {1,2}) folded in as a sweep. num_warps NOT re-swept (settled at 1, FR-4). Post-G1 the only meaningful remaining item is G2 (routing prelude, ~20 us best case, waiver-gated, topk ~41.6 us frozen); if G2 looks sub-gate, recommend honest convergence.`
- artifact_read_hashes: `see Artifact Read Hashes ledger below (24 artifacts hashed this epoch)`

## Current Bottleneck

- **CANON: wall `0.219792 ms`** (`report_001.md`, round-001 accepted). Round-000's 3.255288 ms is **superseded as the comparison anchor**. 5% adoption gate = **10.99 us/call**.
- **The bottleneck has INVERTED.** The graph/launch lever is nearly exhausted: **0 python Triton launcher executions per timed call**, **2.0 submissions/call** (1 cudaGraphLaunch + 1 copy-out memcpy), 0 recaptures in 100 calls. There is essentially no launch-side overhead left to compress, so round 002 cannot win the way round 001 did.

### Where the 219.792 us/call goes (measured)

| component | us/call | ours? |
|---|---:|---|
| `cudaDeviceSynchronize` (CPU) | ~122.1 | **HARNESS** — `time_forward` calls `sync_devices()` inside the timed region; not addressable |
| `_grouped_expert_kernel` (isolated CUDA-event) | 30.192 | ours |
| `_counting_sort_kernel` (isolated CUDA-event) | 28.038 | ours |
| routing prelude (softmax+topk+sum+div) | 33.704 | ours (topk frozen inside) |
| `aten::empty_strided` + `aten::empty_like` (CPU) | 16.219 | ours — fresh output alloc |
| `aten::copy_` (CPU) | 13.936 | ours — copy-out |
| `cudaGraphLaunch` (CPU) | 7.281 | ours — replay submit |
| `cudaMemcpyAsync` (CPU) | 6.160 | ours — copy-out |
| `Memcpy DtoD` (device) | 2.401 | ours — copy-out |
| out zero-init | 1.871 | ours |
| *unattributed* | ~-42 | intra-graph launch latency / interleaving |

- **Ours = ~139.8 us/call; the harness sync is ~122 us/call and is NOT addressable.**
- **The replay boundary is the largest addressable item: ~38.7 us/call (17.6% of wall)** — `empty_like`+`empty_strided` 16.219 + `copy_` 13.936 + `cudaMemcpyAsync` 6.160 + `Memcpy DtoD` 2.401. It exists solely because `out_ws` is graph-pool memory and must not be returned, so `forward` allocates a fresh tensor and copies into it.

### Corrected priors (round-001 measurement overrides the cross-campaign model)

1. **Re-price the graph at the MEASURED 423 us, not the modeled `N_triton x 85 = 170 us`.** Collapsing the 9.82 aten launches into the graph removes far more than the two Triton launcher taxes; R+F (112 us) are dwarfed by it.
2. **The `BLOCK_M 256->16` arithmetic argument DID NOT convert to device time.** 12.34x replicated GEMM removed, yet isolated device is 58.231 vs 55.954 us — device-neutral. **Do not re-derive a device win from an arithmetic-reduction argument on this rig.**
3. **`best_num_warps = 1` on this kernel with a 24.4% margin** (92.855 vs 122.253 us), far outside the 0.5 us tie band. The sibling nw2 prior does **not** transfer (FR-4 fires). Do not import nw2.
4. **fp16-dot exactness is POSITIVE here** (p01: 2.441e-04 vs 1e-2 tolerance, ~40x margin at M16/M32 K128 N64 and M16/M32 K64 N128). Unlike the mm_encoder negative. Still needs in-probe re-qualification at any new tile.
5. **METHODOLOGY: a forced-eager control is not a valid device control.** Disabling the tier guards also bypasses `_alloc_workspace`, so `_pipeline` re-allocates its sort buffers every call (5 `torch.zeros`/forward, `aten::fill_` at 6.00/call / 23.797 us/call). That ~49 us of churn is why FR-2 fired and is NOT a property of the shipped code. Any future eager device control must pre-bind the workspaces first.
6. **METHODOLOGY: fp16-extreme suites must cap at 32, not 1024.** At 1024 `silu(gate)*up` reaches ~1.5e5 and overflows fp16 **in base.py itself**, making `allclose` vacuous (NaN vs NaN). Always assert a finite BASE before reading a candidate FAIL.
- **The device-time target is dispatch/indexing, not the GEMMs.** Dispatch/indexing = **635.313 us/call = 65.6% of device** over ~95 of 124 launches; the two GEMMs together = **118.831 us/call = 12.27%**; all remaining real math under 320 us/call. No single kernel dominates (largest 127.402 us/call = 13.16% of device = 3.91% of wall).
- Round 001 targets the epoch-1 terminal candidate (`9.82` launches, `140.84 us` device, `N_triton = 1`), because that is what a fresh Triton file is built on. See *Two-Lever Pricing* — both levers are live but **neither clears 5% alone**.

**Orchestrator rulings incorporated into `decision_001.md`:** (a) the stale `tl.dot` claim is immutable for this epoch — re-qualify the exact tiles as Decision-scoped probes under `log/probes/`, never edit the snapshot; (b) fp16-dot exactness is operator-dependent — re-verify at the NEW tile, import neither mm_encoder's negative nor epoch-1's pass; (c) `reduction.sum` waiver remains NOT granted — softmax/renorm/cast stay aten, `N_triton` held at 2; (d) `num_warps` swept in-round over {1,2,4}; (e) the mixed kernel+host change is APPROVED under the inseparability clause with two-sided observables; (f) `target_mode`/`target_value` stay null, DELIVERABLE RULE binding.
- Epoch-1 terminal state (`triton_fused_moe_002.py`, wall `0.493474 ms`, device `140.84 us/call`, `9.82 kernels/call`, `device_ratio 0.2854`): the operator is **host/launch-bound**, ~71% of wall is host/launch/other.
- **Both levers look live but neither clears 5% alone.** The decomposition below shows why: there is exactly **1** Triton launch/call (so the graph's break-even is not reached), and ~141 us of device of which ~55.8 us is a single kernel doing **12.3x more GEMM work than necessary** on **half the SMs**.

### Decomposition — base vs epoch-1 candidate

**base.py (`baseline_adapter.py`), epoch-1 measured: 123.9 launches/call, 968.16 us device, 3.258671 ms wall**

| stage | launches/call | device us/call | note |
|---|---:|---:|---|
| softmax + topk + renorm(sum/div) + cast | ~6 | ~43 | topk 2 launches is invariant-frozen |
| `x_rep` expand/reshape + zeros_like | ~2 | — | |
| **per-expert Python loop (x8)** | **~114** | **~700** | the 123.9-launch source |
| — `flat_ids == e` (eq) | 8 | — | |
| — **`mask.any()`** | 8 `or_kernel` + 16 `DeviceReduceSingleTile` | 86.64 + 81.35 = **167.99** | **host-side data-dependent branch → D2H sync per expert** |
| — `x_rep[mask]` CUB DeviceSelect | ~32 (CompactInit 15.98 + Sweep 15.98 + gather 8) | 56.47 + 125.98 + 127.34 | |
| — `x_e @ w1[e].T` (gate/up GEMM) | 8 | 61.17 | |
| — chunk + SiLU + mul | ~16 | ~48 + 40 | |
| — `act @ w2[e].T` (down GEMM) | 8 | 57.82 | |
| — `expert_out[mask] = ...` index_put scatter | 7.98 | 127.92 | |
| weighted scale + sum(dim=1) | ~2 | — | |

**epoch-1 candidate `triton_fused_moe_002.py`: 9.82 launches/call, 140.84 us device, 0.493474 ms wall**

| op | launches/call | device us/call | launch class |
|---|---:|---:|---|
| `_fused_moe_expert_kernel` (Triton, grid=(8,)) | **1.00** | **55.80** | **Triton — carries the 85 us tax** |
| `sbtopk::gatherTopK` (torch.topk) | 0.98 | 21.59 | aten — frozen |
| `bitonicSortKVInPlace` (topk) | 0.98 | 17.85 | aten — frozen |
| `reduce_kernel` renorm sum | 0.98 | 13.74 | aten |
| `elementwise DivFunctor` renorm div | 0.98 | 6.71 | aten |
| `float16_copy_kernel` (w1 + w2 + topk_w cast) | 2.94 | 15.56 | aten |
| `FillFunctor` out zero-init | 0.98 | 4.60 | aten |
| `softmax_warp_forward` | 0.98 | 5.00 | aten |
| **TOTAL** | **9.82** | **140.84** | **N_triton = 1** |

Device budget of the remaining 140.84 us:
- `_fused_moe_expert_kernel` 55.80 (39.6%) — **the only compressible large item**
- `torch.topk` 39.44 (28.0%) — **invariant-frozen** (tie semantics; reimplementation is the groupedtopk failure mode)
- routing (softmax + renorm sum + renorm div) 25.45 (18.1%)
- w1/w2 fp16 cast 15.56 (11.0%)
- out zero-init 4.60 (3.3%)

### Is the 55.80 us bound by GEMM arithmetic or scattered indexing?

**By replicated GEMM arithmetic, at 50% SM occupancy, in a register-spill regime.** Concretely:

- Useful work: `T*K = 166` rows x `(128x64 + 128x64 + 64x128)` MACs = `166 x 24576 = 4.08 M MACs` = **8.16 MFLOP**.
- Actual work: `grid = (E,) = 8` programs, each over `BLOCK_M = next_pow2(166) = 256` rows:
  `8 x 256 x 24576 = 50.3 M MACs` = **100.66 MFLOP**.
- **Replication factor = 12.34x** (= E=8 replication x 256/166 = 1.54 padding). The `is_e` mask zeroes the *loads* and the *atomic_add*, but all three `tl.dot`s are executed over the full 256 rows regardless.
- **SM occupancy**: grid = 8 programs on `multi_processor_count = 16` → **50% of the GPU idle**.
- **Register pressure**: the `x` tile is `[256, 128] fp16` = 32768 elements. At `num_warps=1` (32 threads) that is 1024 elements/thread = **512 x 4B registers for `x` alone, against a 255-register budget** — the cross-campaign prior classifies exactly this as spill-class (mm_encoder nw1 measured ~288 regs/thread vs 255).
- **Not memory-bound**: per program the real traffic is ~5.4 KB (`x`, masked) + 32 KB (`gate_w`,`up_w`) + 16 KB (`w2e`) ~ 53 KB; x8 = ~427 KB/call, ~2 us at small-kernel bandwidth.
- **Not scattered-indexing-bound**: addressing is static `token[:,None]*H + rk[None,:]` with `tl.arange`, no gather, no `tl.gather` (anti-pattern Entry 013 does not apply).

Floor estimate for a well-formed fused MoE: topk 39.44 (frozen) + routing ~8 + cast ~5 + grouped GEMM ~12 = **~64 us**, i.e. ~**-77 us** of addressable device time.

## Two-Lever Pricing (us/call)

Cross-campaign constants, all measured on this rig by sibling campaigns (LABELED priors, not epoch-2 facts):

| symbol | value | source |
|---|---|---|
| `T_launcher` (Triton python launcher tax) | **85** per call, grid-independent (84.77 / 84.57 / 86–89) | mm_encoder r001-r002, flexattention r002 |
| `R` (graph replay sync) | **66** (65.76 @bsz=2; 69.02 @bsz=1 — **transfers**) | mm_encoder r003, flexattention r003 |
| `F` (graph frontend, build-intrinsic) | **46** (device-visible; kernel math unchanged) | mm_encoder r003 |
| `R`/in-graph round-trip **OVERLAP** | **-7** (measured: costs overlap, do not add) | mm_encoder r003 |
| `d_aten` (aten host dispatch) | **0 … 5** per op | groupedtopk fit ~0; flexattention fit ~5 |
| boundary copy | **4 … 12** per copy | mm_encoder lean tier ~11 total |

### Lever A — manual CUDA-graph replay over the epoch-1 candidate AS-IS

```
Δ_wall(A) = -(N_triton x 85) - (N_aten x d_aten) + (R + F) - overlap + boundary
          = -(1    x 85) - (8.82 x 0..5) + (66 + 46) - 7 + ~10
          = -85 - (0 .. 44) + 112 - 7 + 10
          = +30 .. -14        (central ≈ +8, i.e. slightly WORSE)
```
Groupedtopk cross-check (N_triton=3): `-255 + 112 - 7 - 3 = -153` predicted vs **-142 measured** (8 us error).
mm_encoder cross-check (N_triton=1): `-85 + 112 - 7 + 11 = +31` predicted vs **-7.6 measured** (model error ~30 us in the favourable direction — the overlap and the replaceable base host stack were both larger than assumed).

**Verdict: Lever A alone is a WASH-CLASS result, band -3% … +3% of 493 us (mm_encoder scraped +5.08% clearing the bar by 0.077 pp; flexattention was a wash at +0.22%). It does NOT reliably clear the 5% adoption threshold. Do not spend a round on A alone.**

**Graph-capture correctness analysis (the `mask.any()` question):**
- **base.py is NOT capturable.** `if not mask.any(): continue` runs a D2H memcpy + host sync on every one of the 8 loop iterations (`or_kernel` + 2x `DeviceReduceSingleTile` + sync = 167.99 us/call). Host sync inside `torch.cuda.graph()` is an illegal capture.
- **The epoch-1 candidate IS perfectly capturable.** It has **zero host-side data-dependent control flow**: `grid=(E,)` launches all 8 programs unconditionally, `BLOCK_M = next_power_of_2(num_tokens*K)` derives from a *shape* not from data, and expert membership is handled on-device by the `is_e` mask (masked loads + masked `atomic_add`). No `.item()`, no D2H, no data-dependent branch. This is the same property that made groupedtopk r004 capturable.
- **Pointer/shape stability across warmup vs timed calls: STABLE.** `auto_bench.time_forward` (lines 459-475) binds `inputs` once and reuses the *same tensor objects* for all 50 warmup + 100 timed iterations; `set_seed(seed)` calls only `torch.manual_seed` + `manual_seed_all` and never reallocates, clones, or moves tensors. Therefore `hidden_states.data_ptr()` and `router_logits.data_ptr()` are **constant across all 150 calls** — exactly the condition under which mm_encoder's direct-address tier-1 hit **100/100**. Tier-1 (zero copy-ins, one copy-out) should dominate.
- **Expert-activation patterns do NOT vary between calls.** `router_logits` bytes are fixed for the whole timed loop, so `flat_ids` and therefore the activation pattern are byte-identical every call. And it would not matter if they varied: the captured control flow is data-independent, so results stay correct for any activation pattern.
- **Real hazards to bind in the Host Plan** (not the `mask.any()` branch, which is already gone):
  1. **Never let the grid depend on data.** A restructure needing a host-visible `n_e` per expert would be a D2H read → capture failure. Use a **static grid** `(E, ceil(T*K/BLOCK_M))` and let empty tiles early-exit **on-device**.
  2. **Never return graph-resident memory.** `out = torch.zeros(...)` allocated inside the capture region comes from the graph-private pool; returning it aliases memory that the next replay overwrites. Mandatory sibling-campaign pattern: static `out_ws` workspace allocated before capture, graph writes there, **copy-out into a fresh invocation-owned buffer after the replay**.
  3. **Harness arity**: `make_profile_call` calls `run_out(inputs[-1], *output_args)` = `run_out(router_logits, out)` (2 args) against our 3-arg `run_out(hidden_states, router_logits, out)`. **Arity mismatch → kernel-mode profiling is impossible; forward-mode profiling is canonical** for this project.
  4. **Kineto graph-interior blindness**: a replayed route emits **zero `cat=kernel` events** (groupedtopk report_004 took "branch B"). Level-1 `kernel_count_per_call` / `device_us_per_call` become **unattributable**; the Evaluation Contract must substitute a **host API census** (`log/diagnostic_scope_census_NNN.json`) and use wall time as the sole adoption basis.

### Lever B — device-side restructuring (without the graph)

| sub-lever | device us/call | extra launches | notes |
|---|---:|---:|---|
| **B1** `num_warps` 1→2 | **-14 … -18** | **0** | cross-campaign: -31% device, bitwise-identical, nw4 no further gain. Here the spill argument is stronger (512 vs 255 regs). Probe-gated (`resource.num-warps` = constrained). |
| **B2** counting-sort + grouped GEMM (`BLOCK_M=16`, static grid `(8,11)`) | **-36 … -45** | **+1** | 100.66 → ~6.29 MFLOP (16x); active programs 8 → ~16 = full 16 SMs; `M=16` is exactly the proven `tl.dot` minimum. |
| **B4** fused routing (softmax + renorm + cast) | **-17** | **+1** | 25.45 → ~8. **`tl.sum` is WAIVER-GATED** — the safe variant keeps renorm as aten. |
| **B3** pre-cast w1/w2 to fp16 | **-10 … -16** | **-2** | bitwise-safe (fp32→fp16 once == fp32→fp16 every call), but **changes `state_dict()` dtype** → `build_profile_reference`'s `load_state_dict` (try/except, silent) can leave a derived fp16 cache stale. Lifecycle risk. |
| **B5** zero-init folded into the graph | **-5** | 0 (moved in-graph) | 4.60 → ~2; saves 1 aten launch. |
| **device total** | **-68 … -85** (140.84 → ~56 … 73) | | |

**Lever B alone is strongly NEGATIVE.** If B2 (+1) and B4 (+1) are added eagerly the launcher tax is `+2 x 85 = +170 us` (or `+3 x 85 = +255` if routing and sort are separate kernels) against only `-68 … -85 us` of device saving:

```
Δ_wall(B alone) = +170 … +255 (taxes) - 76 (device) = +94 … +179 us  →  -19% … -36% (WORSE)
```

**This is the trap of the round. B must never be attempted without A.**

### BREAK-EVEN ARITHMETIC (the decisive identity)

```
Δ_wall = -(N_triton x 85) - (N_aten x d_aten) + 112 - 7 + boundary - (device savings)
```
Graph-only break-even (`d_aten = 3`, `boundary = 10`):

```
-(N_triton x 85) - 26.5 + 105 = 0   =>  N_triton x 85 = 78.5   =>  N_triton >= 0.93
```
but to clear the **5% adoption bar** on a 493 us wall we need `Δ_wall <= -24.6 us`:

```
-(N_triton x 85) + 78.5 <= -24.6    =>  N_triton x 85 >= 103.1  =>  N_triton >= 1.22
```
Rounded to whole kernels and carrying the +-30 us model error:

| `N_triton` | host-only Δ (us) | host-only % | with maximal device savings (us) | % |
|---:|---:|---:|---:|---:|
| **1 (today)** | **+8** | **-1.6%** | **-68** | **+14%** (only if B adds 0 launches — impossible) |
| **2** (B2 folded: one counting-sort kernel + grouped GEMM) | **-77** | **+16%** | **-153** | **+31%** |
| **3** (routing kernel split from sort) | **-162** | **+33%** | **-240** | **+49%** |
| **4** | -247 | +50% | -325 | +66% |

**Break-even statement: the graph breaks even at `N_triton >= 1` on pure host arithmetic but needs `N_triton >= 2` to clear the 5% bar with margin. The epoch-1 candidate has `N_triton = 1`, so the graph alone is a wash and `B` alone is a large loss. The two levers are MULTIPLICATIVE, not independent: `B` creates the launches that `A` monetizes, and `A` is the only thing that makes `B`'s launches affordable.**

The counting sort can be done in **ONE kernel, grid=(1,), BLOCK=256** (all 166 rows in a single program: 8 masked `tl.sum` counts + an 8-element exclusive scan + scatter into a static `[166]` index buffer). So `B2` costs **+1 launch, not +2**, and folding renorm+cast in front of it keeps `N_triton = 2`.

## Ranked Backlog

| # | family | change_family (normalized) | expected band | device | risk | validation cost | verdict |
|---:|---|---|---:|---:|---|---|---|
| **F3** | graph x counting-sort restructure | `manual-cuda-graph-workspace-replay` x `moe-counting-sort-grouped-gemm` | **+26% … +49%** (central **+32%**) | 140.84 → ~64 us | **high** (multi-kernel + graph + nw probe + `tl.sum` waiver) | 1 round | **HIGHEST CEILING — recommended** |
| **F1** | launch-config tuning | `launch-config-tuning` (`num_warps` 1→2) | **+2% … +4%** | -14 … -18 us | **low** (one literal; probe-gated) | 1 cheap probe | **FASTEST-PAYING — sub-5%, free rider only** |
| **F2** | graph replay on candidate-002 as-is | `manual-cuda-graph-workspace-replay` | **-3% … +3%** | flat | medium | 1 round | wash-class; **do not spend a round alone** |
| **F5** | host lifecycle (pre-cast weights, aten pruning) | `host-lifecycle-optimization` | **+3% … +6%** | -10 … -16 us | medium (`state_dict` dtype hazard) | 1 round | marginal; only as a rider on F3 |
| **F4** | device restructure WITHOUT graph | `kernel-fusion` (eager multi-launch) | **-19% … -36%** | -76 us | certain loss | — | **TRAP — excluded** |

- **FASTEST-PAYING: F1 (`num_warps` 1→2).** One literal, zero extra launches, zero host cost, zero structural risk, cross-campaign measured -31% device with bitwise-identical outputs. It converts to wall at ~1:1 (the harness sync sits inside the timed region, so the GPU tail is on the critical path). But it yields only **+3%** and therefore **cannot carry a round alone** — it is a mandatory free rider on F3.
- **HIGHEST-CEILING: F3.** `N_triton = 2` (counting-sort+routing kernel, grouped GEMM kernel) inside a direct-address manual-graph tier-1, with `num_warps=2` and `BLOCK_M=16` static grid `(8,11)`. Realistic band **+26% … +49%**; this is the groupedtopk pattern (+42.5% from host compression alone) **plus** a ~-77 us device win that groupedtopk never obtained.
- **DISPATCHED as Round 001** (`decision_001.md` @`62820af4…`, `sketch_001.json` @`6a46d4fd…`, both schema-v2 self-validated green — `validate_decision.py`, `validate_sketch.py`, `validate_profile.py` all exit 0). F3 was selected as the round to spend; **F5 (`num_warps` 1→2) is FOLDED IN as an in-round pre-adoption sweep over {1,2,4}** (the mm_encoder r002 precedent), not given its own round. The mixed kernel+host change was authorized under the inseparability clause: B's launches are unaffordable without A, A alone is a wash, and both sub-effects stay separately observable (host API census + submission count vs the `KS_E2_REPLAY=0` device tier).
- **If Round 001 fails**, the two-sided observables attribute the half: FR-1 fires (host) ⇒ replay hit rate / `R` larger than the 66 us prior; FR-2 fires (device) ⇒ counting-sort landing or `BLOCK_M` 16 vs 32; both ⇒ fall back to the epoch-1 candidate plus the sweep result as a standalone ~+3% bank.

## Capability and Input Discrepancies (flagged to Orchestrator)

1. **`profile_snapshot/capability_claim.json` is STALE on `tl.dot`.** It records only `matrix.dot.fp32-fp32-fp32.small-blocked-tiles` (status `constrained`, "(32,32)@(32,32), larger tiles unproven"). But epoch-1 `report_002.md#evidence_for_next_round` states fp16-operand `tl.dot` with contraction 128/64 and `M>=16` is **PROVEN on this exact profile** (max_abs_diff `1.53e-05` at the output, far inside the `1e-2` tolerance). The claim should be promoted; otherwise round-001 will re-litigate a settled capability.
2. **`fp16-operand tl.dot` exactness is operator-dependent.** mm_encoder measured it **exactness-NEGATIVE** on attention (vendor-saturation tie flips, max_abs 1459 vs vendor 1457), while fused_moe already runs fp16-operand dots and passes at `1.5e-05`. The attention failure mode does **not** reproduce here, but any tile-shape change (`BLOCK_M` 256 → 16) must be **re-verified for exactness at the new tile**, not assumed.
3. **`reduction.sum` waiver is NOT granted.** F3's maximal form (fold the renorm `sum(-1)` into the routing kernel via `tl.sum`) is therefore **gated**. Safe default: keep `softmax` + renorm + cast as aten (they are capturable, and aten host dispatch is worth ~0 per the groupedtopk fit), which costs only the B4 device item (-17 us) and keeps `N_triton = 2`.
4. **`resource.num-warps` = `constrained`** ("num_warps=1 recorded stable; every other value is Unknown until probed"). F1/F3's `num_warps=2` therefore requires a **before-fallback probe**; treat the cross-campaign -31%/bitwise-identical result as a LABELED prior, not as profile evidence.
5. **No `reduction.argmax` / `tl.sum` reliance is needed for F3's minimal form** — the counting sort uses `tl.sum` over a *boolean mask* (a popcount-class reduction), not `reduction.sum` in the waiver sense. Confirm with Orchestrator at Decision time whether a mask-popcount `tl.sum` falls inside the waiver gate; if it does, the counting sort can be written with an 8-iteration `tl.static_range` masked add instead (zero waiver exposure, ~8 extra ALU ops on 256 lanes).
6. **No optional user target is supplied** in `project.md` (`target_mode`/`target_value` are `null` in `team-state.md`). The DELIVERABLE RULE binds: submission is always the best correctness-PASS Triton candidate even if it does not beat base.

## Round-002 Family Ranking (re-priced on the 0.219792 ms canon)

Gate: **10.99 us/call** (= 5% of 219.792). Anything below that cannot be adopted.

| # | family | mechanism | expected | risk | verdict |
|---:|---|---|---:|---:|---|
| **G1** | **graph-boundary elimination** | fold the copy-out into the graph + preallocate the returned buffer | **-16 … -39 us (7% … 18%)** | medium (**lifecycle**: returning a persistent buffer aliases across calls) | **HIGHEST CEILING / FASTEST PAYING — but needs a lifecycle ruling** |
| **G2** | **device: routing prelude → Triton** | fold softmax + renorm + casts into one Triton kernel; topk stays aten | **-10 … -20 us (5% … 9%)** | medium (`reduction.sum` waiver-gated; softmax reduction) | live, but touches the waiver |
| **G3** | **device: kernel tuning** | `BLOCK_M` 32, `num_stages`, dot layout | **-3 … -8 us (1% … 4%)** | low | **sub-gate alone; free rider only** |
| G4 | deeper launch fusion | — none left — | 0 | — | **excluded**: 0 launcher executions, 2.0 submissions |
| G5 | weight pre-cast | fp32→fp16 once in `__init__` | -3 us device, -2 launches (worth ~0 in-graph) | `state_dict` dtype hazard | **excluded**: inside the graph a launch is nearly free, so this buys almost nothing |
| G6 | re-derive a device win from arithmetic reduction | — | ~0 | — | **excluded**: falsified by round 001 (prior #2) |

- **FASTEST-PAYING and HIGHEST-CEILING are now the same family: G1 (the boundary).** It is the only item large enough (38.7 us = 17.6%) to clear the gate on its own, it is pure host work, and it needs no new capability.
- **G1 has one blocking question:** `forward()` currently returns a fresh tensor because `out_ws` is graph-pool memory. Returning a persistent buffer instead would make two consecutive calls **alias**, which the existing `run_out poisoned x2` suite already probes. The safe variants, in order of preference:
  1. **Return a preallocated non-graph-pool buffer that the graph writes into directly** (allocate once outside the capture, have the graph's final store target it) — saves all 38.7 us, but the returned tensor aliases across calls.
  2. **Keep the fresh tensor but preallocate it once and have the graph write into it, then return it** — same aliasing question.
  3. **Keep the copy-out but kill the per-call allocation** (reuse one destination buffer, still `copy_`) — saves only 16.2 us (7.4%), still clears the gate, **no aliasing introduced** if the copy is retained... but the returned tensor is then persistent, so aliasing returns.
  - **Any variant that returns a persistent buffer must be gated on a lifecycle ruling.** The conservative fallback that introduces NO aliasing is: keep `torch.empty_like` + `copy_`, and instead attack G2/G3.
- **G2 is the best device target** and is what the Verifier's evidence points at. Its achievable size is bounded: the prelude is 33.7–41.6 us/call of which **topk alone is ~41.6 us and is frozen by the tie-semantics invariant**, so a Triton fold can only realistically reclaim the softmax+sum+div+cast portion (~20 us at best, and `reduction.sum` is waiver-gated).

### Orchestrator ruling on G1 aliasing (binding)

The Orchestrator checked the harness source directly:
- `time_forward` (459-475) **discards** the returned value, so aliasing is harmless on the wall path.
- BUT `compare_case` (735) computes `v1_output = run_forward(model_new, ...)`, compares it, and then **retains that very tensor** at line 743 into `export_profile` as `(f"candidate_{v1_path.stem}", model_new, v1_inputs, v1_output)`. `time_forward` runs `warmup 50 + repeat 100` forwards in between, so a persistent returned buffer would silently overwrite the retained profile reference.

**RULING: option (i) DENIED, option (ii) APPROVED.** Option (i) saves all 38.7 us but corrupts a tensor the harness demonstrably retains across compare → time → profile — a correctness-of-evidence defect, not a style choice. Option (ii) removes the 16.219 us per-call `empty_strided`+`empty_like` (7.4% of wall, clears the gate alone) while keeping a fresh non-aliased returned tensor. **The residual copy (13.936 + 6.160 + 2.401 = 22.497 us) is the price of correctness and stays.**

### Round 002 as dispatched

**G1 option (ii), with G3 folded in as a free-rider sweep.** `decision_002.md` @`dc782254…`, `sketch_002.json` @`015da345…`, both schema-v2 self-validated green (`validate_decision.py`, `validate_sketch.py`, `validate_profile.py` all exit 0).

- **Scope:** allocate `out_dest` once outside the capture region and reuse it as the fixed `copy_` target. `out_dest` is a **destination only and is never returned**; `forward()` still returns a non-aliased tensor.
- **Retention test (new, added per the ruling):** call `forward`, retain the returned tensor, run 50 further forwards, assert the retained tensor is **byte-identical**. This directly guards the `compare_case` failure mode. Encoded as observable `retained_output_unchanged` with **FR-2 firing on ANY change at all** — a correctness failure here is disqualifying regardless of the speed.
- **Carried forward:** the existing `run_out poisoned x2` suite; `num_warps` held at 1; 3-arg `run_out`; forward-mode profiling canonical; branch-B observability.
- **FR-4 is deliberately tight (device must not move >15 us).** Since this round is host-only, any device movement signals an unintended kernel edit. Any eager device control must **pre-bind the workspaces first** or the ~49 us `aten::fill_` churn will produce a spurious firing.

### Deferred / not selected

- `num_warps` is **settled at 1** — do not re-sweep (FR-4, 24.4% margin).
- `num_stages` remains Unknown in the claim; worth one exploratory probe only as part of G3.
- No re-derivation of the device lever from arithmetic-reduction arguments (falsified).

## Recent Three-round Evidence

- `(none — fresh epoch-2 campaign; Phase 0)`
- Non-canonical epoch-1 lineage (archive at `../`, read-only):
  - round 001 `accept`, family `kernel-fusion` — argsort bucketing + Triton weighted-reduce replaced the CUB per-expert dispatch; 123.9 → 54.1 kernels/call, 968.16 → 504.31 us/call, +21.44% wall. Evidence `../rounds/report_001.md`.
  - round 002 `accept`, family `gemm-fusion` — `tl.dot` fused the per-expert GEMM + chunk + SiLU + mul + down + weighted-reduce into ONE kernel, eliminating the argsort; 54.0 → 9.82 kernels/call, 500.65 → 140.84 us/call, **+79.98%** wall. Evidence `../rounds/report_002.md`.
  - round 003 `aborted`, family `no-change` — Designer judged measurement-bound: device_ratio 0.2854, topk 39.44 us frozen by the tie-semantics invariant, fused kernel 55.80 us judged "already single-kernel optimal". Evidence `../rounds/decision_003.md`.
- **Disagreement with epoch-1's round-003 abort rationale, recorded for the Round-N Designer:** decision_003 accepted `BLOCK_M=256` as "required by `tl.dot`'s `M>=16` power-of-two constraint" and therefore treated 55.80 us as irreducible. That inference is wrong. `M>=16` forces `BLOCK_M >= 16`, **not** `BLOCK_M >= 166`; epoch-1 chose 256 because it needed one tile to cover *all* rows of an expert in a *single un-grouped* pass. The 12.34x replication, the 50% SM occupancy, and the nw1 register spill are all consequences of that choice and are all addressable once rows are grouped by expert (which is precisely what the cross-campaign graph lever now makes affordable). This is the specific gap epoch-2 exists to exploit.

## Open Hypotheses or Checks

1. **H-A1 (primary, F3)**: manual `torch.cuda.CUDAGraph` workspace capture (direct-address tier-1, zero copy-ins, one copy-out) of a **2-Triton-launch** restructured pipeline — `_route_sort_kernel` (grid=(1,), BLOCK=256: renorm + fp16 cast + 8-bucket counting sort of the 166 rows) + `_grouped_expert_kernel` (static grid `(8,11)`, `BLOCK_M=16`, `num_warps=2`, gate/up/down dots, `atomic_add` into a static `out_ws`). Predicted `N_triton` 1 → 2, device 140.84 → ~64 us, wall **+26% … +49%**.
2. **H-A2 (F1, free rider)**: `num_warps` 1 → 2 on the restructured kernels; reuses the cross-campaign -31%/bitwise-identical prior; needs the `resource.num-warps` before-fallback probe. Fold into H-A1 rather than spending its own round.
3. **H-A3 (F5, rider)**: pre-cast `w1`/`w2` to fp16 at construction, removing 2 of the 2.94 cast launches (-10 … -16 us device, -2 launches). **Blocked pending an Orchestrator ruling on the `state_dict()` dtype hazard** (`build_profile_reference` calls `load_state_dict` inside a silent try/except on the profiling path).
4. **Checks owed by Verifier before F3 is priced with confidence** (Level-2, host): in-regime `T_launcher`, `R`, `F`, and `d_aten` for this operator, plus an in-regime `device_us_per_call` confirmation of the 140.84 us figure under the epoch-2 measurement fingerprint `fe73bc58…`. Until then every us-number in this document is a labeled prior or an epoch-1 measurement, never an epoch-2 fact.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../base.py` (immutable; matches `project.md#base_sha256`) | `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d` | 000 |
| `../../auto_bench.py` (matches `project.md#harness_sha256`) | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 000 |
| `project.md` | `702a4ab52c346deb2ddf1c79a211bab2a0a665bfa7f755182324b7cbe2fbe1a6` | 000 |
| `baseline_adapter.py` | `752a25033b7629459c6eb128c60a4bdc3ab77b9c7cc97f5d3592bdff4cd45a47` | 000 |
| `profile_snapshot/triton_cuda.yaml` (matches pinned ref) | `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae` | 000 |
| `profile_snapshot/capability_claim.json` (matches pinned ref) | `fcba080f084be2791c43bbe45baaaff695cb2b4a72cc4053a3e070ae6912cff5` | 000 |
| `../triton_fused_moe_002.py` (epoch-1 canonical) | `6ac1f44b111285f5bf746110c51f6486868b12beb2deae3390663d74233f8ae5` | 000 |
| `../triton_fused_moe_001.py` | `8424c7a01bc1d293c2b0ef509dd895950112cfb71dedd145053b4ac3f7eb9ad6` | 000 |
| `../rounds/report_002.md` | `b95a0a3d32600a935df646ff08600d6633eb4559858397504410facaf014f4e4` | 000 |
| `../rounds/report_001.md` | `5813f546684133f699ab2961c4fce1589af76b112714c7fadfb8b3bb6a7a7bf2` | 000 |
| `../rounds/decision_003.md` | `32665b920468f2be2b0f122087b3bb86c4b2b51342c056d92a762c475b0f37a1` | 000 |
| `../final_summary.md` | `d6af5f4ed2be654f7b75a7411e9a242488036d9aef017de3dd5fc66c1647ebc2` | 000 |
| `prompts/designer.md` (role contract) | `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5` | 000 |
| **`rounds/report_000.md`** (canonical baseline — Phase 0) | `48bb7f670fb297d2b385a811c27665cabf6f3ed670537214d3d92f03d9efe23a` | 000 |
| **`rounds/decision_001.md`** (authored this epoch) | `62820af457c7b0b84232dc28bffd07009b5bc1ee482059728da06761381fd1d5` | 001 |
| **`rounds/sketch_001.json`** (authored this epoch) | `6a46d4fd67b0cbce7a34ce41eac0c2b4cc19f00dd6e6098cf91a60e879634cb4` | 001 |
| `state/verifier_context.md` (peer ledger, read-only) | `45cefc40afa95c5320b7ae7038a23c9e70c5f20bf1c47485acf1d0a3e432cf5d` | 000 |
| **`triton_fused_moe_e2_001.py`** (accepted at round 001) | `da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7` | 001 |
| **`rounds/report_001.md`** | `532fe3ea8f461c608bf15efd96c8c5d527ac4a0098d0eb4009b26d21c1fbb8a5` | 001 |
| `log/probes/p01_r001_capability_result.json` (fp16-dot POSITIVE) | `94e1cdc9d3469cc29fa4c15018c78c03fcc86102857f8e5f36f518a15c28b05f` | 001 |
| `log/probes/p02_r001_config_sweep_result.json` (nw1, BLOCK_M16) | `28f10951f9f8983be8307b545dace86ed07c13f7474b5915c5c50b23b9c9f17b` | 001 |
| `log/probes/p02b_r001_perkernel_attribution_result.json` | `56da0fc7b48f87a79c2d4bbe63d85f3968dfac77255fba1de729d3e982fe33a9` | 001 |
| `log/diagnostic_scope_census_001.json` (branch-B census) | read | 001 |
| **`rounds/decision_002.md`** (authored this epoch) | `dc782254a54331454290fac6791b7f583fff81d8de9699f03f5d06722fd7637e` | 002 |
| **`rounds/sketch_002.json`** (authored this epoch) | `015da3456f18582ad6114d3f5a0bfd14c5122a365bfbdd8031b1e543ecfe7ebe` | 002 |
| `auto_bench.py` lines 455-475 (time_forward discards result) | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 002 |
| `auto_bench.py` lines 700-745 (compare_case RETAINS v1_output) | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 002 |
| **cross-campaign** `groupedtopk/bi150-round2/triton_grouped_topk_r2_004.py` | `c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb` | 000 |
| **cross-campaign** `groupedtopk/bi150-round2/final_summary.md` | `7278f1f8d09cda4e22b2ee24e0eade6611c13b047b65f1f81866a5ab70829c4e` | 000 |
| **cross-campaign** `mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_003.py` | `d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81` | 000 |
| **cross-campaign** `mm_encoder_attention/bi150/epoch2/final_summary.md` | `ebc63a6ff918fbfb280d8f8f5d7dc91599aae82a8acf807576ea242b42740a8d` | 000 |
| **cross-campaign** `flexattention/bi150/epoch2/triton_flexattention_e2_003.py` | `6ffb0c94bf6b126317acddcf14119bfd27fab5709c20a1f33cfdf8883d58bf1e` | 000 |
| **cross-campaign** `flexattention/bi150/epoch2/final_summary.md` | `5046a291230561d7d473e55923eeece4870f25554050ce289a8577fd7c8028f1` | 000 |
