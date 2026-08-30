# Verifier Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f`
- context_epoch: `2`
- last_completed_round: `003`
- accepted_kernel_as_of_this_file: `triton_mm_encoder_attention_e2_001.py`
  (Verifier never updates canonical pointers; Orchestrator promotes the
  round-003 candidate `triton_mm_encoder_attention_e2_003.py` if it accepts
  `rounds/report_003.md`, which classifies `accepted`.)
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: see below
- open_hypotheses: `H-003` confirmed and closed. Next families: residual host
  wrapper work (`22.635 us/call`) and `launch-path-reduction`
  (`183.740 us/call`), which needs an Ascend launch-ABI probe first.
- artifact_read_hashes: `see the table below`

## Current Bottleneck

Facts this role established in `rounds/report_003.md`, in one process and one
regime, against the round-003 candidate's `297.410 us` wall:

| Slice | us/call | Share of wall |
|---|---:|---:|
| Triton launch path (bare `_fused_attention_kernel[grid](...)`) | 183.740 | 61.78% |
| harness-fixed, outside `ModelNew.forward` | 91.035 | 30.61% |
| residual `ModelNew.forward` wrapper at a cache hit | 22.635 | 7.61% |
| device kernel time | 13.4224 | 4.51% |

- Device time is now the **smallest** term. `device_us_per_call` is `13.4224` at
  `kernel_count_per_call` `1.00`; the device-only ceiling is `4.09%`, below the
  5% threshold. Device work is closed.
- The **launch path is the largest term** and it allocates: a bare direct launch
  with a preallocated output issues exactly one `aten.empty.memory_format` per
  call (20/20 launches) while the Python-level `torch.empty` count is zero.
- The **harness-fixed floor is `91.035 us/call` (30.61%)** and no host round can
  touch it. It is `51.815 us` of synchronize plus `39.220 us` of seed drain and
  `sync_devices()` accelerator probing.

## Recent Three-round Evidence

- `001` / `accepted` / `rounds/report_001.md`: reference `0.365400` ms ->
  candidate `0.327770` ms, `improvement_pct 10.2983`. Device `118.892` ->
  `13.4064 us/call`, kernels `6.98` -> `1.00`, `device_ratio` `0.0407`.
  Hypothesis confirmed on device and launch, partially-confirmed on host.
- `002` / `aborted` / no report: the complete device budget (`13.4064 us/call`)
  is below the 5% adoption budget (`16.3885 us/call`), so device-only work caps
  at `4.0902%`.
- `003` / `accepted` / `rounds/report_003.md`: reference `0.361050` ms ->
  candidate `0.298240` ms, `improvement_pct 17.3965`. Interleaved control
  against the accepted kernel: `11.2080%` raw, `8.8072%` base-normalized.
  Host lever `ModelNew.forward` `233.645` -> `206.375 us` (`-27.270 us`);
  `output_allocations_per_call` `1.00` -> `0.00`. Control observables held:
  `device_us_per_call` `13.4096` -> `13.4224` (`+0.095%`),
  `kernel_count_per_call` `1.00` -> `1.00`. Hypothesis `H-003` **confirmed**.

## Measurement Regime Notes (learned, do not re-derive)

- The harness requires `--v0_file` to define `Model`, so a paired run is always
  `base.py` versus the candidate. Never compare against a historical baseline
  number: within a single turn `base.py` medians ranged `0.346350`-`0.370825`
  (`~7%`). Always re-measure the reference in the same turn.
- `set_seed` runs **before** `start = time.perf_counter()` and is untimed, but
  `manual_seed_all` enqueues device work that the timed `sync_devices()` then
  drains, so the seed op is billed inside the timed region
  (`(a) - (c) = 39.220 us`).
- `sync_devices()` calls `_iter_accelerators()` -> `torch.npu.is_available()` on
  every call, costing `11.96 us/call` more than a bare
  `torch.npu.synchronize()` (`33.870` vs `21.910 us` idle).
- The profiler run does **not** pass `--warmup`/`--repeat` in this project's
  commands, so its printed `v0`/`v1` are at the defaults `200`/`500`, and the
  `v0` is `base.py`, **not** the `--profile-reference-file` scope. When the
  reference scope is not `base.py`, measure the reference scope's wall
  separately at `200`/`500`.
- `export_chrome_trace` is called once per scope inside the capture loop, so the
  candidate export overwrites the reference one. Only the candidate
  `record_function` survives in the `.pt.trace.json`; do not expect to read the
  reference scope's wall from it.
- The reference scope directory can accumulate several
  `ai_core_op_summary.db` files across rounds. Pass the explicit round-003 `.db`
  path to `summarize_cann_trace.py`.
- A `TorchDispatchMode` op count is **not** a reliable allocation counter here:
  `aten.empty_like` decomposes to `aten.empty.memory_format`, and the Triton
  launch path itself issues one `aten.empty.memory_format` per launch. Use a
  Python-level counter on `torch.empty` / `torch.empty_like` for
  `output_allocations_per_call`.
- A `forward` returning a stable `data_ptr()` does **not** prove a cache hit;
  the caching allocator hands back the same freed block. Count factory calls.

## Open Hypotheses or Checks

- `launch-path-reduction` is the largest remaining lever (`183.740 us/call`) but
  `lifecycle.fast-launcher` is `Unknown` in the frozen profile with no Ascend
  probe. Decision 003 section 5 requires an Ascend launch-ABI probe before
  attempting it as a code experiment. Do not declare the launcher normative.
- Residual host wrapper (`22.635 us/call`) contains the per-call
  `query.device` construction, the four-component cache key, and the grid tuple.
  Clearing 5% again needs `14.871 us` of that `22.635 us` (~66%).
- The reuse invariant depends on full store coverage. Re-verify by NaN-poisoning
  the cached buffer if any future round introduces a masked or partial store.
- Device evidence on this target requires `torch_npu.profiler` plus the CANN
  sqlite at `device_0/sqlite/ai_core_op_summary.db`. A raw `torch.profiler`
  chrome trace alone is host-side `cpu_op` events only, not device evidence.
- Report device and synchronized wall measurements separately; never present a
  device win as a wall win. `device_ratio` must always state which wall value it
  uses.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 003 |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 003 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `triton_mm_encoder_attention_e2_001.py` | `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124` | 003 |
| `triton_mm_encoder_attention_e2_003.py` | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` | 003 |
| `rounds/decision_003.md` | `a4956891de5fef4b9bd629fb3cceb270db5a247ba18b591aecee9480d96c5455` | 003 |
| `rounds/coder_result_003.md` | `d60e74e94f5e87ffbe2c535f8caea8d58c1fc7d4b104e1b0351fb9d854ac948d` | 003 |
| `rounds/sketch_003.json` | `51ebe3a735c7659309e781fd2f35286fd4e67acc86b5d0a9f6676f08f08af69c` | 003 |
