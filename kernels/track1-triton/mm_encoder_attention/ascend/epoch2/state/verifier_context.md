# Verifier Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f`
- context_epoch: `2`
- last_completed_round: `004`
- accepted_kernel_as_of_this_file: `triton_mm_encoder_attention_e2_003.py`
  (round 004 classified `no-improvement`, so the canonical is unchanged; the
  Orchestrator may revisit this on the epoch-1 bar — see `rounds/report_004.md`.
  Verifier never updates canonical pointers.)
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: see below
- open_hypotheses: **`launch-path-reduction` M2 / M3 — the highest-value live
  lever in the epoch.** `lifecycle.fast-launcher` is now *proven* on this
  runtime by round-local evidence, and the probe measured M2 at `-119.360 us`
  and M3 at `-139.580 us` against M1's `-19.305 us`. Needs a decision naming one
  by name; the evidence already exists and no new probe is required.
- artifact_read_hashes: `see the table below`

## Current Bottleneck

Against the canonical `e2_003`, measured in-process (Level 2, round 004):

| Slice | us/call | Share of wall (317.325) |
|---|---:|---:|
| harness-fixed, outside `ModelNew.forward` | 95.720 | 30.2% |
| Triton launch path (bare M0 launch) | 192.255 | 60.6% |
| residual `forward` wrapper | 29.350 | 9.2% |
| device kernel time | 13.3272 | sub-component of the sync term |

- The launch path is the dominant term **and it is now a proven-compressible
  one**. Bare M1 `fast_libentry` is `172.950 us`; bare M2 cached `CompiledKernel`
  is `66.895 us`; bare M3 `NPULauncher` C entry is `46.675 us` (Coder probe,
  round 004). All launch the same compiled kernel, bit-identical, `1.00`
  launches per call.
- Device is closed: `13.3228-13.3272 us/call` at `1.00` kernels,
  `device_ratio` `0.0474`, against a `4.0902%` device-only ceiling.
- Harness-fixed is `95.7-100.4 us/call` (30-33%) and unreachable. Its share
  rises as wall falls.
- Residual wrapper is `~29.7 us/call` (9.2%), unchanged by round 004.

## Recent Four-round Evidence

- `001` / `accepted` / `rounds/report_001.md`: `0.365400` → `0.327770` ms,
  `improvement_pct 10.2983`. Device `118.892` → `13.4064 us/call`, kernels
  `6.98` → `1.00`. Fused attention into one Triton kernel.
- `002` / `aborted`: complete device budget (`13.4064 us/call`) below the 5%
  adoption budget (`16.3885 us/call`) → device-only ceiling `4.0902%`.
- `003` / `accepted` / `rounds/report_003.md`: `0.361050` → `0.298240` ms,
  `improvement_pct 17.3965`; vs accepted kernel `11.2080%` raw / `8.8072%`
  normalized. `output_allocations_per_call` `1.00 → 0.00`; host
  `233.645 → 206.375 us` (`-27.270`). Device `13.4096 → 13.4224` (+0.095%).
- `004` / **`no-improvement`** / `rounds/report_004.md`: protocol vs `base.py`
  `21.3249%` (median speedup `1.270171`), but **vs the canonical `e2_003` only
  `2.5939%` raw / `1.9317%` normalized / `4.35%` in-process — below the
  `14.871 us` threshold by `1.056 us`**. Mechanism fully confirmed (bare M0
  `192.255` → M1 `172.950 us`, `-19.305`; host `221.605 → 202.640 us`), device
  flat (`13.3272 → 13.3228`, `-0.03%`), kernel count `1.00`. Hypothesis
  `partially-confirmed`: three of four causal links held, the ≥5% wall link did
  not.

## Measurement Regime Notes (learned, do not re-derive)

- **The `base.py`-referenced protocol number is cumulative and must never be
  read as the round's contribution.** Round 003 reported `17.40%` and round 004
  `21.32%` against `base.py`, but round 004's own contribution over the canonical
  kernel was `2.59%`. `base.py` drifted `4.15%` between those two turns
  (`0.361050` → `0.376040`). **Always run the interleaved control against the
  current canonical kernel — that is the only number that can decide an
  adoption.**
- **Run interleaved controls in both orders.** Round 004's blocks 1-6
  (`e2_003` first) gave a median paired diff of `-10.145 us`; blocks 7-12
  (`e2_004` first) gave `-6.405 us`. Both favoured the candidate, so the
  direction was real, but the `3.7 us` disagreement between orderings is itself
  the noise scale of the cross-process comparison.
- **Speedup does NOT reliably cancel drift on this machine.** The premise is
  that reference and candidate drift together. Round 004 disproved it: between
  two windows of the *same turn*, `base.py`'s median moved `-6.0%`
  (`0.376040` -> `0.353615`) while the candidate moved only `-2.3%`
  (`0.295850` -> `0.289150`), so the measured speedup swung **`4.31%`**
  (`1.270171` -> `1.217744`). Consequences: never compare speedups across
  windows or turns; if a ratio-of-speedups bar is used, measure both candidates
  in strict pair-by-pair alternation in one window. A gap smaller than a few
  percent of ratio carries no information.
- **Wall conversion from forward gain is lossy at ~75%.** Round 004: forward
  `-18.965 us`, wall `-14.330 us`. Do not predict wall gain from a forward-level
  lever at 100%.
- The harness requires `--v0_file` to define `Model`, so a paired run is always
  `base.py` versus the candidate. Never compare against a historical baseline
  number; `base.py` medians ranged `~7%` within a single turn in round 003.
- `set_seed` runs before `start = time.perf_counter()` and is untimed, but
  `manual_seed_all` enqueues device work that the timed `sync_devices()` then
  drains, so the seed op is billed inside the timed region.
- `sync_devices()` costs `11.96 us/call` more than a bare
  `torch.npu.synchronize()` because `_iter_accelerators()` calls
  `torch.npu.is_available()` every time. Pre-resolving the accelerator list
  costs `0.600 us`.
- The profiler run's printed `v0`/`v1` are at defaults `200`/`500`, and the `v0`
  is `base.py`, **not** the `--profile-reference-file` scope. Measure the
  reference scope's wall separately at `200`/`500`.
- `export_chrome_trace` is called once per scope inside the capture loop, so the
  candidate export overwrites the reference one. Only the candidate
  `record_function` survives in the `.pt.trace.json`.
- Pass the explicit `ai_core_op_summary.db` path to `summarize_cann_trace.py`;
  scope directories accumulate databases across rounds.

## Instrumentation Traps (each cost real debugging time)

- **Launch counting:** `triton.backends.ascend.driver.NPULauncher` is shadowed
  and patching it silently reports zero launches. The real class is the compiled
  `ascend.NPULauncher`, reachable as
  `triton.runtime.driver.active.launcher_cls`; patch its `__call__`. Confirmed
  directly: `shadowed is real → False`.
- **Allocation counting:** a `TorchDispatchMode` op count is not a valid
  allocation counter. `aten.empty_like` decomposes to
  `aten.empty.memory_format`, and the Triton launch path issues one
  `aten.empty.memory_format` per launch on its own. Count at the Python level
  (`torch.empty` / `torch.empty_like`) for `output_allocations_per_call`.
- **Cache-hit detection:** a stable `data_ptr()` does not prove a cache hit —
  the caching allocator returns the same freed block. Count factory calls.
- **Model comparison:** `get_inputs()` draws fresh random tensors. Two models
  must be given `clone_value` of the *same* input set or their outputs will
  differ for no reason.

## Open Hypotheses or Checks

- **M2 / M3 launch paths** are the live lever. `-119.360` and `-139.580 us/call`
  against M1's `-19.305`. Both are already proven capable, bit-identical, and
  `1.00` launches per call by `log/probes/round_004_launch_abi_probe.json`. A
  decision naming one by name is the cheapest next step; no new probe needed.
- **The frozen profile still records `lifecycle.fast-launcher` as `Unknown`.**
  Round 004's evidence is round-local and does not amend it (hash-pinned at
  `a2c3e2e4…`). A later round must re-establish legality on its own evidence.
- **The epoch-1 bar question is unresolved at the policy level.** The
  maintainer's stated bar is "beat the epoch-1 deliverable" (speedup `1.02626`).
  Both `e2_003` (`1.189874`) and `e2_004` (`1.270171`) clear it, so it does not
  discriminate adopt/reject. Verifier flagged this in `report_004.md`;
  Orchestrator adjudicates.
- Residual wrapper (`~29.7 us/call`) is the fallback family and is small.
- Re-verify the buffer-reuse invariant by NaN-poisoning whenever the output
  cache or store coverage changes.
- Carried forward: the kernel requires `S <= 128` (campaign shape `S=83`); the
  second `tl.dot` tile `(128,64,128)` compiles and is numerically correct but
  was not one of the eleven probed tiles; `../triton_attn_001.py` is preserved
  intact.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 004 |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 004 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `triton_mm_encoder_attention_e2_001.py` | `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124` | 003 |
| `triton_mm_encoder_attention_e2_003.py` | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` | 004 |
| `triton_mm_encoder_attention_e2_004.py` | `f5aa1d709e4deeb1562757d795dd4da41217238dd15c53d33c2c338da1938020` | 004 |
| `rounds/decision_004.md` | `30758ad4dd30ccb0087534e47f61ea0443bdeead40ba64d41c28dd052c397088` | 004 |
| `rounds/coder_result_004.md` | `9c8c46ef1b58233e464a30022fd2b0dedf2fce7b95410a501d95e2e24ac59e0e` | 004 |
| `rounds/sketch_004.json` | `d3e52f6af032014381908e03e87a6b1c3f5694090686df2af3bfe3a6d9474dbf` | 004 |
