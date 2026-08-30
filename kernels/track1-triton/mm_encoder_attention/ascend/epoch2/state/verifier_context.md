# Verifier Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f`
- context_epoch: `2`
- last_completed_round: `005`
- accepted_kernel_as_of_this_file: `triton_mm_encoder_attention_e2_003.py`
  (round 005 classified `accepted`, so Orchestrator promotes
  `triton_mm_encoder_attention_e2_005.py`. Verifier never updates canonical
  pointers.)
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: see below
- open_hypotheses: **the `23.000 us` per-call stream resolution.** M2a (cached
  stream) measures `66.220 us` against the shipped M2b's `89.220 us`. Taking it
  needs a decision amending `device_stream_behavior`. Two independent probes
  agree within `1 us`. Beyond that, the launch path's remaining `66.220 us` and
  a `25.965 us` wrapper.
- artifact_read_hashes: `see the table below`

## Current Bottleneck

Against the round-005 candidate's in-process `212.445 us` wall:

| Slice | us/call | Share of wall |
|---|---:|---:|
| **harness-fixed, outside `ModelNew.forward`** | **97.260** | **45.8%** |
| Triton launch path (bare M2b launch) | 89.220 | 42.0% |
| residual `forward` wrapper | 25.965 | 12.2% |
| device kernel time | 13.478 | inside the sync term |

**The harness-fixed term is now the largest slice and it is unreachable.** It
grew from `30.6%` of wall to `45.8%` not by growing in absolute terms
(`94.645 → 97.260 us`) but because everything else shrank. Every further
microsecond a round removes is a larger fraction of a smaller host budget but a
smaller fraction of a wall that is now nearly half harness overhead. Expect
sharply diminishing returns from here.

Device is closed and nearly irrelevant: `13.478 us/call` at `1.00` kernels.

## Recent Five-round Evidence

- `001` / `accepted` / `rounds/report_001.md`: `0.365400` → `0.327770` ms,
  `improvement_pct 10.2983`. Device `118.892` → `13.4064 us/call`, kernels
  `6.98` → `1.00`. Fused attention into one Triton kernel.
- `002` / `aborted`: complete device budget (`13.4064 us/call`) below the 5%
  adoption budget (`16.3885 us/call`) → device-only ceiling `4.0902%`.
- `003` / `accepted` / `rounds/report_003.md`: `0.361050` → `0.298240` ms;
  vs accepted kernel `11.2080%` raw / `8.8072%` normalized.
  `output_allocations_per_call` `1.00 → 0.00`; host `233.645 → 206.375 us`.
- `004` / `no-improvement` / `rounds/report_004.md`: M1 `fast_libentry`.
  Mechanism confirmed (bare M0 `192.255` → M1 `172.950 us`) but the governing
  bar gave only **`+2.8874%`**; all estimators `1.9-4.4%`, below 5%. Lesson
  recorded below: M1's lever (`~19-22 us`) was too small for this machine's
  noise band.
- `005` / **`accepted`** / `rounds/report_005.md`: M2 cached `CompiledKernel`,
  shipped as M2b. Governing bar **`+41.5498%`** (5 of 5 pairs clear, weakest
  `+34.31%`). Wall `298.130 → 212.445 us` in-process; bare launch
  `178.915 → 89.220 us`. Device `13.4816 → 13.4780` (`-0.027%`). Hypothesis
  **`confirmed`**.

## Measurement Regime Notes (learned, do not re-derive)

- **The governing bar is `speedup(candidate) / speedup(previous_accepted) - 1 >= 5%`,
  and both candidates must be measured in strict pair-by-pair alternation inside
  ONE window.** This is the round-004 campaign rule.
- **Speedup does NOT reliably cancel drift on this machine.** In round 004,
  between two windows of the *same turn*, `base.py` moved `-6.0%` while the
  candidate moved `-2.3%`, so the measured speedup swung `4.31%`. Never compare
  speedups across windows or turns. A gap under a few percent of ratio carries
  no information.
- **The bar's output is a ratio of ratios, not a wall-time improvement.** Round
  005: bar metric `+41.5498%`, actual wall improvement `28.83%`. Both clear 5%
  here, but on a small effect they can diverge enough to matter. **Report both.**
- **The `base.py`-referenced protocol number is cumulative** (`40.71%` in round
  005, of which roughly `17` points are round 003's). Never read it as a round's
  contribution.
- The harness requires `--v0_file` to define `Model`, so a paired run is always
  `base.py` versus the candidate. Never compare against a historical baseline;
  `base.py` medians ranged `~7%` within a single turn in round 003.
- **Wall propagation from a forward lever is not a constant.** Round 004 saw
  `~75%`, round 005 saw `97.0%`. Do not assume either.
- `set_seed` runs before `start = time.perf_counter()` and is untimed, but
  `manual_seed_all` enqueues device work that the timed `sync_devices()` then
  drains, so the seed op is billed inside the timed region.
- `sync_devices()` costs `11.96 us/call` more than a bare
  `torch.npu.synchronize()` because `_iter_accelerators()` calls
  `torch.npu.is_available()` every time.
- The profiler run's printed `v0`/`v1` are at defaults `200`/`500`, and the `v0`
  is `base.py`, **not** the `--profile-reference-file` scope. Measure the
  reference scope's wall separately at `200`/`500`.
- `export_chrome_trace` runs per scope inside the capture loop, so the candidate
  export overwrites the reference one. Only the candidate `record_function`
  survives in the `.pt.trace.json`.
- Pass the explicit `ai_core_op_summary.db` path to `summarize_cann_trace.py`.
  Scope directories accumulate databases across rounds — the `e2_003` reference
  directory already holds round-004 and round-005 captures.

## Instrumentation Traps (each cost real debugging time)

- **Launch counting:** `triton.backends.ascend.driver.NPULauncher` is shadowed
  and reports zero. Patch `triton.runtime.driver.active.launcher_cls.__call__`
  (the compiled `ascend.NPULauncher`). Counts are `1.00` for every candidate.
- **Fast-path detection for M2:** the identity check `kernel is
  self._proven_kernel` is a zero-cost `is` comparison and is invisible to launch
  counting. Install the counting proxy on **both** `_kernel` and
  `_proven_kernel` simultaneously — if they differ, the candidate's own check
  disables the path and the probe measures the wrong thing.
- **Stream variant detection (M2a vs M2b):** instrument
  `driver.active.get_current_stream`. `1.00`/call means per-call resolution
  (M2b). Note that M2a **cannot** be written with `torch.npu.current_stream()`
  on this runtime — it raises `TypeError: argument 4 must be int, not Stream`;
  it needs the raw handle from
  `driver.active.get_current_stream(driver.active.get_current_device())`.
- **Allocation counting:** a `TorchDispatchMode` count is not a valid allocator
  counter. `aten.empty_like` decomposes to `aten.empty.memory_format`, and the
  Triton launch path issues one `aten.empty.memory_format` per launch on its
  own. Count at the Python level (`torch.empty` / `torch.empty_like`).
- **Cache-hit detection:** a stable `data_ptr()` does not prove a cache hit —
  the caching allocator returns the same freed block. Count factory calls.
- **Model comparison:** `get_inputs()` draws fresh random tensors. Two models
  must be given `clone_value` of the *same* input set.

## Open Hypotheses or Checks

- **Cached stream (`23.000 us`)** is the largest identified remaining host lever
  and needs a decision amending `device_stream_behavior`. The amendment must
  also specify how the stream handle is obtained, given the `TypeError` above.
- **M3 (`NPULauncher` C entry)** was measured at `-139.580 us` in round 004 and
  rejected there on coupling-risk grounds. Its marginal gain over M2 is now
  smaller still in relative terms. Re-establishing it would require new evidence
  and a fresh justification of the coupling cost.
- **The frozen profile still records `lifecycle.fast-launcher` as `Unknown`.**
  All evidence is round-local and does not amend it (hash-pinned `a2c3e2e4…`).
  A later round must re-establish legality on its own evidence.
- Residual wrapper is `~26 us` and is the fallback family.
- Re-verify the buffer-reuse invariant by NaN-poisoning whenever the output
  cache or store coverage changes.
- Carried forward: the kernel requires `S <= 128` (campaign shape `S=83`); the
  second `tl.dot` tile `(128,64,128)` compiles and is numerically correct but
  was not one of the eleven probed tiles; `../triton_attn_001.py` is preserved
  intact.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 005 |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 005 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `triton_mm_encoder_attention_e2_001.py` | `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124` | 003 |
| `triton_mm_encoder_attention_e2_003.py` | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` | 005 |
| `triton_mm_encoder_attention_e2_004.py` | `f5aa1d709e4deeb1562757d795dd4da41217238dd15c53d33c2c338da1938020` | 004 |
| `triton_mm_encoder_attention_e2_005.py` | `bf54cea2a1fcdafd8916c2e0bf607766a6e7ffc2981fd956e18e92bf51b88b26` | 005 |
| `rounds/decision_005.md` | `1fdd16d7ddca961760260b9e6130c7e6d2fb17b689728474ee9e5bea9b8ce551` | 005 |
| `rounds/coder_result_005.md` | `b8f8a06fddaa4328dc340ece21af997acab7250b6a4a0db33df335f68a087268` | 005 |
| `rounds/sketch_005.json` | `f44ed2bfbef80e9dc603494221bbc2cd47db40a9d8d48d85ee2ae344cd11c4ee` | 005 |
