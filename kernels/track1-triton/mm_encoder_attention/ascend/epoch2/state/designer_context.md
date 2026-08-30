# Designer Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5`
- context_epoch: `2`
- last_completed_round: `003`
- accepted_kernel: `triton_mm_encoder_attention_e2_003.py`
- accepted_report: `rounds/report_003.md`
- current_decision: `rounds/decision_005.md` (draft, `proceed`, `change_scope` `host`, `change_family` `launch-path-reduction`, mechanism **M2 cached `CompiledKernel`**)
- round_004_status: **not terminated.** `rounds/report_004.md` is not on disk. Round 004 produced `coder_result_004.md` (`candidate-ready`, M1 `fast_libentry`, `f5aa1d70…`) and has not been measured. **No round-004 result, wall figure, or terminal classification is recorded here.**
- recent_three_round_evidence: `003 accepted at +17.3965% against base.py (0.361050 -> 0.298240 ms); +11.2080% raw / +8.8072% base-normalized against e2_001. 002 aborted on the 4.0902% device ceiling. 001 accepted at +10.2983%. Epoch-1 history under ../ is labeled noncanonical and is not a source baseline.`
- open_hypotheses: `The round-004 probe proved lifecycle.fast-launcher and measured three realizations. M2 (cached CompiledKernel, 66.895 us) is selected for round 005: 119.360 us saving needing only 12.5% propagation, versus M1's 22.030 us needing 67.5%. M3 is dominated. The 22.635 us wrapper is the fallback family. The device side stays closed at 4.0902%.`
- artifact_read_hashes: `see the table below`

## Maintainer Constraint

Host-side code is authorized. Recorded in `team-state.md` Policy Revisions at
commit `de1b9b7`:

> `maintainer_constraint.host_code`: host-side code out of scope -> host-side
> code authorized; `launch-path-reduction` and `allocation-reuse` Host Plan
> rounds permitted.

The device-only ceiling of `4.0902%` still closes every device-side mechanism.
Counters and run epoch were untouched by the revision.

## Current Bottleneck — measured, not inferred

The Level 2 host decomposition in `rounds/report_003.md` is a Verifier
measurement in one process and one regime, against a wall median of `297.410 us`.
This replaces every earlier residual arithmetic; do not reuse the round-002
`~316 us` figure.

| Quantity | us/call |
|---|---:|
| (a) harness wall (`auto_bench.time_forward`) | 297.410 |
| (b) `ModelNew.forward` alone, no synchronize | 206.375 |
| (c) `forward` + `torch.npu.synchronize()` | 258.190 |
| (d) allocation-free direct launch, preallocated output, no wrapper | 183.740 |
| (a) - (b) harness-fixed term | 91.035 |
| (b) - (d) residual `forward` wrapper | 22.635 |

### The slice table

| Slice | us/call | Share of wall | Reachable by a host round? |
|---|---:|---:|---|
| harness-fixed (outside `ModelNew.forward`) | 91.035 | 30.61% | **no** |
| Triton launch path (bare launch) | 183.740 | 61.78% | only by `launch-path-reduction`, capability Unknown |
| residual `forward` wrapper | 22.635 | 7.61% | yes, no new capability needed |
| device kernel time | 13.4224 | — | sub-component of the sync term; bounded at 4.09% |

**Device time is not a fourth independent slice.** The `13.4224 us` of device work
sits inside the `51.815 us` synchronize term, which sits inside the harness-fixed
`91.035 us`. The shares therefore sum above 100%.

### Where the harness-fixed 91.035 us goes

```text
(a) - (c) = 39.220 us   seed drain + harness dispatch
(c) - (b) = 51.815 us   synchronize term
```

`set_seed` is called before `start = time.perf_counter()` and so is untimed, but
`mod.manual_seed_all(seed)` enqueues device work that the timed `sync_devices()`
then waits for, so the seed op is billed inside the timed region. On top of that
`sync_devices()` costs `11.96 us/call` more than a bare `torch.npu.synchronize()`
because `_iter_accelerators()` calls `torch.npu.is_available()` every time. This
is harness code and is out of scope for any host round.

### The 5% budget against each slice

```text
0.05 * 297.410 = 14.871 us
14.871 / 183.740 =  8.09% of the launch path
14.871 /  22.635 = 65.71% of the residual wrapper
```

### Ordinary device bound (unchanged)

```text
5% adoption budget = 0.05 * 327.770          = 16.3885 us/call   (round-001 wall)
complete device budget                       = 13.4064 us/call
best possible device-only wall improvement   =  4.0902%
```

## Round-004 Launch-ABI Probe (retained evidence)

Coder-produced Decision-scoped probe under `log/probes/`. **Legality evidence, not
a Verifier adoption measurement.** Reused by citation; it does not amend the
frozen snapshot (`lifecycle.fast-launcher` stays `Unknown`, hash-pinned) and does
not license later rounds beyond those that re-establish legality on their own.

Artifacts: `round_004_launch_abi_probe.json`,
`round_004_probe_evidence.md`, `round_004_candidate_conformance.json`.

In-process, warmup 50 / repeat 100 / 3 blocks, M0 baseline reproduced in-script:

| Mechanism | us/call | saving vs M0 | bit-identical | same kernel object |
|---|---:|---:|---|---|
| M0 proven `kernel[grid](...)` | 186.255 | — | control | control |
| M1 `fast_libentry` | 164.225 | 22.030 | yes (`0.0`) | yes |
| M2 cached `CompiledKernel` | 66.895 | 119.360 | yes (`0.0`) | yes |
| M3 `NPULauncher.launch` C entry | 46.675 | 139.580 | yes (`0.0`) | yes |

- All four share kernel hash `18db9f0320830a397f740d02078551aeea898355fd7e06d59bb3a7bca2e1c903`, so the same-compiled-kernel criterion holds for every mechanism.
- M0 reproduces Verifier's `183.740 us` at `186.255 us` (`+1.37%`), so the regime matches and **no new probe is needed** for round 005.
- M1 forward-level lever measured at `-18.470 us` (`223.505 -> 205.035`) against a `14.871 us` threshold: margin `3.599 us` = `1.21%` of wall. Coder's attempt ledger shows it was `-11.715 us` before hoisting the launch kwargs bundle, so ~10 us of the lever is implementation-detail fragile.
- Propagation needed to clear 5%: M1 `67.5%`, M2 `12.5%`, M3 `10.7%`. M1 observed `83.8%`.
- M3 vs M2 is only `20.220 us` more, against materially deeper coupling (compiled C++ `ascend.NPULauncher` C entry, hand-marshalled `function` + `packed_metadata`, no `launch_metadata`, failure mode is a wrong successful call rather than an exception). **M3 is dominated.**

## Recent Three-round Evidence

- `003` / `accepted` / `rounds/report_003.md` / change family `allocation-reuse`:
  `0.361050 -> 0.298240 ms` against `base.py`, `improvement_pct = 17.3965`.
  Against `e2_001`: `11.2080%` raw / `8.8072%` base-normalized at
  `warmup 50 / repeat 100`, `10.1299%` / `11.5476%` at `warmup 200 / repeat 500`.
  Observables: `output_allocations_per_call` `1.00 -> 0.00`; `forward` alone
  `233.645 -> 206.375 us` (`-27.270`); `device_us_per_call` `13.4096 -> 13.4224`
  (`+0.095%`, wrong direction, so device explains `0.035%` of the change);
  `kernel_count_per_call` `1.00 -> 1.00`. Verdict `confirmed`. Output is
  bit-identical to the accepted kernel.
- `002` / `aborted` / `rounds/decision_002.md` / change family `no-change`: no
  kernel-side intervention can clear 5%. Accepted as correct; triggered the
  maintainer authorization.
- `001` / `accepted` / `rounds/report_001.md` / change family `kernel-fusion`:
  `+10.2983%`. Device `118.892 -> 13.4064 us/call`; launches `6.98 -> 1.00`.
- `000` / `baseline` / `rounds/report_000.md`: `baseline_adapter.py`, 0.52%,
  within noise. Baseline drifted +9.04% versus epoch 1 under an identical
  fingerprint.

## Open Hypotheses or Checks

1. **`launch-path-reduction` / M2 cached `CompiledKernel`** — SELECTED for round
   005. Legality already discharged by the retained round-004 probe; no new probe
   work. M2 saves `119.360 us/call` at the bare-launch level and needs only
   `12.5%` propagation to clear the `14.871 us` threshold, against M1's `67.5%`.
   Expected wall `32.9-39.3%` (declared `30.0` conservatively). **M1 rejected on
   margin** (`3.599 us`, `1.21%` of wall, against ~7% intra-turn drift, and its
   lever was below threshold before an implementation detail was fixed).
   **M3 rejected as dominated**: `20.220 us` more than M2 for a jump from
   Triton-runtime `CompiledKernel.__getitem__` to the backend-internal compiled
   C++ `ascend.NPULauncher` C entry, whose failure mode is a wrong successful
   call that the sticky fallback cannot detect.
2. **`launch-path-reduction` / M3** — held in reserve. Only `20.220 us` above M2.
   A future round wanting it must re-establish legality on its own evidence and
   must answer the coupling cost set out above.
3. **`host-wrapper-reduction`** — fallback family. Only `22.635 us` remain and
   clearing 5% needs `65.71%` of it. Contents at a cache hit: `query.shape`
   unpacking, the four-component cache-key tuple including a fresh
   `query.device` construction per call, the key comparison, and the grid tuple.
   Needs no new capability, which is its entire merit.
4. **`kernel-config-tuning`** — `num_warps` 1/2/4/8 and `num_stages` 1/2/3/4.
   **Rejected on the merits:** bounded by the `4.0902%` device ceiling. Also
   unavailable as `final-autotune` while `last_completed_binding` is `null`.
5. **`kernel-tiling`** — row-blocked loop to lift `S <= 128`. Correctness and
   generality only; no wall upside at the campaign shape `S=83`.
6. **`harness-fixed reduction`** — **permanently out of scope.** `91.035 us` is
   harness code; `bottleneck-judgment.md` forbids altering the harness to
   manufacture a speedup.

## Carry-forward Contract Facts

- **Schema-v2 always validates a Sketch**, including at `change_scope: host`. The
  v1 `N/A: host-only change` marker is not accepted. Resolution used in rounds
  003 and 004: `host` scope + a Sketch declaring the unchanged computation
  boundary (`scope.kind = unchanged-computation-boundary`) + a `required`
  eight-field Host Plan.
- **`fallback_provenance` is unavailable in this project.**
  `state/project_capability_claim.json` carries `qualification_dispositions: []`,
  and the validator requires the named disposition to be embedded in the claim
  with `fallback_authorized: true`. It is also restricted to
  `algorithm-substitution`, which no host round is. Do not attempt it.
- **An abort is only expressible at schema_version 1.** `_validate_metadata_v2`
  requires `decision == "proceed"`. A v1 abort validates with
  `--expected-profile`, not `--expected-implementation-profile`.
- **Allocation counting trap.** A `TorchDispatchMode` count still reports one
  `aten.empty.memory_format` per call because the Triton launch path allocates on
  its own. Read `output_allocations_per_call` at the Python level. Never treat a
  dispatch-mode count as the output-allocation observable.
- **Drift is large.** Within a single turn `base.py` medians ranged
  `0.346350`-`0.370825` (~7%). Always re-measure the reference in the same turn.

## Candidate Limitations to Carry Forward

- The kernel requires `S <= 128` and raises rather than silently producing wrong
  output. Campaign shape is `S=83`.
- The second `tl.dot` tile `(128,64,128)` compiles and is numerically correct but
  was not among the eleven probed tiles.
- The kernel definition is byte-identical across `e2_001` and `e2_003`. Round 004
  keeps it byte-identical; only the invocation site and a cached launcher handle
  change.
- The reuse invariant depends on full store coverage. Any future masked or
  partial store invalidates the round-003 Host Plan.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 000 |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 003 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `rounds/report_000.md` | `64fe68820ac2b5b45211477dca5de66ac53b9bdbadbbc96297b0b6ae925dfb55` | 000 |
| `rounds/report_001.md` | `89ee8b2a3861e84eda32ed8198906ffcfeaa8e99bf22a6d97d4738c525542af3` | 002 |
| `rounds/decision_001.md` | `3775c9548afc7070898ee73ead2e6ecad19225525b58052946f2ff5e3c4c0167` | 001 |
| `rounds/sketch_001.json` | `76818c21a7502a68b6ec5c6230607fa24bddf3e342e61d4d333990d16d639738` | 001 |
| `rounds/coder_result_001.md` | `e3c1b57193230fa47187f491a0f3946f19981b53a0a675749625ad1beb62d4e0` | 002 |
| `rounds/decision_002.md` | `8b8d36508920e310f35a55a8459742d187a8d313f8b302920a93103ec8dbebc7` | 002 |
| `rounds/decision_003.md` | `a4956891de5fef4b9bd629fb3cceb270db5a247ba18b591aecee9480d96c5455` | 003 |
| `rounds/sketch_003.json` | `51ebe3a735c7659309e781fd2f35286fd4e67acc86b5d0a9f6676f08f08af69c` | 003 |
| `rounds/coder_result_003.md` | `d60e74e94f5e87ffbe2c535f8caea8d58c1fc7d4b104e1b0351fb9d854ac948d` | 003 |
| `rounds/report_003.md` | `f5dbb4dfefadd88ee8b7ea1f98efb657334143cf18050706f424585f9cd9dcef` | 003 |
| `rounds/decision_004.md` | `30758ad4dd30ccb0087534e47f61ea0443bdeead40ba64d41c28dd052c397088` | 003 |
| `rounds/sketch_004.json` | `d3e52f6af032014381908e03e87a6b1c3f5694090686df2af3bfe3a6d9474dbf` | 003 |
| `rounds/coder_result_004.md` | `9c8c46ef1b58233e464a30022fd2b0dedf2fce7b95410a501d95e2e24ac59e0e` | 003 |
| `log/probes/round_004_launch_abi_probe.json` | `ec9c7f61560cfd07bd6bff24ad9b801045d97ba9ce8d0f45fb3b01feb4325fdd` | 003 |
| `rounds/decision_005.md` | `1fdd16d7ddca961760260b9e6130c7e6d2fb17b689728474ee9e5bea9b8ce551` | 003 |
| `rounds/sketch_005.json` | `f44ed2bfbef80e9dc603494221bbc2cd47db40a9d8d48d85ee2ae344cd11c4ee` | 003 |
| `triton_mm_encoder_attention_e2_001.py` | `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124` | 003 |
| `triton_mm_encoder_attention_e2_003.py` | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` | 003 |
| `project.md` | `914eb006c9132b39f12787f816f42d76ef2803a1aaba371954e5ee81083c3ab1` | 003 |
| `state/runtime-snapshot.json` | `6004296625865f2aea0ed6e72b1ff0e0d2b6122b9eff7567de3382d53dfb4ad1` | 001 |
| `state/implementation_profile_snapshot/profile.yaml` | `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321` | 003 |
| `state/project_capability_claim.json` | `a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d` | 003 |
| `skills/kernel-opt-loop/prompts/designer.md` | `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5` | 003 |
| `skills/kernel-opt-loop/references/decision-template.md` | `a081503562fa30751f8df63ba3553e1766b9707d9af663810d800f829409ffa0` | 003 |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | 003 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | 003 |
