# Designer Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5`
- context_epoch: `2`
- last_completed_round: `002`
- accepted_kernel: `triton_mm_encoder_attention_e2_001.py`
- accepted_report: `rounds/report_001.md`
- current_decision: `rounds/decision_003.md` (uncommitted, `proceed`, `change_scope` `host`, `change_family` `allocation-reuse`)
- recent_three_round_evidence: `001 accepted at +10.2983% wall; device 118.892 -> 13.4064 us/call, launches 6.98 -> 1.00, device_ratio 0.0407. 002 aborted: the whole device budget is 13.4064 us against a 327.770 us wall, so a perfect device elimination yields 4.0902% and misses the 5% budget of 16.3885 us by 2.98 us. Epoch-1 history under ../ is labeled noncanonical and is not a source baseline.`
- open_hypotheses: `Host side is now authorized. Round 003 attacks the per-call output allocation (allocation-reuse). launch-path-reduction is the fallback family and needs an Ascend launch-ABI probe first. The device side stays closed by the 4.0902% bound.`
- artifact_read_hashes: `see the table below`

## Maintainer Constraint (updated at round 003)

Superseded. The round-002 constraint (host-side code out of scope; host gain only
indirectly via launch-count reduction) was lifted by the maintainer and recorded
in `team-state.md` Policy Revisions at commit `de1b9b7`:

> `maintainer_constraint.host_code`: host-side code out of scope -> host-side
> code authorized; `launch-path-reduction` and `allocation-reuse` Host Plan
> rounds permitted.

The device-only ceiling of `4.0902%` is unchanged and still closes every
device-side mechanism. Counters and run epoch were untouched by the revision.

## Current Bottleneck

Verifier-backed facts from `rounds/report_001.md` only:

- Wall median `0.327770 ms` = `327.770 us/call` against `13.4064 us/call` of
  device time, so `device_ratio` is `0.0407`. About `95.9%` of wall time is not
  device compute.
- Launch count is `1.00` kernel per call, down from `6.98`. Floor for a correct
  attention kernel.
- Removing six of seven launches while cutting device time by `105.4856 us/call`
  recovered only about `38 us` of wall time, so the dominant per-call host term
  does not scale with launch count.

Classification is `host-bound`. Under `bottleneck-judgment.md` a ratio below 0.05
is the `measurement-bound candidate` band, but that label requires Level 2
evidence that the residual is harness-fixed, and no such evidence exists yet.

### The device-side bound (still closing)

```text
5% adoption budget = 0.05 * 327.770          = 16.3885 us/call
complete device budget                       = 13.4064 us/call
best possible device-only wall improvement   =  4.0902%
deficit against the threshold                =  2.9821 us/call
```

### Residual, derived not measured

Under the report's own `device_ratio` convention the non-device residual is
`315.9586 us/call` (candidate) and `239.8280 us/call` (reference). This is
arithmetic on two Verifier numbers under an additive device-plus-non-device
assumption. It is **not** a measured host decomposition and must never be cited
as one. Its internal split between harness synchronize, Triton launch path, and
output allocation is unknown.

**Confirming evidence to request from Verifier:** a Level 2 host decomposition in
one process and regime measuring (a) harness wall, (b) `ModelNew.forward` alone,
(c) `forward` plus the harness synchronize boundary, (d) an allocation-free
`forward` variant. (b)-(d) sizes the allocation lever; (c)-(b) sizes the
harness-fixed term no host round can touch.

### Harness timing structure (read from source, not measured)

From `auto_bench.py` `time_forward` (harness sha256 `71fb3ad0…`): `set_seed` runs
**before** the timer starts, so it is not timed. The timed region is
`torch.no_grad(): model.forward(*inputs)` followed by `sync_devices()`
(`torch.npu.synchronize()`). Every microsecond of Python and allocator work
inside `ModelNew.forward` is therefore directly billable, which is what makes a
host round able to move wall time at all.

## Recent Three-round Evidence

- `002` / `aborted` / `rounds/decision_002.md` / change family `no-change`:
  no kernel-side intervention can clear 5%. Device budget `13.4064 us` is below
  the `16.3885 us` budget even at zero. Launch count already `1.00`. Accepted as
  correct by Orchestrator; the maintainer then authorized host-side code.
- `001` / `accepted` / `rounds/report_001.md` / change family `kernel-fusion`:
  wall median `0.327770` ms versus re-measured reference `0.365400` ms,
  improvement `10.2983%`. Device `118.892 -> 13.4064 us/call`; launches
  `6.98 -> 1.00`; transposes `4.00 -> 0.00`; `device_ratio` `0.3314 -> 0.0407`.
  Verdict `confirmed` on device and launch links, `partially-confirmed` on host.
- `000` / `baseline` / `rounds/report_000.md` / change family `not-applicable`:
  `baseline_adapter.py` is a faithful reproduction; improvement 0.52%, within
  noise. Baseline drifted +9.04% versus epoch 1 under an identical fingerprint.

## Open Hypotheses or Checks

1. **`allocation-reuse`** — SELECTED for round 003. Cache the output buffer on the
   `ModelNew` instance under an explicit cache key so the steady-state forward
   performs zero allocations; allocate with `torch.empty` rather than
   `torch.empty_like`, which also removes the per-call internal-format warning
   path recorded in `rounds/coder_result_001.md`. Zero new capability required.
   Expected gain is a judgment (declared 8.0%) with wide uncertainty.
2. **`launch-path-reduction`** — the fallback family and the larger prize on paper.
   Its high-value form bypasses `JITFunction.run` and invokes the cached compiled
   kernel directly. **Not taken first for a capability reason:** the frozen
   profile records `launch_abi: "kernel[(grid)](args)"` with direct launch as the
   proven path and `lifecycle.fast-launcher` as `Unknown`, so declaring an
   unproven launcher normative converts the round into a `capability-miss`.
   Requires an Ascend launch-ABI probe before it can be declared normative.
3. **`measurement-decomposition`** — the Level 2 host decomposition described
   above. Diagnostic only. Highest-value evidence in the epoch: it sizes both
   levers and settles whether the residual is harness-fixed.
4. **`kernel-config-tuning`** — `num_warps` 1/2/4/8 and `num_stages` 1/2/3/4
   search. **Rejected on the merits:** bounded by the same `4.0902%` ceiling, so
   predicted to fail before it runs. Also unavailable as `final-autotune` while
   `last_completed_binding` is `null`.
5. **`kernel-tiling`** — row-blocked loop to lift the `S <= 128` restriction.
   Correctness and generality only; no wall upside at the campaign shape `S=83`.

## Round 003 Contract Resolution (carry forward)

Schema-version 2 calls `_validate_v2_sketch` unconditionally, so a `host`-scope
decision still needs a real Sketch artifact; the v1 `N/A: host-only change`
marker is not accepted. Resolution used and validated: `change_scope: "host"` with
a Sketch that declares the **unchanged** computation boundary
(`scope.kind = unchanged-computation-boundary`) plus a `required` eight-field Host
Plan. Both validators exit 0 on this form.

## Candidate Limitations to Carry Forward

- The accepted kernel requires `S <= 128` and raises rather than silently
  producing wrong output. Campaign shape is `S=83`.
- The second `tl.dot` shape `(128,64,128)` compiles and is numerically correct but
  was not among the eleven probed tiles.
- `torch.empty_like` emits an internal-format warning on this runtime. Coder
  asserted it does not affect the measured path; that was never measured. Round
  003 removes the call, which settles it either way.
- Output-buffer reuse is only safe while the store covers the whole buffer. The
  `B*NH` = 16 programs together write every row `0..S-1` of every head slice, so
  full coverage holds today. Any future masked or partial store invalidates this
  Host Plan.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 000 |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 002 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `rounds/report_000.md` | `64fe68820ac2b5b45211477dca5de66ac53b9bdbadbbc96297b0b6ae925dfb55` | 000 |
| `rounds/report_001.md` | `89ee8b2a3861e84eda32ed8198906ffcfeaa8e99bf22a6d97d4738c525542af3` | 002 |
| `rounds/decision_001.md` | `3775c9548afc7070898ee73ead2e6ecad19225525b58052946f2ff5e3c4c0167` | 001 |
| `rounds/sketch_001.json` | `76818c21a7502a68b6ec5c6230607fa24bddf3e342e61d4d333990d16d639738` | 001 |
| `rounds/coder_result_001.md` | `e3c1b57193230fa47187f491a0f3946f19981b53a0a675749625ad1beb62d4e0` | 002 |
| `rounds/decision_002.md` | `8b8d36508920e310f35a55a8459742d187a8d313f8b302920a93103ec8dbebc7` | 002 |
| `rounds/decision_003.md` | `a4956891de5fef4b9bd629fb3cceb270db5a247ba18b591aecee9480d96c5455` | 002 |
| `rounds/sketch_003.json` | `51ebe3a735c7659309e781fd2f35286fd4e67acc86b5d0a9f6676f08f08af69c` | 002 |
| `triton_mm_encoder_attention_e2_001.py` | `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124` | 002 |
| `project.md` | `914eb006c9132b39f12787f816f42d76ef2803a1aaba371954e5ee81083c3ab1` | 002 |
| `state/runtime-snapshot.json` | `6004296625865f2aea0ed6e72b1ff0e0d2b6122b9eff7567de3382d53dfb4ad1` | 001 |
| `state/implementation_profile_snapshot/profile.yaml` | `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321` | 002 |
| `state/project_capability_claim.json` | `a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d` | 002 |
| `skills/kernel-opt-loop/prompts/designer.md` | `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5` | 002 |
| `skills/kernel-opt-loop/references/decision-template.md` | `a081503562fa30751f8df63ba3553e1766b9707d9af663810d800f829409ffa0` | 002 |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | 002 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | 002 |
