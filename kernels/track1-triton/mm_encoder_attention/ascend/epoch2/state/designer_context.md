# Designer Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5`
- context_epoch: `2`
- last_completed_round: `001`
- accepted_kernel: `triton_mm_encoder_attention_e2_001.py`
- accepted_report: `rounds/report_001.md`
- current_decision: `rounds/decision_002.md` (uncommitted, `abort`, `change_family` `no-change`)
- recent_three_round_evidence: `001 accepted at +10.2983% wall; device collapsed 118.892 -> 13.4064 us/call and launches 6.98 -> 1.00, leaving device_ratio 0.0407 and roughly 316 us/call of non-device residual. Epoch-1 history under ../ is labeled noncanonical and is not a source baseline.`
- open_hypotheses: `No kernel-side hypothesis survives the round-001 arithmetic bound: the whole device budget is 13.4064 us against a 327.770 us wall, so a perfect device elimination yields 4.0902% and misses the 5% threshold by 2.98 us/call. Every surviving candidate lives on the host side and is blocked by the maintainer constraint.`
- artifact_read_hashes: `see the table below`

## Maintainer Constraint (user directive for this epoch)

Host-side code is not to be touched. Host gain may arrive only indirectly from
launch-count reduction. Round 001 drove launch count to `1.00`, so that indirect
lever is now at its floor. Every remaining mechanism with more than `16.3885 us`
of headroom is host-side and therefore out of scope until the maintainer lifts
this constraint. This is an authorization boundary, not a measurement, and it is
the reason round 002 aborts rather than proceeding.

## Current Bottleneck

Verifier-backed facts from `rounds/report_001.md` only:

- Wall median is `0.327770 ms` = `327.770 us/call` against `13.4064 us/call` of
  device time, so `device_ratio` is `0.0407`. About `95.9%` of wall time is not
  device compute.
- Launch count is `1.00` kernel per call, down from `6.98`. This is the floor for
  a correct attention kernel.
- The candidate still emits exactly one kernel, `_fused_attention_kernel`, at
  `13.4064 us/call`. There are zero transpose kernels left; the round-000
  `63.35 us/call` of materialized layout conversion is gone.
- Removing six of seven launches while cutting device time by `105.4856 us/call`
  recovered only about `38 us` of wall time, so the dominant per-call host term
  does not scale with launch count.

Implication: the bottleneck is `host-bound` and the device side is exhausted.
`bottleneck-judgment.md` puts a `device_ratio` below 0.05 in the
`measurement-bound candidate` band, but that label requires targeted Level 2
evidence that the residual is harness-fixed; no such evidence exists for this
epoch, so the declared class stays `host-bound`.

### The arithmetic bound that closes the kernel-side search

```text
5% adoption budget = 0.05 * 327.770          = 16.3885 us/call
complete device budget                       = 13.4064 us/call
best possible device-only wall improvement   =  4.0902%
deficit against the threshold                =  2.9821 us/call
```

A kernel-side change that removed 100% of device time would still miss the
threshold. No tiling, fusion, precision, hint, or dataflow change inside the
kernel can escape this, because the bound is on the total device budget.

### Residual, derived not measured

Under the report's own `device_ratio` convention, the non-device residual is
`315.9586 us/call` for the candidate and `239.8280 us/call` for the reference, a
`76.1306 us/call` increase. This is arithmetic on two Verifier-reported numbers
under an additive device-plus-non-device assumption. It is not a measured host
decomposition and must never be cited as one. Its direction agrees with the
report's own statement that roughly `316 us/call` of host cost remains.

## Recent Three-round Evidence

- `001` / `accepted` / `rounds/report_001.md` / change family `kernel-fusion`:
  wall median `0.327770` ms versus re-measured reference `0.365400` ms,
  improvement `10.2983%`, above the 5% threshold. Device `118.892 -> 13.4064
  us/call`; launches `6.98 -> 1.00`; transposes `4.00 -> 0.00`; `device_ratio`
  `0.3314 -> 0.0407`. Hypothesis verdict `confirmed` on the device and launch
  links, `partially-confirmed` on the host link. First accepted round for this
  operator on Ascend.
- `000` / `baseline` / `rounds/report_000.md` / change family `not-applicable`:
  `baseline_adapter.py` is a faithful reproduction; improvement 0.52%, within
  noise. Baseline drifted +9.04% versus epoch 1 (`0.320635 -> 0.349625`) under an
  identical measurement fingerprint.

## Open Hypotheses or Checks

Ranked backlog. Every item is currently blocked; the ranking is by theoretical
headroom, not by availability.

1. **`launch-path-reduction`** — attack the Triton dispatch path itself. Derived
   residual suggests `>76 us/call` of headroom versus the reference, far above the
   `16.3885 us` budget. **Blocked twice:** it is host-side code, which the
   maintainer constraint forbids, and `lifecycle.fast-launcher` is `Unknown` in
   the frozen profile, so declaring it normative would be a `capability-miss`.
   This is the single highest-value item in the epoch and it is not actionable.
2. **`allocation-reuse`** — reuse the per-call `torch.empty_like` output buffer
   under a shape/dtype/device cache key. Host-side. **Blocked by the maintainer
   constraint.** Would need a full Host Plan (owner, lifetime, cache key,
   invalidation, concurrency, stream behavior).
3. **`measurement-decomposition`** — Verifier-owned Level 2 host decomposition to
   decide whether the residual is harness-fixed. Diagnostic only, not an
   intervention, and it becomes actionable only alongside item 1. Request this if
   Orchestrator wants the epoch's terminal stop reason to be `measurement-bound`
   rather than `host-blocked`.
4. **`kernel-config-tuning`** — profile-legal `num_warps` 1/2/4/8 and
   `num_stages` 1/2/3/4 search. **Rejected on the merits:** it is bounded by the
   same `4.0902%` ceiling, so it is predicted to fail before it runs. Also
   unavailable as `final-autotune` because `last_completed_binding` is `null` and
   the `binding_sha256` anchor cannot be resolved.
5. **`kernel-tiling`** — row-blocked loop to lift the `S <= 128` restriction.
   Correctness and generality only. The campaign shape is `S=83`, so this carries
   no wall-time upside and is not an adoption candidate.

## Candidate Limitations to Carry Forward

- The accepted kernel requires `S <= 128` and raises rather than silently
  producing wrong output. Campaign shape is `S=83`.
- The second `tl.dot` shape `(128,64,128)` compiles and is numerically correct but
  was not among the eleven probed tiles.
- `torch.empty_like` emits an internal-format warning on this runtime; warning
  only, does not affect the measured path.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 000 |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 000 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `rounds/report_000.md` | `64fe68820ac2b5b45211477dca5de66ac53b9bdbadbbc96297b0b6ae925dfb55` | 000 |
| `rounds/report_001.md` | `89ee8b2a3861e84eda32ed8198906ffcfeaa8e99bf22a6d97d4738c525542af3` | 001 |
| `rounds/decision_001.md` | `3775c9548afc7070898ee73ead2e6ecad19225525b58052946f2ff5e3c4c0167` | 001 |
| `rounds/sketch_001.json` | `76818c21a7502a68b6ec5c6230607fa24bddf3e342e61d4d333990d16d639738` | 001 |
| `rounds/coder_result_001.md` | `e3c1b57193230fa47187f491a0f3946f19981b53a0a675749625ad1beb62d4e0` | 001 |
| `triton_mm_encoder_attention_e2_001.py` | `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124` | 001 |
| `project.md` | `914eb006c9132b39f12787f816f42d76ef2803a1aaba371954e5ee81083c3ab1` | 001 |
| `state/runtime-snapshot.json` | `6004296625865f2aea0ed6e72b1ff0e0d2b6122b9eff7567de3382d53dfb4ad1` | 001 |
| `state/implementation_profile_snapshot/profile.yaml` | `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321` | 001 |
| `state/project_capability_claim.json` | `a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d` | 001 |
| `skills/kernel-opt-loop/prompts/designer.md` | `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5` | 001 |
| `skills/kernel-opt-loop/references/decision-template.md` | `a081503562fa30751f8df63ba3553e1766b9707d9af663810d800f829409ffa0` | 001 |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | 001 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | 001 |
