# Final Summary — mm_encoder_attention / ascend910b / epoch 2

## Outcome

| | Value |
|---|---|
| Final canonical kernel | `triton_mm_encoder_attention_e2_005.py` |
| Candidate SHA-256 | `bf54cea2a1fcdafd8916c2e0bf607766a6e7ffc2981fd956e18e92bf51b88b26` |
| Accepted report | `rounds/report_005.md` |
| Terminal rounds | 5 (`accepted`, `aborted`, `accepted`, `no-improvement`, `accepted`) |
| Stop reason | `user-intervention` (maintainer ended the epoch after round 005) |
| Kernel definition | byte-identical across all four candidates (`c741834a…`) |

The deliverable rule is satisfied: the submission is a correctness-PASS Triton
implementation. The kernel is a single fused self-attention kernel with one launch
per call, no `@triton.autotune`, no runtime search, and no autotune-cache
dependency. Configuration is pinned at `num_warps=4`, `num_stages=1`, and the
launcher's cache key pins them further by checking
`proven_kernel.metadata.num_warps/num_stages` before taking the fast path.

## Measured progression

Wall medians below are drawn from each round's own window. **Cross-turn comparison
is not sound on this machine** (see the methodology note), so the per-round figure
that matters is the last column, measured by strict pair-by-pair alternation against
the previous accepted kernel inside one window.

| Round | Intervention | Result | Same-window gain |
|---:|---|---|---:|
| 000 | Phase 0 baseline (`baseline_adapter.py`) | baseline | — |
| 001 | fused single-kernel attention | accepted | +10.90% |
| 002 | device-side search | aborted | device-only ceiling 4.0902% < 5% |
| 003 | host output-buffer reuse | accepted | +8.59% |
| 004 | M1 `fast_libentry` launch path | no-improvement | +2.89% (bar +5%) |
| 005 | M2 cached `CompiledKernel` launch path | accepted | **+41.55%** |

Round 005 in absolute terms: `e2_003` `0.296900 ms` → `e2_005` `0.211295 ms` in one
window, a real **28.83%** wall improvement. Five of five alternating pairs cleared
the bar; the weakest was `+34.31%` and the spread was `+34.31%`–`+49.00%`.

Indicative only, **not a sound comparison**: against the epoch-2 starting point
(`baseline_adapter.py` at `0.347800 ms`), the final candidate is about `39%` faster.
Against the epoch-1 deliverable (`triton_attn_001.py` at `0.339685 ms`), about `38%`
faster. Both are cross-turn and inherit the drift discussed below.

## Where the time goes now

Against the final wall of `212.445 us`:

| Term | us/call | Share | Reachable |
|---|---:|---:|---|
| harness-fixed | 97.260 | 45.8% | no |
| Triton launch path | 89.220 | 42.0% | partly |
| residual Python wrapper | 25.965 | 12.2% | yes |
| device | 13.478 | 6.3% | yes |

The harness-fixed term did not grow in absolute terms (`94.645 → 97.260`); it became
the largest slice because everything else shrank. It decomposes into `51.8 us` of
synchronize plus seed drain and `sync_devices()` overhead, and no host or kernel
round can touch it.

## Why no final-autotune was run

This is recorded deliberately, so the omission is auditable rather than an
oversight.

1. **It is blocked by the frozen profile.** `validate_configuration_domain` requires
   `configuration_constraints` in the profile; the frozen `triton_ascend` snapshot
   has none, and validation returns `profile-legality-unavailable`. Adding the field
   now would change the profile hash and invalidate the `implementation_profile_snapshot_sha256`
   pinned in every decision from round 001 to 005, effectively requiring a new Phase 0.
   This was an onboarding omission: the capability matrix was built, but configuration
   legality was not.
2. **It could not have improved anything.** Configuration search acts only on the
   device side, and device is now `6.34%` of wall. Even eliminating device time
   entirely yields `6.34%`; a realistic `10–30%` device gain yields `0.63%–1.90%`.
   No point in the domain clears `5%`, so the outcome would have been
   `fallback-retained` with the configuration unchanged.
3. **The compliance goal is already met.** The final candidate hard-codes
   `num_warps=4` and `num_stages=1`, contains no autotune decorator, and performs no
   runtime or cache-dependent configuration selection, which is what the
   finalization rules exist to guarantee.

If configuration tuning is wanted later, it must be preceded by a profile promotion
adding `configuration_constraints` for `num_warps` (`1/2/4/8`) and `num_stages`
(`1/2/3/4`), both of which already have approved probe evidence in the profile.

## Known limitations

- **The kernel requires `S <= 128`.** The campaign shape is `S=83`, so this is fine
  here; longer sequences need a row-blocked loop. That is a generality fix with no
  wall benefit.
- **The second `tl.dot` shape `(128,64,128)` was not among the eleven probed tiles.**
  It compiles and is numerically correct here, but it is outside the probed envelope.
- **The launcher fast path is a runtime optimization, not a profile capability.**
  `lifecycle.fast-launcher` remains `Unknown` in the frozen profile. The candidate
  works because the fast path degrades to the proven launch on any failure, never
  because the capability was promoted. Any new campaign must re-establish legality on
  its own evidence.
- **Stream resolution costs `23.000 us` per call** and is mandated by
  `device_stream_behavior`. Two independent probes agree within `1 us`. Recovering it
  requires a decision amending that field, and must specify how the raw handle is
  obtained: `torch.npu.current_stream()` cannot be passed to
  `CompiledKernel.__getitem__` on this runtime (it raises
  `TypeError: argument 4 must be int, not Stream`).
- **The cache key includes `key.stride()` and `value.stride()`**, which the kernel does
  not use (it indexes all three tensors with `query.stride`). This is conservative per
  the Host Plan and costs roughly `1 us`.

## Methodology finding, carried forward

Established in round 004 and confirmed in round 005: **a speedup cancels drift only
against another speedup drawn from the same reference measurements.** Within a single
turn, `base.py`'s median moved `-5.96%` while the candidate moved only `-2.26%`,
swinging one candidate's measured speedup by `4.13%`. Separately, the same kernel
(`e2_003`) read `1.210602` in round 003's turn and `1.188531` in round 005's turn, a
`1.8%` difference.

Consequence: compare candidates only by strict pair-by-pair alternation inside one
window. Cross-window and cross-turn speedup ratios are not decisive and must not be
defended.

## Evidence ledger

All artifacts are committed on `kernel-opt/mmenc-attn-e2-ascend-20260830`:

- Decisions `rounds/decision_001..005.md`, typed sketches `rounds/sketch_001/003/004/005.json`
- Coder results `rounds/coder_result_001/003/004/005.md`, reports `rounds/report_000..005.md`
- Frozen profile `state/implementation_profile_snapshot/`, claim `state/project_capability_claim.json`
- Capability probe evidence `log/probes/` (gitignored by design, referenced by hash in the decisions)
- Team state `team-state.md` with the append-only transition log and policy revisions

## Knowledge lift — candidates for promotion

Generic patterns from this epoch, in the order I would promote them:

1. **Device-ceiling abort.** When the entire device budget is smaller than the
   adoption budget, a device-side round is arithmetically impossible, not merely
   unlikely. The bound is `device_us_per_call / wall_us < threshold`. This is a
   cheap, decisive check that saved a wasted round here.
2. **Launch path dominates small-shape kernels.** For `S=83` attention, the Triton
   launch path cost more than six times the kernel's own device time. Any backend
   with a heavyweight launch path should be measured this way before optimizing
   kernels.
3. **The `device-win-wall-loss` counterexample generalizes.** An 88.7% device
   reduction produced only a 10.3% wall gain. Device wins must never be reported as
   wall wins.
4. **Measurement drift defeats naive ratios.** See the methodology finding above.
5. **Mechanism ordering must separate legality from selection.** "Stop at the first
   mechanism that satisfies the criteria" is correct for proving a capability exists
   and wrong for choosing among mechanisms once magnitudes are known; it is
   indifferent to a 6x lever difference.

Items 1–4 are backend-independent and would be worth promoting to KernelWiki as
cards. Item 5 is a Designer-contract lesson.
