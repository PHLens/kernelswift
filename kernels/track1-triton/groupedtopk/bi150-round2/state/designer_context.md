# Designer Context

> Materialized from `references/role-context-template.md` (contract_version 3,
> semantic_contract typed-sketch-v1). Written during Phase 0 (phase
> `0-initializing`). Contains no runtime claim for THIS campaign; historical
> epoch-1 (`../bi150`, contract_version 2) evidence is admissible only when
> explicitly labeled NONCANONICAL. Epoch-1 has different base/harness bytes and
> a different `measurement_fingerprint`; its wall numbers are never in-regime
> comparables.

- role_contract_sha256: `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5`
- context_epoch: `1`
- last_completed_round: `null`
- accepted_kernel: `null`
- accepted_report: `null`
- recent_three_round_evidence: `none in this campaign; see Recent Three-round Evidence for labeled epoch-1 history`
- open_hypotheses: `5-item bounded backlog (SEL-FUSE-01, DISPATCH-02, FUSION-TOPK-03, CHECK-TIE-audit, HOST-SLIM-04)`
- artifact_read_hashes: `see Artifact Read Hashes ledger`

## Current Bottleneck

- NO campaign-local Verifier-backed fact exists yet: team-state records
  `last_accepted_kernel: null`, `current_round: 000`; the round-000 baseline
  run is dispatched separately by Orchestrator. Any bottleneck classification
  below is a Phase-0 PRIOR derived from labeled historical evidence, to be
  confirmed or replaced by the round-000 report before any Round-1 decision.
- [NONCANONICAL, epoch-1 verifier reports] At the epoch-1 baseline
  (`report_000.md`: eager `base.py` @`d57ace7d…`, old fingerprint `57bf01…`),
  wall was `0.474612 ms` with `14.8 kernels/call`, device `177.18 us/call`,
  `device_ratio 0.373` — mixed, tending host-bound: many tiny framework
  kernels plus launch/dispatch dominated before any fusion. Largest device
  contributors: `gatherTopK` `48.7 us/call` + `bitonicSortKVInPlace`
  `36.9 us/call` (~85.6 us/call combined, both tied to exact `torch.topk`
  retention), then reduce/copy elementwise chain.
- [NONCANONICAL, epoch-1 verifier reports] Interventions moved wall mainly by
  collapsing kernel count/dispatch, not by shrinking selection math: r004
  (two Triton preprocessing stages, exact topk retained) `-7.46%`;
  r008 (`torch.compile` default dispatch) `-19.99%` over r004; r009
  (`mode="reduce-overhead"` CUDA Graph capture) `-22.51%` over r008, ending at
  `0.277234 ms`. Report_009 names the end-state bottleneck as graph-replay +
  retained exact top-k path; under reduce-overhead the BI150 trace cannot
  attribute graph-internal `cat=kernel` events (attribution caveat).
- Design constraint carrying over: exact-ID equality versus live
  library-reference output forces keeping the selection path semantically
  faithful to `torch.topk` OR proving equivalent tie behavior on-device.

## Phase 0 Semantic Analysis (groupedtopk)

### A. Exact operator decomposition (from `../base.py` @`12f33248…`, read-only)

Per token row `j ∈ [0,83)`; `T=83`, `E=256`, `num_expert_group=8`,
`topk_group=4`, `topk=8`, `renormalize=True`, `routed_scaling_factor=1.0`,
`scoring_func="softmax"` (public constructor permits other values):

1. `scores = softmax(gating_output[j], dim=-1)` — full-row normalization over
   all 256 fp32 logits (`[83,256]` input; row-wise).
2. Reshape view `[256] -> [8 groups x 32 contiguous experts]`
   (`view(num_token, 8, -1)`); group g covers experts `32g … 32g+31`.
   `group_scores[g] = max over the group's 32 scores` → `[T,8]`.
3. `group_idx = torch.topk(group_scores, k=4)[1]` — best 4 of 8 groups,
   descending value order, torch tie behavior.
4. Mask build: zeros scatter → expand-to-experts → boolean mask `[T,256]`;
   unselected groups' 128 surviving-slot complement zeroed out.
5. `tmp_scores = masked_fill(~mask, -inf)` — 128 of 256 lanes set `-inf`.
6. `(topk_weights, topk_ids) = torch.topk(tmp_scores, k=8)` — descending;
   second torch.topk call; survivors pool is 128 lanes (4 groups x 32).
7. `renormalize: weights /= weights.sum(dim=-1, keepdim=True)` — NOTE: two
   distinct normalizers exist (step-1 full-row softmax denominator vs step-7
   sum over only the 8 selected values); they are NOT interchangeable.
8. `routed_scaling_factor != 1.0` multiplies weights (identity at `1.0`;
   public contract allows other values — cannot hardcode away).
9. Outputs: `topk_weights[83,8] fp32`, `topk_ids[83,8] int32`.

Critical metadata-only dependency: `hidden_states[83,7168] fp16` participates
ONLY in `assert hidden_states.size(0) == gating_output.size(0)`; its payload is
never loaded. An optimized forward may leave it untouched (no fp16 load —
which is unproven on this profile anyway); the batch-size assertion must stay
observable-behavior compatible.

### B. Tie-semantics constraints vs torch.topk

- Contract bar: integer IDs compared by EXACT equality against the live
  reference; float outputs `atol=rtol=1e-2`. Floating proximity cannot rescue
  an ID flip. `capability_claim.json` primary_contract
  `reduction.argmax-grouped-selection`, `tie_semantics: torch.topk-exact-order`.
- Profile classification: `tl.argmax` Supported ONLY for axis-0 argmax over an
  `(8,)` fp32 vector with a UNIQUE maximum; "argmax tie and repeated selection"
  is Constrained (repeated top-k selection and PyTorch-compatible tie ordering
  remain unproven). Argmax over `(256,)` or axis-1 over `(8,32)` is unproven —
  a normative unproven requirement is `capability-miss`.
- Softmax monotonicity hazard: exp/div are weakly monotone in fp32, so score
  order matches logit order except where distinct logits collapse to equal
  scores (or reassociation/different exp lowering changes comparisons). Ties
  created by rounding differ from ties in raw logits; selection must operate
  exactly like the reference (which selects on post-softmax scores).
- Empirical evidence that torch.topk CUDA tie order is NOT plain
  ascending-index (NONCANONICAL epoch-1 guardrail suites, which PASSED because
  candidates kept the exact library calls): all-equal case ids
  `[7,6,4,5,1,0,2,3]`; two-expert-tie case `[1,0,2,3,4,5,6,7]` (tied pair 1>0);
  structured-group case `[32,0,64,96,4,3,1,2]` (equal group-maxima selected at
  stride-32 offsets in descending-id order, tail ascending). Any custom
  in-kernel selection must reproduce such orders bit-exactly ⇒ a value/index
  total-order key alone is insufficient until the ACTUAL ordering rule is
  audited on the current runtime/fingerprint (see CHECK-TIE-audit).
- Safe-by-construction alternative within the claim: retain BOTH
  `torch.topk` calls (library-exact tie behavior unchanged) while fusing only
  the surrounding reductions/masking — this is what passed epoch-1 and remains
  the fallback-compatible primary path for early rounds.

### C. Dataflow amenable to single-kernel fusion on triton_cuda (BI-V150, cc 7.1)

- Fully parallel over tokens; no cross-token dependence anywhere in steps 1–9.
  Per-row working set is one KB-class fp32 vector (256 logits ≈ 1 KB) —
  trivially block-resident; 83-row grid fits a 1-D launch
  (`tl.program_id` axis 0 is the proven configuration).
- Proven-primitive coverage for a fused pipeline (snapshot §Supported):
  `tl.load/store` contiguous fp32 extents ≤256 (stores at 256/8/1 incl. int32),
  `tl.arange` 256/8, `tl.reshape` (256,)→(8,32) and (8,)→(8,1),
  `tl.max`/`tl.sum` axis-1 over (8,32) and axis-0 over (256,) fp32, `tl.exp`,
  `tl.where`, `tl.broadcast_to` (8,1)→(8,32), `tl.zeros`/`tl.full`,
  `tl.static_range` (≥4 iterations), elementwise transcendentals `sqrt/sin/cos`
  lowering bit-identically to torch, and `tl.dot` (probe-backed fp32 exact /
  bf16 near-exact — though dot is superfluous for pure selection; using it
  adds unproven-shape risk without a causal mechanism for wall time).
- Boundaries a fusion design must respect (snapshot §Constrained/§Unknown):
  (i) argmax beyond `(8,)` unique-maximum + repeated selection unproven;
  (ii) masked indexing limited to contiguous (8,32)-style masking —
  gather/scatter/arbitrary multidimensional indexing/aliasing unproven, so
  DYNAMIC COMPACTION DESIGNS ARE NORMATIVE-RISK (formulate mask+argmax instead
  of gathers); (iii) `num_warps`/`num_stages` Unknown — may appear in a
  decision only as OPTIONAL exploratory fields backed by a matched local
  probe, never normatively; (iv) non-contiguous/mixed-dtype regimes unproven
  (inputs here are fp32-contiguous gating + untouched fp16 hidden_states);
  (v) stream/context and `fast_libentry` lifecycle properties unproven — Host
  Plan must preserve caller device/stream and use direct launch syntax;
  (vi) no matched profiler export claim beyond measured `cat=kernel`
  durations actually observed during verification.
- Host-side regime (project.md Measurement Regime): warmup 50 / repeat 100 /
  profile 100 iters kernel-mode; adoption requires ≥5% UNROUNDED paired median
  wall improvement vs `last_accepted_kernel` plus all guardrails.
- Anti-pattern consultation (references/anti-patterns.md; all entries are
  CONDITIONAL evidence from MLU590-H8/Triton-3.2.0 fingerprints — preconditions
  do NOT match this runtime; treat as cautions, never as rules):
  Entry 011 hierarchical parallel group-argmax rounds regressed 20→45 us;
  Entry 012 full bitonic sort-32+sort-64 networks regressed ~752%;
  Entry 013 dynamic `tl.gather` compaction regressed 9.37%; Entry 016 cumsum
  prefix compaction regressed 5.02%. Shared lesson on THIS operator class:
  prefer narrow fixed-mask reductions over gathers/sorts/prefix scans unless a
  matched local microbenchmark proves otherwise.

## Recent Three-round Evidence

Campaign-local rounds completed: NONE (Phase 0). Labeled historical record —
final three TERMINAL rounds of the read-only epoch-1 lineage
(`../bi150`, same operator/device class, DIFFERENT measurement_fingerprint
`57bf01…`; all values below are noncanonical priors only):

| Round | Result | Change family | Key Verifier-backed numbers | Pointer |
|---|---|---|---|---|
| 004 | accepted | preprocess-fusion (two direct Triton stages around exact torch.topk) | wall 0.466908→0.432098 ms (-7.46%); kernels/call 14.86→9.9; device 178.99→127.26 us/call | `../bi150/rounds/report_004.md` |
| 008 | accepted | compile-graph-default (`torch.compile` dispatch, unchanged dataflow) | wall 0.430385→0.344360 ms (-19.99%); device 127.46→111.12 us/call | `../bi150/rounds/report_008.md` |
| 009 | accepted | compile-graph-reduce-overhead (CUDA Graph capture) | wall 0.344360→0.277234 ms (-22.51%); graph replay hides internal kernels (attribution caveat) | `../bi150/rounds/report_009.md` |

Epoch-1 aborted/rejected rounds (002 candidate-failed, 003 design-rejected;
001/005–007 aborted) left canonical pointers unchanged and are irrelevant as
baselines. Rounds 000–003 details were sampled only as needed; the transition
log above is authoritative.

## Open Hypotheses or Checks

Ranked backlog (each item = candidate Round-N intervention; none is committed;
expected gains are priors, never measurements):

- **SEL-FUSE-01** — change_family `preprocess-fusion.triton-stages`. Replace
  eager softmax/group-max/mask framework ops with direct Triton stages while
  retaining BOTH exact `torch.topk` calls (tie-safe-by-construction).
  Bottleneck basis: fragmentation 14.8 kernels/call + 179 us/call device at
  eager baseline [NONCANONICAL r000]. Expected wall gain: ≥5% (same-family
  lineage delivered -7.46% in-regime once). Risk: low-medium (no new selection
  semantics; new probes unneeded). Evidence: `../bi150/rounds/report_004.md`.
  Validation cost: medium (full correctness + authoritative timing).
- **DISPATCH-02** — change_family `compile-graph.capture`. Wrap the accepted
  fixed-shape forward in `torch.compile` (default mode first, reduce-overhead
  only as a separate follow-on decision given its profiler-attribution
  caveat). Bottleneck basis: host launch/dispatch share at device_ratio
  ~0.30 (mixed/host-bound) [NONCANONICAL r004–r009]. Expected gain: ≥10%
  cumulative; family validated twice in lineage (-19.99%, -22.51%). Risk:
  medium — profile marks `torch.compile` Constrained (graph coverage/lifecycle
  unproven); MUST fall back to eager on non-target shapes/errors; interplay
  with direct Triton stage launches must hold. Evidence:
  `../bi150/rounds/report_008.md`, `report_009.md`. Validation cost: medium.
- **FUSION-TOPK-03** — change_family `kernel-fusion.single-pass-selection`.
  ONE Triton kernel implementing steps 1–9 fully, eliminating library topk
  entirely. Bottleneck basis: retained `gatherTopK`+`bitonicSortKVInPlace`
  ~85.6 us/call are the largest attributable device contributors in every
  epoch-1 trace [NONCANONICAL]. Expected gain: large (single launch; removes
  dominant device + most dispatch), IF correctness achievable. Risk: HIGH —
  hinges on (a) on-device proof of repeated top-k/argmax selection with
  torch-exact tie order (profile Constrained ⇒ capability-miss exposure),
  (b) avoid gather/sort/prefix formulations (anti-pattern entries). Gate:
  CHECK-TIE-audit must resolve tie rule first. Validation cost: high.
- **CHECK-TIE-audit** — non-round investigation to schedule BEFORE any
  FUSION-TOPK-03 decision: derive torch.topk's deterministic tie-order rule on
  THIS runtime for the exact regime (sorted=True, k∈{4-of-8} and {8-of-256},
  post-softmax fp32 inputs incl. manufactured duplicate logits/values), via
  matched local file-backed probes agreed through Orchestrator (Designer
  writes no runtime facts itself). Output feeds Sketch fallback provenance;
  until then any custom selection stays non-normative fallback.
- **HOST-SLIM-04** — change_family `host.allocation-minimization`. Trim
  per-forward temporaries/wrapper work around whatever dataflow survives
  earlier rounds (allocation/cache-key discipline per invariants; no
  cross-instance state). Bottleneck basis: device_ratio <0.4 residual host
  time [NONCANONICAL classification heuristics]. Expected gain: likely <5%
  standalone ⇒ bundle only into a mixed change whose device piece is
  separately observable, else hold. Risk: low. Validation cost: low-medium.

Standing checks:
- Input-path discrepancy (reported to Orchestrator):
  `/root/CodeBuddy/20260818191200/kernelswift/kernels/track1-triton/groupedtopk/bi150/final_summary.md`
  DOES NOT EXIST (verified by directory listing and repo-wide search; sibling
  operators have one, groupedtopk/bi150 does not). Not blocking; lineage
  evidence taken from `team-state.md` + `rounds/report_000/004/008/009.md`.
- This epoch's `base.py` (@`12f33248…`) and harness (@`71fb3ad0…`) differ from
  epoch-1 bytes (`d57ace7d…`/`3d4fa4ee…`): ALL epoch-1 wall/device numbers are
  outside this campaign's measurement fingerprint `8deb1b01…` and cannot be
  presented as in-regime speedups.
- Round 000 baseline is dispatched separately; Designer stays idle during
  verification/measurement exclusivity windows (measurement_exclusive=false at
  context-write time).

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---|
| `skills/kernel-opt-loop/prompts/designer.md` (role contract) | `7227706c7068ad4a20caebb95c045721f643a409473fc9768e73d828fb2e5ab5` | P0 |
| `skills/kernel-opt-loop/adapters/claude-code.md` | `31a161224900c8e7af2c3b9175adbf64165c2f56199790aee264a0cd3d8fb597` | P0 |
| `skills/kernel-opt-loop/references/invariants.md` | `2349247c5653db35ab5af5b22267f6ab813fad1f24076c88d6bf80207cbd8cb7` | P0 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | P0 |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | P0 |
| `skills/kernel-opt-loop/references/role-context-template.md` | `d3eead2d8480975a9a954b104d21c6d9d57e1713edf0fc8096184c09aafe56d2` | P0 |
| `kernels/track1-triton/groupedtopk/base.py` | `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58` | P0 |
| `bi150-round2/project.md` | `9b2483b34789463aa0afe7f86ef2475a2d7926cd654a90b41cbfbf58d3a66821` | P0 |
| `bi150-round2/profile_snapshot/triton_cuda.md` | `8b9cb9836c4abf97141081288d9eb68af7a571309057181e5ec1914827249a2f` | P0 |
| `bi150-round2/profile_snapshot/capability_claim.json` | `bc50f7f974f025e6be49d611e2546b6db6426d0761b794001898482f80f91371` | P0 |
| `../bi150/team-state.md` | `38dda5717a5019ad917689225ced71ebd9a8362082c7d59f60579743be8e6e57` | P0 |
| `../bi150/rounds/report_000.md` | `8198ae7cda910799d7dfc7aa081aaf4280973bce254c3af0c1cb1441a08476f5` | P0 |
| `../bi150/rounds/report_004.md` | `2821208486c00f6add2bac177819fc8fc39c931170cfea2b4efb5dcf26eb6042` | P0 |
| `../bi150/rounds/report_008.md` | `42f6b7a713e09b0adef661c0e24d85e7afd28d253fd72a04a5b721894b773fb5` | P0 |
| `../bi150/rounds/report_009.md` | `015be0aef0d96b09702d393014892862cc84fc68657ae2e848813546e3644f6d` | P0 |
