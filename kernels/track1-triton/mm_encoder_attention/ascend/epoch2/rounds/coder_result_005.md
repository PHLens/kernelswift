# Coder Result 005

## Identity

- Round: `005`
- Decision: `rounds/decision_005.md` (sha256 `1fdd16d7ddca961760260b9e6130c7e6d2fb17b689728474ee9e5bea9b8ce551`)
- Decision kind: `optimization`; change scope `host` / change family `launch-path-reduction`
- Sketch: `rounds/sketch_005.json` (sha256 `f44ed2bfbef80e9dc603494221bbc2cd47db40a9d8d48d85ee2ae344cd11c4ee`)
- Reference implementation: `triton_mm_encoder_attention_e2_003.py` (sha256 `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe`)
- Reference report: `rounds/report_003.md`
- Candidate: `triton_mm_encoder_attention_e2_005.py`
- Candidate SHA256: `bf54cea2a1fcdafd8916c2e0bf607766a6e7ffc2981fd956e18e92bf51b88b26`
- Classification: **`candidate-ready`**
- Capability gate: `lifecycle.fast-launcher` — `round_local_status: proven`, `new_probe_required: false`
- Selected mechanism: **M2, cached `CompiledKernel`**
- Selected profile: `triton_ascend` (snapshot sha256 `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321`, **unamended**)
- Capability claim: sha256 `a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d`
- Runtime fingerprint: `project.md#runtime-fingerprint` (Ascend910B4, torch 2.7.1+cpu / torch_npu 2.7.1.post4 / triton 3.2.0 / CANN 9.0.0)
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged)

## Classification

`candidate-ready`. The probe is not re-run; legality is carried by citation of
the retained round-004 artifacts, per decision 005's
`legality_reestablished_by` field. M2 was implemented as named and works under
every stated constraint. Whether wall clears 5% is Verifier's adoption
measurement and is not claimed here.

## Validation Gate

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 skills/kernel-opt-loop/scripts/validate_decision.py \
  kernels/track1-triton/mm_encoder_attention/ascend/epoch2/rounds/decision_005.md \
  --expected-implementation-profile triton_ascend \
  --project-root kernels/track1-triton/mm_encoder_attention/ascend/epoch2
```

Exit `0`, `"valid":true`, round `005`, scope `host`, family
`launch-path-reduction`, gate `lifecycle.fast-launcher` with
`round_local_status: proven` and `new_probe_required: false`, selected
mechanism `M2 cached CompiledKernel`, and the three round-004 evidence refs.

## In-process per-launch cost: M2 versus a reproduced M0 baseline

Script: `log/probes/round_005_mechanism_probe.py`, result
`log/probes/round_005_mechanism_probe.json`. One process, one regime
(warmup 50 / repeat 100, median of 3 blocks, queue drained before each timed
call), M0 reproduced in the same script.

Two independent runs, because decision 005 quoted a round-004 M2 figure that
turned out to be a different implementation of the stream:

| Quantity | Run 1 | Run 2 |
|---|---:|---:|
| **M0 proven `kernel[grid](...)` — reproduced here** | **173.535** | **182.150** |
| M2a `kernel[grid](*args, stream=cached_stream)` | 63.065 | 64.670 |
| **M2b `kernel[grid](*args)` — stream resolved per call — SHIPPED** | **85.770** | **88.720** |
| M2a saving vs M0 | 110.470 | 117.480 |
| **M2b saving vs M0** | **87.765** | **93.430** |
| cost of per-call stream resolution | +22.705 | +24.050 |

Baseline reproduction: run 2 measured M0 at **182.150 us/call** against
Verifier's **183.740 us/call** — drift **−0.87%**. The regime matches, so the
saving is apples-to-apples.

### A discrepancy in the decision's quoted number, resolved

Decision 005 quotes M2 at **66.895 us/call** from the round-004 probe. That
measurement passed `stream=cached_stream`, i.e. it is **M2a**, not the
Host-Plan-conformant form. This probe reproduces M2a at **63.065-64.670 us**
(consistent with 66.895 within this machine's drift), and measures the
conformant form M2b at **85.770-88.720 us**.

The gap is the per-call stream resolution that `CompiledKernel.__getitem__`
performs: `driver.active.get_current_device()` plus
`driver.active.get_current_stream()`, the latter routing through
`get_backend_func("get_current_stream", device)` on every call. It costs
**22-24 us/call** — more than M1's entire lever.

**M2b ships**, because the Host Plan's `device_stream_behavior` field says the
stream "is resolved per call by `CompiledKernel.__getitem__` exactly as the
proven path does", and because the proven path (`JITFunction.run`) also
resolves the stream per call. Caching it would contradict an explicit field and
would be the same class of hand-marshalling decision 005 section 3 uses to
reject M3. The 22-24 us is recorded rather than silently taken; recovering it
needs a decision amending that field, and the measurement to support it is
already on disk.

This does not change the magnitude argument. M2b saves **~88-93 us/call**
against a `14.871 us` wall requirement: propagation needed is
`14.871 / 87.765 = 16.9%`, versus M1's `67.5%`.

## Bit-identity versus e2_003

`max_abs_diff = 0.0`, `torch.equal == True`, verified on a NaN-poisoned buffer
so full store coverage is re-proved under the new launch path. Re-verified
after an `S` change and after a stride change, both of which produce a correct
finite result and re-prove the handle.

## Kernel definition stayed byte-identical — the check I ran

```bash
cd .../ascend/epoch2
diff <(sed -n '1,76p' triton_mm_encoder_attention_e2_003.py) \
     <(sed -n '1,76p' triton_mm_encoder_attention_e2_005.py)   # exit 0, no output
diff <(sed -n '1,74p' triton_mm_encoder_attention_e2_001.py) \
     <(sed -n '1,74p' triton_mm_encoder_attention_e2_005.py)   # exit 0, no output
```

Both exit `0` with no output, so the `@triton.jit` definition is unchanged from
`e2_003` and from the original `e2_001`. No `LibEntry` or other wrapper is used
(`grep` for `libentry`/`LibEntry` returns nothing), so constraint 1's
"keep `LibEntry`-style wrappers out" is satisfied by construction, not by
assertion.

Only `ModelNew.__init__` and `ModelNew.forward` changed. Two launch sites
exist and are mutually exclusive, so exactly one executes per call.

## Implementation Summary

1. **Cached `CompiledKernel` handle.** The kernel returned by the proven launch
   is stored on the instance and driven directly thereafter as
   `kernel[grid](q, k, v, out, sb, ss, S, scale)`. The stream is deliberately
   **not** passed, so `CompiledKernel.__getitem__` resolves it per call.
2. **Extended cache key.** Beyond the round-003 output key
   `(shape, dtype, device, query stride)`, the key adds `key.stride()`,
   `value.stride()`, `S`, the grid, and a precomputed constant bundle
   `(BLOCK_M, BLOCK_N, HEAD_DIM, NH, num_warps, num_stages, scale)`. The
   constructor-constant components are precomputed in `__init__` so the
   per-call build is a single 6-tuple.
3. **Zero-cost kernel-identity check.** The fast path is taken only when
   `self._launch_key == launch_key` **and** `kernel is self._proven_kernel`.
   The identity comparison adds no launch, so `kernel_count_per_call` stays at
   `1.00`. Verified by instrumenting the real launcher class: `1.00` for both
   `e2_003` and `e2_005`.
4. **Structural proof at resolution.** When the handle is (re)proven, the
   resolved kernel's `metadata.num_warps` and `metadata.num_stages` are checked
   against the configured values. A mismatch is an unproven resolution and
   disables the fast path sticky.
5. **Launch keyword bundle hoisted to `__init__`**, as in round 004: every
   launch argument other than q/k/v/out/strides/S is a constructor constant.
6. **Grid as a 3-tuple.** `grid = (bsz * NH, 1, 1)` serves both paths.
   `JITFunction.run` canonicalizes a 1-tuple to the same `(16,1,1)`, so the
   proven path's launch is unchanged and one tuple construction is saved.

### Fallback behaviour — the decision's five rows, all implemented and tested

| Event | Behaviour | Tested |
|---|---|---|
| handle not yet resolved | proven `_fused_attention_kernel[grid](...)` launch, then resolve and cache for this key | yes (steady state reached) |
| any cache-key component changes | handle discarded, **proven launch for that call**, handle re-proven for the new key, **not** disabled | yes — `S` change and stride change both re-prove with `_launcher_disabled == False` |
| fast path raises | exception swallowed, handle cleared, **disabled**, proven launch in the **same** call | yes — `CompiledKernel.__getitem__` patched to raise |
| fast path resolves a different `CompiledKernel` | handle cleared, **disabled**, proven launch in the **same** call | yes — `_proven_kernel` replaced with a sentinel |
| resolution or construction fails | `_launcher_disabled` set, proven launch for the remainder of the instance lifetime | yes — via the two rows above |

Both destructive tests produced a **correct, bit-identical output in the failing
call itself**, because the proven launch runs in the same call and fully
overwrites `out`. `_launcher_disabled` is sticky, so the worst case is always
accepted round-003 behaviour — never a wrong answer and never a per-call
penalty.

## Sketch Conformance

| Sketch requirement | Status |
|---|---|
| `target=triton_ascend` (required) | honored |
| `BLOCK_M=128`, `BLOCK_N=128`, `HEAD_DIM=64` (required) | honored, unchanged |
| `accumulator_dtype=fp32` (required) | honored, unchanged |
| `num_warps=4` (preferred) | honored, unchanged; asserted from `CompiledKernel.metadata` at resolution |
| `num_stages=1` (preferred) | honored, unchanged; asserted from `CompiledKernel.metadata` at resolution |
| `op_alloc_out` | round-003 instance cache retained unchanged |
| all device-side operations and controls | kernel definition byte-identical |
| `scope.entrypoints = [ModelNew.forward, ModelNew.__init__]` | only those two methods touched |
| `scope.unchanged_boundary` | see below |

### Unchanged computation boundary (verified)

| Boundary element | Evidence |
|---|---|
| `_fused_attention_kernel` `triton.jit` definition | `diff` lines 1-76 vs `e2_003` → exit `0`; also identical to `e2_001` |
| kernel launch count stays at one | instrumented: **1.00** for `e2_003` and `e2_005` |
| `BLOCK_M` / `BLOCK_N` / `HEAD_DIM` | `128` / `128` / `64` |
| accumulator dtype | fp32 |
| `num_warps` / `num_stages` | `4` / `1`; asserted from `CompiledKernel.metadata` when the handle is proven |
| device kernel name | `_fused_attention_kernel`; same `CompiledKernel` object, hash `18db9f0320830a397f740d02078551aeea898355fd7e06d59bb3a7bca2e1c903` |
| output shape / dtype / device / contiguity | `(2, 83, 512)`, `torch.float16`, `npu:0`, `is_contiguous() == True` |
| tolerance `atol=1e-2`, `rtol=1e-2` | smoke `PASS`; bit-identical (`max_abs_diff = 0.0`) |

## Host Plan Conformance

| Host Plan field | Implementation | Verdict |
|---|---|---|
| `state_owner` | ordinary instance attributes (`_kernel`, `_proven_kernel`, `_launch_key`); `state_dict()` is `[]` | conformant |
| `lifetime` | resolved through the proven launch on the first call after construction or any key change, reused until a key component changes | conformant |
| `allocation_reuse` | round-003 output buffer cache retained unchanged; no per-call allocation added | conformant |
| `cache_key` | output shape/dtype/device/query-stride, plus key and value strides, `S`, `scale`, grid, `BLOCK_M`/`BLOCK_N`/`HEAD_DIM`/`NH`, `num_warps`/`num_stages` | conformant |
| `invalidation` | any key change discards the handle and routes that call through the proven launch, re-proving for the new key; an unproven, mismatched, or failed resolution is a miss that never launches | conformant |
| `concurrency` | no lock, thread-local, or per-call state introduced | conformant |
| `device_stream_behavior` | stream resolved per call by `CompiledKernel.__getitem__`; no stream created, captured, or switched | conformant — this is why M2b ships over the faster M2a |
| `unchanged_behavior` | all eight items preserved and observed | conformant |

### Observed host invariants

```text
e2_005 kernel cached                : True
e2_005 launcher disabled            : False
e2_005 kernel is proven             : True
bit-identical to e2_003             : True (max_abs_diff=0.0)
alias of query/key/value            : False
state_dict                          : []
cache hit reuses buffer             : True
poisoned buffer leaks NaN           : False
output still bit-equal              : True
kernel launches per call e2_003     : 1.00
kernel launches per call e2_005     : 1.00
S change -> re-proven               : True
back on original key, bit-equal     : True
stride change -> re-proven          : True
identity mismatch -> disabled       : True
identity mismatch -> correct out    : True
identity mismatch -> handle clear   : True
forced exception -> disabled        : True
forced exception -> correct out     : True
forced exception -> handle clear     : True
```

Forward-level lever, one process, identical inputs:

```text
e2_003 forward alone : 219.610 us/call
e2_005 forward alone : 128.655 us/call
lever                : -90.955 us/call  (-41.42%)
```

## Deviations

None. Three conformance notes:

1. **M2b rather than the quoted M2a figure.** The decision quotes M2 at
   `66.895 us`; that figure is the cached-stream variant. The shipped form
   resolves the stream per call at `~88 us`, per the Host Plan's
   `device_stream_behavior`. This is a conformance note, not a deviation, but
   it is a `22-24 us` magnitude difference and is flagged as such.
2. **Launch keyword bundle and the constant part of the key hoisted to
   `__init__`.** Consequence: post-construction mutation of `scale`,
   `num_heads`, `head_size`, or the block/warp/stage counts would not affect
   the launch. Same trade-off recorded in rounds 003 and 004; nothing in the
   harness mutates them.
3. **Grid carried as a 3-tuple** `(bsz*NH, 1, 1)` rather than a 1-tuple,
   because `CompiledKernel.__getitem__` indexes `grid[0..2]` directly.
   `JITFunction.run` canonicalizes a 1-tuple to the identical triple, so the
   proven launch is bit-for-bit the same launch as before.

## Compile and Smoke Evidence

- Command:
  ```bash
  cd /workspace/kernelswift-dev-4ff2094
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_005.py \
    --warmup 5 --repeat 10 --full-traceback
  ```
- Exit status: `0`
- Observation: `PASS accuracy; v0=0.363515 ms, v1=0.224585 ms, speedup=1.619x`
  followed by `Summary: 1 passed, 0 failed, 1 total.`
- A second run of the same command gave
  `PASS accuracy; v0=0.377260 ms, v1=0.221005 ms, speedup=1.707x`.
- **Smoke-tier only, and not to be read as the round's effect.** Per the
  caution Verifier established in round 004, a speedup only cancels drift
  against another speedup from the same reference draws; within one round-004
  turn `base.py` moved `-5.96%` while the candidate moved `-2.26%`, swinging
  the measured speedup by `4.13%`. These two smoke runs are a correctness and
  conformance signal only. Verifier runs the decisive strict pair-by-pair
  alternation against `e2_003`.
- I did not run `auto_bench.py --warmup 50 --repeat 100` and did not run
  `--profile`.

### Local gate (Coder contract step 6)

| Gate | Result |
|---|---|
| `validate_decision.py --expected-implementation-profile triton_ascend` | exit `0`, `"valid":true` |
| kernel definition vs `e2_003` (lines 1-76) and `e2_001` (1-74) | exit `0`, byte-identical |
| `ast.parse` | ok; `_fused_attention_kernel`, `ModelNew`, `get_inputs`, `get_init_inputs` |
| real harness AST loader | ok; `get_init_inputs() == [8, 64, 8]` |
| public signatures | `__init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8)`; `forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor` — identical to `e2_003` |
| `read_lints` | `totalCount: 0` |
| forbidden constructs (`import triton_ascend`, `make_block_ptr`, `async_copy`, `vectorize`, `"cuda"`, `LibEntry`) | none |
| warm-up / compile smoke | `PASS accuracy`, exit `0` (twice) |

## Attempt Ledger

| Attempt | Command | Exit | Defect | Candidate before | Candidate after |
|---:|---|---:|---|---|---|
| 1 | `validate_decision.py`; mechanism probe | 0 | **probe defect**: summary block raised `KeyError` because `rows` was keyed `M2a`/`M2b` while `measured` used `M2a_cached_stream`/`M2b_percall_stream`. Timing printed correctly; the JSON was not written. | `not-applicable` | n/a |
| 2 | mechanism probe re-run | 0 | none — clean artifact, second independent sample | n/a | n/a |
| 3 | write candidate; kernel `diff`; smoke | 0 | none | `not-applicable` | (interim) |
| 4 | restructure the identity-mismatch branch | 0 | **design gap**: the preliminary version self-healed on an identity mismatch instead of disabling sticky, contradicting row 4 of the decision's five-row table. Fixed so a mismatch disables; a pure key change still re-proves. | (interim) | `bf54cea2a1fcdafd8916c2e0bf607766a6e7ffc2981fd956e18e92bf51b88b26` |

No Verifier repair was requested. Attempts 1, 3 and 4 are Coder's own
in-round defects, all found and fixed before `candidate-ready`.

## Risks Noted for Verifier

1. **The quoted M2 figure and the shipped mechanism differ by 22-24 us.**
   Decision 005 quotes `66.895 us`; the shipped M2b measures `~88 us` because
   the stream is resolved per call as the Host Plan requires. If Verifier's
   in-turn `launch_path_us_per_call` reads near `88` rather than `67`, that is
   expected and is not a regression. The measurement to recover the difference
   is in `log/probes/round_005_mechanism_probe.json`.
2. **Adoption is Verifier's.** The forward lever is `-90.955 us/call`; Verifier
   established in round 004 that only about `75%` of a forward lever reaches the
   synchronized wall. Even so this implies far more than the `14.871 us`
   threshold. The risk is not magnitude; it is that this machine drifts `5-7%`
   within a turn, so only the strictly interleaved same-window comparison against
   `e2_003` is decisive.
3. **`kernel_count_per_call` must be measured through the real launcher class.**
   `triton.backends.ascend.driver.NPULauncher` is shadowed by the compiled C++
   `ascend.NPULauncher` reachable via `triton.runtime.driver.active.launcher_cls`.
   Patching the shadowed class silently counts zero. My instrumented count is
   `1.00` for both candidates.
4. **The identity check is the whole safety net.** It is a zero-cost
   `is` comparison, so it cannot be observed by counting launches. If it ever
   fires, the observable is `model._launcher_disabled == True` with
   `model._kernel is None`. Worth asserting those two after a steady-state run
   to confirm the fast path is live rather than silently degraded.
5. **The handle is per-instance and per-key.** It is resolved on the first
   successful launch, so the first call in a fresh process pays the proven path.
   With warmup ≥ 1 this is invisible; with warmup 0 it would not be.
6. **The extended key includes `key.stride()` and `value.stride()`,** which the
   kernel does not actually use (it indexes all three tensors with
   `query.stride`). Including them is conservative and follows the Host Plan;
   it costs two tuple constructions per call. If Verifier wants that ~1 us
   back, it needs a decision amending the `cache_key` field.
7. **Carried forward unchanged:** the kernel requires `S <= 128` (campaign shape
   is `S=83`); the second `tl.dot` tile `(128,64,128)` compiles and is
   numerically correct but was not one of the eleven probed tiles; the
   buffer-reuse invariant depends on full store coverage and must be revisited
   if a later round introduces a masked or partial store.

## Profile and Campaign Integrity

- The frozen snapshot is **not** amended. `lifecycle.fast-launcher` remains
  `Unknown` and hash-pinned at
  `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321`.
- No new probe was run for legality; it is carried by citation of the round-004
  artifacts, per decision 005.
- `state/project_capability_claim.json` is untouched; no
  `qualification_disposition` was created.
- This evidence is round-local and does not license a later round.
- Files written by Coder this round: the candidate, this result,
  `state/coder_context.md`, and five files under `log/probes/`. Nothing else.

## Probe Evidence Paths

- `log/probes/round_005_mechanism_probe.py` — M0 vs the two M2 stream variants, baseline reproduced
- `log/probes/round_005_mechanism_probe.json` — machine-readable result
- `log/probes/round_005_candidate_conformance.py` — bit-identity, fallback, exact launch count, lever
- `log/probes/round_005_candidate_conformance.json` — machine-readable result
- `log/probes/round_005_probe_evidence.md` — narrative summary
- Retained and cited: `round_004_launch_abi_probe.{py,json}`, `round_004_probe_evidence.md`, `round_004_candidate_conformance.{py,json}`
