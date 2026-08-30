# Coder Result 004

## Identity

- Round: `004`
- Decision: `rounds/decision_004.md` (sha256 `30758ad4dd30ccb0087534e47f61ea0443bdeead40ba64d41c28dd052c397088`)
- Decision kind: `optimization`; change scope `host` / change family `launch-path-reduction`
- Sketch: `rounds/sketch_004.json` (sha256 `d3e52f6af032014381908e03e87a6b1c3f5694090686df2af3bfe3a6d9474dbf`)
- Reference implementation: `triton_mm_encoder_attention_e2_003.py` (sha256 `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe`)
- Reference report: `rounds/report_003.md`
- Candidate: `triton_mm_encoder_attention_e2_004.py`
- Candidate SHA256: `f5aa1d709e4deeb1562757d795dd4da41217238dd15c53d33c2c338da1938020`
- Classification: **`candidate-ready`**
- Capability gate: `lifecycle.fast-launcher` — **probe outcome: proven**
- Selected mechanism: **M1, `fast_libentry` (`triton.runtime.libentry.LibEntry`)**
- Selected profile: `triton_ascend` (snapshot sha256 `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321`, **unamended**)
- Capability claim: sha256 `a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d`
- Runtime fingerprint: `project.md#runtime-fingerprint` (Ascend910B4, torch 2.7.1+cpu / torch_npu 2.7.1.post4 / triton 3.2.0 / CANN 9.0.0)
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged)

## Classification

`candidate-ready`. The capability probe is **proven**: a cheaper launch path
exists on this runtime, launches the same compiled kernel under the same
configuration, and produces bit-identical output. The round therefore does not
terminate as `capability-miss`. Whether wall time clears 5% is Verifier's
adoption measurement and is not claimed here.

The Coder result taxonomy has no `accepted` or `no-improvement`; those are
Verifier-driven terminal results for the round, selected by the decision's own
table:

| Probe outcome | Wall outcome | Terminal classification | Canonical after |
|---|---|---|---|
| **proven (this result)** | ≥ 5% | `accepted` | `triton_mm_encoder_attention_e2_004.py` |
| **proven (this result)** | < 5% | `no-improvement` | `triton_mm_encoder_attention_e2_003.py` |

Probe evidence is retained on this outcome under `log/probes/`.

## Validation Gate

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 skills/kernel-opt-loop/scripts/validate_decision.py \
  kernels/track1-triton/mm_encoder_attention/ascend/epoch2/rounds/decision_004.md \
  --expected-implementation-profile triton_ascend \
  --project-root kernels/track1-triton/mm_encoder_attention/ascend/epoch2
```

Exit `0`, `"valid":true`, round `004`, scope `host`, family
`launch-path-reduction`, gate `lifecycle.fast-launcher`. Language, backend,
target profile, and device architecture match the selected profile.

## The Capability Probe

Decision-scoped, run before any candidate code. Script:
`log/probes/round_004_launch_abi_probe.py`, machine-readable result
`log/probes/round_004_launch_abi_probe.json`, narrative
`log/probes/round_004_probe_evidence.md`.

The baseline is reproduced **in the same script and regime** using Verifier's
own helper from `log/round_003_host_decomposition.py` (warmup 50, repeat 100,
median of 3 blocks, queue drained before each timed call).

### The four criteria, per mechanism

| Mechanism | C1 exists | C2 same compiled kernel | C3 bit-identical | C4 us/call | C4 vs M0 |
|---|---|---|---|---:|---:|
| M0 proven `kernel[grid](...)` | yes | control | control | **186.255** | — |
| **M1 `fast_libentry`** | **yes** | **yes** (same object, hash match) | **yes, `max_abs_diff = 0.0`** | **164.225** | **−22.030** |
| M2 cached `CompiledKernel` | yes | yes | yes, `0.0` | 66.895 | −119.360 |
| M3 `NPULauncher.launch` C entry | yes | yes | yes, `0.0` | 46.675 | −139.580 |

**Baseline reproduction:** this process's M0 measured `186.255 us/call` against
Verifier's `183.740 us/call` — drift `+1.37%`. The regime matches, so the
comparison is apples-to-apples. Criterion 4 is therefore decided against a
baseline measured here, not against a historical number.

**Stability** across three independent probe runs (us/call):

| Run | M0 | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
| 1 | 181.745 | 160.955 | 62.075 | 42.460 |
| 2 | 176.040 | n/a | 63.955 | 43.565 |
| 3 | 186.255 | 164.225 | 66.895 | 46.675 |

The ordering M0 > M1 > M2 > M3 and the M1 saving of ~21-22 us are stable; M0
itself drifts ~5%, which is exactly why the baseline is reproduced in-process.

### How each mechanism was found

- **M1** — `triton/runtime/libentry.py` defines `LibEntry`, a `KernelInterface`
  owning a per-device kernel cache plus a C++ `libentryC.ArgProcessor`. On a
  cache hit it skips `JITFunction.run` entirely: the binder, the per-call
  specialization-key string construction, `make_backend`, and the
  `used_global_vals` recheck all disappear, and it issues
  `kernel[grid[0:3]](*k_args)`.
- **M2** — `JITFunction.run` returns the `CompiledKernel`, which can be driven
  directly as `kernel[(gx,gy,gz)](*args, stream=stream)`.
- **M3** — `CompiledKernel._init_handles` sets `kernel.run` to an instance of the
  **compiled C++** `ascend.NPULauncher` class (not the Python `NPULauncher` in
  `triton/backends/ascend/driver.py`, which is shadowed). Its `.launch`
  attribute is the generated per-kernel C entry point.

### Selected mechanism, and the magnitude gap

The decision fixes the order and says to stop at the first mechanism satisfying
all four criteria. **All three satisfy all four.** The first in the specified
order, **M1 `fast_libentry`**, is therefore the selected mechanism. It is also
the one the frozen profile names (`implementation_symbol: fast_libentry`), so it
is the canonical answer to the capability question.

**This is a material magnitude decision and the lead should see it explicitly.**
M2 saves `119.360 us/call` and M3 saves `139.580 us/call`, against M1's
`22.030 us/call` — roughly 5.4x and 6.3x more. Against the `297.410 us` wall,
M1's `22 us` is ~7.4% before overheads, while M3's `139 us` is ~47%. Coder does
not reorder a normative mechanism list, so M1 was implemented. If the lead or
Designer wants the larger lever, that needs a new decision authorizing M2 or M3
by name; the probe evidence to support it already exists in `log/probes/`.

## Implementation Summary

The `_fused_attention_kernel` `@triton.jit` definition is **byte-identical** to
`e2_003` (and therefore to `e2_001`) — `diff` over lines 1-76, exit `0`, no
output. Only `ModelNew.__init__` and `ModelNew.forward` changed. No kernel
signature change was needed, so no `mixed`-scope round is required.

Host changes:

1. **Launch handle.** `self._launcher` holds a `LibEntry(_fused_attention_kernel)`
   constructed at runtime, resolving on the first successful launch after
   construction (and after any cache-key change). Using `LibEntry` as a wrapper
   rather than as a `@libentry()` decorator is what keeps the kernel definition
   byte-identical.
2. **Per-call correctness enforcement.** `LibEntry.run` returns
   `(kernel, constexprs)` on every call, so the compiled kernel it selected is
   available for free. `forward` accepts the fast path only when
   `result[0] is self._proven_kernel`, the exact object the proven launch
   produced. A mismatch, or any exception, drops the handle and redoes that very
   call through the proven launch, which fully overwrites `out`.
3. **Cache-key invalidation.** The round-003 output cache key
   `(shape, dtype, device, stride)` is unchanged. On a miss the buffer is
   reallocated **and the handle is cleared**, so that call uses the proven
   launch and the handle is re-proven for the new key.
4. **Launch keyword bundle hoisted to `__init__`.** Every launch argument other
   than q/k/v/out/strides/S is a constructor constant, so the kwargs dict is
   built once instead of per call. This recovered ~7 us of the lever (see the
   attempt ledger).

Exactly one kernel is launched per call on every path. The first call uses the
proven path; the verification that makes the handle trustworthy is the
zero-cost kernel-identity check, not a second launch.

### Fallback behaviour

| Event | Behaviour |
|---|---|
| handle not yet resolved | proven `_fused_attention_kernel[grid](...)` launch, then resolve |
| cache-key component changes (shape / dtype / device / stride) | buffer reallocated, handle cleared, **proven launch for that call**, handle re-proven for the new key |
| fast path raises | exception swallowed, handle cleared and disabled, proven launch in the **same** call |
| fast path selects a different compiled kernel | handle cleared and disabled, proven launch in the **same** call |
| `LibEntry` import or construction fails | `_launcher_disabled` set, proven launch for the remainder of the instance lifetime |

`_launcher_disabled` is deliberately sticky: a structural failure will not
recur-fix itself, and the worst case at runtime is always the accepted round-003
behaviour, never a wrong answer.

## Sketch Conformance

| Sketch requirement | Status |
|---|---|
| `target=triton_ascend` (required) | honored |
| `BLOCK_M=128`, `BLOCK_N=128`, `HEAD_DIM=64` (required) | honored, unchanged |
| `accumulator_dtype=fp32` (required) | honored, unchanged |
| `num_warps=4` (preferred) | honored, unchanged |
| `num_stages=1` (preferred) | honored, unchanged |
| `op_alloc_out` | round-003 instance cache retained unchanged |
| `op_load_q/k/v`, `op_qk`, `op_mask_and_softmax`, `op_pv`, `op_normalize`, `op_store_out` | kernel body byte-identical |
| `c_parallel_bh`, `c_guard_row`, `c_guard_col` | byte-identical |
| `scope.entrypoints = [ModelNew.forward, ModelNew.__init__]` | only those two methods touched |
| `scope.unchanged_boundary` | see below |

### Unchanged computation boundary (verified)

| Boundary element | Evidence |
|---|---|
| `_fused_attention_kernel` `triton.jit` definition | `diff` lines 1-76 vs `e2_003` → exit `0`, no output; also identical to `e2_001` |
| kernel launch count stays at one | instrumented count: **1.00** for both `e2_003` and `e2_004` |
| `BLOCK_M` / `BLOCK_N` / `HEAD_DIM` | `128` / `128` / `64` |
| accumulator dtype | fp32 |
| `num_warps` / `num_stages` | `4` / `1`, and confirmed from `CompiledKernel.metadata` (`num_warps=4`, `num_stages=1`) |
| device kernel name | `_fused_attention_kernel`, the same `CompiledKernel` object (hash `18db9f0320830a397f740d02078551aeea898355fd7e06d59bb3a7bca2e1c903`) |
| output shape / dtype / device / contiguity | `(2, 83, 512)`, `torch.float16`, `npu:0`, `is_contiguous() == True` |
| tolerance `atol=1e-2`, `rtol=1e-2` | smoke `PASS`; bit-identical (`max_abs_diff = 0.0`) |

## Host Plan Conformance

| Host Plan field | Implementation | Verdict |
|---|---|---|
| `state_owner` | ordinary instance attributes (`_launcher`, `_launch_key`, `_proven_kernel`); `state_dict()` is `[]` | conformant |
| `lifetime` | resolved on the first successful launch after construction, reused until a cache-key component changes | conformant |
| `allocation_reuse` | round-003 output buffer cache retained unchanged; no per-call allocation added | conformant |
| `cache_key` | kernel specialization (enforced by kernel identity), grid, `BLOCK_M`/`BLOCK_N`/`HEAD_DIM`, `num_warps`/`num_stages` (constructor constants), plus the round-003 shape/dtype/device/stride key | conformant |
| `invalidation` | any key change discards the handle and uses the proven launch; a failed, unproven, or mismatched resolution is a miss | conformant |
| `concurrency` | no lock, thread-local, or per-call state introduced | conformant |
| `device_stream_behavior` | the handle resolves the stream internally on every call, exactly as `CompiledKernel.__getitem__` does; no stream created, captured, or switched | conformant |
| `unchanged_behavior` | all eight items preserved and observed | conformant |

### Observed host invariants

From `log/probes/round_004_candidate_conformance.py` (conformance facts, not
adoption timing):

```text
e2_004 launcher installed          : True
e2_004 launcher disabled           : False
e2_004 proven kernel set           : True
bit-identical to e2_003            : True (max_abs_diff=0.0)
alias of query/key/value           : False
state_dict                         : []
cache hit reuses buffer            : True
poisoned buffer leaks NaN          : False
output still bit-equal             : True
kernel launches per call e2_003    : 1.00
kernel launches per call e2_004    : 1.00
after stride change, handle re-proven : True
back on original key, bit-equal    : True
```

Launch counting deserves a note. The obvious interception point,
`triton.backends.ascend.driver.NPULauncher`, is **shadowed**: the class actually
in use is the compiled C++ `ascend.NPULauncher`, reachable through
`triton.runtime.driver.active.launcher_cls`. Patching that class gives an exact
device-launch count, and it is `1.00` for both candidates. An earlier version of
this probe patched the shadowed Python class and reported `0.00` — a silent
false negative that would have looked like "no launches at all".

## Deviations

None. Three conformance notes, all preserving normative semantics:

1. **`LibEntry` used as a runtime wrapper, not a decorator.** The decision
   requires the kernel definition to stay byte-identical; applying `@libentry()`
   would have changed it. Constructing `LibEntry(jit_fn)` at runtime is
   semantically the same launcher over the same JITFunction.
2. **Launch keyword bundle hoisted to `__init__`.** Consequence: a
   post-construction mutation of `scale`, `num_heads`, or `head_size` would not
   affect the launch. This is the same trade-off round 003 recorded for
   `_num_heads` / `_head_dim` / `_block`; nothing in the harness mutates them.
3. **Redundant per-call key comparison dropped.** `_launcher is not None` already
   implies the handle was proven for the current key, because the handle is
   cleared on every cache miss. Keeping the comparison cost ~2-3 us/call for no
   behavioural difference.

## Compile and Smoke Evidence

- Command:
  ```bash
  cd /workspace/kernelswift-dev-4ff2094
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_004.py \
    --warmup 5 --repeat 10 --full-traceback
  ```
- Exit status: `0`
- Observation: `PASS accuracy; v0=0.367450 ms, v1=0.312850 ms, speedup=1.175x`
  followed by `Summary: 1 passed, 0 failed, 1 total.`
- **Smoke-tier number only.** It is not the adoption measurement. Verifier owns
  the 3-pair warmup-50/repeat-100 protocol. Coder did not run
  `--warmup 50 --repeat 100` and did not run `--profile`.
- Corroborating: output bit-identical to `e2_003` (`max_abs_diff = 0.0`), so the
  launch-path change is numerically inert.

### Local gate (Coder contract step 6)

| Gate | Result |
|---|---|
| `validate_decision.py --expected-implementation-profile triton_ascend` | exit `0`, `"valid":true` |
| kernel definition vs `e2_003`, `diff` lines 1-76 | exit `0`, byte-identical |
| kernel definition vs `e2_001`, `diff` lines 1-74 | exit `0`, byte-identical |
| `ast.parse` | ok; `_fused_attention_kernel`, `ModelNew`, `get_inputs`, `get_init_inputs` |
| real harness AST loader | ok |
| `read_lints` | `totalCount: 0` |
| forbidden constructs (`import triton_ascend`, `make_block_ptr`, `async_copy`, `vectorize`, `"cuda"`) | none |
| warm-up / compile smoke | `PASS accuracy`, exit `0` |

## Attempt Ledger

| Attempt | Command | Exit | Defect | Candidate before | Candidate after |
|---:|---|---:|---|---|---|
| 1 | `validate_decision.py`; write candidate; kernel `diff`; `ast.parse`; smoke | 0 | none | `not-applicable` | (interim) |
| 2 | probe + conformance re-run | 0 | **design defect found by probe**: the verification scheme issued two launches on the first call, which would have broken `kernel_count_per_call = 1.00` | (interim) | (interim) |
| 3 | conformance re-run | 0 | **probe defect, not candidate**: launch counter patched the shadowed Python `NPULauncher` and read `0.00`; `restored_bitequal` compared against a live cached buffer that later calls had overwritten | (interim) | (interim) |
| 4 | conformance re-run | 0 | **candidate performance defect**: forward-level lever only `-11.715 us` vs the `-22.030 us` bare-launch saving; ~10 us was added per-call host work | (interim) | `f5aa1d709e4deeb1562757d795dd4da41217238dd15c53d33c2c338da1938020` |

Attempt 1 also included one in-flight refinement (switching from a
launch-into-scratch verification to a zero-cost kernel-identity check) applied
before the first gate run.

The final candidate's forward-level lever after attempt 4:

```text
e2_003 forward alone : 223.505 us/call
e2_004 forward alone : 205.035 us/call
lever                : -18.470 us/call  (-8.26%)
```

measured in one process, warmup 50 / repeat 100, median of 3 blocks.

## Risks Noted for Verifier

1. **Adoption is not yet established.** The probe proves legality, not adoption.
   The `-18.470 us/call` forward-level lever on a `297.410 us` wall is ~6.2%,
   and the 5% threshold needs `14.871 us`. It clears on paper with ~3.6 us of
   margin, which is thin against a machine that drifted ~5% within a single
   turn in round 003. This is the dominant risk.
2. **M2 and M3 are 5-6x larger and were not used.** `119.360` and `139.580 us`
   respectively. If this round lands as `no-improvement` on margin, the cheapest
   next step is a decision naming M2 or M3, not a new probe.
3. **`kernel_count_per_call` must be confirmed by the profiler, not assumed.**
   The instrumented count is `1.00` for both candidates, but it counts launches
   through the active launcher class in a plain Python process. If Verifier
   profiles with warmup 0, the first call is still a single launch (the handle
   installation performs no launch), so no special case should appear.
4. **The identity check is the whole safety net.** The fast path is trusted only
   when `result[0] is self._proven_kernel`. If `LibEntry` ever returned a
   differently-specialized kernel, the candidate silently uses the proven path
   and the fast path is disabled for the instance — visible as
   `model._launcher_disabled == True` with `model._launcher is None`. Worth
   asserting that both are `False`/`not None` after a steady-state run.
5. **The handle is per-instance and per-key.** It is resolved on the first
   successful launch, so the very first timed call in any fresh process pays the
   proven path. With warmup ≥ 1 this is invisible in the measurement; with
   warmup 0 it would not be.
6. **`launch_metadata` is not passed by M3** (it is not used here). If a future
   round adopts M3 and Verifier profiles with hooks enabled, the profiler's
   `record_function` metadata could change. Not applicable to M1.
7. **Carried forward unchanged from round 003:** the kernel requires `S <= 128`
   (campaign shape is `S=83`); the second `tl.dot` tile `(128,64,128)` compiles
   and is numerically correct but was not one of the eleven probed tiles; the
   buffer-reuse invariant depends on full store coverage and must be revisited
   if a later round introduces a masked or partial store.

## Profile and Campaign Integrity

- The frozen snapshot is **not amended**. `lifecycle.fast-launcher` remains
  `Unknown` and hash-pinned at
  `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321`.
- No `run_profile_probe.py` or `validate_probe.py` was run; this is a
  Decision-scoped probe, not a profile qualification.
- `state/project_capability_claim.json` is untouched; no
  `qualification_disposition` was created.
- This evidence is round-local and does not license any later round.
- Files written by Coder this round: the candidate, this result,
  `state/coder_context.md`, and four files under `log/probes/`. Nothing else.

## Probe Evidence Paths

- `log/probes/round_004_launch_abi_probe.py` — the capability probe (four criteria, baseline reproduction)
- `log/probes/round_004_launch_abi_probe.json` — machine-readable result
- `log/probes/round_004_candidate_conformance.py` — bit-identity, fallback, exact launch count, lever
- `log/probes/round_004_candidate_conformance.json` — machine-readable result
- `log/probes/round_004_probe_evidence.md` — narrative summary and mechanism notes
