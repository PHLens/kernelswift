# Triton Profiler Exploration on CoreX BI150

## Scope and evidence

This note records what is actually available in the matched BI150 environment rather than assuming that upstream Triton profiler support is usable on CoreX.

The installed snapshot is:

- CoreX `4.4.0`;
- Triton `3.1.0`;
- device target reported through Triton as `GPUTarget(backend='cuda', arch=71, warp_size=64)`;
- profiler package snapshot collected while the BI150 server was still available, ahead of its expected reclamation, and archived under [`evidence-archive/triton-profiler-package/`](evidence-archive/triton-profiler-package/);
- Route C evidence and conclusions frozen in `evidence-summary.json`, `assessment.md`, `evidence-summary-v2.json`, and `assessment-v2.md`.

The snapshot contains the installed Python sources for `triton.profiler`, the command-entry-point locations, direct import/CLI probe output, and captured distribution metadata. No additional profiler or timing experiment is inferred beyond those artifacts. The profiler Python files are unmodified third-party distribution evidence rather than KernelSwift-authored or maintained code; see [`evidence-archive/THIRD_PARTY_NOTICE.md`](evidence-archive/THIRD_PARTY_NOTICE.md) and [`evidence-archive/triton-profiler-package/distribution-license-info.txt`](evidence-archive/triton-profiler-package/distribution-license-info.txt).

## Executive conclusion

The installed CoreX distribution contains the **Python-facing Triton profiler API and command entry points**, but it does not contain the native `triton._C.libproton` extension required to use them. Consequently, neither `import triton.profiler` nor `proton --help` is functional in this exact environment.

Even if the missing native extension were supplied, the installed implementation is not a BI-specific intra-kernel profiler:

- automatic backend selection treats every Triton target named `cuda` as CUPTI;
- the Triton hook wraps `CompiledKernel` launch enter/exit boundaries;
- the hook attaches only kernel name plus `flops8/16/32/64` and `bytes` metadata when present;
- the viewer's peak-compute tables cover selected NVIDIA CUDA and AMD HIP architectures, not BI arch `71`;
- no installed path provides an optimization-stable, completion-dependent timestamp inside a Triton kernel.

For KernelSwift, Proton should therefore be treated as an **optional launch-level layer with a strict capability preflight**, not as a substitute for whole-kernel benchmark timing, compiler evidence, or a qualified intra-kernel region measurement.

## 1. Installed profiler surface

### 1.1 Python API exists

The installed [`evidence-archive/triton-profiler-package/__init__.py`](evidence-archive/triton-profiler-package/__init__.py) exports:

- `scope`, `enter_scope`, and `exit_scope`;
- `start`, `activate`, `deactivate`, and `finalize`;
- the `profile` decorator;
- `DEFAULT_PROFILE_NAME`.

The corresponding Python modules are present:

```text
evidence-archive/triton-profiler-package/__init__.py
evidence-archive/triton-profiler-package/flags.py
evidence-archive/triton-profiler-package/hook.py
evidence-archive/triton-profiler-package/profile.py
evidence-archive/triton-profiler-package/proton.py
evidence-archive/triton-profiler-package/scope.py
evidence-archive/triton-profiler-package/viewer.py
```

This proves that the API surface was packaged. It does **not** prove that a profiling session can start.

Snapshot references:

- [`evidence-archive/triton-profiler-package/package-info.txt`](evidence-archive/triton-profiler-package/package-info.txt);
- [`evidence-archive/triton-profiler-package/__init__.py`](evidence-archive/triton-profiler-package/__init__.py);
- [`evidence-archive/triton-profiler-package/profile.py`](evidence-archive/triton-profiler-package/profile.py);
- [`evidence-archive/triton-profiler-package/scope.py`](evidence-archive/triton-profiler-package/scope.py).

### 1.2 Command entry points exist

The CoreX installation contains:

```text
proton        /usr/local/corex-4.4.0/lib64/python3/dist-packages/bin/proton
proton-viewer /usr/local/corex-4.4.0/lib64/python3/dist-packages/bin/proton-viewer
```

There is no separate `triton-profiler` executable in the captured installation.

The `proton` Python entry point supports invocation forms such as:

```text
proton [options] script.py ...
proton [options] pytest ...
python -m triton.profiler.proton [options] script.py ...
```

Its CLI exposes `cupti` as the only explicit backend choice, plus `shadow`/`python` contexts, tree data, and an optional `triton` hook.

Snapshot references:

- [`evidence-archive/triton-profiler-package/native-extension-info.txt`](evidence-archive/triton-profiler-package/native-extension-info.txt);
- [`evidence-archive/triton-profiler-package/proton.py`](evidence-archive/triton-profiler-package/proton.py).

## 2. The installed profiler is not runnable

### 2.1 Native Proton extension is absent

The captured package inspection reports:

```text
libproton_exists False
profiler_import failed ModuleNotFoundError No module named 'triton._C.libproton'
```

Both [`evidence-archive/triton-profiler-package/profile.py`](evidence-archive/triton-profiler-package/profile.py) and [`evidence-archive/triton-profiler-package/scope.py`](evidence-archive/triton-profiler-package/scope.py) import the native extension through:

```python
from triton._C.libproton import proton as libproton
```

Because `triton.profiler.__init__` imports `scope` and `profile` eagerly, the missing extension prevents the top-level Python API from importing.

This is a packaging/runtime capability failure, not evidence that BI hardware cannot be profiled.

### 2.2 `proton --help` also fails

The captured CLI probe exited with return code `1` before argument parsing. Its traceback ends with:

```text
ModuleNotFoundError: No module named 'triton._C.libproton'
```

Therefore the existence of the `proton` script is not a usable-capability signal. A capability check must import the native extension or execute a harmless CLI/API probe; checking only files or executable names produces a false positive.

The snapshot proves that the `proton-viewer` entry point exists. It was not independently validated as runnable, and its import chain reaches profiler modules that depend on `libproton`; it must not be described as usable from entry-point presence alone.

Snapshot references:

- [`evidence-archive/triton-profiler-package/package-info.txt`](evidence-archive/triton-profiler-package/package-info.txt);
- [`evidence-archive/triton-profiler-package/proton-cli-probe.txt`](evidence-archive/triton-profiler-package/proton-cli-probe.txt).

## 3. Backend selection is CUDA-name-based, not BI-aware

The installed automatic selector is:

```python
def _select_backend() -> str:
    backend = triton.runtime.driver.active.get_current_target().backend
    if backend == "cuda":
        return "cupti"
    elif backend == "hip":
        return "roctracer"
    else:
        raise ValueError("No backend is available for the current target.")
```

The BI150 CoreX target reports `backend='cuda'`, so this function selects `cupti` solely from the compatibility backend name. It does not inspect:

- the Iluvatar vendor/device identity;
- target triple `bi-iluvatar-ilurt`;
- target CPU `ivcore11`;
- architecture `71` as a BI capability;
- availability of a CoreX trace backend;
- whether CUPTI can observe this device.

This mapping is reasonable for upstream NVIDIA Triton, but it is not a BI-specific backend implementation or capability probe. KernelSwift must not interpret `backend == "cuda"` as proof that CUPTI/Proton is available on a CUDA-compatible vendor stack.

Snapshot reference: [`evidence-archive/triton-profiler-package/profile.py`](evidence-archive/triton-profiler-package/profile.py), lines 12–19.

## 4. What the Triton hook measures

The installed `TritonHook` registers itself through:

```python
CompiledKernel.launch_enter_hook = TritonHook.enter
CompiledKernel.launch_exit_hook = TritonHook.exit
```

At launch entry it:

1. obtains the launch metadata `LazyDict`;
2. extracts the kernel name;
3. copies any available `flops8`, `flops16`, `flops32`, `flops64`, and `bytes` fields;
4. enters a Proton scope marked as a Triton operation.

At launch exit it closes that scope.

This is a **CompiledKernel launch-boundary hook**. It can organize launch-level profiling data and attach static work metadata, but it does not:

- insert timestamps around QK, softmax, PV, or any other intra-kernel region;
- observe individual Triton programs, warps, or lanes;
- prove instruction completion at a source boundary;
- derive region attribution from TTIR, TTGIR, LLIR, or final ISA;
- turn `flops`/`bytes` metadata into an observed hardware duration by itself.

The actual launch duration and trace data would have to come from a working native Proton backend such as CUPTI. The hook only defines launch scopes and metadata.

Snapshot reference: [`evidence-archive/triton-profiler-package/hook.py`](evidence-archive/triton-profiler-package/hook.py).

## 5. Viewer assumptions do not cover BI150

The installed viewer derives compute-bound minimum time from hard-coded peak tables.

Recognized CUDA architectures are:

- `80`;
- `89`;
- `90`.

Recognized HIP architectures are:

- `gfx90a`;
- `gfx941`;
- `gfx942`.

There is no peak-compute entry for BI arch `71`, `ivcore11`, or another Iluvatar identifier. A BI device reported under the generic `CUDA` device type therefore has no valid BI peak-compute model in this viewer. Derived FLOP utilization or roofline-like minimum time would be undefined or misleading unless BI-specific device metadata and peak tables were added and validated.

The bytes-based path derives bandwidth from reported memory clock and bus width and is less architecture-table-specific, but it still depends on a working profiler database with correct BI device information.

Snapshot reference: [`evidence-archive/triton-profiler-package/viewer.py`](evidence-archive/triton-profiler-package/viewer.py), lines 32–75.

## 6. Four measurement layers must remain distinct

| Layer | Available on this snapshot | What it can support | What it must not be called |
|---|---|---|---|
| Whole-kernel timing | Yes | CUDA Event device time, host wall time, graph replay timing, official benchmark comparison | Intra-kernel region duration |
| Launch-level Proton | No: Python/CLI surface exists, native `libproton` is absent | If restored and validated: launch tree/scopes plus backend-observed launch data and Triton flops/bytes metadata | BI-specific profiler support merely because target backend is named `cuda`; QK/softmax/PV timing |
| Compiler evidence | Yes | TTIR/TTGIR/linked LLIR/cubin capture, registers, spills, shared memory, compile identity, external-bitcode linkage | Observed runtime duration or final executed ISA semantics |
| Intra-kernel issue windows | Not qualified | Potential diagnostic selected-program windows only after ordering and dependency qualification | Official timing, execution duration, per-warp timing, or profiler authority |

### 6.1 Whole-kernel timing

The existing KernelSwift and Route C paths can use CUDA Events for device-side whole-kernel timing and can separately record host or graph timing. These are the authoritative performance comparisons for accepted candidate evaluation when run through the approved harness.

They reveal whether a candidate is faster, but they generally do not explain which intra-kernel region changed.

### 6.2 Launch-level Proton

A working Proton backend would add launch-level call-tree and scope context. The Triton hook could label each `CompiledKernel` launch and attach static operation/byte counts. This could be valuable for multi-kernel models, launch overhead, graph structure, and attributing time among separate kernels.

It would still not split one fused attention kernel into QK, softmax, and PV regions.

### 6.3 Compiler evidence

The CoreX Triton stack successfully exposes useful compilation evidence:

- `compiled.asm` artifacts including TTGIR, linked LLIR, assembly/binary keys, and cubin payloads where available;
- resource metadata such as registers, spills, and shared memory after kernel loading;
- content-hashed external LLVM bitcode through `extern_libs`;
- target information including warp size `64`, arch `71`, `ivcore11`, and triple `bi-iluvatar-ilurt` across the validated compiler path.

Route C proved that CoreX CUDA C `clock64()` bitcode can be linked into Triton and materialized as `llvm.nvvm.read.ptx.sreg.clock64()` in linked LLIR without adding a lowering.

Compiler evidence is valuable for mechanism hypotheses, cache identity, resource regressions, and verifying whether instrumentation survived optimization. It remains static evidence, not runtime timing.

### 6.4 Intra-kernel issue windows

Route C attempted selected-program timestamps around synthetic and attention regions. The required semantics were deliberately weaker than execution duration: even a qualified result would have been labeled `measurement_semantics: issue-window` because final ISA ordering was unavailable.

The final experiment did not reach that qualification. The linked-LLIR audit showed that LLVM placed the two clock reads adjacently before the synthetic dependency chain. No attention cycle record was produced.

## 7. Route C ordering and runtime lessons

### 7.1 Clock availability was not the blocker

CoreX CUDA C `clock64()` works on BI150 and lowers through `llvm.nvvm.read.ptx.sreg.clock64()`. External LLVM bitcode linkage also works. The failed PTX-style Triton inline assembly did not prove primitive absence; it proved only that the guessed PTX spelling was rejected by the BI assembler.

### 7.2 A start/end pair needs a real completion dependency

Two side-effecting no-argument clock reads do not necessarily bracket intervening dataflow. In the external-clock experiment, LLVM was free to place the end read before the dependency chain when its value had no completion dependency on the chain result.

The final token-controlled source helper was also insufficient. LLVM collapsed its equivalent branches to `clock64() + (token & 1)` and emitted the start and end intrinsics adjacently before the 16-step chain. The compile-only audit therefore stopped with:

```text
end-dependency-optimized-away
```

A source-level wrapper around a no-argument clock primitive is not an optimization-stable region marker.

### 7.3 Runtime history must be attributed carefully

A retained `noinline` helper linked successfully but hung on the first BI150 kernel execution while the host waited in `torch.cuda.synchronize()`. The process was terminated after approximately seven minutes.

After termination, the container-visible GPU context remained unusable: even a minimal CUDA allocation stalled, and a container-local `ixsmi --gpu-reset` could not reset the device because the owning host PIDs were outside the container PID namespace.

Consequently, the later empty-inline-assembly smoke stall is **not** independent evidence that empty inline assembly or the CoreX backend itself caused that stall. The terminal measurement conclusion comes instead from the independent compile-only LLIR audit, which required no CUDA tensor allocation and directly showed dependency elimination.

### 7.4 Final ISA remains unavailable

The validated tool path could retain TTGIR, LLIR, cubin payloads, and resource metadata, but it could not obtain trustworthy final BI ISA for direct Triton cubins:

- upstream LLVM objdump had no BI disassembler for the direct cubin target;
- `ixobjdump` expected a compatible fatbin/ELF shape that the direct Triton cubin did not provide;
- wrapping attempts did not produce a valid input for the installed disassembler.

Without final ISA or documented target ordering semantics, cycle observations must not be promoted from qualified issue windows to execution-duration claims.

## 8. Implications for KernelSwift

### 8.1 Do not make one profiler path mandatory

KernelSwift spans vendor stacks with different levels of profiler, compiler, and timing support. A mandatory Proton dependency would fail on this exact CoreX snapshot before any kernel runs. A mandatory intra-kernel clock path would also fail qualification even though whole-kernel optimization remains productive.

The optimization loop should continue to work with whole-kernel timing and correctness alone. Additional profiler layers should enrich evidence when available rather than gate the campaign.

### 8.2 Use a layered optional-profiler architecture

A practical architecture has four independent capability layers:

1. **Whole-kernel timer — required**
   - approved benchmark harness;
   - CUDA Events or vendor-equivalent device timing;
   - host/graph timing kept as distinct fields;
   - correctness and accepted-source authority unchanged.

2. **Launch profiler — optional**
   - Proton/CUPTI, ROCTracer, or vendor profiler only after capability validation;
   - separate backend adapter per validated vendor/runtime;
   - launch-level semantics explicitly labeled;
   - no inference of fused-kernel subregions.

3. **Compiler evidence collector — optional but broadly useful**
   - compile identity and target;
   - TTIR/TTGIR/LLIR/binary artifacts when available;
   - registers, spills, shared memory, and external-library hashes;
   - artifact availability represented explicitly rather than treated as failure.

4. **Intra-kernel region probe — experimental**
   - enabled only after synthetic dependency/order qualification;
   - records the selected program/region and perturbation controls;
   - emits `issue-window` or another exact semantics label;
   - never replaces official whole-kernel timing.

Each layer should be independently optional except the approved whole-kernel timer.

### 8.3 Require strict capability preflight

A Proton preflight for a CoreX-like environment should verify all of the following before enabling the adapter:

- `import triton.profiler` succeeds;
- `import triton._C.libproton` succeeds;
- `proton --help` or a no-work API session succeeds;
- the selected native backend is present and can observe the actual vendor device;
- backend selection is based on validated vendor capability, not only the string `cuda`;
- a one-kernel smoke profile produces a readable output database;
- `proton-viewer` can read that database;
- BI device metadata and peak tables are supported before derived utilization metrics are enabled;
- profiler overhead and timing direction are recorded.

If any check fails, the launch-profiler layer should report `unavailable` with a concrete reason and allow the optimization run to continue.

### 8.4 Preserve semantics labels and authority boundaries

Recommended labels include:

- `whole-kernel-device-time`;
- `host-wall-time`;
- `graph-replay-time`;
- `launch-profile-time`;
- `compiler-static-evidence`;
- `selected-program-issue-window`.

These values should not be merged into a generic `time` field. Every report should identify:

- measurement layer;
- backend and device;
- program/kernel/region ownership;
- whether the result is observed or derived;
- perturbation and availability status;
- whether it participates in official acceptance authority.

### 8.5 Prioritize a compiler/vendor token-dependent timestamp hook

The highest-value future primitive is not another spelling of a no-argument clock read. It is a compiler- or vendor-supported marker that accepts a dependency token, or otherwise specifies ordering after the measured dataflow and survives optimization.

Preferred order of investment:

1. documented CoreX/vendor profiler or timestamp API with completion semantics;
2. compiler-supported Triton/CoreX hook or intrinsic that consumes a region-derived token;
3. documented final-ISA and ordering support sufficient to qualify a lower-level marker;
4. only after demonstrated optimization value, consideration of a maintained backend integration.

Do not prioritize:

- guessed BI assembly copied from pseudo-disassembly;
- generated-LLIR surgery;
- relying on `is_pure=False` alone as a completion barrier;
- source wrappers around a no-argument clock primitive;
- a maintained Triton lowering or Proton/Iluvatar backend before a bounded experiment demonstrates incremental KernelSwift value.

## 9. Recommendation

For the current CoreX 4.4.0/Triton 3.1.0 snapshot:

- mark Proton launch profiling as `unavailable: native-libproton-missing`;
- retain whole-kernel device timing as the required performance signal;
- retain compiler artifacts/resources as optional mechanism evidence;
- keep intra-kernel cycle instrumentation disabled because no optimization-stable completion-dependent end timestamp was qualified;
- preserve the Route C result as `inconclusive`, not `unsupported`;
- revisit intra-kernel profiling only when CoreX supplies a documented token-dependent marker, ordering primitive, or validated vendor profiler path.

This layered approach gives KernelSwift useful evidence where the platform supports it without making vendor-specific profiler gaps block kernel optimization or weakening measurement semantics.
