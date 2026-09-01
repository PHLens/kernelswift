# BI150 Route C External-Clock Design

**Status:** approved addendum to the Route C design, including the final token-control fallback
**Scope:** rerun the disposable BI150 value probe using the confirmed CoreX `clock64()` compiler path, without adding or modifying Triton lowering
**Supersedes:** only the failed PTX-style inline-assembly clock source in `2026-08-30-bi150-kperfir-value-probe-design.md`

## 1. Objective

Determine whether selected-program cycle observations from the accepted BI150 `mm_encoder_attention` lineage add actionable information beyond existing whole-kernel, host, graph, resource, and source evidence.

The experiment must first obtain a clock inside the same Triton kernel. It will do so by linking a tiny CoreX device bitcode helper that calls the already validated CUDA C `clock64()` builtin. It will not add a Triton op, modify the Iluvatar backend, patch generated LLIR, or modify accepted candidates.

## 2. Established facts

The matched CoreX 4.4.0 environment establishes all of the following:

- CUDA C `clock64()` compiles and runs on BI150 with target `ivcore11`.
- CoreX Clang lowers `clock64()` to `llvm.nvvm.read.ptx.sreg.clock64()` in module triple `bi-iluvatar-ilurt`.
- CoreX Triton 3.1.0 exposes `triton.language.core.extern_elementwise`.
- The Iluvatar backend accepts `extern_libs`, hashes their contents, and links each path into the generated LLVM module before optimization.
- The PTX spelling `mov.u64 $0, %clock64;` is rejected by the BI target assembler.
- Public BI pseudo-assembly documents `TIME`, but does not document source-level inline-assembly operands or clock ordering semantics.
- Final disassembly of direct Triton cubins remains unavailable with the currently validated tool path.
- A first external-clock implementation retained `noinline` helper calls. It linked successfully but hung on the first BI150 kernel execution while the host waited in `torch.cuda.synchronize()`; the process was terminated after approximately seven minutes.
- After terminating the noinline run, the container-visible GPU context remained unusable: a minimal `torch.ones(..., device="cuda")` allocation stalled, and container-local `ixsmi --gpu-reset` could not reset the device because the owning host PIDs were outside the container PID namespace. Consequently, the subsequent empty-inline-assembly variant did not establish a separate backend or runtime result.
- The final token-controlled helper did produce compile-only linked LLIR. LLVM optimization collapsed both branches to `clock64() + (token & 1)`, and the two kernel clock intrinsics became adjacent before the unrelated writer branch, eliminating the intended completion dependency.

Therefore external bitcode linkage remains a bounded source-level experiment using a supported compiler primitive, but retained runtime device calls are excluded. The token-dependent control flow below is the final attempted dependency mechanism. Its linked-LLIR optimization is terminal for this rerun: a compile-only prelaunch audit records `end-dependency-optimized-away`, and no additional barrier, manual BI assembly, generated-LLIR surgery, or compiler-lowering work is attempted.

## 3. Architecture

### 3.1 Device helper

A disposable helper source defines two externally visible device functions:

```cpp
extern "C" __device__ __attribute__((always_inline, used))
unsigned long long corex_clock64_start() {
    return clock64();
}

extern "C" __device__ __attribute__((always_inline, used))
unsigned long long corex_clock64_after_u64(unsigned long long token) {
    if (token & 1ULL) {
        return clock64() + 1ULL;
    }
    return clock64();
}
```

The start helper supplies the first timestamp. The end helper creates a token-derived conditional branch and reads `clock64()` only in the branch successors. The true arm returns `clock64() + 1`; the false arm returns `clock64()`. The caller decodes either result by subtracting `token & 1`.

`always_inline` is required because retained `noinline` device calls hung at first execution on the matched BI150 stack. This final helper contains no inline assembly; the empty-assembly runtime variant was not independently qualified after that hang left the container-visible GPU context unusable. Linked LLIR must contain the start clock, token-derived conditional branch, and one or more reachable end clocks directly in the diagnostic function, with no runtime calls to either helper symbol. The empty-work control would measure the inline clock/control floor if the static dependency audit passed.

CoreX Clang compiles the helpers as device-only LLVM bitcode for `ivcore11`. The build records:

- compiler path and version;
- command line;
- input source SHA256;
- output bitcode SHA256;
- target triple;
- presence of symbols `corex_clock64_start` and `corex_clock64_after_u64`;
- `alwaysinline` on both definitions;
- no inline assembly in the end helper;
- a token-derived conditional branch;
- clock reads in both branch arms after the branch;
- true-arm encoding as `clock64() + 1`;
- presence of `llvm.nvvm.read.ptx.sreg.clock64()` in both helpers.

The helper is diagnostic-only and is rebuilt on the matched host. No generated binary is treated as portable across CoreX versions.

### 3.2 Triton external calls

A custom Python `@core.extern` wrapper outside the `triton` package is rejected by the installed JIT dependency scanner. The diagnostic `@triton.jit` function therefore calls the installed builtin `core.extern_elementwise` directly.

The start call uses an empty signature:

```python
start = core.extern_elementwise(
    "",
    "",
    [],
    {(): ("corex_clock64_start", core.dtype("int64"))},
    is_pure=False,
)
```

The end call consumes a scalar unsigned token derived from the measured region:

```python
encoded_end = core.extern_elementwise(
    "",
    "",
    [token],
    {(core.dtype("uint64"),):
        ("corex_clock64_after_u64", core.dtype("uint64"))},
    is_pure=False,
)
end = encoded_end - (token & 1)
```

The kernel launch supplies:

```python
extern_libs={"corex_clock": clock_bitcode_path}
```

`is_pure=False` prevents front-end common-subexpression elimination from treating clock reads as interchangeable, but the matched CoreX LLVM optimizer still proved both token-control arms equivalent and moved the end read ahead of the branch. Backend linking must resolve and inline both helper symbols. A compile-only audit therefore inspects linked LLIR before launch; runtime qualification is permitted only if the diagnostic function contains a start clock intrinsic followed by a token-derived conditional branch and one or more reachable end clock intrinsics, with no inline assembly and no retained runtime calls to either helper symbol.

### 3.3 Timestamp retention

This rerun is scoped to one pair per selected Triton program:

```text
[generation, start, end]
```

The external function returns a scalar value. Supplying a tensor-shaped dummy argument would create lane-shaped elementwise calls, not exactly one independently owned call per local warp, and would add unacceptable clock-call volume. Therefore this experiment makes no warp-0/warp-1 imbalance claim.

The pair stays live until one selected writer flushes it after the measured region. Accepted-source arithmetic and official outputs remain unchanged. Raw values are decoded with unsigned 64-bit subtraction.

### 3.4 Completion token

Every interpreted end marker consumes a scalar token produced after the measured work. The synthetic probe uses the final dependency-chain value. Real-kernel variants reuse an existing scalar when one exists; otherwise they add one bounded scalar reduction, such as a reduction of the region result, and compile a matched dependency-control variant that performs the same reduction without clock calls.

The result records the token expression and its control overhead. A region is not interpreted when the token machinery materially changes correctness, spills, resource class, or whole-kernel direction. The empty clock/control and dependency-control costs are reported as perturbation evidence; they are not blindly subtracted from raw issue windows.

### 3.5 Semantics

Without validated final target ordering, every successful observation is reported as:

```text
measurement_semantics: issue-window
```

The experiment may compare selected-program issue windows, distributions, and directionality. It must not convert them into authoritative region execution duration, warp imbalance, or official benchmark timing.

## 4. Execution stages

### Stage A: external-clock qualification

Before allocating CUDA tensors, the worker constructs a Triton `ASTSource` for the existing synthetic JIT kernel with manual `*i64/*i64/*i64/i32` argument types, `chain_iters=16`, `GPUTarget("cuda", 71, 64)`, one warp, and the helper bitcode in `extern_libs`. It compiles only, persists linked LLIR, hashes the artifact, and runs the same dependency inspection used by runtime qualification.

The prelaunch audit must establish:

1. external bitcode links successfully;
2. emitted linked LLIR contains a start clock and at least one reachable end clock in the diagnostic function;
3. no runtime call to either helper symbol remains;
4. no inline assembly remains;
5. a conditional branch derived from the chain token appears after the start clock;
6. one or more end clock calls remain in branch-successor or merge blocks reachable after that branch;
7. the branch operand traces to the 16-step chain-derived scalar token.

If item 5–7 fails, the worker writes an inconclusive result with cause `end-dependency-optimized-away`, static-audit metadata, linked-LLIR artifact hash, and no cycle regions before returning. The parent retains this file even if CoreX aborts during interpreter teardown. Only a passing audit may allocate CUDA tensors and continue with writeback, positive-delta, empty/16/256 sensitivity, resource, and stability checks.

The matched final mechanism fails this audit because LLVM emits adjacent clock intrinsics before the token branch. This is the terminal Stage-A result; generated-LLIR surgery and another dependency mechanism are not attempted.

### Stage B: real-kernel whole-span control

Create diagnostic copies equivalent to:

```text
r001: num_warps = 1
r002: num_warps = 2
```

Instrument one sampled-kernel span and record selected first, middle, and edge query programs. Compare:

- correctness against the immutable accepted source;
- independent uninstrumented CUDA Event direction;
- instrumented Event overhead;
- registers, spills, and shared memory;
- selected-program issue-window distributions;
- the residual between selected-program behavior and whole-grid direction, explicitly without a warp-imbalance claim.

A selected-program span is not required to follow the whole-grid timing direction. An unchanged selected-program span alongside faster whole-kernel time is interpreted as evidence for concurrency, residency, or latency-hiding effects rather than a failed timer.

### Stage C: minimum coarse attribution

Only if Stage B remains interpretable, compile separate one-pair variants for:

```text
first full key tile
middle full key tile
edge key tile
prelude
loop body total
epilogue
```

This stage answers the two highest-value questions:

1. Does the 19-of-32 edge tile cost nearly as much as a full tile, supporting padding/tail waste as an optimization target?
2. Does the r001-to-r002 gain appear in selected-program issue windows or only in whole-grid behavior?

### Stage D: conditional deep attribution

Run only if Stage C produces stable, non-perturbing observations that leave an actionable ambiguity. Separate one-region variants may measure:

```text
K load
QK score
V load
online-softmax update
PV accumulate
```

This stage is optional. It is not required merely to declare that the external clock works.

### Stage E: graph replay

Run r003 graph replay only if eager evidence is already useful. Compare eager and replay records from the same r002-equivalent diagnostic kernel. Graph collection remains outside the recorded issue window.

## 5. Minimal validity checks

Only checks needed to prevent a false mechanism conclusion are mandatory:

- accepted source hashes match the approved ledger;
- output correctness is unchanged;
- external clock helper and bitcode hashes are recorded;
- no spill is introduced;
- resource and Event overhead are reported;
- one-pair ownership and generations decode correctly;
- synthetic short/long sensitivity passes;
- interpreted data remains labeled `issue-window` without final ISA proof.

Exact occupancy modeling, product schemas, large matrix runners, and maintained evidence infrastructure remain out of scope.

## 6. Incremental-value decision

The rerun is `valuable` only if it adds at least one fact that changes or narrows a KernelSwift optimization decision, for example:

- edge key-tile issue windows are approximately full-tile cost, prioritizing tail specialization;
- edge issue windows are materially cheaper, deprioritizing tail work;
- r001/r002 local spans remain similar while whole-kernel time improves, prioritizing occupancy/concurrency rather than region instruction tuning;
- QK, softmax, PV, or loads show an unexpected dominant issue window that changes optimization priority;
- graph replay recovers stable internal records not available from the existing profiler and exposes a real eager/replay difference.

The result is `technically-valid-low-value` when clock, buffer, correctness, resource, and sensitivity checks pass but observations only repeat existing source or whole-kernel evidence.

The result is `inconclusive` when the helper cannot link, the synthetic control cannot distinguish work, perturbation prevents interpretation, or ordering limitations make even issue-window comparison misleading.

No result from this experiment changes official benchmark authority. A `valuable` result only justifies a separate maintained-backend/source-access assessment.

## 7. Repository changes

Expected implementation files are limited to the existing experiment tree:

```text
experiments/bi150-kperfir-value/
├── device/corex_clock.cu
├── lib/corex_clock.py
├── scripts/build_corex_clock.py
├── scripts/clock64_probe.py              # replace failed inline-asm source
├── scripts/route_c_attention_probe.py    # minimum real-kernel stages
├── diagnostic/                           # disposable source-equivalent kernels
├── tests/                                # focused local tests
├── evidence-summary-v2.json
└── assessment-v2.md
```

Raw device artifacts remain ignored. Accepted candidates, `auto_bench.py`, official campaign artifacts, and `submissions/` remain untouched.

## 8. Stop boundary

The experiment stops without LLIR surgery or compiler modification if the external helper cannot be linked and invoked from the installed CoreX Triton. It also stops before deep or graph stages when synthetic sensitivity, correctness, or basic perturbation checks fail.
