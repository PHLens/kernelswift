# BI150 Route C External-Clock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Link the confirmed CoreX CUDA C `clock64()` path into Triton without new lowering, qualify honest selected-program issue windows, rerun the minimum r001/r002 attention attribution matrix, and decide whether the new evidence adds KernelSwift value.

**Architecture:** A tiny CoreX device library provides always-inline start and token-dependent end clock helpers as LLVM bitcode. The end helper places clock reads behind a token-derived conditional branch and contains no inline assembly. Triton diagnostic kernels call those symbols directly through `core.extern_elementwise` and pass the bitcode through the Iluvatar backend’s `extern_libs` option. Synthetic controls first prove inlining, branch dependency, buffer writeback, and short/long sensitivity; only then do source-equivalent attention variants measure whole span, full-versus-edge key tiles, and representative QK/softmax/PV issue windows.

**Tech Stack:** Python 3 `unittest`, CoreX 4.4.0 Clang/LLVM, Triton 3.1.0 Iluvatar backend, PyTorch CUDA-compatible runtime, JSON evidence, SHA256 source guards.

**Spec:** `docs/superpowers/specs/2026-08-30-bi150-route-c-external-clock-design.md`

## Global Constraints

- Accepted candidates, `auto_bench.py`, official campaign artifacts, and `submissions/` are immutable.
- Build and run only on the matched BI150/CoreX 4.4.0 host after accepted-source verification.
- Do not add or modify Triton lowering, patch generated LLIR, or guess BI assembly syntax.
- Stop if external bitcode cannot link/invoke or synthetic short/long sensitivity fails; do not fall back to LLIR surgery.
- External calls are program-level. Do not claim per-warp timing or warp imbalance.
- Every interpreted end marker must consume a scalar token derived after the measured work.
- Every cycle result is `measurement_semantics: issue-window`; never call it authoritative execution duration or official timing.
- Keep one timestamp pair per selected program and flush it after the measured region.
- Record inline clock/control and dependency-token controls; do not blindly subtract them from raw cycles.
- Do not retain runtime device calls to clock helpers: the noinline implementation linked but hung on first BI150 execution and was terminated after approximately seven minutes.
- After the noinline runtime hang, the container-visible GPU context remained unusable and could not be reset from the container; therefore do not attribute the subsequent empty-inline-assembly smoke-test stall to the backend as an independent result.
- The token-dependent conditional branch is the final attempted dependency mechanism. CoreX LLVM compile-only evidence shows it is optimized to adjacent clock reads before the unrelated writer branch; freeze `end-dependency-optimized-away` instead of adding another barrier variant.
- Use one combined review and at most one correction pass per task. Avoid product schemas, exact occupancy models, generic matrix frameworks, and review loops.

---

## File Structure

- `experiments/bi150-kperfir-value/device/corex_clock.cu` — confirmed CoreX CUDA C clock helpers only.
- `experiments/bi150-kperfir-value/lib/corex_clock.py` — build metadata, LLIR checks, result classification, and command construction; imports no Torch/Triton.
- `experiments/bi150-kperfir-value/scripts/build_corex_clock.py` — CLI that builds `.bc`/`.ll` on the matched host and writes one metadata JSON.
- `experiments/bi150-kperfir-value/scripts/clock64_probe.py` — replace the failed PTX-inline-asm worker with external-library synthetic qualification.
- `experiments/bi150-kperfir-value/diagnostic/mm_encoder_attention_external_clock.py` — source-equivalent r001/r002 arithmetic plus one program-level timestamp pair.
- `experiments/bi150-kperfir-value/scripts/route_c_attention_probe.py` — run the bounded real-kernel matrix and emit one result JSON.
- `experiments/bi150-kperfir-value/scripts/build_evidence_summary_v2.py` — validate and summarize the new run.
- `experiments/bi150-kperfir-value/evidence-summary-v2.json` — versioned compact evidence.
- `experiments/bi150-kperfir-value/assessment-v2.md` — final incremental-value decision.
- Focused tests under `experiments/bi150-kperfir-value/tests/`.

---

### Task 1: Build and validate the CoreX clock helper

**Files:**
- Create: `experiments/bi150-kperfir-value/device/corex_clock.cu`
- Create: `experiments/bi150-kperfir-value/lib/corex_clock.py`
- Create: `experiments/bi150-kperfir-value/scripts/build_corex_clock.py`
- Create: `experiments/bi150-kperfir-value/tests/test_corex_clock.py`
- Modify: `experiments/bi150-kperfir-value/README.md`

**Interfaces:**
- Produces `build_corex_clock(corex_root: Path, source: Path, output_dir: Path, runner=subprocess.run) -> dict`.
- Produces `validate_clock_helper_ir(ir_text: str) -> dict[str, bool]`.
- Produces artifacts `corex-clock.bc`, `corex-clock.ll`, and `clock-helper.json`.
- Later tasks consume the absolute `.bc` path and metadata SHA256.

- [ ] **Step 1: Write the device helper source**

```cpp
#include <cuda_runtime.h>

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

The end helper uses token-dependent control flow only. It contains no inline assembly or target instruction text. The caller subtracts `token & 1` from the encoded return.

- [ ] **Step 2: Write failing command/IR tests**

Test exact command properties without CoreX installed locally:

```python
command = clock_compile_command(
    Path("/usr/local/corex-4.4.0"),
    Path("device/corex_clock.cu"),
    Path("out/corex-clock.bc"),
)
self.assertIn("-x", command)
self.assertIn("ivcore", command)
self.assertIn("--cuda-device-only", command)
self.assertIn("--cuda-gpu-arch=ivcore11", command)
self.assertIn("-emit-llvm", command)
self.assertIn("-c", command)
```

Test IR requirements:

```python
checks = validate_clock_helper_ir("""
target triple = "bi-iluvatar-ilurt"
define i64 @corex_clock64_start() alwaysinline {
  %start = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
  ret i64 %start
}
define i64 @corex_clock64_after_u64(i64 %token) alwaysinline {
  %bit = and i64 %token, 1
  %condition = icmp ne i64 %bit, 0
  br i1 %condition, label %odd, label %even
odd:
  %odd_clock = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
  %encoded = add i64 %odd_clock, 1
  ret i64 %encoded
even:
  %even_clock = call i64 @llvm.nvvm.read.ptx.sreg.clock64()
  ret i64 %even_clock
}
declare i64 @llvm.nvvm.read.ptx.sreg.clock64()
""")
self.assertTrue(all(checks.values()))
```

Test that a missing end symbol or intrinsic raises `ClockHelperBuildError`.

- [ ] **Step 3: Run the focused test and confirm failure**

Run:

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_corex_clock.py -v
```

Expected: import failure because `lib/corex_clock.py` does not yet exist.

- [ ] **Step 4: Implement command construction and metadata validation**

Use these compile commands:

```text
clang++ -x ivcore --cuda-path=<corex-root> --cuda-gpu-arch=ivcore11 \
  --cuda-device-only -I<corex-root>/include -Wno-unused-command-line-argument \
  -emit-llvm -c device/corex_clock.cu -o corex-clock.bc

clang++ -x ivcore --cuda-path=<corex-root> --cuda-gpu-arch=ivcore11 \
  --cuda-device-only -I<corex-root>/include -Wno-unused-command-line-argument \
  -S -emit-llvm device/corex_clock.cu -o corex-clock.ll
```

The result JSON must contain:

```json
{
  "document_type": "corex-clock-helper",
  "status": "valid",
  "corex_root": "/usr/local/corex-4.4.0",
  "target": "ivcore11",
  "target_triple": "bi-iluvatar-ilurt",
  "source_sha256": "...",
  "bitcode_sha256": "...",
  "bitcode_path": "corex-clock.bc",
  "ir_path": "corex-clock.ll",
  "ir_checks": {
    "start_symbol": true,
    "dependent_end_symbol": true,
    "clock64_intrinsic": true,
    "target_triple": true
  }
}
```

Write files only after both commands succeed; malformed IR is a build failure.

- [ ] **Step 5: Run focused and complete local tests**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_corex_clock.py -v
python3 -m unittest discover -s experiments/bi150-kperfir-value/tests -p 'test_*.py'
python3 -m py_compile experiments/bi150-kperfir-value/lib/corex_clock.py \
  experiments/bi150-kperfir-value/scripts/build_corex_clock.py
git diff --check
```

Expected: all pass.

- [ ] **Step 6: Build on the matched host**

Deploy the exact commit with `git archive`, set `ROUTE_C_COMMIT`, source `/usr/local/corex/enable`, and run:

```bash
python3 experiments/bi150-kperfir-value/scripts/build_corex_clock.py \
  --corex-root /usr/local/corex-4.4.0 \
  --output-dir experiments/bi150-kperfir-value/artifacts/external-clock/helper
```

Expected: valid metadata, readable `.bc`, and `.ll` containing both symbols and the clock intrinsic.

- [ ] **Step 7: Perform one combined review and at most one correction**

Review only build correctness, ABI, target, and evidence integrity. Do not request packaging frameworks.

- [ ] **Step 8: Commit**

```bash
git add experiments/bi150-kperfir-value/device/corex_clock.cu \
  experiments/bi150-kperfir-value/lib/corex_clock.py \
  experiments/bi150-kperfir-value/scripts/build_corex_clock.py \
  experiments/bi150-kperfir-value/tests/test_corex_clock.py \
  experiments/bi150-kperfir-value/README.md
git commit -m "feat(experiment): build CoreX external clock helper"
```

---

### Task 2: Qualify program-level issue windows through Triton external calls

**Files:**
- Modify: `experiments/bi150-kperfir-value/scripts/clock64_probe.py`
- Modify: `experiments/bi150-kperfir-value/tests/test_clock64_probe.py`
- Modify: `experiments/bi150-kperfir-value/lib/result_contract.py`
- Modify: `experiments/bi150-kperfir-value/tests/test_result_contract.py`

**Interfaces:**
- Consumes helper `.bc` and `clock-helper.json` from Task 1.
- Produces `artifacts/external-clock/synthetic/result.json`.
- Produces `qualification_status` of `valid`, `inconclusive`, or `invalid`.
- Later tasks run only when qualification is `valid`.

- [ ] **Step 1: Write failing direct-extern source contract tests**

Assert that the script:

- contains `core.extern_elementwise` directly inside `@triton.jit` helpers;
- does not contain `@core.extern`;
- references `corex_clock64_start` and `corex_clock64_after_u64`;
- passes `extern_libs={"corex_clock": ...}`;
- uses one profile slot `[generation,start,end]` per program;
- labels semantics `issue-window`;
- contains no `%clock64` inline assembly.

Add classification tests:

```python
self.assertEqual(
    ("valid", []),
    classify_external_clock(
        linked=True,
        intrinsic_count_ok=True,
        writeback_ok=True,
        positive_deltas=True,
        short_long_sensitive=True,
        dependency_verified=True,
    ),
)
```

A link failure or short/long failure must be `inconclusive`, not `unsupported`.

- [ ] **Step 2: Run focused tests and confirm failure**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_clock64_probe.py -v
```

Expected: failures because the old script still contains PTX inline assembly.

- [ ] **Step 3: Replace inline-assembly helpers with JIT helpers**

The JIT helpers must call the builtin directly while referencing dependency-scanner-safe global dispatch maps. Inline dictionary literals are rejected by the installed Triton AST parser:

```python
uint64 = core.dtype("uint64")
CLOCK_START_DISPATCH: "constexpr" = {
    (): ("corex_clock64_start", uint64),
}
CLOCK_AFTER_DISPATCH: "constexpr" = {
    (uint64,): ("corex_clock64_after_u64", uint64),
}

@triton.jit
def read_clock_start():
    return core.extern_elementwise(
        "", "", [], CLOCK_START_DISPATCH, is_pure=False,
    )

@triton.jit
def read_clock_after(token):
    encoded = core.extern_elementwise(
        "", "", [token], CLOCK_AFTER_DISPATCH, is_pure=False,
    )
    return encoded - (token & 1)
```

If the installed CoreX type parser requires `int64` instead of `uint64`, use a consistent signed ABI in both helper declaration and dispatch, while preserving unsigned host decoding. Record the observed ABI in the result.

- [ ] **Step 4: Implement the synthetic control kernel**

Use one program and one profile slot. Run fixed `CHAIN_ITERS` values:

```text
0    empty inline clock/control floor
16   short control
256  long control
```

Build a tensor dependency chain, keep it live through an output store, reduce its final value to one scalar `uint64` token, then invoke the dependent end helper. Only element zero writes:

```text
profile[0] = generation
profile[1] = start
profile[2] = end
```

Run 20 warmups and 50 samples for each specialization. Record raw deltas and `count/minimum/median/p10/p90/maximum/CV`.

- [ ] **Step 5: Run the prelaunch compile-only dependency audit**

Before any CUDA tensor allocation, construct `triton.compiler.ASTSource` for the existing synthetic JIT kernel with this manual specialization:

```python
signature = {
    "seed_ptr": "*i64",
    "profile_ptr": "*i64",
    "output_ptr": "*i64",
    "generation": "i32",
}
constants = {"chain_iters": 16}
target = GPUTarget("cuda", 71, 64)
options = {
    "num_warps": 1,
    "extern_libs": {"corex_clock": str(clock_bitcode)},
}
```

Compile without allocating tensors, persist linked LLIR, and run `inspect_linked_llir`. Record helper SHA256, compiled hash, target/signature/constants, linked-LLIR SHA256/path, clock/branch/dependency checks, no inline assembly, and no retained helper runtime calls.

If the dependency is absent, immediately write an inconclusive result with:

```json
{
  "status_causes": ["end-dependency-optimized-away"],
  "regions": [],
  "static_dependency_audit": {
    "status": "optimized-away",
    "mode": "compile-only-prelaunch"
  }
}
```

The worker must write this result before returning, and the parent must retain it even when CoreX exits nonzero during interpreter teardown. The matched final helper is expected to take this terminal path because LLVM emits adjacent start/end clocks before the token branch. Do not allocate CUDA tensors or add another dependency mechanism after this result.

Only if the static audit unexpectedly passes may the probe continue to runtime resource and cycle qualification. Final ISA remains optional and successful runtime measurements remain `issue-window` evidence.

- [ ] **Step 6: Define the minimal sensitivity rule**

Qualification is valid when:

```text
all generations match
all decoded deltas are positive
median(256) > median(16) > median(0)
linked LLIR shows a chain-derived conditional branch after the start clock, reachable end clocks, no inline assembly, and no helper runtime calls
no spill
```

Report CV and overlap, but do not add a hard CV threshold unless the distributions cannot distinguish the three controls.

- [ ] **Step 7: Run local tests**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_clock64_probe.py -v
python3 -m unittest discover -s experiments/bi150-kperfir-value/tests -p 'test_*.py'
python3 -m py_compile experiments/bi150-kperfir-value/scripts/clock64_probe.py
git diff --check
```

- [ ] **Step 8: Deploy and run synthetic qualification on BI150**

```bash
python3 experiments/bi150-kperfir-value/scripts/clock64_probe.py \
  --clock-bitcode experiments/bi150-kperfir-value/artifacts/external-clock/helper/corex-clock.bc \
  --clock-metadata experiments/bi150-kperfir-value/artifacts/external-clock/helper/clock-helper.json \
  --output experiments/bi150-kperfir-value/artifacts/external-clock/synthetic/result.json \
  --samples 50
```

If status is not `valid`, stop the remaining plan and write a blocker assessment. Do not implement Task 3.

- [ ] **Step 9: Perform one combined review and at most one correction**

Review only linkage, dependency, writeback, sensitivity, and classification.

- [ ] **Step 10: Commit**

```bash
git add experiments/bi150-kperfir-value/scripts/clock64_probe.py \
  experiments/bi150-kperfir-value/tests/test_clock64_probe.py \
  experiments/bi150-kperfir-value/lib/result_contract.py \
  experiments/bi150-kperfir-value/tests/test_result_contract.py
git commit -m "feat(experiment): qualify CoreX external clock cycles"
```

---

### Task 3: Add source-equivalent attention diagnostic variants

**Files:**
- Create: `experiments/bi150-kperfir-value/diagnostic/__init__.py`
- Create: `experiments/bi150-kperfir-value/diagnostic/mm_encoder_attention_external_clock.py`
- Create: `experiments/bi150-kperfir-value/tests/test_attention_diagnostic.py`

**Interfaces:**
- Produces `launch_diagnostic(query, key, value, out, profile, generation, sample_pid, region, key_tile, num_warps, clock_bitcode)`.
- Region names are exactly `kernel-span`, `prelude`, `loop-total`, `key-tile`, `epilogue`, `qk`, `softmax`, and `pv`.
- Produces one profile slot for the selected program only.

- [ ] **Step 1: Write structural tests before copying kernel code**

Tests must assert:

- accepted hashes still match;
- `_BM=_BN=_BD=32`;
- diagnostic arithmetic contains the same four `tl.dot` expressions and online-softmax updates as r002;
- grid remains `B * H * ceil(S / BM)`;
- launch supports `num_warps=1` and `2` only;
- only `pid == sample_pid` executes/writes the clock pair;
- exactly one generation/start/end slot exists;
- no accepted path is imported and mutated;
- every end-marker branch passes a token to `read_clock_after`;
- no region variant stores intermediate tensors to global memory.

- [ ] **Step 2: Run the structural test and confirm failure**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_attention_diagnostic.py -v
```

Expected: missing diagnostic module.

- [ ] **Step 3: Copy the accepted arithmetic into an isolated diagnostic kernel**

Copy the arithmetic body from `triton_mm_encoder_attention_e2_002.py` without changing:

- Q/K/V addressing;
- fp16-to-fp32 widening;
- four dot operations per key tile;
- online max/sum recurrence;
- masks;
- normalization;
- final fp16 store.

Add only:

- profile pointer;
- generation;
- selected PID;
- constexpr region ID/key tile;
- direct external start/end calls;
- a scalar dependency token;
- selected writer flush.

Use launch `num_warps` to reproduce r001/r002 rather than maintaining duplicate arithmetic functions.

- [ ] **Step 4: Define exact region boundaries and tokens**

Use these boundaries:

```text
kernel-span: start before Q loads; end after output values are produced
prelude:     start before Q loads; end after Q loads/state initialization
loop-total:  start before static key loop; end after the final acc/l_run update
key-tile:    start before selected K loads; end after selected PV acc update
qk:          start before selected K loads/dots; end after scaled/masked score exists
softmax:     start before m_new/alpha/p; end after l_run update
pv:          start before selected V loads/dots; end after acc update
epilogue:    start before normalization; end after final output values exist
```

For each region, derive one scalar token from a value produced after the boundary. When this requires a new reduction, expose a constexpr `CLOCK_ENABLED` so the matched dependency-control specialization performs the same token calculation without clock calls.

- [ ] **Step 5: Implement output and profile launch surfaces**

The host wrapper must allocate no official output internally. It accepts preallocated output/profile tensors and returns metadata only. Pass:

```python
extern_libs={"corex_clock": str(clock_bitcode)}
```

Only selected PID writes the profile slot. Non-selected programs must execute accepted arithmetic without clock calls.

- [ ] **Step 6: Run local structural tests and source guard**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_attention_diagnostic.py -v
python3 -m unittest discover -s experiments/bi150-kperfir-value/tests -p 'test_*.py'
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'experiments/bi150-kperfir-value/lib')
from source_guard import verify_accepted_sources
verify_accepted_sources(Path.cwd())
print('accepted sources unchanged')
PY
git diff --check
```

- [ ] **Step 7: Perform one combined review and at most one correction**

Review arithmetic equivalence, one-pair retention, selected-PID predicate, token dependency, and accepted-source isolation only.

- [ ] **Step 8: Commit**

```bash
git add experiments/bi150-kperfir-value/diagnostic \
  experiments/bi150-kperfir-value/tests/test_attention_diagnostic.py
git commit -m "feat(experiment): add external-clock attention diagnostics"
```

---

### Task 4: Run the bounded r001/r002 attribution matrix

**Files:**
- Create: `experiments/bi150-kperfir-value/scripts/route_c_attention_probe.py`
- Create: `experiments/bi150-kperfir-value/tests/test_route_c_attention_probe.py`
- Modify: `experiments/bi150-kperfir-value/README.md`

**Interfaces:**
- Consumes valid Task-1 helper metadata and valid Task-2 qualification result.
- Produces `artifacts/external-clock/attention/result.json`.
- Emits no automatic `valuable` verdict; it emits normalized observations and stop reasons for Task 5.

- [ ] **Step 1: Write failing matrix and classification tests**

The fixed matrix is:

```python
WHOLE_SPAN_CASES = [
    ("r001-nw1", 1, 0), ("r001-nw1", 1, 16), ("r001-nw1", 1, 32),
    ("r002-nw2", 2, 0), ("r002-nw2", 2, 16), ("r002-nw2", 2, 32),
]
KEY_TILE_CASES = [("r002-nw2", 2, 16, tile) for tile in (0, 1, 2)]
DEEP_CASES = [
    ("r002-nw2", 2, 16, 1, "qk"),
    ("r002-nw2", 2, 16, 1, "softmax"),
    ("r002-nw2", 2, 16, 1, "pv"),
]
```

Also test PID meaning:

```text
pid 0  = batch 0, head 0, query tile 0
pid 16 = batch 0, head 0, query tile 1
pid 32 = batch 0, head 0, query tile 2 (19 valid rows)
```

Test that invalid synthetic qualification prevents all device imports and returns `inconclusive`.

- [ ] **Step 2: Run the focused test and confirm failure**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_route_c_attention_probe.py -v
```

Expected: missing runner module.

- [ ] **Step 3: Implement preflight and immutable-source checks**

Before importing Torch/Triton:

1. verify five accepted hashes;
2. validate helper metadata;
3. validate synthetic result;
4. require the same route commit and helper bitcode hash;
5. reject missing bitcode.

Then lazily import device libraries and diagnostic module.

- [ ] **Step 4: Reproduce accepted correctness and uninstrumented direction**

Using one fixed seed/input set, run accepted r001 and r002 and record:

- bitwise comparison between their outputs;
- comparison to base with the existing tolerance only as context;
- 20 warmups and 50 CUDA Event samples per accepted variant;
- median/p10/p90/CV;
- reproduced direction `median(r002) < median(r001)`.

A failed correctness comparison makes the real probe invalid. A failed timing direction is recorded and stops mechanism interpretation.

- [ ] **Step 5: Run dependency controls and instrumented cases**

For every matrix row:

1. run the `CLOCK_ENABLED=False` dependency-control specialization;
2. verify output bitwise-equal to the accepted matching warp-count candidate;
3. record resources and Event timing;
4. run the clock-enabled specialization;
5. verify output again;
6. collect 50 profile generations/deltas;
7. summarize raw issue windows;
8. record inline clock/control floor, token expression, control overhead, register/spill/shared changes, and clock-enabled Event overhead.

Do not subtract the inline clock/control floor or dependency-control cycles from the raw issue window.

- [ ] **Step 6: Preserve only interpretable rows**

Mark a row `observed` only when:

```text
correctness unchanged
generation matches for all retained samples
all deltas positive
no spill introduced
whole-kernel direction is not reversed by instrumentation
linked LLIR shows a chain-derived conditional branch after the start clock, reachable end clocks, no inline assembly, and no helper runtime calls
```

Otherwise mark the row `unavailable` or `invalid` with a concrete cause and no fabricated summary.

- [ ] **Step 7: Emit one normalized result**

Include:

```json
{
  "document_type": "route-c-attention-result-v2",
  "route_c_commit": "...",
  "helper": {"bitcode_sha256": "..."},
  "synthetic_qualification": "valid",
  "accepted_timing": {},
  "whole_span": [],
  "key_tiles": [],
  "deep_regions": [],
  "limitations": [
    "program-level-only",
    "issue-window-not-execution-duration",
    "final-isa-unavailable"
  ]
}
```

- [ ] **Step 8: Run local tests and compile checks**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_route_c_attention_probe.py -v
python3 -m unittest discover -s experiments/bi150-kperfir-value/tests -p 'test_*.py'
python3 -m py_compile experiments/bi150-kperfir-value/scripts/route_c_attention_probe.py
git diff --check
```

- [ ] **Step 9: Deploy the exact commit and run on BI150**

```bash
python3 experiments/bi150-kperfir-value/scripts/route_c_attention_probe.py \
  --clock-bitcode experiments/bi150-kperfir-value/artifacts/external-clock/helper/corex-clock.bc \
  --clock-metadata experiments/bi150-kperfir-value/artifacts/external-clock/helper/clock-helper.json \
  --synthetic-result experiments/bi150-kperfir-value/artifacts/external-clock/synthetic/result.json \
  --output experiments/bi150-kperfir-value/artifacts/external-clock/attention/result.json \
  --samples 50
```

Collect the ignored result and any linked LLIR/resource artifacts locally. Validate hashes and document contract.

- [ ] **Step 10: Perform one combined review and at most one correction**

Review only whether observed rows are honest, source-equivalent, correctly decoded, and sufficient for value assessment.

- [ ] **Step 11: Commit**

```bash
git add experiments/bi150-kperfir-value/scripts/route_c_attention_probe.py \
  experiments/bi150-kperfir-value/tests/test_route_c_attention_probe.py \
  experiments/bi150-kperfir-value/README.md
git commit -m "feat(experiment): run BI150 Route C attention matrix"
```

---

### Task 5: Freeze v2 evidence and decide incremental KernelSwift value

**Files:**
- Create: `experiments/bi150-kperfir-value/scripts/build_evidence_summary_v2.py`
- Create: `experiments/bi150-kperfir-value/tests/test_evidence_summary_v2.py`
- Create: `experiments/bi150-kperfir-value/evidence-summary-v2.json`
- Create: `experiments/bi150-kperfir-value/assessment-v2.md`
- Modify: `experiments/bi150-kperfir-value/README.md`

**Interfaces:**
- Consumes helper metadata, synthetic qualification, and attention result from one frozen evidence commit.
- Produces final classification `valuable`, `technically-valid-low-value`, `perturbation-invalid`, or `inconclusive`.

- [ ] **Step 1: Write failing evidence-binding tests**

Tests must require:

- all three raw documents have the same route commit;
- accepted-source ledger matches exactly;
- helper bitcode SHA256 matches in all documents;
- raw JSON SHA256 values are recorded;
- every observed row says `issue-window`;
- program-level-only limitation is retained;
- no per-warp or execution-duration statement appears.

- [ ] **Step 2: Run the focused test and confirm failure**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_evidence_summary_v2.py -v
```

Expected: missing v2 builder.

- [ ] **Step 3: Implement the compact v2 summary builder**

Copy only:

- environment and evidence commit;
- accepted hashes;
- helper build identity;
- synthetic empty/short/long summaries;
- accepted r001/r002 Event timing;
- observed whole-span rows;
- full/middle/edge key-tile rows;
- QK/softmax/PV rows;
- resource/overhead/control facts;
- raw evidence hashes;
- limitations and stop reasons.

Do not create a general bundle schema or root digest framework.

- [ ] **Step 4: Write the assessment using explicit decision rules**

Classify `valuable` when at least one observed fact changes or narrows an optimization decision, including:

- edge tile approximately full-tile issue window, prioritizing tail specialization;
- edge tile materially cheaper, deprioritizing tail work;
- r001/r002 selected-program spans similar despite faster whole-grid r002, prioritizing residency/concurrency;
- an unexpected QK/softmax/PV ordering changes optimization priority.

Classify `technically-valid-low-value` when the external clock is technically valid but all observed facts only repeat source inspection or existing whole-kernel evidence.

Classify `perturbation-invalid` when the token/helper machinery demonstrably changes the relevant execution/resource class so the original mechanism cannot be described.

Classify `inconclusive` when the external helper, synthetic sensitivity, correctness, timing direction, or enough real rows fail.

The assessment must distinguish:

```text
new measurement capability
new kernel fact
new optimization decision
```

A working clock alone is not incremental KernelSwift value.

- [ ] **Step 5: Generate and reproduce the versioned summary**

```bash
python3 experiments/bi150-kperfir-value/scripts/build_evidence_summary_v2.py \
  --helper experiments/bi150-kperfir-value/artifacts/external-clock/helper/clock-helper.json \
  --synthetic experiments/bi150-kperfir-value/artifacts/external-clock/synthetic/result.json \
  --attention experiments/bi150-kperfir-value/artifacts/external-clock/attention/result.json \
  --output experiments/bi150-kperfir-value/evidence-summary-v2.json
```

Run it again to `/tmp/evidence-summary-v2.json` and require byte-for-byte equality.

- [ ] **Step 6: Run final tests and regressions**

```bash
python3 -m unittest discover -s experiments/bi150-kperfir-value/tests -p 'test_*.py'
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py \
  skills/kernel-opt-loop/tests/test_run_profile_probe.py
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'experiments/bi150-kperfir-value/lib')
from source_guard import verify_accepted_sources
print(verify_accepted_sources(Path.cwd()))
PY
sha256sum experiments/bi150-kperfir-value/evidence-summary-v2.json
git diff --check
git status --short
```

Expected: all tests pass; accepted hashes unchanged; only the pre-existing untracked `submissions/` remains outside tracked changes.

- [ ] **Step 7: Perform one combined final review and at most one correction**

Review factual consistency, raw hash binding, and whether the final classification follows the explicit rules. Do not demand more experiments after the bounded matrix is complete.

- [ ] **Step 8: Commit**

```bash
git add experiments/bi150-kperfir-value/scripts/build_evidence_summary_v2.py \
  experiments/bi150-kperfir-value/tests/test_evidence_summary_v2.py \
  experiments/bi150-kperfir-value/evidence-summary-v2.json \
  experiments/bi150-kperfir-value/assessment-v2.md \
  experiments/bi150-kperfir-value/README.md
git commit -m "docs(experiment): assess external-clock Route C rerun"
```

---

## Completion Criteria

The plan is complete when one of these terminal outcomes is committed:

1. **Qualified rerun:** helper links, synthetic sensitivity passes, real attention matrix runs, v2 evidence is frozen, and assessment states the incremental-value classification.
2. **Bounded blocker:** helper linkage or synthetic dependency sensitivity fails, no LLIR surgery is attempted, and v2 assessment records the concrete blocker and retained facts.

In both outcomes accepted sources and official authority remain unchanged.
