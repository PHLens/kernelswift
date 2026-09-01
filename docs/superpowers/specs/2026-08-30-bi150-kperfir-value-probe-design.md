# BI150 KPerfIR Value Probe (Route C) Design

**Created**: 2026-08-30<br>
**Status**: Proposed<br>
**Scope**: A bounded, disposable Triton source-instrumentation experiment on Iluvatar BI-V150 / CoreX that measures whether intra-kernel region facts add actionable value beyond the existing vendor profiler evidence<br>
**Related designs**: `2026-08-13-kernel-opt-loop-restructure-design.md`, `2026-08-14-kernel-opt-loop-v2-continuous-run-design.md`, and `2026-08-19-kernel-opt-loop-vnext-semantic-attribution-design.md`

## 1. Decision summary

KPerfIR and Triton Proton have already established that compiler-integrated profiling IR can provide accurate intra-kernel region timing on modern NVIDIA and AMD SIMT GPUs. This design does not repeat that research result.

The remaining KernelSwift-specific question is narrower:

> Can a BI150-local, KPerfIR-inspired region-timing probe expose actionable facts that the current CoreX profiler and CUDA Event probes do not expose, strongly enough to justify requesting the matching CoreX Triton source and conducting a separate maintained-backend port-cost assessment?

The approved answer path is Route C:

```text
accepted BI150 candidate remains immutable
    -> create isolated diagnostic copies
    -> read CoreX clock64 through Triton inline assembly
    -> sample selected programs and every local warp present in each variant
    -> compile one measured region pair per diagnostic variant
    -> retain that start/end pair in registers
    -> flush the pair after the measured region
    -> verify ordering through TTGIR, LLIR, and final CoreX ISA
    -> compare known r001/r002 device-time behavior
    -> test graph-replay visibility on r003
    -> classify incremental value
```

Route C is a disposable value probe, not a production profiler backend. A `valuable` result only opens a separate port-cost assessment. A formal `ProtonIluvatarGPUToLLVM` design is permitted only after that assessment, source access, substrate, ownership, and generality gates also pass.

## 2. Motivation

The current BI150 campaign evidence is strong at the host, launch, graph, and whole-kernel levels but weak inside a generated Triton kernel.

The accepted `mm_encoder_attention` epoch-2 candidate established:

- direct-launch Triton kernel device time around 18–20 microseconds;
- a graph-replayed route that improves official wall time;
- complete host API, graph launch, synchronization, and copy-out census;
- no `cat=kernel` events for graph-interior kernels on the matched CoreX profiler build;
- a CUDA Event graph round trip around 64 microseconds, combining about 46 microseconds of graph/frontend behavior with about 18 microseconds of kernel math;
- source-based attribution that QK, online softmax, PV accumulation, padding, and warp count are plausible internal mechanisms.

The missing evidence is:

```text
Within the approximately 18–20 microsecond Triton attention kernel,
which regions dominate,
which regions changed when num_warps changed from one to two,
and whether graph replay preserves the same internal device behavior?
```

Vendor profiling already answers whole-kernel duration and launch census. Route C is worthwhile only if it answers an internal mechanism question that can change optimization priority, eliminate a speculative direction, recover graph-interior attribution, or strengthen a device-floor bound.

## 3. Established BI150 environment facts

A read-only audit of the matched BI150 environment established the following facts.

### 3.1 Runtime identity

```text
device: Iluvatar BI-V150
compute compatibility surface: 7.1
multiprocessors: 16
CoreX: 4.4.0
PyTorch: 2.7.1+corex.4.4.0
Triton: 3.1.0+corex.4.4.0
warp size: 64
target processor: ivcore11
```

### 3.2 CoreX is SIMT-like but not the NVIDIA Triton backend

The Python backend uses Triton programs, warps, CUDA-style launch, and CUDA-compatible runtime APIs. Its compiler path is nevertheless target-specific:

```text
TritonGPU IR
    -> iluvatar.passes.ttgpuir.add_to_llvmir
    -> target triple bi-iluvatar-ilurt
    -> processor ivcore11
    -> Iluvatar kernel calling convention
```

A future production port can reuse the Proton and ProtonGPU common dialects and transforms, but it cannot assume that `ProtonNvidiaGPUToLLVM` is a drop-in backend.

### 3.3 Existing Proton package is incomplete

The installed wheel includes the early `triton.profiler` Python package and Proton command-line entry points. Importing `triton.profiler` fails because `triton._C.libproton` is absent. The installed `libtriton.so` does not expose the current ProtonGPU instrumentation dialect and conversion-pass bindings.

Therefore the environment does not have an instrumentation backend that can be enabled through a runtime flag.

### 3.4 Required source is unavailable in the environment

The machine contains:

- the vendor Triton wheel;
- `libtriton.so`;
- Python backend files;
- CoreX LLVM/MLIR libraries and headers;
- CoreX runtime and sample code.

It does not contain the matching CoreX Triton compiler source tree, Iluvatar Triton conversion sources, or a build tree for the installed wheel. A maintained full KPerfIR port is blocked until the matching source revision and build instructions are obtained.

### 3.5 A source-level clock probe is feasible

The CoreX SDK and compiler contain evidence for SIMT and timer primitives including `clock`, `clock64`, lane ID, warp ID, thread ID, block ID, and SM-ID-related intrinsics. The installed Triton language supports `tl.inline_asm_elementwise`.

This is sufficient to justify a source-level feasibility probe. It is not yet runtime proof that every primitive has the required semantics; Route C must validate monotonicity, ordering, ID ranges, graph compatibility, and perturbation before interpreting results.

## 4. Goals

1. Validate that CoreX `clock64` can produce monotonic, repeatable local-warp cycle deltas inside a Triton kernel.
2. Validate a bounded source-instrumentation and result-writeback path without rebuilding the CoreX compiler.
3. Preserve accepted competition candidates byte-for-byte and isolate all diagnostic code and cache state.
4. Measure selected query-tile programs and every local warp present in each launch variant without instrumenting all 48 programs.
5. Establish trustworthy coarse region timing before attempting deeper QK, softmax, and PV attribution.
6. Verify marker placement and target instruction ordering through TTGIR, LLIR, and final CoreX ISA or disassembly.
7. Use the already-proven `num_warps=1` versus `num_warps=2` behavior as a real sensitivity control.
8. Determine whether instrumentation remains visible inside CoreX graph replay when Kineto omits graph-interior kernel events.
9. Produce one bounded classification of technical validity and incremental KernelSwift value.
10. Define exact gates for requesting CoreX compiler source and designing a maintained Proton Iluvatar backend.

## 5. Non-goals

- Re-proving the general KPerfIR result on NVIDIA or AMD.
- Implementing or claiming a production KPerfIR/Proton CoreX backend.
- Modifying `skills/kernel-opt-loop`, Verifier authority, report schemas, implementation profiles, or profiler Levels 0–3.
- Changing official competition timing, correctness, adoption, or stop rules.
- Modifying accepted `mm_encoder_attention` candidates or their official campaign artifacts.
- Producing synchronized cross-core timelines.
- Treating `clock64` cycles as nanoseconds or authoritative wall time.
- Unifying vendor hardware counters, occupancy, cache metrics, or bandwidth metrics.
- Adding instruction sampling, PC sampling, shared-buffer mode, or automatic region discovery.
- Making SOL, KernelWiki, or cross-project knowledge decisions.
- Maintaining a binary plugin against the installed vendor `libtriton.so`.
- Extending KPerfIR to Ascend or another non-SIMT NPU architecture.

## 6. Design principles

### 6.1 Measure incremental value, not generic profiler novelty

A successful clock read is insufficient. The probe is valuable only when it supplies a fact not already provided by CoreX kernel timing, CUDA Event timing, graph API census, or source inspection.

### 6.2 Accepted sources remain authoritative and immutable

Instrumentation is diagnostic code. It never becomes the accepted candidate, official benchmark input, or campaign report authority.

### 6.3 Raw cycles are observations; converted time is an estimate

Route C records local `clock64` values. A same-session cycles-per-microsecond estimate can support sanity checks, but raw cycles remain the primary fact.

### 6.4 Ordering proof precedes timing interpretation

Low variance cannot prove that markers remained around the intended instructions. Every interpreted region must pass IR and ISA ordering review.

### 6.5 Perturbation can invalidate an otherwise correct probe

A probe that changes spill behavior, register class, occupancy, scheduling class, or the measured kernel's critical path cannot describe the original candidate.

### 6.6 Stop early on a failed gate

Synthetic failure, ordering failure, resource-class change, or unstable timing stops deeper work. The five-day experiment is not allowed to become a profiler implementation project.

## 7. Artifact and repository boundary

Route C uses an isolated experiment tree rather than production skill paths:

```text
experiments/bi150-kperfir-value/
├── README.md
├── synthetic/
├── mm_encoder_attention/
├── scripts/
└── artifacts/                 # gitignored raw outputs
```

Versioned content may include:

- experiment definitions;
- diagnostic source copies;
- fixed result schemas;
- deterministic decoders and summary scripts;
- small reviewed IR or ISA excerpts necessary for the ordering audit;
- a final `assessment.md`.

The experiment must not commit:

- raw large traces;
- compiler caches;
- raw profile-buffer dumps unless reduced to a bounded fixture;
- credentials, access commands, private hostnames, or secrets;
- official campaign reports rewritten with diagnostic findings;
- mutable tuning history or profiler state families.

All diagnostic kernels use distinct function names, distinct source hashes, and an isolated Triton cache namespace.

## 8. Diagnostic source variants

The experiment creates isolated diagnostic equivalents of:

```text
diag_r001_nw1     source-equivalent to epoch-2 round 001, num_warps=1
diag_r002_nw2     source-equivalent to epoch-2 round 002, num_warps=2
diag_r003_graph   the instrumented r002 kernel used through the round-003 graph route
```

The accepted files remain unchanged:

```text
triton_mm_encoder_attention_e2_001.py
triton_mm_encoder_attention_e2_002.py
triton_mm_encoder_attention_e2_003.py
```

Each diagnostic source records both the accepted-source SHA256 and diagnostic-source SHA256. Source equivalence claims apply to mathematical behavior and selected launch configuration; they do not imply byte identity after instrumentation.

## 9. Instrumentation primitive

The first implementation candidate reads a 64-bit local cycle counter through a side-effecting Triton inline-assembly operation conceptually equivalent to:

```python
tl.inline_asm_elementwise(
    asm="mov.u64 $0, %clock64;",
    constraints="=l",
    args=[],
    dtype=tl.int64,
    is_pure=False,
    pack=1,
)
```

The exact accepted syntax is a probe result, not a predeclared capability. If the target only accepts separate low and high 32-bit reads, the probe may combine `clock` and `clock_hi` into one unsigned 64-bit value.

Local thread identity is read through a target-supported thread-ID primitive. For the matched warp size:

```text
local_warp = tid_x / 64
lane = tid_x % 64
```

Physical warp ID is not used as a profile-buffer index because it is not required to be local, stable, or contiguous within one Triton program.

Every marker is side-effecting. No marker is interpreted until final target ordering is audited.

## 10. Sampling contract

The kernel grid contains 48 programs. Route C samples only the programs for:

```text
batch = 0
head = 0
query tile = first, middle, edge
```

Under the accepted PID mapping, these are expected to be:

```text
pid = 0, 16, 32
```

The implementation must verify, rather than assume, this mapping in the diagnostic source and generated IR.

Each variant records every local warp that actually exists under its launch configuration:

```text
diag_r001_nw1:   local warp 0
diag_r002_nw2:   local warps 0 and 1
diag_r003_graph: local warps 0 and 1
```

Non-selected programs execute the original mathematical kernel and write no records. Results are named `selected-program samples`; they are never reported as an average over all 48 programs.

## 11. Profile buffer and record ownership

The profile buffer is allocated before compilation warmup or graph capture and remains at a fixed device address for the lifetime of one diagnostic run.

Route C compiles a separate diagnostic variant for each measured region. A variant retains only one start/end pair per sampled warp. The logical layout is:

```text
[query_tile_class][local_warp][start_or_end]
```

The result metadata carries the region ID, launch variant, and execution mode. Each record set includes enough validity metadata to distinguish an unwritten slot from a legitimate zero or wrapped counter value. A bounded preamble or generation marker is required.

The start/end pair remains in registers through the measured region. At the diagnostic kernel's final writeback point, lane zero of each sampled local warp writes that pair to the global profile buffer.

This strategy avoids global stores between the measured start and end while limiting retained timer values to one pair. It still creates register-pressure risk, which remains an explicit validity gate.

The first version uses global-buffer writeback only. Route C does not implement Proton shared-buffer allocation, circular storage, or flush strategies.

## 12. Region model

### 12.1 Coarse pass

The coarse pass consists of separate variants for:

```text
prelude
key tile 0 total
key tile 1 total
key tile 2 total
epilogue
sampled kernel span
```

Each variant contains only the start/end pair for its named region. This tests query-tile and warp variability without retaining a complete multi-boundary timeline in one register-sensitive kernel.

### 12.2 Deep pass

The deep pass is permitted only after the coarse pass satisfies correctness, ordering, resource, overhead, and stability gates.

It uses separate one-region variants for:

```text
K load
QK score
V load
online-softmax update
PV accumulate
```

The exact boundaries follow the accepted algorithm, not an automatic semantic classifier. The generated IR and final ISA must establish that the selected marker pair still brackets the claimed target operations.

A region end marker must have an explicit data dependency on a value produced by the measured region, sufficient to prevent the compiler and target from issuing the end read before the region result is available. The start side must be protected by side-effect ordering and final-ISA review. When a region lacks a validated completion dependency or target retirement guarantee, its result is named an `issue-window`, not execution duration, and cannot support a dominant-region claim.

If any one-region variant changes the resource or occupancy class, that region result is `perturbation-invalid`; other independently valid region variants may remain usable.

## 13. Time semantics

The primary data is:

```text
raw clock64 start
raw clock64 end
unsigned raw cycle delta
```

The decoder must handle wraparound according to the validated counter width.

The experiment may derive `estimated_us` from a same-session long-running synthetic calibration that has both raw cycle delta and CUDA Event duration. Such a conversion is labeled `estimated` and cannot be used as:

- official device time;
- official wall time;
- cross-session time;
- cross-core alignment;
- a claim that CoreX cycles are globally synchronized nanoseconds.

CUDA Events remain an independent whole-kernel or graph-round-trip reference. They do not replace internal cycle records.

## 14. Ordering and semantic validation

Before diagnostic-kernel implementation begins, a tooling preflight must confirm access to:

- emitted TTGIR and LLIR;
- final CoreX ISA or a target disassembler for the generated binary;
- independent eager kernel-event timing;
- register and spill evidence, or another reviewed resource-class report.

The preflight records each capability as `available`, `unavailable`, or `unsupported`. Missing mandatory ordering or perturbation evidence produces an `inconclusive` assessment before deeper attribution work; it is not mislabeled as a hardware capability absence.

Every interpreted region must pass:

```text
source intent
    -> TTGIR
    -> LLIR
    -> final CoreX ISA or disassembly
```

The audit must verify:

1. the start marker remains before the intended operations;
2. the end marker remains after the intended operations;
3. the target scheduler does not move the main region work across either boundary;
4. the end marker retains the declared data dependency on a measured-region result;
5. marker results remain live until the final writeback;
6. markers are not unexpectedly duplicated, merged, or removed;
7. final global stores occur after the measured region;
8. the selected-program predicate and lane-zero writer predicate lower as intended.

Textual order alone does not prove asynchronous operation completion. A region may be reported as execution duration only when the end marker has a validated completion dependency or the target documents equivalent retirement semantics. Otherwise it is reported as an `issue-window` and cannot support claims about dominant execution time.

A region without target-order proof is reported:

```text
status: unavailable
cause: instrumentation-order-unproven
```

Empirical repeatability cannot override this status.

## 15. Resource and perturbation validation

For synthetic and real kernels, compare an uninstrumented control against entry/exit-only and separate one-region diagnostic variants.

Record at least:

- independent kernel-event median and distribution;
- generated register count or reviewed equivalent;
- spill status;
- shared-memory use;
- active-warp or occupancy class when available;
- generated instruction count or bounded ISA diff;
- raw region-cycle distribution.

A timing result is valid only when:

```text
correctness is unchanged
no spill is introduced
register/occupancy class is unchanged
kernel-event overhead is at most 10 percent
region coefficient of variation is at most 5 percent
```

If exact occupancy is unavailable, an unchanged reviewed compiler resource class plus no spill is the minimum acceptable substitute. If neither exact occupancy nor a reviewed substitute exists, the result is `inconclusive`, not valid.

Overhead above 10 percent can still be decomposed for diagnosis, but the instrumented cycles cannot be presented as the original kernel's region timing.

## 16. Experiment matrix

The entire Route C experiment is timeboxed to five engineer-days.

| Stage | Maximum | Stop or downgrade condition |
|---|---:|---|
| Tooling preflight plus synthetic clock/buffer bring-up | 1 day | mandatory ordering/resource evidence unavailable, invalid clock or IDs, writeback failure, or correctness failure |
| IR/ISA ordering and resource audit | 1 day | target ordering unproven, completion semantics unproven, or resource class changed |
| r001/r002 eager exploratory control | 1 day | independent device-time direction fails to reproduce; local-cycle mismatch is recorded rather than treated as automatic profiler failure |
| Coarse and conditional deep attribution | 1 day | coarse gates fail or an individual region variant perturbs the kernel |
| r003 graph replay and final assessment | 1 day | graph failure downgrades the graph experiment but does not erase independently valid eager evidence |

### 16.1 Synthetic A0: monotonicity

A minimal kernel records `t0`, executes fixed work, records `t1`, and flushes both values.

Required observations:

- `t1 > t0` for each sampled warp;
- valid nonzero records;
- independent slots for every local warp present in the launch variant;
- stable repeated deltas;
- correct counter-width and wrap handling.

### 16.2 Synthetic A1: sensitivity

Two correctness-checked variants have a short and long dependency chain. The long variant must have a larger cycle delta beyond the noise band.

### 16.3 Synthetic A2: writeback isolation

Compare uninstrumented, entry/exit-only, and coarse-boundary variants to verify correctness, preamble validity, slot ownership, and writeback placement.

Any synthetic failure stops the real-kernel experiment.

### 16.4 Perturbation calibration

Use interleaved uninstrumented and instrumented device-event measurements. Official competition wall time is not part of this calibration.

A longer synthetic kernel provides same-session cycle-to-event sanity calibration. Raw cycles remain the primary diagnostic observation; they never become campaign authority.

### 16.5 Real B: r001 versus r002

The primary real mechanism case is:

```text
r001: num_warps = 1
r002: num_warps = 2
```

The repository already establishes unchanged mathematics, bitwise output behavior for the tested regime, and improved independent device time.

Route C must show:

1. the independent uninstrumented whole-kernel device-time direction still reproduces;
2. selected-program local cycle spans are reported with their own noise bands;
3. any local region change or non-change is stated without assuming it must follow the whole-grid device-time direction;
4. the different one-warp and two-warp launch structures are explicit in the comparison;
5. unexplained residual and concurrency effects are quantified or declared unavailable rather than hidden;
6. selected-program results are not extrapolated into an unsupported full-grid average.

The r001-to-r002 gain may arise from occupancy, concurrency, or latency hiding even when a sampled local-warp span does not shrink. Such a mismatch is a substantive result, not automatic clock-probe failure. Synthetic dependency-chain experiments remain the hard sensitivity gate.

### 16.6 Real C: coarse attribution

The coarse result evaluates:

- first, middle, and edge key-tile costs;
- padding or edge-tile waste;
- local-warp imbalance;
- prelude and epilogue share;
- stability of the selected program classes.

Examples of potentially actionable facts include:

```text
edge tile costs nearly the same as a full tile
    -> direct evidence for padding waste

local warps are persistently imbalanced
    -> work partition or launch configuration remains a lever

all key tiles have similar cost
    -> the loop body, not prelude or epilogue, dominates

epilogue is unexpectedly large
    -> normalization or final layout/store deserves targeted analysis
```

### 16.7 Real D: conditional deep attribution

The deep result must answer, when valid:

- whether QK and PV dot regions dominate;
- whether the online-softmax dependency chain dominates;
- whether load time is material;
- where padding waste appears;
- whether any valid local region evidence helps characterize the `num_warps=1` to `num_warps=2` improvement, without claiming that local cycles alone explain whole-grid occupancy or concurrency effects.

Operation count and source inspection alone are not accepted as the answer.

### 16.8 Real E: graph replay

`diag_r003_graph` captures the instrumented r002-equivalent kernel using the existing graph route.

The buffer is allocated before capture. After each replay, the host synchronizes and reads the fixed slots before the next replay. Host collection cost is outside the recorded device cycle spans.

The result compares eager and graph-replay raw region cycles. A valuable graph result demonstrates that:

- graph replay produces valid region records;
- region ordering and cycles remain stable;
- internal kernel cycles do not absorb the graph/frontend round trip;
- instrumentation recovers device-internal evidence despite missing Kineto graph-interior kernel events.

## 17. Normalized experiment result

Each experiment emits one compact structured result. Measurements that are unavailable are represented by a status and cause, never by a fabricated zero.

```json
{
  "experiment_id": "bi150-mm-attn-r002-qk-score",
  "environment": {
    "device": "Iluvatar BI-V150",
    "corex": "4.4.0",
    "triton": "3.1.0+corex.4.4.0",
    "target": "ivcore11",
    "warp_size": 64
  },
  "variant": {
    "kernel_variant": "r002-nw2",
    "num_warps": 2,
    "execution_mode": "eager"
  },
  "source": {
    "accepted_kernel_sha256": "...",
    "accepted_host_sha256": null,
    "diagnostic_sha256": "..."
  },
  "instrumentation": {
    "mode": "deep-one-region",
    "region_id": "qk-score",
    "selected_pids": [0, 16, 32],
    "selected_local_warps": [0, 1],
    "time_unit": "raw-cycle",
    "storage": "one-pair-register-then-global-flush",
    "measurement_semantics": "execution-duration"
  },
  "validation": {
    "correctness": {"status": "pass", "evidence": ["..."]},
    "ttgir_ordering": {"status": "pass", "evidence": ["..."]},
    "llir_ordering": {"status": "pass", "evidence": ["..."]},
    "isa_ordering": {"status": "pass", "evidence": ["..."]},
    "completion_dependency": {"status": "pass", "evidence": ["..."]},
    "graph_capture": {"status": "not-applicable", "evidence": []}
  },
  "perturbation": {
    "kernel_event_overhead_pct": {"status": "observed", "value": 4.2, "unit": "percent", "evidence": ["..."]},
    "register_delta": {"status": "observed", "value": 2, "unit": "registers-per-thread", "evidence": ["..."]},
    "spill_delta": {"status": "observed", "value": 0, "unit": "bytes", "evidence": ["..."]},
    "occupancy_class_changed": {"status": "observed", "value": false, "evidence": ["..."]}
  },
  "regions": [
    {
      "pid": 0,
      "query_tile_class": "first",
      "local_warp": 0,
      "status": "observed",
      "cause": "none",
      "measurement_semantics": "execution-duration",
      "start_boundary": "qk-score-start",
      "end_boundary": "qk-score-end",
      "raw_cycle_median": 1234,
      "raw_cycle_p10": 1200,
      "raw_cycle_p90": 1280,
      "coefficient_of_variation": 0.02,
      "estimated_us": {"status": "estimated", "value": 1.5, "unit": "us"},
      "evidence": ["..."]
    }
  ],
  "status_causes": [],
  "experiment_status": "valid"
}
```

Graph variants record both the accepted mathematical-kernel source hash and accepted host graph-route source hash. Per-experiment `experiment_status` is limited to `valid`, `invalid`, `unsupported`, or `inconclusive`. The overall Route C classification is written once in `assessment.md` after all applicable experiments complete.

`measurement_semantics` is `execution-duration` only after completion-dependency or target-retirement validation. Otherwise it is `issue-window`.

Each region entry has `status: observed|unavailable|invalid` plus a stable cause. Numeric cycle fields are present only for `observed`; unwritten, unavailable, or invalid PID/warp slots do not use zero as a placeholder. Each entry retains selected PID and query-tile class, local warp, boundaries, optional estimated time, validity, and evidence paths.

## 18. Final classification

Route C permits exactly five final classifications.

### 18.1 `valuable`

All facts used by the conclusion pass their measurement gates, and at least one of these outcomes occurs:

- valid local-region evidence materially characterizes the r001-to-r002 change without claiming to explain whole-grid occupancy from local cycles alone;
- padding or local-warp imbalance receives direct diagnostic evidence;
- a valid dominant-region result differs from the current source-based hypothesis;
- graph replay gains stable internal attribution absent from Kineto;
- valid internal timing supports a diagnostic upper-bound argument that remaining device-only work is unlikely to move official wall time materially.

`valuable` means valuable enough to request source and perform a separate port-cost assessment. It does not authorize productization, adoption, stop, or an official campaign-report change. A failed optional graph experiment does not erase independently valid eager value; its limitation remains explicit.

### 18.2 `technically-valid-low-value`

Clock, ordering, correctness, and perturbation gates pass, but the result only repeats existing whole-kernel evidence and changes no optimization priority.

This result rejects productization for current KernelSwift needs.

### 18.3 `perturbation-invalid`

The probe demonstrably changes occupancy or resource class, introduces spill, exceeds the overhead limit, alters the measured critical path through its dependency machinery, or exceeds the stability limit.

This result does not prove that compiler-integrated KPerfIR lacks value. It proves that this Route C measurement perturbs the original kernel too strongly to answer the question reliably.

### 18.4 `unsupported`

A required platform primitive is demonstrated absent or rejected, such as the local clock, thread/warp identity, legal writeback, or basic graph-capture compatibility when graph profiling is the experiment under test.

An optional graph-only `unsupported` result does not force the whole assessment to `unsupported` when valid eager evidence exists.

### 18.5 `inconclusive`

The timebox expires, mandatory audit tooling or resource evidence is unavailable, target ordering cannot be proved, marker reordering invalidates the intended bracket, only `issue-window` evidence exists where execution duration is required, completed experiments conflict without a valid attribution, or only partial evidence exists without enough support for either value or low-value classification.

`inconclusive` preserves completed valid observations and names the missing gate. It is distinct from demonstrated platform `unsupported` and measured `perturbation-invalid`.

## 19. Upgrade gate for a port-cost assessment and maintained Proton Iluvatar backend

Route C measures incremental value, not compiler-port cost. A production design begins only after a separate port-cost assessment and all applicable gates pass.

### 19.1 Value gate

Route C must report `valuable`. This opens a source request and port-cost assessment; it does not by itself authorize backend implementation. Technical feasibility without decision value is insufficient.

The port-cost assessment must estimate common Proton reuse, Iluvatar-specific lowering work, Triton 3.1 backport work, runtime/package work, test scope, and expected version-maintenance ownership.

### 19.2 Source gate

Obtain the exact CoreX Triton source revision corresponding to the deployed wheel, including:

- Iluvatar TritonGPU-to-LLVM conversion sources;
- target-info and utility headers;
- build instructions;
- package construction;
- runtime tests;
- expected compiler and SDK versions.

Installed LLVM/MLIR libraries and a vendor `libtriton.so` do not satisfy this gate.

### 19.3 Backend substrate gate

Confirm support for:

- local 64-bit cycle reads;
- local thread and warp identity;
- legal global scratch and final writeback;
- target ordering or scheduler-barrier semantics;
- graph-compatible profile buffers;
- a physical processor ID, or an explicit v0 limitation excluding cross-core timelines;
- synchronized global time only if cross-core timelines are later required.

### 19.4 Ownership gate

Preferred ownership order is:

1. vendor-maintained Iluvatar Proton lowering;
2. public CoreX Triton fork with maintained Proton integration;
3. KernelSwift-maintained source fork pinned to a reviewed CoreX version.

A binary ABI plugin against the installed vendor `libtriton.so` is not an acceptable production architecture.

### 19.5 Generality gate

After a positive port-cost assessment and before committing to a reusable backend, either:

- `mm_encoder_attention` must demonstrate both graph-interior recovery and actionable region attribution; or
- a second dominant BI150 Triton kernel must reproduce incremental value.

A second-kernel experiment is outside Route C's five-day timebox.

## 20. Risks and mitigations

### 20.1 Inline assembly syntax differs from NVIDIA PTX

Mitigation: synthetic bring-up validates the exact accepted `clock64` form before real-kernel work.

### 20.2 Local clocks are not globally synchronized

Mitigation: report local-warp deltas only; do not build cross-core timelines.

### 20.3 Even one retained timestamp pair can increase register pressure

Mitigation: compile one measured region per diagnostic variant, retain only one pair, compare resource metadata and occupancy class, and reject a variant that changes the class or introduces spill.

### 20.4 Final scheduler moves operations across markers or clocks before completion

Mitigation: require final CoreX ISA or disassembly ordering proof plus a validated end-marker data dependency or documented target-retirement semantics. Otherwise report only an `issue-window`.

### 20.5 Selected programs are not representative of all programs

Mitigation: sample first, middle, and edge query tiles; name the result as selected-program evidence; do not extrapolate a full-grid average.

### 20.6 Host collection changes graph timing

Mitigation: clock spans are recorded inside the kernel; host synchronization and buffer reads happen after replay and outside the recorded spans.

### 20.7 Diagnostic work contaminates official evidence

Mitigation: immutable accepted files, separate function names, separate hashes, separate cache namespace, separate experiment directory, and no campaign report rewrite.

### 20.8 A useful result triggers premature productization

Mitigation: require the source, substrate, ownership, and generality gates before a production design.

## 21. Acceptance criteria for this design

The design is complete when it establishes all of the following contracts:

1. Route C is a disposable value probe and not a production profiler.
2. Accepted competition candidates remain byte-identical.
3. BI150 evidence is limited to local SIMT warp/program cycle deltas.
4. Selected programs and every local warp present in each launch variant are sampled explicitly.
5. Coarse profiling precedes conditional deep profiling.
6. Each diagnostic variant measures one region pair, retains only that pair in registers, and flushes after the measured region.
7. Raw cycles are primary diagnostic observations; converted time is estimated.
8. TTGIR, LLIR, final target ordering, and completion semantics are mandatory evidence for execution-duration claims.
9. Correctness, resource class, spill, overhead, and stability are hard gates.
10. Synthetic dependency chains are the hard sensitivity control; r001 versus r002 is an exploratory real mechanism case that may expose concurrency effects not visible in local cycles.
11. r003 graph replay is the graph-interior visibility test.
12. The experiment is timeboxed to five engineer-days and may finish `inconclusive`.
13. Final output uses exactly one of five classifications.
14. A maintained Proton Iluvatar backend requires positive value, a separate port-cost assessment, and source, substrate, ownership, and generality gates.
15. Ascend and other non-SIMT profiler research remain out of scope.
