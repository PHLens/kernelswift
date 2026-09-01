# BI150 KPerfIR Value Probe

This directory contains a disposable, diagnostic-only BI150 Triton profiling experiment. It does not change accepted competition kernels, official benchmark timing, Verifier authority, SOL, or KernelWiki.

Ordinary raw device outputs and compiler artifacts belong under `artifacts/` and remain gitignored. The curated [`evidence-archive/`](evidence-archive/README.md) is the tracked exception: it contains selected byte-exact evidence, a deterministic [`manifest.json`](evidence-archive/manifest.json), and third-party provenance notes collected for review before the BI150 server's expected reclamation. Verify the tracked archive with [`scripts/verify_evidence_archive.py`](scripts/verify_evidence_archive.py):

```bash
python3 experiments/bi150-kperfir-value/scripts/verify_evidence_archive.py
```

Remote access must use a locally configured `BI150_SSH` alias or environment variable; credentials and private endpoints must never be committed.

Accepted kernel sources and `auto_bench.py` are immutable and checked by SHA256 before device work. The experiment is timeboxed to five engineer-days and ends with one of: `valuable`, `technically-valid-low-value`, `perturbation-invalid`, `unsupported`, or `inconclusive`.

## Final status

Both bounded runs ended `inconclusive`. The original PTX-inline-assembly result remains frozen in [`assessment.md`](assessment.md) and [`evidence-summary.json`](evidence-summary.json) (SHA256 `5b68295dc8c1b17163f010ff706fc55a7c0171b1a6b3f9d350b85d8a7e89f598`). The external-clock rerun is frozen separately in [`assessment-v2.md`](assessment-v2.md) and [`evidence-summary-v2.json`](evidence-summary-v2.json) (SHA256 `1cbc1d6fab926e43c3861088c8bb9ab78e9c4d20d7efda16a2a139960416088b`). V2 proved external CoreX bitcode linkage, but LLVM optimized away the completion dependency before the end clock; no attention stages were run.

## Triton profiler exploration

See [`triton-profiler-exploration.md`](triton-profiler-exploration.md) for the exact CoreX 4.4.0/Triton 3.1.0 profiler snapshot and KernelSwift implications. The Python API and `proton`/`proton-viewer` entry points are packaged, but native `libproton` is absent; launch-level Proton, compiler evidence, whole-kernel timing, and experimental intra-kernel issue windows must remain separate optional measurement layers.

## CoreX external clock helper

On the matched BI150/CoreX 4.4.0 host, build the disposable device-only helper before importing Torch or Triton:

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
python3 experiments/bi150-kperfir-value/scripts/build_corex_clock.py \
  --corex-root /usr/local/corex-4.4.0 \
  --output-dir experiments/bi150-kperfir-value/artifacts/external-clock/helper
```

The CLI verifies all accepted-source SHA256 values first, then compiles `device/corex_clock.cu` for `ivcore11` into `corex-clock.bc` and `corex-clock.ll`. It validates the two external helper symbols, the `llvm.nvvm.read.ptx.sreg.clock64` intrinsic, and target triple `bi-iluvatar-ilurt`, then writes `clock-helper.json`. These raw build artifacts are gitignored and are valid only for the matched CoreX toolchain.

## External-clock terminal evidence

The final synthetic result is terminal `inconclusive: end-dependency-optimized-away`. A compile-only prelaunch audit proved that the helper bitcode linked and materialized two clock intrinsics, but CoreX LLVM placed both clock reads before the dependency chain. Tasks 3 and 4 were therefore skipped and no attention cycle record exists.

Regenerate the deterministic v2 summary from the collected raw evidence with:

```bash
python3 experiments/bi150-kperfir-value/scripts/build_evidence_summary_v2.py \
  --helper experiments/bi150-kperfir-value/artifacts/external-clock/helper/clock-helper.json \
  --synthetic experiments/bi150-kperfir-value/artifacts/external-clock/synthetic/result.json \
  --attention experiments/bi150-kperfir-value/artifacts/external-clock/attention/result.json \
  --output experiments/bi150-kperfir-value/evidence-summary-v2.json
```

A missing attention result is accepted only because synthetic qualification is not `valid`. The builder verifies the accepted-source ledger, evidence commit, helper metadata/bitcode binding, linked LLIR path/hash/size, and program-level `issue-window` limitations.

The earlier noinline helper hung during its first kernel execution. After terminating that run, the container-visible GPU context was unusable, so a later smoke stall is not attributed to the inline-assembly variant. The terminal conclusion comes from the independent compile-only LLVM audit.

## Minimal Stage-0 probe

On the matched BI150/CoreX host, run:

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
python3 experiments/bi150-kperfir-value/scripts/preflight_corex.py \
  --output-dir experiments/bi150-kperfir-value/artifacts/preflight
```

The probe checks accepted-source hashes before importing Torch/Triton, compiles and verifies a stock vector-add kernel, records compiler artifacts/resources and CUDA Event timing, and attempts final disassembly with CoreX `llvm-objdump` and `ixobjdump`. A missing final disassembler is reported as `inconclusive`, not `unsupported`.

## External clock64 qualification probe

After building the helper, run the current one-program external-bitcode probe with:

```bash
python3 experiments/bi150-kperfir-value/scripts/clock64_probe.py \
  --clock-bitcode experiments/bi150-kperfir-value/artifacts/external-clock/helper/corex-clock.bc \
  --clock-metadata experiments/bi150-kperfir-value/artifacts/external-clock/helper/clock-helper.json \
  --output experiments/bi150-kperfir-value/artifacts/external-clock/synthetic/result.json \
  --samples 50
```

The probe first performs a compile-only linked-LLIR audit before allocating CUDA tensors. It requires a chain-derived completion dependency between the start and end clock reads. On the frozen evidence commit, CoreX LLVM optimized that dependency away, so the probe stopped with `inconclusive: end-dependency-optimized-away`, emitted no cycle regions, and did not run attention instrumentation.

The original PTX-inline-assembly/per-warp probe is historical v1 evidence only. Its rejected `mov.u64 $0, %clock64;` result remains frozen in `evidence-summary.json` and `assessment.md`; the current `clock64_probe.py` no longer implements that path.
