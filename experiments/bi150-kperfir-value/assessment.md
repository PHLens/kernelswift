# BI150 Route C Minimal Experiment Assessment

**Final classification: `inconclusive`**

## Evidence identity

- Evidence commit: `e61443606746959ea537a20190308d20af93234c`
- Canonical summary: `experiments/bi150-kperfir-value/evidence-summary.json`
- Summary SHA256: `5b68295dc8c1b17163f010ff706fc55a7c0171b1a6b3f9d350b85d8a7e89f598`
- Raw preflight evidence: `experiments/bi150-kperfir-value/artifacts/evidence/preflight/stage0-result.json`, SHA256 `41897cc64ac739128c6d04aa4aa390bc01367c7ad23df8a3f6f9aee53bd18f4b`
- Raw clock64 evidence: `experiments/bi150-kperfir-value/artifacts/evidence/clock64/result.json`, SHA256 `6699c0535ee78b36141aa8d86c64c635261e92db1a5fc859faefe3b2342fba7b`

The accepted-source ledger remained immutable:

- `auto_bench.py`: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- `kernels/track1-triton/mm_encoder_attention/base.py`: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- `triton_mm_encoder_attention_e2_001.py`: `4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2`
- `triton_mm_encoder_attention_e2_002.py`: `cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078`
- `triton_mm_encoder_attention_e2_003.py`: `d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81`

## Findings

The matched BI150/CoreX stock Triton path works and is correct. The stock vector-add emitted TTGIR, LLIR, and cubin artifacts, used 13 registers with 0 spills and 0 shared bytes, and produced a CUDA Event median of approximately 5.63 microseconds.

Final target ordering could not be audited. CoreX `llvm-objdump` lacks a BI disassembler for the direct Triton cubin, while `ixobjdump --sass` rejects the direct cubin because it does not contain the expected Fatbin format.

The canonical inline-assembly form `mov.u64 $0, %clock64;` was rejected by the target assembler. This establishes failure of that syntax path only; it does not establish that BI150 lacks a usable clock primitive.

No interpretable clock path was obtained. Consequently, the experiment produced no raw region timing and did not run or interpret the real attention kernel, r001/r002 sensitivity case, or graph replay case.

## Decision

Route C did not establish incremental profiler value beyond existing whole-kernel evidence. It therefore does not yet justify implementing or maintaining a Proton Iluvatar backend, nor beginning a backend port-cost assessment.

The bounded experiment stopped early within its five-engineer-day timebox because final ISA evidence was unavailable and the attempted clock64 syntax was rejected. The next low-cost action is to obtain at least one of:

1. a vendor-supported Triton/CoreX clock intrinsic or lowering;
2. a supported final-ISA disassembly path for direct Triton cubins; or
3. the matching CoreX Triton compiler source and build instructions.

After obtaining that support, rerun this probe before assessing maintained-backend port cost.

## Authority boundary

This diagnostic experiment made no changes to accepted kernels, official benchmark timing, correctness/adoption/stop authority, Verifier reports, SOL, KernelWiki, or campaign conclusions. Its findings remain isolated Route C diagnostic evidence.
