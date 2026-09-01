# BI150 Route C External-Clock Rerun Assessment

**Final classification: `inconclusive`**

This v2 assessment is additive. The frozen v1 `evidence-summary.json` and `assessment.md` remain unchanged.

## Evidence identity

- Raw evidence commit: `8e4d99b89407ecb3d35bac1c276d3cc73de27699`
- Canonical v2 summary: `experiments/bi150-kperfir-value/evidence-summary-v2.json`
- V2 summary SHA256: `1cbc1d6fab926e43c3861088c8bb9ab78e9c4d20d7efda16a2a139960416088b`
- Helper metadata: `experiments/bi150-kperfir-value/artifacts/external-clock/helper/clock-helper.json`, SHA256 `83bcf97370791b65b7b2a0cd9608d4c681f83c3eb3c5ca50ad6973de89b9f648`
- Helper bitcode: `experiments/bi150-kperfir-value/artifacts/external-clock/helper/corex-clock.bc`, SHA256 `3855554cfcf6e42a3239448f23ca96af8c783b48b66bcc587fea72b8595030ae`
- Synthetic qualification: `experiments/bi150-kperfir-value/artifacts/external-clock/synthetic/result.json`, SHA256 `c621cee0704b95fe2c0426d115d2b9506290443d5c22fe7a8f7a42255a749723`
- Compile-only linked LLIR: `experiments/bi150-kperfir-value/artifacts/external-clock/synthetic/linked-llir/clock64-static-dependency-audit.ll`, SHA256 `ae558835d4611a89f5c3536262ec73e7c7ff9081fd5f2be50c0efd293ff6d542`

The accepted-source ledger remained unchanged and exactly matched the approved hashes for r001, r002, r003, base, and `auto_bench.py`.

## What was proven

The external-library capability is real. CoreX Clang 4.4.0 produced valid `ivcore11` device bitcode containing the two helper symbols and `llvm.nvvm.read.ptx.sreg.clock64()`. CoreX Triton accepted that bitcode through `extern_libs`, linked it into the diagnostic module, and materialized two clock intrinsics in the kernel LLIR. No unresolved runtime helper call remained in the final compile-only audit.

This is a new **partial measurement capability**: external CoreX device bitcode and the CUDA C clock builtin can be introduced into a Triton compilation without adding a lowering or editing generated LLIR.

## What was not proven

An honest completion-dependent end marker was not preserved.

The final helper placed `clock64()` in two token-controlled source branches. CoreX LLVM optimization proved the branches equivalent, collapsed the helper to `clock64() + (token & 1)`, and emitted the kernel's two clock intrinsics adjacently before the dependency chain. In the frozen LLIR, the start and end reads appear at lines 15–16, while the 16-step chain begins later at line 22. The static audit therefore reported:

- `status: optimized-away`;
- `chain_dependency_verified: false`;
- `chain_operation_count: 0` on the end-marker dependency path;
- `runtime_seed_dependency: false`;
- terminal cause `end-dependency-optimized-away`.

Reading a clock is not enough: without the completion dependency, the two values do not bracket the intended work. Reporting their delta as an attention-region duration or even as a useful region issue window would be misleading.

## Runtime history and attribution boundary

A prior `noinline` helper version linked successfully but hung on its first BI150 kernel execution while the host waited in `torch.cuda.synchronize()`. The process was terminated after approximately seven minutes.

After that termination, the GPU context visible inside the container was unusable: even a minimal CUDA allocation stalled. A container-local reset was unavailable because the owning host PIDs existed outside the container PID namespace. Consequently, the later empty-inline-assembly smoke stall did **not** provide an independent result and is not attributed to inline assembly or to that barrier mechanism.

The terminal conclusion instead comes from the subsequent compile-only token-control audit, which did not allocate CUDA tensors and directly demonstrated the LLVM dependency collapse.

## Attention stages and KernelSwift value

Tasks 3 and 4 were correctly skipped because Stage A never qualified an interpretable end marker. Therefore the rerun produced:

- no attention cycle records;
- no r001/r002 selected-program comparison;
- no full-versus-edge key-tile comparison;
- no QK/softmax/PV attribution;
- no graph-replay internal record.

The three value layers are therefore:

1. **New measurement capability:** partial — external bitcode linkage and clock intrinsic materialization were proven.
2. **New kernel fact:** none — no honest attention cycle observation was obtained.
3. **New optimization decision:** none — existing KernelSwift priorities are unchanged.

This result is not `unsupported`: BI150 has a working CUDA C `clock64()` primitive, and external linkage was proven. It is not `technically-valid-low-value`: the measurement path never passed synthetic qualification, so there was no technically valid attention observation whose usefulness could be judged. The correct classification is `inconclusive`.

## Decision

Route C still does not establish incremental KernelSwift optimization value beyond existing whole-kernel, source, graph, and resource evidence. No maintained Proton/Iluvatar backend, port-cost assessment, attention-instrumentation matrix, LLIR surgery, guessed BI assembly, or new lowering is justified by this run.

A future retry requires a compiler-supported primitive that accepts a completion token or documented target ordering semantics that survive optimization. Repeating source-level no-argument clock wrappers is not sufficient.

## Authority boundary

This diagnostic rerun changed no accepted kernel, official benchmark timing, correctness/adoption/stop authority, Verifier report, SOL, KernelWiki, or campaign conclusion. `submissions/` and the frozen v1 evidence remain outside this v2 change.
