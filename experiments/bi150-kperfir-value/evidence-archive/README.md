# BI150 Route C Evidence Archive

This directory is a curated, tracked copy of the disposable BI150 Route C artifacts collected while the matched server was still available, ahead of its expected reclamation. Payload files were copied byte-for-byte from the staging tree `/tmp/bi150-route-c-remote-snapshots`; `manifest.json` records the original source root/commit, source path, role, status, byte count, and SHA256 for every archived file other than the manifest itself.

## Authority boundary

This archive is diagnostic evidence only. It does not change accepted kernels, official benchmark timing, correctness/adoption/stop authority, Verifier conclusions, SOL, KernelWiki, or campaign results. The intermediate external-clock failures are retained as implementation history and are **not** final authority. The final Route C result remains the `8e4d99b` terminal synthetic result: external LLVM bitcode linkage worked, but LLVM optimized away the completion dependency before the end clock, so no attention-region cycle record was interpreted.

## Manifest source identity

Every manifest entry has a `source_commit` and normalized relative `source_path`. For repository-derived files, `source_commit` is the full Git SHA of the source tree. For curated archive documentation or installed-distribution snapshots, it is a stable origin/version label such as `archive-curation` or `installed-corex-4.4.0-triton-3.1.0`, not a Git SHA. The top-level `source_commit_semantics` field is the verifier-required definition of these sentinel rules.

## Contents

### `stage0-c3d006b/`

The Stage-0 stock vector-add run from commit `c3d006b779c9c0f915a53bad0290dd6c4b70f1c5`:

- `stage0-result.json` — correctness, CUDA Event timing, resources, compiler keys, and final-disassembly attempts;
- `compiler/stock-vector-add.ttgir`;
- `compiler/stock-vector-add.llir`;
- `compiler/stock-vector-add.cubin`.

The kernel compiled and ran correctly. The result remained `inconclusive` only because neither validated tool produced final BI instruction disassembly.

### `v1-e614436/`

The frozen v1 canonical PTX-inline-assembly clock probe from commit `e61443606746959ea537a20190308d20af93234c`. The BI assembler rejected `mov.u64 $0, %clock64;`; this is diagnostic evidence about source syntax, not evidence that BI150 lacks a clock primitive.

### `external-noinline-4d74549/`

The first valid external CoreX helper build from commit `4d74549e6747e289ee80989bcdd7fe45770ef07c`. It contains the helper metadata, LLVM text, and bitcode for the `noinline` start/end helper ABI. The build itself was valid; a later runtime attempt with retained device calls hung, so these files are historical build evidence rather than the final measurement path.

### `external-history-edb3a57/`

Two intermediate diagnostic failures from the initial external-clock probe:

- `initial-compile-result.json` — the JIT helper used a dictionary literal that the installed Triton AST rejected;
- `scalar-mask-diagnostic-result.json` — after fixing dispatch construction, a vector mask was incorrectly applied to scalar pointers.

Both are intentionally labeled diagnostic history. They explain implementation corrections and must not be read as hardware or profiler capability conclusions.

### `final-8e4d99b/`

The final frozen external-clock evidence from commit `8e4d99b89407ecb3d35bac1c276d3cc73de27699`:

- final helper metadata, LLVM text, and bitcode;
- terminal synthetic `result.json`;
- compile-only linked LLIR `clock64-static-dependency-audit.ll`.

The linked module contains two materialized `llvm.nvvm.read.ptx.sreg.clock64()` calls and no retained helper runtime calls. However, both clock reads appear before the 16-step dependency chain. The terminal cause is `end-dependency-optimized-away`; Tasks 3 and 4 were not run and no attention cycle records exist.

### `triton-profiler-package/`

Byte-exact snapshots of the installed CoreX 4.4.0 / Triton 3.1.0 `triton.profiler` Python package plus package/native/CLI/distribution-metadata probes. Python bytecode (`.pyc` and `__pycache__`) is intentionally excluded. See [`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md) for the provenance boundary of the unmodified profiler Python distribution evidence.

## Triton profiler exploration and implications

### What the installed Python surface intends to provide

The installed sources expose Triton Proton-style profiling:

- `profile.py` selects `cupti` for a CUDA backend, starts/finalizes sessions, and supports `shadow` or `python` context with tree data;
- `scope.py` exposes explicit scopes and operation scopes with metrics/properties;
- `hook.py` attaches `CompiledKernel.launch_enter_hook` and `launch_exit_hook`, uses launch metadata for operation names, and forwards available `flops8/16/32/64` and `bytes` metrics;
- `viewer.py` consumes the emitted Hatchet-style profile data;
- the `proton` and `proton-viewer` command entry points are installed.

This design is useful for hierarchical operation/kernel attribution and launch metadata. It is conceptually complementary to CUDA Event timing and compiler resource records.

### What is actually available on the matched CoreX installation

The package is incomplete at runtime:

- Triton version is `3.1.0` under `/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton`;
- `triton.profiler` import fails with `ModuleNotFoundError: No module named 'triton._C.libproton'`;
- no separate native `libproton` extension was found;
- the installed `proton` CLI exits with the same missing-module error;
- `libtriton.so` exists, but that does not satisfy the Python package's explicit `triton._C.libproton` dependency.

Therefore Proton was **not a usable profiler on this matched CoreX image**, despite the Python package and CLI entry points being present.

### What Proton would and would not answer

If a matching native extension were supplied, the captured sources indicate that Proton could provide coarse kernel/operation scopes, hierarchical context, launch metadata, and aggregate backend activity. That could improve attribution between host-level calls and Triton kernel launches.

It would not by itself establish the Route C questions that required an in-kernel completion marker:

- selected-program issue windows;
- full versus edge key-tile internal cost;
- QK versus softmax versus PV regions inside one fused kernel;
- per-warp imbalance;
- documented BI instruction ordering or final-ISA duration.

Those require either a compiler-supported timestamp/profiler operation with a real data/completion dependency, a vendor profiler API that exposes internal program/region events, or documented target ordering semantics. A no-argument clock wrapper is insufficient because optimization may issue both reads before the work.

### KernelSwift recommendation

Treat Triton Proton as an optional **coarse profiling layer**, gated by an explicit capability probe that checks the Python import, native extension, CLI, output production, and backend compatibility. Continue using compiler artifacts/resources and CUDA Events as independent layers. For fused-kernel internals, prioritize a vendor/compiler-supported token-dependent timestamp or KPerfIR/profiler hook; do not infer region duration from source-level clock wrappers, guessed BI assembly, or patched generated LLIR.

## Deliberate omission

The `d583012` missing-bitcode smoke result is omitted. It was produced by a command sequence in which helper construction had already failed, so the probe only reported that the expected `.bc` file was absent. It adds no information beyond the helper-build failure and is not useful diagnostic evidence.

A later empty-inline-assembly smoke stall is also not promoted to an independent backend conclusion: after the earlier `noinline` runtime hang, the container-visible GPU context was already unusable and could not be reset from inside the container.

## Verification

Run:

```bash
python3 experiments/bi150-kperfir-value/scripts/verify_evidence_archive.py
```

The verifier checks every manifest entry, rejects missing or extra files, verifies byte counts and SHA256 values, and rejects archived `.pyc`/`__pycache__` content.
