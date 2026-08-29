# KernelWiki Index

KernelWiki provides a reviewed, source-backed Ascend seed corpus for Designer exploration. Cards are advisory and do not constitute exact-profile Coder recipes.

## Generated query views

- [By problem](queries/by-problem.md)
- [By technique](queries/by-technique.md)
- [By hardware feature](queries/by-hardware-feature.md)
- [By kernel type](queries/by-kernel-type.md)
- [By language](queries/by-language.md)
- [By target](queries/by-target.md)
- [By source repository](queries/by-source-repo.md)
- [By version](queries/by-version.md)
- [By evidence level](queries/by-evidence-level.md)

## Reviewed seed Cards

### Hardware

- [Ascend execution and memory evidence boundary](wiki/hardware/ascend-execution-and-memory.md)

### Languages and runtimes

- [Triton language on the Ascend backend](wiki/languages/triton-ascend-backend.md)
- [MSKL kernel authoring and invocation](wiki/languages/mskl-kernel-authoring.md)
- [Ascend C programming-model evidence boundary](wiki/languages/ascendc-programming-model.md)
- [Ascend kernel integration boundaries](wiki/runtimes/ascend-kernel-integration.md)

### Techniques and patterns

- [Kernel fusion as a review hypothesis](wiki/techniques/kernel-fusion.md)
- [Tiling and work partitioning](wiki/techniques/tiling-and-work-partitioning.md)
- [Top-k selection and reduction evidence boundary](wiki/techniques/topk-selection-and-reduction.md)
- [Launch-bound materialization pattern](wiki/patterns/launch-bound-materialization.md)
- [Device win with wall-time loss](wiki/patterns/device-win-wall-loss.md)
- [Ascend capability-gap handling](wiki/patterns/ascend-capability-gap.md)

### Measurement

- [CANN device-time attribution boundary](wiki/measurement/cann-device-attribution.md)

## Reviewed Sources

- [Triton Ascend pinned README](sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md)
- [vLLM Ascend pinned README](sources/commits/vllm-ascend/7702ccd7d8dea6b4dabdacb0118adb522dedbec7.md)
- [MSKL pinned user guide](sources/docs/source-mskl-user-guide-f9fbf4d2.md)
- [Official Ascend C CANN 9.0.0-beta.1 metadata](sources/docs/source-ascendc-programming-model-cann-900beta1.md)

PR 814 remains `defer` in the [reviewed candidate ledger](candidates/repos/vllm-ascend.yaml) because complete changed-file accounting and a publication license decision were not established for this task.

## Role-aware query

- [Role-query contract](references/role-query-contract.md)
- Designer: broad classified evidence with admission before ranking.
- Coder: exact profile/runtime/Sketch-bound guidance only; Card and asset admission remain separate.
- Missing canonical AscendC authority is expected to return empty implementation guidance without fallback.
- Saved receipts are advisory and do not mutate prompts, campaigns, project state, or `kernel-opt-loop`.

## Evaluation boundary

- [Sealed holdout manifest](data/evaluation-holdouts.yaml)
- [Evaluation protocol](references/evaluation-protocol.md)

## Standalone maintenance

```text
discover candidates -> curator edits reviewed ledger -> capture immutable Source -> author/review generic Card -> validate -> generate views -> review diff -> commit
```

Run the complete hardware-free contract suite and production smoke commands from [`README.md`](README.md) before committing corpus or generated-output changes.

## Deferred plans

Phase C role-aware query admission is available through the documented standalone query context. Phase D offline knowledge lift and a Phase E `kernel-opt-loop` adapter remain excluded.
