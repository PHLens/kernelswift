# BI150 KPerfIR Value Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a five-engineer-day, diagnostic-only BI150 Route C experiment that determines whether local-warp intra-kernel cycle windows add actionable KernelSwift value beyond current CoreX profiler and CUDA Event evidence.

**Architecture:** Keep accepted competition files immutable. Add an isolated `experiments/bi150-kperfir-value/` tree containing status-bearing result contracts, source guards, a Stage-0 CoreX observability gate, an exact compilation-manifest/audit path, a `clock64` synthetic sensitivity probe, one-region-at-a-time instrumented attention copies, eager and graph runners, and a final assessment. Hardware execution is conditional: if final ISA/disassembly or reviewed resource evidence is unavailable, stop before diagnostic-kernel implementation and classify the probe `inconclusive` rather than weakening the approved gate.

**Tech Stack:** Python 3.10 standard library, `unittest`, PyTorch 2.7.1+corex.4.4.0, Triton 3.1.0+corex.4.4.0, `tl.inline_asm_elementwise`, CUDA Events, Triton TTGIR/LLIR/binary artifacts, CoreX `llvm-objdump`/`ixobjdump` when supported, JSON.

**Spec:** `docs/superpowers/specs/2026-08-30-bi150-kperfir-value-probe-design.md`

## Global Constraints

- Do not modify or stage `submissions/`.
- Do not modify these accepted sources:
  - `kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_001.py` — SHA256 `4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2`.
  - `kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_002.py` — SHA256 `cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078`.
  - `kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_003.py` — SHA256 `d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81`.
  - `kernels/track1-triton/mm_encoder_attention/base.py` — SHA256 `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`.
  - `auto_bench.py` — SHA256 `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`.
- Route C is diagnostic only. It must not change official correctness, timing, adoption, stop authority, Verifier reports, implementation profiles, SOL, or KernelWiki.
- Store raw device outputs, compiler artifacts, disassembly, profile-buffer dumps, and remote logs under `experiments/bi150-kperfir-value/artifacts/`; keep them gitignored.
- Never write SSH endpoints, usernames, passwords, private hostnames, or credentials into repository files, fixtures, commits, or assessment text. Execution uses a locally configured `BI150_SSH` environment variable or secure SSH alias.
- Every diagnostic specialization measures one region start/end pair. Do not retain a multi-boundary timeline.
- Sample PIDs `0`, `16`, and `32`. Record every local warp present: `[0]` for `num_warps=1`, `[0, 1]` for `num_warps=2`.
- Raw local `clock64` cycles are primary diagnostic observations. Estimated microseconds are optional and never authoritative.
- An execution-duration claim requires source, TTGIR, LLIR, final CoreX ISA/disassembly ordering evidence plus completion-dependency or documented retirement semantics. Otherwise report `measurement_semantics: issue-window`.
- Hard validity gates are unchanged correctness, no new spill, unchanged reviewed resource/occupancy class, at most 10% independent kernel-event overhead, and at most 5% cycle coefficient of variation.
- Resource-class rule: exact equality of `n_regs`, `n_spills`, `shared_bytes` as reported by `CompiledKernel.metadata.shared`, launch threads, and launch warps is an acceptable conservative substitute for occupancy evidence. If resources differ and no documented occupancy API/report proves the same class, the measurement is `inconclusive`; introduced spill is `perturbation-invalid`.
- Synthetic dependency-chain probes are the hard sensitivity control. The r001/r002 comparison is exploratory because whole-grid occupancy/concurrency gains need not reduce a selected local-warp span.
- If a mandatory gate fails or the timebox expires, stop deeper work and write `unsupported`, `perturbation-invalid`, or `inconclusive`. Do not work around a failed gate by accepting LLIR-only ordering or presence-only resource metadata.
- Use one isolated branch/worktree. Commit test-passing source before every remote run. Hardware-driven corrections require a new commit and a fresh deployment.

## Exact Remote Deployment Protocol

Before each hardware stage, deploy the reviewed commit rather than relying on an unsynchronized remote checkout:

```bash
export ROUTE_C_COMMIT="$(git rev-parse HEAD)"
export ROUTE_C_REMOTE_ROOT="/tmp/kernelswift-route-c-${ROUTE_C_COMMIT}"
git archive "$ROUTE_C_COMMIT" | ssh "$BI150_SSH" \
  "rm -rf '$ROUTE_C_REMOTE_ROOT' && mkdir -p '$ROUTE_C_REMOTE_ROOT' && tar -x -C '$ROUTE_C_REMOTE_ROOT'"
```

Every remote result records `ROUTE_C_COMMIT`, the diagnostic source SHA, and the accepted-source ledger. Every output path is namespaced under `artifacts/by-commit/$ROUTE_C_COMMIT/`. Final comparisons must use one declared `evidence_commit`; after a hardware-driven source correction, rerun every compared control/diagnostic input under the new commit. Older commit directories remain preserved but are excluded unless a validated summary records an explicit supersession ledger.

Collect ignored artifacts back with:

```bash
rsync -a "$BI150_SSH:$ROUTE_C_REMOTE_ROOT/experiments/bi150-kperfir-value/artifacts/" \
  experiments/bi150-kperfir-value/artifacts/
```

If code changes after hardware feedback, rerun tests, commit the fix, create a new `ROUTE_C_REMOTE_ROOT`, and regenerate evidence. Never mix artifacts from different commits in one result.

---

## File Structure

```text
experiments/bi150-kperfir-value/
├── .gitignore
├── README.md
├── assessment.md
├── lib/
│   ├── __init__.py
│   ├── compiler_manifest.py
│   ├── evidence_bundle.py
│   ├── invocation_manifest.py
│   ├── profile_buffer.py
│   ├── result_contract.py
│   └── source_guard.py
├── synthetic/
│   └── clock_probe.py
├── mm_encoder_attention/
│   └── diagnostic_kernel.py
├── scripts/
│   ├── audit_compiler_evidence.py
│   ├── preflight_corex.py
│   ├── run_eager.py
│   ├── run_evidence_matrix.py
│   ├── run_graph.py
│   ├── run_synthetic.py
│   └── summarize_results.py
├── tests/
│   ├── fixtures/
│   │   ├── valid_audit.json
│   │   ├── valid_invocation.json
│   │   ├── valid_manifest.json
│   │   ├── valid_preflight.json
│   │   ├── valid_result.json
│   │   └── valid_summary.json
│   ├── test_compiler_audit.py
│   ├── test_diagnostic_source.py
│   ├── test_evidence_bundle.py
│   ├── test_graph_runner.py
│   ├── test_profile_buffer.py
│   ├── test_result_contract.py
│   ├── test_runner_metadata.py
│   ├── test_source_guard.py
│   └── test_summary.py
└── artifacts/
    └── .gitignore
```

- `compiler_manifest.py`: materialize exact `CompiledKernel.asm` entries, binary bytes, hashes, source hash, compile hash, constants, resources, and artifact paths.
- `evidence_bundle.py`: cross-document hash, role, identity, compile-constant, resource, path, and commit binding.
- `invocation_manifest.py`: write validated eager/graph invocation identity and stream/input/output bindings.
- `profile_buffer.py`: fixed record decoding, wrap handling, variant-dependent warp layout, and cycle statistics.
- `result_contract.py`: complete experiment/audit/summary contracts and final-classification vocabulary.
- `source_guard.py`: exact accepted-source hash verification before importing device libraries.
- `preflight_corex.py`: compile a stock non-instrumented Triton kernel and test final-ISA/resource observability before Route C instrumentation.
- `audit_compiler_evidence.py`: consume one exact compilation manifest; never search by kernel name.
- `clock_probe.py`/`run_synthetic.py`: dependency-bounded clock sensitivity and perturbation controls.
- `diagnostic_kernel.py`: r002-equivalent mathematics with one compile-time region pair.
- `run_eager.py`: correctness, independent events, resources, cycles, and exact manifests for r001/r002-equivalent launches.
- `run_evidence_matrix.py`: rerun every applicable gate and experiment under one frozen evidence commit.
- `run_graph.py`: fixed-address graph replay with ordered sentinel/output poisoning and correctness validation.
- `summarize_results.py`: aggregate valid diagnostic evidence without official authority.

---

### Task 1: Lock Result Contracts, Buffer Layout, and Accepted-Source Guards

**Files:**
- Create: `experiments/bi150-kperfir-value/.gitignore`
- Create: `experiments/bi150-kperfir-value/artifacts/.gitignore`
- Create: `experiments/bi150-kperfir-value/README.md`
- Create: `experiments/bi150-kperfir-value/lib/__init__.py`
- Create: `experiments/bi150-kperfir-value/lib/profile_buffer.py`
- Create: `experiments/bi150-kperfir-value/lib/result_contract.py`
- Create: `experiments/bi150-kperfir-value/lib/source_guard.py`
- Create: `experiments/bi150-kperfir-value/tests/fixtures/valid_result.json`
- Create: `experiments/bi150-kperfir-value/tests/fixtures/valid_manifest.json`
- Create: `experiments/bi150-kperfir-value/tests/fixtures/valid_invocation.json`
- Create: `experiments/bi150-kperfir-value/tests/fixtures/valid_preflight.json`
- Create: `experiments/bi150-kperfir-value/tests/fixtures/valid_audit.json`
- Create: `experiments/bi150-kperfir-value/tests/fixtures/valid_summary.json`
- Create: `experiments/bi150-kperfir-value/tests/test_profile_buffer.py`
- Create: `experiments/bi150-kperfir-value/tests/test_result_contract.py`
- Create: `experiments/bi150-kperfir-value/tests/test_source_guard.py`

**Interfaces:**
- Produces: `unsigned_cycle_delta(start: int, end: int, bits: int = 64) -> int`
- Produces: `summarize_cycles(samples: list[int]) -> dict[str, int | float]`
- Produces: `decode_profile_buffer(values: list[int], selected_pids: tuple[int, ...], selected_local_warps: tuple[int, ...], generation: int) -> list[dict]`
- Produces: `validate_document(payload: dict) -> None`
- Produces: `document_route_c_commit(payload: dict) -> str`
- Produces: `validate_final_classification(value: str) -> None`
- Produces: `verify_accepted_sources(repo_root: Path) -> dict[str, str]`

- [ ] **Step 1: Create the ignored artifact boundary**

Write experiment `.gitignore`:

```gitignore
artifacts/*
!artifacts/.gitignore
```

Write `artifacts/.gitignore`:

```gitignore
*
!.gitignore
```

README must state the diagnostic-only boundary, five-day timebox, secure `BI150_SSH` requirement, accepted-source immutability, exact deployment protocol, and five final classifications.

- [ ] **Step 2: Write failing profile-buffer tests**

Use the fixed three-word signed-int64 slot:

```text
generation, start_clock, end_clock
```

Tests must cover:

```python
self.assertEqual(4, unsigned_cycle_delta((1 << 64) - 2, 2))
self.assertEqual(11.0, summarize_cycles([10, 10, 12, 12])["median"])
self.assertEqual((0,), selected_local_warps_for_num_warps(1))
self.assertEqual((0, 1), selected_local_warps_for_num_warps(2))
```

For a generation mismatch, require `status: unavailable`, cause `generation-mismatch`, and no numeric cycle fields. For a matching generation, require `status: observed`, boundaries, raw delta, and local-warp identity.

- [ ] **Step 3: Run buffer tests and verify failure**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_profile_buffer.py -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 4: Implement buffer decoding and statistics**

Use standard-library `statistics`. Compute population CV. Compute deterministic p10/p90 from sorted samples. Reject empty samples and nonpositive counter width.

- [ ] **Step 5: Write the complete result-contract fixtures and failing tests**

Every experiment result must require:

```text
document_type = experiment-result
experiment_id
environment: device, corex, torch, triton, target, warp_size, route_c_commit
variant: kernel_variant, num_warps, execution_mode
source: accepted_kernel_sha256, accepted_host_sha256, diagnostic_sha256
instrumentation: mode, region_id, tile, selected_pids, selected_local_warps,
                 time_unit, storage, measurement_semantics
validation: correctness, ttgir_ordering, llir_ordering, isa_ordering,
            completion_dependency, graph_capture
perturbation: kernel_event_overhead_pct, n_regs, n_spills,
              shared_bytes, launch_threads, resource_class
regions
status_causes
experiment_status
compilation_manifests: accepted_control and diagnostic
invocation_manifest: path and sha256
launch_binding: control_launch_role and measured_launch_role
secondary_comparisons: optional named launch-role pairs for diagnostics such as writeback isolation
```

Each validation/perturbation measurement is status-bearing and carries evidence paths. Every region requires PID, query-tile class, local warp, status, cause, measurement semantics, start/end boundary IDs, and evidence. `observed` requires raw median/p10/p90/CV; `unavailable` and `invalid` reject numeric cycle fields.

Also define and test:

- `document_type = compilation-manifest` with Route C commit, role `stock-preflight|accepted-control|diagnostic`, kernel variant, common mathematical specialization/launch configuration, role-specific instrumentation semantics, `compiled_kernel_hash`, `triton_source_hash`, source SHA, TTGIR/LLIR/binary paths and hashes, resource tuple, and manifest status. `common_specialization` carries B/S/H/D/NM/NT/BM/BN/BD, `num_warps`, and launch threads. `instrumentation` is `mode:none` with null region/tile/generation for accepted controls, and `mode:one-region` with actual region/tile/generation/profile constants for diagnostics;
- `document_type = invocation-manifest` with Route C commit, experiment ID, input tensor fingerprint, accepted host-route SHA when applicable, named stream definitions, and a `launches` ledger. Each launch entry requires launch role, compilation-manifest path/SHA, `compiled_kernel_hash`, execution mode, stream role, output destination ID, optional profile destination ID, optional generation, and specialization identity. Control/measured role selection lives in each experiment result's `launch_binding`, allowing one invocation ledger to support multiple comparisons. Optional `secondary_comparisons` may bind two non-authoritative diagnostic launch roles without redefining the accepted-control role;
- `document_type = preflight-result` with Route C commit, environment, stock-manifest path/SHA, capability statuses, disassembler attempts, evidence paths, status causes, and complete preflight status;
- `document_type = compiler-audit` with exact compilation-manifest SHA, tool attempts, ordering offsets, resources, and audit status;
- `document_type = experiment-summary` with input document SHA/commit ledger, supersession ledger, duplicate-experiment-ID rejection, and no final Route C classification; subtype `evidence-matrix` additionally requires completed/skipped stages, actual stop stage, complete document/artifact ledgers, and bundle payload digest;
- final classes `valuable|technically-valid-low-value|perturbation-invalid|unsupported|inconclusive`.

- [ ] **Step 6: Run contract tests and verify failure**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_result_contract.py -v
```

Expected: FAIL because `result_contract.py` does not exist.

- [ ] **Step 7: Implement complete document validation**

Use only the standard library. Validation failures include a JSON-style field path. `validate_document` dispatches by `document_type`; no completed JSON consumed by assessment may be silently skipped.

- [ ] **Step 8: Implement exact source guards and tests**

Store the five path/hash pairs from Global Constraints. Hash bytes with SHA256. Raise before any Torch/Triton import or device work. Test the current repository and a temporary altered candidate copy.

- [ ] **Step 9: Run all Task-1 tests**

```bash
python3 -m unittest discover -s experiments/bi150-kperfir-value/tests -p 'test_*.py' -v
```

Expected: all present tests PASS.

- [ ] **Step 10: Commit Task 1**

```bash
git add experiments/bi150-kperfir-value/.gitignore \
        experiments/bi150-kperfir-value/artifacts/.gitignore \
        experiments/bi150-kperfir-value/README.md \
        experiments/bi150-kperfir-value/lib \
        experiments/bi150-kperfir-value/tests
git commit -m "test(experiment): lock BI150 profiler contracts"
```

---

### Task 2: Stage-0 CoreX Final-ISA and Resource Observability Gate

**Files:**
- Create: `experiments/bi150-kperfir-value/lib/compiler_manifest.py`
- Create: `experiments/bi150-kperfir-value/lib/evidence_bundle.py`
- Create: `experiments/bi150-kperfir-value/scripts/preflight_corex.py`
- Create: `experiments/bi150-kperfir-value/scripts/audit_compiler_evidence.py`
- Create: `experiments/bi150-kperfir-value/scripts/run_evidence_matrix.py`
- Create: `experiments/bi150-kperfir-value/scripts/summarize_results.py`
- Create: `experiments/bi150-kperfir-value/tests/test_compiler_audit.py`
- Create: `experiments/bi150-kperfir-value/tests/test_evidence_bundle.py`
- Modify: `experiments/bi150-kperfir-value/README.md`

**Interfaces:**
- Produces: `materialize_compilation_manifest(compiled_kernel, output_dir: Path, identity: dict) -> Path`
- Produces: `launch_compiled_kernel(compiled_kernel, grid: tuple[int, ...], stream, non_constexpr_args: tuple) -> None`
- Produces: `classify_preflight(...) -> tuple[str, list[str]]`
- Produces preflight-capable `validate_evidence_documents` and `validate_evidence_bundle`
- Produces a lazy `run_evidence_matrix.py` that can seal `max-stage=preflight` without importing later-stage modules
- Produces CLI: `python3 scripts/preflight_corex.py --output-dir PATH`
- Produces CLI: `python3 scripts/audit_compiler_evidence.py --manifest PATH --output PATH`

- [ ] **Step 1: Write failing preflight/audit tests**

Require:

- all TTGIR/LLIR/binary/resource/final-disassembly capabilities plus positive finite repeatable CUDA Event stock-kernel timing -> `valid`;
- missing final disassembly -> `inconclusive`, cause `final-isa-unavailable`;
- missing raw resource metadata -> `inconclusive`, cause `resource-evidence-unavailable`;
- binary missing from a manifest -> `invalid`, cause `compiled-binary-missing`;
- auditor consumes an exact manifest path and rejects a `--kernel` name-search interface;
- manifest records `document_type: compilation-manifest`, semantic role, kernel variant, `common_specialization`, role-specific `instrumentation`, `compiled_kernel_hash`, `triton_source_hash`, binary/TTGIR/LLIR hashes, source SHA, resource tuple, evidence paths, status causes, and Route C commit; accepted controls must never fabricate diagnostic region/tile/generation constants;
- preflight output validates as `document_type: preflight-result`, binds the stock-manifest path/SHA, and records status-bearing stock-kernel CUDA Event median/p10/p90/CV evidence.

- [ ] **Step 2: Implement exact compilation materialization**

Obtain the reviewed object with `compiled_kernel = jit_function.warmup(..., grid=grid, num_warps=num_warps)` using the exact constexpr specialization. Materialize the manifest from that object. All execution must call the same object through:

```python
def launch_compiled_kernel(compiled_kernel, grid, stream, non_constexpr_args):
    grid_0 = grid[0]
    grid_1 = grid[1] if len(grid) > 1 else 1
    grid_2 = grid[2] if len(grid) > 2 else 1
    run = compiled_kernel.run
    launch_metadata = compiled_kernel.launch_metadata(grid, stream, *non_constexpr_args)
    run(
        grid_0,
        grid_1,
        grid_2,
        stream,
        compiled_kernel.function,
        compiled_kernel.packed_metadata,
        launch_metadata,
        type(compiled_kernel).launch_enter_hook,
        type(compiled_kernel).launch_exit_hook,
        *non_constexpr_args,
    )
```

Do not relaunch through `jit_function[grid](...)` after binding evidence, because that could resolve another cached specialization. From `CompiledKernel.asm`, write each text/binary artifact into the ignored output directory and hash its bytes. Require `identity` to supply semantic role, kernel variant, common mathematical specialization/launch configuration, and role-specific instrumentation semantics. Accepted controls supply `instrumentation.mode=none` and null diagnostic fields; diagnostics supply their actual one-region constants. Record those fields plus `CompiledKernel.hash`, `CompiledKernel.src.hash()`, metadata, `n_regs`, `n_spills`, shared bytes, and exact paths. Never locate artifacts by function name.

- [ ] **Step 3: Implement stock Triton preflight**

`preflight_corex.py` must verify accepted sources before importing Torch/Triton, then compile and launch a minimal vector-add kernel. After correctness, warm up 20 times and collect at least 50 independent CUDA Event samples around the stock launch; require positive finite samples, report median/p10/p90/CV, and require CV at most 5%. Missing, zero, non-finite, or less-stable event timing makes preflight `inconclusive` with `kernel-event-timing-unavailable`. It must materialize the exact compilation manifest and attempt final ISA extraction with both:

```text
/usr/local/corex-4.4.0/bin/llvm-objdump -d BINARY
/usr/local/corex-4.4.0/bin/ixobjdump --sass BINARY
```

Record command, exit code, stderr tail, output path, and whether output contains a disassembled function body. Do not treat tool presence or an ELF header alone as success.

- [ ] **Step 4: Implement conservative resource decision helpers**

Test and implement:

```text
exact resource tuple equal and no spill -> pass
introduced spill -> perturbation-invalid
resource tuple differs and documented occupancy class equal -> pass
resource tuple differs without documented occupancy evidence -> inconclusive
raw resource metadata missing -> inconclusive
```

Resource tuple is `n_regs, n_spills, shared_bytes, launch_threads, num_warps`.

- [ ] **Step 5: Implement the preflight-only sealing foundation**

Implement `evidence_bundle.py`, `run_evidence_matrix.py`, and `summarize_results.py` sufficiently for `--max-stage preflight`: later-stage modules are imported only after their gate is selected; pre-seal validation covers the stock manifest/audit/preflight; the runner emits one closed evidence-matrix summary; strict validation seals the preflight-only bundle. Add tests proving this path succeeds when synthetic, eager, and graph scripts do not exist.

- [ ] **Step 6: Run local tests**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_compiler_audit.py -v
python3 -m unittest experiments/bi150-kperfir-value/tests/test_evidence_bundle.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit before hardware execution**

```bash
git add experiments/bi150-kperfir-value/lib/compiler_manifest.py \
        experiments/bi150-kperfir-value/lib/evidence_bundle.py \
        experiments/bi150-kperfir-value/scripts/preflight_corex.py \
        experiments/bi150-kperfir-value/scripts/audit_compiler_evidence.py \
        experiments/bi150-kperfir-value/scripts/run_evidence_matrix.py \
        experiments/bi150-kperfir-value/scripts/summarize_results.py \
        experiments/bi150-kperfir-value/tests/test_compiler_audit.py \
        experiments/bi150-kperfir-value/tests/test_evidence_bundle.py \
        experiments/bi150-kperfir-value/README.md
git commit -m "feat(experiment): add CoreX observability preflight"
```

- [ ] **Step 8: Deploy the exact commit and run Stage 0 remotely**

Use the Exact Remote Deployment Protocol, then:

```bash
ssh "$BI150_SSH" "export ROUTE_C_COMMIT='$ROUTE_C_COMMIT'; \
  export COREX_VERSION=4.4.0; \
  . /usr/local/corex/enable; \
  cd '$ROUTE_C_REMOTE_ROOT'; \
  python3 experiments/bi150-kperfir-value/scripts/preflight_corex.py \
    --output-dir experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/preflight"
```

Collect artifacts and validate the preflight document locally with `validate_document`.

- [ ] **Step 9: Apply the Stage-0 stop gate**

Known matched-host behavior must be recorded, not hidden: current CoreX `llvm-objdump` reports no BI disassembler for direct Triton cubins, and `ixobjdump --sass` rejects cubins without a Fatbin section. Spend at most the remaining Stage-0 day checking a documented vendor disassembler/API or supported wrapper. Do not reverse-engineer a binary plugin.

Branch on this complete decision table:

| Condition | Action |
|---|---|
| accepted-source hash mismatch | hard stop, restore the approved checkout, and rerun; if unrecoverable within the timebox, final `inconclusive` with `source-ledger-unrecoverable` |
| SSH/bootstrap/device/runtime fingerprint unavailable or mismatched | skip deeper work; final `inconclusive` with the exact environment cause |
| stock kernel launch/correctness failure or invalid independent CUDA Event timing | skip deeper work; final `inconclusive` with `stock-kernel-preflight-failed` or `kernel-event-timing-unavailable` |
| compiler artifact missing, malformed manifest, or malformed preflight document | repair/test/commit/redeploy within Stage 0; if unresolved at the one-day limit, final `inconclusive` |
| disassembler absent or rejects the direct CoreX Triton binary format | final `inconclusive` with `final-isa-unavailable`; this is tooling evidence failure, not hardware primitive absence |
| raw resource metadata absent or malformed | final `inconclusive` with `resource-evidence-unavailable` |
| a specifically tested required platform primitive such as `clock64` is rejected in its dedicated probe | final `unsupported` |
| all mandatory environment, correctness, artifact, ISA, and resource checks pass | record exact evidence and continue |

A working final-ISA path alone is insufficient when another mandatory preflight capability is not valid.

---

### Task 3: Dependency-Bounded Synthetic Clock Sensitivity and Perturbation Controls

**Condition:** Execute only after Task 2 preflight is `valid`.

**Files:**
- Create: `experiments/bi150-kperfir-value/lib/invocation_manifest.py`
- Create: `experiments/bi150-kperfir-value/synthetic/clock_probe.py`
- Create: `experiments/bi150-kperfir-value/scripts/run_synthetic.py`
- Create: `experiments/bi150-kperfir-value/tests/test_runner_metadata.py`
- Modify: `experiments/bi150-kperfir-value/README.md`

**Interfaces:**
- Produces: `write_invocation_manifest(output_path: Path, payload: dict) -> Path`
- Produces timer helpers: `read_clock64()` and `read_clock64_after(dep)`
- Produces three synthetic variants: `control`, `clock-pair-output`, `clock-pair-profile`
- Produces CLI: `python3 scripts/run_synthetic.py --output-dir PATH --samples 100`

- [ ] **Step 1: Write failing static and metadata tests**

Require:

- `%clock64` and `%tid.x` inline assembly with `is_pure=False`;
- `read_clock64_after(dep)` carries an inline-assembly input operand;
- the chain begins with a zero-valued dependency derived from the start clock;
- the end clock consumes the chain result;
- all three variants emit a bitwise-identical computation output;
- clock delta uses a separate diagnostic output or profile slots and never replaces the computation output;
- only one timer pair is live;
- only lane zero writes independent local-warp slots;
- short and long specializations produce six separate compilation manifests and one invocation manifest with roles `synthetic-control-short`, `clock-pair-output-short`, `clock-pair-profile-short`, `synthetic-control-long`, `clock-pair-output-long`, and `clock-pair-profile-long`;
- control roles emit no standalone experiment result;
- four experiment results bind output/profile measured roles to the matching short/long control role;
- profile results record same-length writeback isolation as secondary comparisons from `clock-pair-output-short` to `clock-pair-profile-short` and from `clock-pair-output-long` to `clock-pair-profile-long`;
- the long profile result records the hard sensitivity secondary comparison from `clock-pair-profile-short` to `clock-pair-profile-long`, with both manifest hashes and CHAIN_ITERS values bound explicitly;
- source guards run before Torch/Triton imports.

- [ ] **Step 2: Implement the three synthetic variants**

Use short `CHAIN_ITERS=16` and long `CHAIN_ITERS=256` specializations.

- `control`: dependency chain and deterministic computation output, no clock reads.
- `clock-pair-output`: start clock, start-derived zero dependency, chain, dependency-consuming end clock, the same computation output as control, and a separate diagnostic-delta output; no profile-buffer slots.
- `clock-pair-profile`: same computation output and timer pair, with fixed `generation,start,end` profile slots written by lane zero after the region.

The dependency operand may be accepted only if TTGIR, LLIR, and final ISA show the intended chain-to-end ordering. Otherwise report `issue-window` and fail the hard execution-duration sensitivity gate.

- [ ] **Step 3: Implement the synthetic runner**

The runner must:

- verify accepted sources before importing device libraries;
- assert device/CoreX/Torch/Triton/target/warp identity;
- warm up 20 and sample 100 interleaved control/instrumented launches;
- require bitwise equality of the computation output across control, clock-pair-output, and clock-pair-profile before timing interpretation;
- compare short versus long cycle distributions;
- compare control versus instrumented CUDA Event timing and exact resource tuples;
- materialize one exact compilation manifest per specialization;
- emit an invocation manifest whose `launches` ledger has all six specialization-qualified short/long roles, each bound to its exact manifest SHA, `CHAIN_ITERS`, eager stream role, computation/diagnostic output IDs, and generation;
- emit exactly four normalized experiment results with the fixed short/long launch bindings above, and only after every compilation/invocation/result document passes `validate_document`.

- [ ] **Step 4: Run local tests and commit before hardware**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_runner_metadata.py -v
git add experiments/bi150-kperfir-value/lib/invocation_manifest.py \
        experiments/bi150-kperfir-value/synthetic/clock_probe.py \
        experiments/bi150-kperfir-value/scripts/run_synthetic.py \
        experiments/bi150-kperfir-value/tests/test_runner_metadata.py \
        experiments/bi150-kperfir-value/README.md
git commit -m "feat(experiment): add dependency-bounded clock probe"
```

- [ ] **Step 5: Deploy and run remotely**

Use a fresh exact-commit deployment, then:

```bash
ssh "$BI150_SSH" "export ROUTE_C_COMMIT='$ROUTE_C_COMMIT'; \
  export COREX_VERSION=4.4.0; \
  . /usr/local/corex/enable; \
  cd '$ROUTE_C_REMOTE_ROOT'; \
  python3 experiments/bi150-kperfir-value/scripts/run_synthetic.py \
    --output-dir experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/synthetic \
    --samples 100"
```

Collect artifacts locally.

- [ ] **Step 6: Audit exact synthetic manifests**

For every manifest, run this exact loop:

```bash
for manifest in experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/synthetic/manifests/*.json; do
  name="$(basename "$manifest" .json)"
  python3 experiments/bi150-kperfir-value/scripts/audit_compiler_evidence.py \
    --manifest "$manifest" \
    --output "experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/synthetic/audits/${name}.json"
done
```

Never search by kernel name. Validate every audit document.

- [ ] **Step 7: Apply synthetic gates**

Continue only when:

- short/long outputs are correct;
- clock values are monotonic;
- long dependency-bounded cycles exceed short beyond both noise bands;
- final ISA proves start-before-chain, chain-before-end, and end-before-profile-store;
- no spill is introduced;
- resource class passes the exact-equality/documented-occupancy rule;
- event overhead is at most 10%;
- CV is at most 5%.

Clock syntax rejection after the approved `clock64` and `clock`/`clock_hi` attempts is `unsupported`. Unproven dependency/order or changed resources without occupancy evidence is `inconclusive`. A measured perturbation violation is `perturbation-invalid`.

---

### Task 4: One-Region Attention Diagnostic Kernel

**Condition:** Execute only after Task 3 synthetic gates pass.

**Files:**
- Create: `experiments/bi150-kperfir-value/mm_encoder_attention/diagnostic_kernel.py`
- Create: `experiments/bi150-kperfir-value/tests/test_diagnostic_source.py`
- Modify: `experiments/bi150-kperfir-value/README.md`

**Interfaces:**
- Produces: `SELECTED_PIDS = (0, 16, 32)`
- Produces: `decode_pid(pid: int, *, batches: int = 2, heads: int = 8) -> tuple[int, int, int]`
- Produces region IDs: `kernel-span`, `prelude`, `tile-total`, `epilogue`, `k-load`, `qk-score`, `v-load`, `softmax-update`, `pv-accumulate`
- Produces: `launch_diagnostic(..., region: str, tile: int, num_warps: int) -> object`, pinning compile-time `GENERATION=1`

- [ ] **Step 1: Write failing source/PID tests**

Require:

```python
self.assertEqual((0, 0, 0), decode_pid(0))
self.assertEqual((0, 0, 1), decode_pid(16))
self.assertEqual((0, 0, 2), decode_pid(32))
```

Also require:

- r002 accepted hash literal and live hash match;
- mathematical operation order matches r002 lines 42–96;
- one compile-time region pair only;
- r001 local warps `(0,)`, r002 local warps `(0,1)`;
- no campaign report/state writes;
- source guard runs before Torch/Triton imports in all runners.

- [ ] **Step 2: Copy the accepted mathematical kernel**

Copy `triton_mm_encoder_attention_e2_002.py:12-96`. Add `profile_ptr`, `GENERATION: tl.constexpr`, `PROFILE_REGION: tl.constexpr`, `PROFILE_TILE: tl.constexpr`, and `PROFILE_NUM_WARPS: tl.constexpr`. Do not alter BM/BN/BD, loads, four fp32-widened dots, online softmax, or final output stores.

- [ ] **Step 3: Implement selected-program ownership**

Read `%tid.x`; derive local warp/lane using warp size 64. Only selected PIDs and lane zero write. Fixed slot order is `generation,start,end`.

- [ ] **Step 4: Implement exact one-region brackets**

- `kernel-span`: start after PID/TID setup and end after both final output stores; default `issue-window` unless store retirement is proven.
- `prelude`: before Q loads through running-state initialization.
- `tile-total`: selected tile from K-load start through `m_run = m_new`.
- `epilogue`: normalization through both final stores; default `issue-window` unless store retirement is proven.
- `k-load`: r002 lines 69–72.
- `qk-score`: lines 74–76.
- `v-load`: lines 78–81.
- `softmax-update`: lines 83–86.
- `pv-accumulate`: lines 87–88 only; keep line 89 outside this bracket.

Route C v0 does not invent tensor reductions or checksum work solely to manufacture a completion dependency, because that would extend the measured critical path. Therefore all real attention regions initially use side-effecting `read_clock64()` without `read_clock64_after(dep)`, set `completion_dependency.status: unavailable` with cause `real-region-completion-token-not-proven`, and report `measurement_semantics: issue-window`.

An individual region may be upgraded only if an already-existing value can be consumed without adding a new reduction or changing mathematics/resources, and exact TTGIR/LLIR/ISA evidence proves the dependency. The required complete dependency coverage is: both Q halves for prelude, both K halves for K load, scaled/masked `s` for QK, both V halves for V load, `m_new`, `l_run`, and probability state for softmax, both accumulator halves for PV, and accumulators plus updated running state for tile total. Kernel-span and store-ending epilogue remain issue windows unless store retirement is independently proven.

- [ ] **Step 5: Run tests and commit before hardware**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_diagnostic_source.py -v
git add experiments/bi150-kperfir-value/mm_encoder_attention/diagnostic_kernel.py \
        experiments/bi150-kperfir-value/tests/test_diagnostic_source.py \
        experiments/bi150-kperfir-value/README.md
git commit -m "feat(experiment): add one-region BI150 attention diagnostics"
```

---

### Task 5: Eager Correctness, Events, Resources, and r001/r002 Exploration

**Condition:** Execute only after Task 4 is committed.

**Files:**
- Create: `experiments/bi150-kperfir-value/scripts/run_eager.py`
- Extend: `experiments/bi150-kperfir-value/tests/test_runner_metadata.py`
- Modify: `experiments/bi150-kperfir-value/README.md`

**Interfaces:**
- Produces CLI: `run_eager.py --variant r001|r002 --region REGION --tile -1|0|1|2 --samples 100 --output-dir PATH`
- Produces one result plus one exact compilation manifest per specialization

- [ ] **Step 1: Write failing runner tests**

Require:

- region/tile validation;
- r001 metadata has `num_warps=1`, selected warps `[0]`;
- r002 metadata has `num_warps=2`, selected warps `[0,1]`;
- eager accepted host SHA is null;
- accepted kernel SHA, diagnostic SHA, Route C commit, boundary IDs, exact accepted-control and diagnostic manifest paths/SHAs, invocation-manifest path/SHA, status-bearing ordering/resources, p10/p90/CV, and evidence paths are mandatory;
- source guards execute before device imports.

- [ ] **Step 2: Implement deterministic loading and correctness**

Use `importlib.util.spec_from_file_location` for accepted base/r001/r002. Seed once, clone identical target-regime inputs `(2,83,512)` fp16, execute accepted and diagnostic direct launches into preallocated outputs, require `torch.equal` to the accepted variant, and require `torch.testing.assert_close(..., atol=1e-2, rtol=1e-2, equal_nan=True)` to base.

A failure emits `invalid: diagnostic-correctness-failed` and stops timing.

- [ ] **Step 3: Implement interleaved independent event timing**

Warm up 20. Collect 100 accepted/diagnostic A/B pairs with events around direct launches only. Exclude allocations, JSON, synchronization setup, and profile-buffer reads. Report distributions and overhead percentage.

- [ ] **Step 4: Implement resources and exact manifests**

Record accepted and diagnostic resource tuples. Apply the exact-equality/documented-occupancy rule. Materialize exact TTGIR/LLIR/binary artifacts and separate manifests for both the accepted control specialization and diagnostic specialization. Emit one eager invocation manifest with `accepted-direct` and `eager-diagnostic` launch entries. Each entry binds its manifest SHA, eager execution mode, caller stream, output/profile destination IDs, specialization identity, and generation; name those entries as control/measured launch roles. Bind all three documents into the experiment result; resource comparison consumes only those bound manifests.

- [ ] **Step 5: Implement profile collection**

For each sample, fill the profile buffer with generation-mismatch zero, launch with compile-time `GENERATION=1`, synchronize, copy the small buffer, and decode expected generation one. Do this in a pass separate from independent event timing.

- [ ] **Step 6: Run tests and commit before hardware**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_runner_metadata.py -v
git add experiments/bi150-kperfir-value/scripts/run_eager.py \
        experiments/bi150-kperfir-value/tests/test_runner_metadata.py \
        experiments/bi150-kperfir-value/README.md
git commit -m "feat(experiment): add BI150 eager region runner"
```

- [ ] **Step 7: Deploy exact commit and run kernel-span experiments**

```bash
for variant in r001 r002; do
  ssh "$BI150_SSH" "export ROUTE_C_COMMIT='$ROUTE_C_COMMIT'; \
    export COREX_VERSION=4.4.0; \
    . /usr/local/corex/enable; \
    cd '$ROUTE_C_REMOTE_ROOT'; \
    python3 experiments/bi150-kperfir-value/scripts/run_eager.py \
      --variant '$variant' --region kernel-span --tile -1 --samples 100 \
      --output-dir experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/eager/${variant}-kernel-span"
done
```

Collect artifacts and audit both the accepted-control and diagnostic exact manifests for each variant.

- [ ] **Step 8: Apply eager gates without forcing local/whole-grid agreement**

Require accepted whole-kernel direction to reproduce. Report local cycle changes or non-changes against their own noise bands. A mismatch is not automatic probe failure because occupancy/concurrency can change whole-grid time. Stop if correctness, ordering, resources, overhead, or CV gates fail.

---

### Task 6: Coarse/Deep Orchestration and Non-Authoritative Summary

**Condition:** Execute only after valid eager kernel-span evidence.

**Files:**
- Modify: `experiments/bi150-kperfir-value/scripts/summarize_results.py`
- Create: `experiments/bi150-kperfir-value/tests/test_summary.py`
- Modify: `experiments/bi150-kperfir-value/README.md`

**Interfaces:**
- Produces CLI: `summarize_results.py --inputs PATH... --output PATH`
- Produces validated `experiment-summary`; never emits a final Route C classification

- [ ] **Step 1: Write failing summary tests**

Require grouping by variant, region, tile, PID, and local warp; input SHA ledger; no full-grid average; separate event/local-cycle deltas; explicit `not-comparable-as-whole-grid` for r001/r002 local spans; status propagation; and no `final_classification` field.

- [ ] **Step 2: Implement and test the summary**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_summary.py -v
```

Expected: PASS after implementation.

- [ ] **Step 3: Commit before remote coarse/deep runs**

```bash
git add experiments/bi150-kperfir-value/scripts/summarize_results.py \
        experiments/bi150-kperfir-value/tests/test_summary.py \
        experiments/bi150-kperfir-value/README.md
git commit -m "feat(experiment): summarize BI150 region evidence"
```

Deploy this exact commit before continuing.

- [ ] **Step 4: Run coarse variants remotely**

After a fresh exact-commit deployment, run:

```bash
for variant in r001 r002; do
  for region_tile in 'prelude -1' 'tile-total 0' 'tile-total 1' 'tile-total 2' 'epilogue -1'; do
    set -- $region_tile
    region="$1"
    tile="$2"
    ssh "$BI150_SSH" "export ROUTE_C_COMMIT='$ROUTE_C_COMMIT'; \
      export COREX_VERSION=4.4.0; \
      . /usr/local/corex/enable; \
      cd '$ROUTE_C_REMOTE_ROOT'; \
      python3 experiments/bi150-kperfir-value/scripts/run_eager.py \
        --variant '$variant' --region '$region' --tile '$tile' --samples 100 \
        --output-dir experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/eager/${variant}-${region}-tile-${tile}"
  done
done
```

Collect artifacts, audit every exact manifest, and validate every result/audit document. Stop before deep runs if any required coarse gate fails; preserve independently valid prior results.

- [ ] **Step 5: Run conditional deep variants**

First run middle tile 1 for both variants:

```bash
for variant in r001 r002; do
  for region in k-load qk-score v-load softmax-update pv-accumulate; do
    ssh "$BI150_SSH" "export ROUTE_C_COMMIT='$ROUTE_C_COMMIT'; \
      export COREX_VERSION=4.4.0; \
      . /usr/local/corex/enable; \
      cd '$ROUTE_C_REMOTE_ROOT'; \
      python3 experiments/bi150-kperfir-value/scripts/run_eager.py \
        --variant '$variant' --region '$region' --tile 1 --samples 100 \
        --output-dir experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/eager/${variant}-${region}-tile-1"
  done
done
```

Only if those r002 results remain valid, repeat the r002 loop for tiles 0 and 2. Audit every exact manifest. A failed deep specialization does not invalidate valid coarse evidence.

- [ ] **Step 6: Generate and validate eager summary**

```bash
python3 experiments/bi150-kperfir-value/scripts/summarize_results.py \
  --inputs experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/eager/*/result.json \
  --output experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/summary/eager-summary.json
```

Validate the summary document. The summary must expose `recommended_graph_probe.region` and `.tile` only when that result is valid. Use `dominant`, region-cost share, padding-cost, warp-cost imbalance, or device-upper-bound language only for `execution-duration` evidence. For `issue-window`, report only ordering/issue behavior and `largest observed issue-window`.

---

### Task 7: Ordered and Correctness-Checked Graph Replay Probe

**Condition:** Execute only after at least one valid eager region result.

**Files:**
- Create: `experiments/bi150-kperfir-value/scripts/run_graph.py`
- Create: `experiments/bi150-kperfir-value/tests/test_graph_runner.py`
- Modify: `experiments/bi150-kperfir-value/README.md`

**Interfaces:**
- Produces CLI: `run_graph.py --region REGION --tile -1|0|1|2 --samples 100 --output-dir PATH`
- Records accepted r002 mathematical SHA and r003 host-route SHA separately

- [ ] **Step 1: Write failing graph tests**

Require `kernel_variant=r002-nw2`, `num_warps=2`, `execution_mode=graph`, selected warps `[0,1]`, both accepted hashes, diagnostic hash, exact accepted-control and diagnostic manifests/SHAs, exact graph invocation-manifest path/SHA, and source-guard-before-device-import ordering. The same invocation must compile and execute the accepted direct control, eager diagnostic control, and graph diagnostic using identical input tensors and specialization constants.

Pure sequence tests must require one retained stream and this order:

```text
profile sentinel fill
output poison fill
start event
single graph replay
end event
synchronize
profile decode
output correctness checks
```

- [ ] **Step 2: Implement fixed-address graph capture with named stream roles**

Follow accepted r003 lines 214–229 with three retained roles:

- `caller_stream = torch.cuda.current_stream()`: owns control execution, post-replay copies, and host-read synchronization;
- `warmup_stream`: used only for three pre-capture JIT warmups, then joined into `caller_stream`;
- `graph_stream`: retained for graph capture and every replay sequence.

Use identical fixed Q/K/V tensors for the accepted direct control, eager diagnostic control, and graph diagnostic. Materialize/bind accepted-control and diagnostic manifests in the same invocation. Emit one graph invocation manifest with `accepted-direct`, `eager-diagnostic`, and `graph-diagnostic` launch entries. Bind each entry to its exact manifest SHA, execution mode, stream role, output/profile destination IDs, specialization, and generation; bind top-level input fingerprint, r003 host-route SHA, and named `caller_stream|warmup_stream|graph_stream` definitions. The eager and graph diagnostic entries may reference the same diagnostic compilation manifest and same reviewed `CompiledKernel` object/hash but must remain distinct launch entries. Capture and replay must invoke that bound object's `run` callable, not the JIT wrapper. After joining `warmup_stream`, make `graph_stream` wait on `caller_stream`, enter `torch.cuda.stream(graph_stream)`, and capture exactly one instrumented kernel launch. Do not copy the three-tier service state machine.

- [ ] **Step 3: Implement race-free replay and correctness**

On the retained `graph_stream`, execute profile sentinel fill, output poisoning, start event, `graph.replay()`, and end event in that order. Then call `caller_stream.wait_event(end_event)`, enqueue profile/output copies on `caller_stream`, and synchronize `caller_stream` before any host read. Record the start event after both fills so fills are outside timing.

For every replay:

- require generation-one profile records;
- require output changed from the poison;
- require bitwise equality between accepted direct control and eager diagnostic output produced in the same invocation;
- require bitwise equality between that eager diagnostic output and graph diagnostic output;
- require closeness of accepted direct, eager diagnostic, and graph diagnostic outputs to the base reference;
- keep profile host copy/read outside the event interval.

- [ ] **Step 4: Run tests and commit before hardware**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_graph_runner.py -v
git add experiments/bi150-kperfir-value/scripts/run_graph.py \
        experiments/bi150-kperfir-value/tests/test_graph_runner.py \
        experiments/bi150-kperfir-value/README.md
git commit -m "feat(experiment): probe BI150 graph-interior regions"
```

- [ ] **Step 5: Deploy and run graph kernel-span first**

After a fresh exact-commit deployment, run:

```bash
ssh "$BI150_SSH" "export ROUTE_C_COMMIT='$ROUTE_C_COMMIT'; \
  export COREX_VERSION=4.4.0; \
  . /usr/local/corex/enable; \
  cd '$ROUTE_C_REMOTE_ROOT'; \
  python3 experiments/bi150-kperfir-value/scripts/run_graph.py \
    --region kernel-span --tile -1 --samples 100 \
    --output-dir experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/graph/kernel-span"
```

Collect artifacts, audit the exact accepted-control and captured diagnostic manifests, and validate correctness. If valid, read the strongest valid eager probe selected by the summary:

```bash
read -r region tile < <(python3 - <<'PY'
import json
import os
from pathlib import Path
commit = os.environ['ROUTE_C_COMMIT']
payload = json.loads(Path(f'experiments/bi150-kperfir-value/artifacts/by-commit/{commit}/summary/eager-summary.json').read_text())
probe = payload['recommended_graph_probe']
print(probe['region'], probe['tile'])
PY
)
```

Run the selected probe:

```bash
ssh "$BI150_SSH" "export ROUTE_C_COMMIT='$ROUTE_C_COMMIT'; \
  export COREX_VERSION=4.4.0; \
  . /usr/local/corex/enable; \
  cd '$ROUTE_C_REMOTE_ROOT'; \
  python3 experiments/bi150-kperfir-value/scripts/run_graph.py \
    --region '$region' --tile '$tile' --samples 100 \
    --output-dir experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/graph/${region}-tile-${tile}"
```

Do not select perturbation-invalid evidence or use issue-window-only evidence for an execution-duration claim.

- [ ] **Step 6: Generate graph/eager summary**

```bash
python3 experiments/bi150-kperfir-value/scripts/summarize_results.py \
  --inputs experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/eager/*/result.json \
           experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/graph/*/result.json \
  --output experiments/bi150-kperfir-value/artifacts/by-commit/$ROUTE_C_COMMIT/summary/all-summary.json
```

Validate the summary. Keep graph round-trip event time separate from local cycles. Graph-only failure is recorded but does not erase valid eager evidence.

---

### Task 8: Freeze One Evidence Commit, Validate Cross-Document Bindings, and Rerun the Matrix

**Condition:** Execute before final assessment whether the experiment stopped at Stage 0, synthetic, eager, deep, or graph.

**Files:**
- Modify: `experiments/bi150-kperfir-value/lib/evidence_bundle.py`
- Modify: `experiments/bi150-kperfir-value/scripts/run_evidence_matrix.py`
- Modify: `experiments/bi150-kperfir-value/tests/test_evidence_bundle.py`
- Modify: `experiments/bi150-kperfir-value/scripts/summarize_results.py`
- Modify: `experiments/bi150-kperfir-value/README.md`

**Interfaces:**
- Produces: `validate_evidence_documents(evidence_root: Path) -> dict` for pre-seal semantic validation
- Produces: `validate_evidence_bundle(evidence_root: Path) -> dict` for strict sealed closure validation
- Produces CLI: `run_evidence_matrix.py --output-root PATH --samples 100 --max-stage preflight|synthetic|eager|deep|graph`
- Produces a validated `experiment-summary` with subtype `evidence-matrix`; it records stop stage but never assigns the final Route C classification

- [ ] **Step 1: Write failing cross-document bundle tests**

Create a bounded fixture bundle and require the validator to prove:

- every document belongs to the bundle commit and commit namespace;
- result `accepted_kernel_sha256` equals the accepted-control manifest source SHA;
- result `diagnostic_sha256` equals the diagnostic manifest source SHA;
- manifest roles `accepted-control` and `diagnostic` cannot be swapped, and invocation launch roles cannot reference the wrong manifest role;
- accepted and diagnostic manifests have equal common mathematical specialization/launch configuration, while accepted instrumentation is `mode:none` and diagnostic instrumentation matches the result's region/tile/generation/profile constants;
- result resource tuple matches the role-correct referenced manifests;
- result `control_launch_role` and `measured_launch_role` resolve to invocation `launches` entries whose manifest SHA, `compiled_kernel_hash`, execution mode, stream role, specialization, output/profile ownership, and generation match the result and referenced manifest;
- synthetic results use specialization-qualified approved control/measured pairs; writeback-isolation comparisons stay within equal `CHAIN_ITERS`, and the hard sensitivity comparison resolves specifically from `clock-pair-profile-short` to `clock-pair-profile-long`;
- invocation input fingerprint and host-route identity match the result source/variant metadata;
- compiler-audit `manifest_sha256` matches the exact referenced manifest bytes;
- every evidence path stays under the evidence root, exists, and matches its recorded SHA;
- graph results bind r002 mathematical source and r003 host-route source separately;
- duplicate experiment IDs in one commit are rejected;
- summaries carry the exact input document SHA ledger;
- exactly one `experiment-summary` with subtype `evidence-matrix` exists;
- its document/artifact ledgers cover every other regular file under the evidence root exactly once;
- no orphan or stale document/file remains;
- its `bundle_payload_digest` equals the deterministic SHA256 over sorted `(relative_path, file_sha256)` pairs for every file except the evidence-matrix summary itself.

Add one failing test for each mismatch class before implementation. Also test that pre-seal validation succeeds without an evidence-matrix summary, while strict bundle validation fails until the summary/closed ledgers are added and then succeeds.

- [ ] **Step 2: Implement bundle validation**

Use standard-library paths, hashes, and JSON. Reject path traversal and symlink escape.

- `validate_evidence_documents` validates all currently present non-summary documents, hashes, commits, paths, invocation-to-compilation roles, and result-to-invocation semantics without requiring an evidence-matrix summary or ledger closure. It returns a pre-seal index for summary generation.
- `validate_evidence_bundle` runs only after sealing. It additionally requires exactly one evidence-matrix summary, complete ledger closure, and no orphan files, then returns the validated index, evidence-matrix path/SHA, actual completed/stop stage, and deterministic `bundle_payload_digest`.

- [ ] **Step 3: Make summaries consume validated bundles**

Before aggregation, `summarize_results.py` must call `validate_evidence_documents` on the common unsealed evidence root. It computes only from that pre-seal validated index and preserves input hashes/commit in the output. It must not call the strict closure validator before the evidence-matrix summary exists.

- [ ] **Step 4: Implement the one-commit evidence matrix runner**

`run_evidence_matrix.py` runs all applicable stages under its `ROUTE_C_COMMIT`:

1. verify accepted sources before device imports;
2. run and validate preflight;
3. stop on the complete preflight decision table;
4. when allowed, run/audit synthetic controls;
5. when allowed, run/audit r001/r002 kernel-span and coarse variants;
6. when allowed, run/audit conditional deep variants;
7. when allowed, run accepted control, eager diagnostic control, and graph diagnostic probes;
8. run `validate_evidence_documents` and generate non-matrix summaries from its pre-seal index;
9. build complete document/artifact ledgers and the non-circular bundle payload digest;
10. emit exactly one evidence-matrix summary with completed/skipped stages, actual stop stage, exact causes, ledgers, and digest;
11. run `validate_evidence_bundle` on the sealed closed bundle.

The runner invokes the existing scripts as subprocesses, passes one output root, and never searches global Triton caches by kernel name. Development artifacts from earlier commits are not inputs.

- [ ] **Step 5: Run local tests**

```bash
python3 -m unittest experiments/bi150-kperfir-value/tests/test_evidence_bundle.py -v
python3 -m unittest experiments/bi150-kperfir-value/tests/test_summary.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit all final executable code**

```bash
git add experiments/bi150-kperfir-value/lib/evidence_bundle.py \
        experiments/bi150-kperfir-value/scripts/run_evidence_matrix.py \
        experiments/bi150-kperfir-value/scripts/summarize_results.py \
        experiments/bi150-kperfir-value/tests/test_evidence_bundle.py \
        experiments/bi150-kperfir-value/README.md
git commit -m "feat(experiment): bind BI150 evidence under one commit"
```

Set this immutable revision as the only final evidence revision:

```bash
export EVIDENCE_COMMIT="$(git rev-parse HEAD)"
export ROUTE_C_COMMIT="$EVIDENCE_COMMIT"
export ROUTE_C_REMOTE_ROOT="/tmp/kernelswift-route-c-${EVIDENCE_COMMIT}"
```

- [ ] **Step 7: Deploy the evidence commit and rerun every applicable stage**

Use the Exact Remote Deployment Protocol, then select `max-stage` from the furthest stage whose implementation and prior development gates completed. Run:

```bash
ssh "$BI150_SSH" "export ROUTE_C_COMMIT='$EVIDENCE_COMMIT'; \
  export COREX_VERSION=4.4.0; \
  . /usr/local/corex/enable; \
  cd '$ROUTE_C_REMOTE_ROOT'; \
  python3 experiments/bi150-kperfir-value/scripts/run_evidence_matrix.py \
    --output-root experiments/bi150-kperfir-value/artifacts/by-commit/$EVIDENCE_COMMIT \
    --samples 100 \
    --max-stage '$MAX_VALIDATED_STAGE'"
```

`MAX_VALIDATED_STAGE` is exactly one of `preflight`, `synthetic`, `eager`, `deep`, or `graph`, recorded in the matrix summary. If a gate fails earlier during the frozen rerun, the matrix stops there and records that actual stage.

- [ ] **Step 8: Collect and validate the frozen evidence bundle**

```bash
rm -rf "experiments/bi150-kperfir-value/artifacts/by-commit/$EVIDENCE_COMMIT"
mkdir -p "experiments/bi150-kperfir-value/artifacts/by-commit/$EVIDENCE_COMMIT"
rsync -a --delete "$BI150_SSH:$ROUTE_C_REMOTE_ROOT/experiments/bi150-kperfir-value/artifacts/by-commit/$EVIDENCE_COMMIT/" \
  "experiments/bi150-kperfir-value/artifacts/by-commit/$EVIDENCE_COMMIT/"
python3 - <<'PY'
import os
import sys
from pathlib import Path
root = Path('experiments/bi150-kperfir-value')
sys.path.insert(0, str(root / 'lib'))
from evidence_bundle import validate_evidence_bundle
commit = os.environ['EVIDENCE_COMMIT']
bundle = validate_evidence_bundle(root / 'artifacts' / 'by-commit' / commit)
print(bundle['evidence_matrix_path'])
print(bundle['evidence_matrix_sha256'])
print(bundle['bundle_payload_digest'])
print(bundle['actual_stop_stage'])
PY
```

Only this validated directory may support Task 9. If the frozen rerun cannot complete within the five-day timebox, final classification is `inconclusive` and the exact last valid stage is preserved.

---

### Task 9: Final Assessment and Repository Verification

**Files:**
- Create: `experiments/bi150-kperfir-value/assessment.md`
- Modify: `experiments/bi150-kperfir-value/README.md`

**Interfaces:**
- Consumes only the frozen `EVIDENCE_COMMIT` bundle through `validate_document` and `validate_evidence_bundle`
- Produces exactly one classification: `valuable`, `technically-valid-low-value`, `perturbation-invalid`, `unsupported`, or `inconclusive`

- [ ] **Step 1: Validate every completed JSON document**

```bash
python3 - <<'PY'
import json
import os
import sys
from pathlib import Path
root = Path('experiments/bi150-kperfir-value')
sys.path.insert(0, str(root / 'lib'))
from evidence_bundle import validate_evidence_bundle
from result_contract import document_route_c_commit, validate_document
commit = os.environ['EVIDENCE_COMMIT']
evidence_root = root / 'artifacts' / 'by-commit' / commit
seen = set()
for path in sorted(evidence_root.rglob('*.json')):
    payload = json.loads(path.read_text(encoding='utf-8'))
    validate_document(payload)
    if document_route_c_commit(payload) != commit:
        raise SystemExit(f'{path}: document commit mismatch')
    if payload['document_type'] == 'experiment-result':
        if payload['experiment_id'] in seen:
            raise SystemExit(f'{path}: duplicate experiment id')
        seen.add(payload['experiment_id'])
    print(f'valid {path}')
validate_evidence_bundle(evidence_root)
print(f'valid evidence bundle {evidence_root}')
PY
```

Expected: every frozen-commit JSON has a recognized `document_type`, every cross-document binding validates, and no development-commit artifact is consumed.

- [ ] **Step 2: Write the fixed assessment sections**

```text
Environment fingerprint
Route C commit ledger
Evidence-matrix relative path and SHA256
Bundle payload digest
Actual completed/stop stage
Accepted and diagnostic source hashes
Stage-0 observability result
Completed experiments
Skipped experiments and stop gate
Correctness
Clock monotonicity and synthetic sensitivity
Ordering and completion semantics
Resources and perturbation
r001/r002 local-versus-whole-grid interpretation
Coarse/deep findings
Graph visibility
Incremental decision value
Five-day timebox
Final classification
Port-cost assessment recommendation
```

Declare one `evidence_commit` in the assessment, export the same value as `EVIDENCE_COMMIT`, and consume evidence only from `artifacts/by-commit/$EVIDENCE_COMMIT/`. Record the validator-returned evidence-matrix relative path/SHA256, bundle payload digest, and actual completed/stop stage verbatim. Older commit directories are development provenance only and cannot support final comparisons. Use `issue-window` terminology where required. Reject any assessment sentence that derives region cost, dominance, padding cost, warp cost, or a device upper bound from issue-window-only evidence. Do not assert official SOL, campaign stop, or report changes.

- [ ] **Step 3: Apply the exact classification rule**

- `valuable`: execution-duration evidence may change optimization priority through region cost/dominance, padding-cost, warp-cost imbalance, or a diagnostic device-only upper bound. Valid issue-window evidence may qualify only by recovering graph-interior observability or revealing actionable instruction-issue/ordering behavior; it cannot support cost, dominance, padding-cost, warp-cost, or upper-bound claims.
- `technically-valid-low-value`: all required gates pass but evidence only repeats existing whole-kernel facts.
- `perturbation-invalid`: measured spill/resource/critical-path/overhead/CV perturbation invalidates the probe.
- `unsupported`: a required platform primitive is demonstrated absent or rejected.
- `inconclusive`: final ISA/resource evidence is unavailable, ordering/completion is unproven, the timebox expires, or partial/conflicting evidence prevents a value decision.

A `valuable` result only recommends requesting matching CoreX Triton source and conducting a separate port-cost assessment.

- [ ] **Step 4: Run experiment tests**

```bash
python3 -m unittest discover -s experiments/bi150-kperfir-value/tests -p 'test_*.py' -v
```

Expected: PASS.

- [ ] **Step 5: Run immutable-boundary regressions**

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_run_profile_probe.py -v
```

Expected: PASS.

- [ ] **Step 6: Recheck accepted hashes and ignored artifacts**

```bash
python3 - <<'PY'
import sys
from pathlib import Path
root = Path('experiments/bi150-kperfir-value')
sys.path.insert(0, str(root / 'lib'))
from source_guard import verify_accepted_sources
print(verify_accepted_sources(Path('.').resolve()))
PY
git status --short
git check-ignore experiments/bi150-kperfir-value/artifacts/by-commit/$EVIDENCE_COMMIT/preflight/preflight.json
git diff --check
```

Expected: accepted hashes match, raw artifacts are ignored, and `submissions/` remains untouched.

- [ ] **Step 7: Scan versioned experiment text for unresolved placeholders and credentials**

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('experiments/bi150-kperfir-value')
for path in root.rglob('*'):
    if not path.is_file() or 'artifacts' in path.parts:
        continue
    text = path.read_text(encoding='utf-8', errors='ignore')
    for token in ('T' + 'BD', 'T' + 'ODO', 'fill in ' + 'details'):
        if token in text:
            raise SystemExit(f'{path}: unresolved placeholder')
    lowered = text.lower()
    for token in ('sshpass', 'password=', 'strictHostKeyChecking=no'.lower(), '@zibo.', 'iluvatar.com.cn'):
        if token in lowered:
            raise SystemExit(f'{path}: credential or private endpoint marker')
print('versioned text scan pass')
PY
```

Expected: PASS; no credential or private endpoint is committed.

- [ ] **Step 8: Commit final documentation**

```bash
git add experiments/bi150-kperfir-value/assessment.md \
        experiments/bi150-kperfir-value/README.md
git commit -m "docs(experiment): assess BI150 profiler value"
```

- [ ] **Step 9: Final execution handoff**

Report final classification, strongest valid finding, primary missing/failed gate, source-request recommendation, exact committed files, and confirmation that accepted candidates, official reports, and `submissions/` were not modified.
