# Coder Result 003

## Identity

- Round: `003`
- Decision: `rounds/decision_003.md` (sha256 `a4956891de5fef4b9bd629fb3cceb270db5a247ba18b591aecee9480d96c5455`)
- Decision kind: `optimization`
- Change scope: `host` / change family: `allocation-reuse`
- Sketch: `rounds/sketch_003.json` (sha256 `51ebe3a735c7659309e781fd2f35286fd4e67acc86b5d0a9f6676f08f08af69c`)
- Reference implementation: `triton_mm_encoder_attention_e2_001.py` (sha256 `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124`)
- Reference report: `rounds/report_001.md`
- Candidate: `triton_mm_encoder_attention_e2_003.py`
- Candidate SHA256: `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe`
- Classification: **`candidate-ready`**
- Selected profile: `triton_ascend` (snapshot `state/implementation_profile_snapshot/profile.yaml`, sha256 `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321`)
- Project capability claim: sha256 `a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d`
- Runtime fingerprint: `project.md#runtime-fingerprint` (Ascend910B4, torch 2.7.1+cpu / torch_npu 2.7.1.post4 / triton 3.2.0 / CANN 9.0.0)
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged)

## Validation Gate

Step 1 of the Coder contract ran before any code was written:

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 skills/kernel-opt-loop/scripts/validate_decision.py \
  kernels/track1-triton/mm_encoder_attention/ascend/epoch2/rounds/decision_003.md \
  --expected-implementation-profile triton_ascend \
  --project-root kernels/track1-triton/mm_encoder_attention/ascend/epoch2
```

Exit status `0`; normalized contract reports `"valid":true`. The validator also
re-checked the `sketch_ref` hash, the frozen profile snapshot hash, and the
project capability claim hash and found all three to match the Metadata block.
Language (`triton`), backend (`ascend`), target profile (`triton_ascend`), and
device architecture (`ascend-910b4`) all match the selected profile's Identity
and Match rules. No profile switch occurred.

## Capability Conformance

Every Sketch primitive and target hint was checked against the frozen snapshot
(`profile_status: partial`).

| Construct | Profile status | Used here | Verdict |
|---|---|---|---|
| `tl.load` / `tl.store` (contiguous) | `supported` | yes | conformant |
| `tl.arange` extents 64 / 128 | `constrained` | 64 (`HEAD_DIM`), 128 (`BLOCK_M`, `BLOCK_N`) | conformant, inside the probed set |
| `tl.program_id` axis 0 | `supported` | yes | conformant |
| `tl.where`, `tl.max`, `tl.sum`, `tl.exp` | `supported` | yes | conformant |
| `tl.dot` fp16 x fp16 -> fp32 | `constrained` | yes, tiles `(128,64,128)` and `(128,128,64)` | conformant; unchanged from the accepted round-001 kernel |
| `num_warps=4` | `constrained` (1/2/4/8 legal) | yes | conformant |
| `num_stages=1` | `constrained` (1/2/3/4 legal) | yes | conformant |
| `make_block_ptr`, `async_copy`, `vectorize`, `lifecycle.fast-launcher` | `unknown` | **not used** | no Unknown construct declared normative |

The host-side change introduces no Triton construct at all. `torch.empty` is
host allocation and is outside the capability matrix, so no capability in the
frozen profile is newly required. This matches the decision's own finding and
means there is no basis for a `capability-miss` classification.

## Implementation Summary

The candidate is the round-001 kernel plus a host-side output buffer cache. The
`_fused_attention_kernel` body is **byte-identical** to the canonical source,
verified by diff over lines 1-74 (exit status `0`, no output).

Host changes, all confined to `ModelNew.__init__` and `ModelNew.forward`:

1. **Output allocation moved to an instance cache.** `torch.empty_like(query)`
   is replaced by a lazily created `torch.empty(...)` buffer held on the
   instance. The cache key `(query.shape, query.dtype, query.device,
   query.stride())` is built and compared on **every** call; a mismatch
   discards the buffer and reallocates. A cache hit performs zero allocations.
2. **Launch constants hoisted to `__init__`.** `self._num_heads`,
   `self._head_dim`, and `self._block = 128` replace the per-call
   `self.num_heads`, `self.head_size` local rebinding, and the `block = 128`
   literal. The public attributes `num_heads`, `head_size`, and `num_kv_heads`
   remain readable and unchanged.
3. **Stride unpacking retained.** `bsz, q_len, hidden = query.shape` and the
   `query.stride(0)` / `query.stride(1)` launch arguments are unchanged, per
   decision section 4, because the strides are simultaneously launch arguments
   and cache-key components.

No lock, thread-local, stream, module state, or extra launch was introduced.
No file outside the declared outputs was written.

## Sketch Conformance

| Sketch requirement | Status |
|---|---|
| `target=triton_ascend` (required) | honored |
| `BLOCK_M=128` (required) | honored, unchanged |
| `BLOCK_N=128` (required) | honored, unchanged |
| `HEAD_DIM=64` (required) | honored, unchanged |
| `accumulator_dtype=fp32` (required) | honored, unchanged |
| `num_warps=4` (preferred) | honored, unchanged |
| `num_stages=1` (preferred) | honored, unchanged |
| `op_alloc_out`: "alloc out ... on the ModelNew instance; a cache hit performs no allocation" | implemented exactly |
| `op_load_q` / `op_load_k` / `op_load_v` | kernel body byte-identical |
| `op_qk`, `op_mask_and_softmax`, `op_pv`, `op_normalize` | kernel body byte-identical |
| `op_store_out` | kernel body byte-identical; store coverage unchanged, so the reuse invariant holds |
| `c_parallel_bh` (`parallel bh over B*NH`) | grid `(bsz * self._num_heads,)` = `(16,)` |
| `c_guard_row` (`row_idx < S`), `c_guard_col` (`offs_n < S`) | byte-identical |
| `scope.kind = unchanged-computation-boundary` | respected; see the boundary table below |
| `scope.entrypoints = [ModelNew.forward, ModelNew.__init__]` | only those two methods touched |

### Unchanged computation boundary (verified)

| Boundary element | Evidence |
|---|---|
| fused attention kernel body | `diff` of lines 1-74 vs `triton_mm_encoder_attention_e2_001.py` → exit `0`, no output |
| kernel launch count stays at one | exactly one `_fused_attention_kernel[grid](...)` site in the file, same as canonical |
| `BLOCK_M` / `BLOCK_N` | `128` / `128`, from `self._block` |
| `HEAD_DIM` | `64`, from `self._head_dim` |
| accumulator dtype | fp32, `tl.dot` on fp16 inputs unchanged |
| `num_warps` / `num_stages` | `4` / `1` |
| output shape / dtype / device / contiguity | observed `[2, 83, 512]`, `torch.float16`, `npu:0`, `is_contiguous() == True` |
| numerical tolerance `atol=1e-2 rtol=1e-2` | smoke `PASS accuracy`; max abs diff vs round-001 output `0.0` |

## Host Plan Conformance

| Host Plan field | Implementation | Verdict |
|---|---|---|
| `state_owner` | `self._out_buffer` / `self._out_cache_key`, ordinary instance attributes | conformant |
| `lifetime` | created on the first forward or after a key change, held until instance release | conformant |
| `allocation_reuse` | `torch.empty(query.shape, dtype=query.dtype, device=query.device)`; never `empty_like` | conformant |
| `cache_key` | `(query.shape, query.dtype, query.device, query.stride())` — shape, dtype, device, stride | conformant |
| `invalidation` | the full key tuple is compared on every call; any difference reallocates | conformant |
| `concurrency` | no lock, thread-local, or per-call state; single sequential call stream | conformant |
| `device_stream_behavior` | buffer on `query.device`; no stream created, captured, or switched; harness `torch.npu.synchronize()` boundary untouched | conformant |
| `unchanged_behavior` | all eight items preserved; observed below | conformant |

### Observed host invariants (probe, no timing)

Run in one process through the real harness AST loader. These are conformance
facts, **not** measurements; no wall time was sampled.

```text
shape (2, 83, 512) torch.float16 npu:0 contig True
cache_hit_same_ptr True
state_dict {}
alias_of_qkv False
max_abs_diff_vs_001 0.0
poisoned_buffer_leaks_nan False
ptr_stable_after_poison True
stride_miss_realloc True
shape_miss_realloc True (2, 40, 512)
return_to_original_shape_realloc True
```

Reading of each line:

- `cache_hit_same_ptr True` — two consecutive forwards at fixed shape return the
  same `data_ptr()`, so steady state performs zero allocations.
- `state_dict {}` — the buffer is an ordinary attribute, not module state, and
  is never serialized. This is the `state_owner` clause made observable.
- `alias_of_qkv False` — the returned tensor does not share storage with
  `query`, `key`, or `value`.
- `max_abs_diff_vs_001 0.0` — bit-identical to the accepted round-001 output,
  so the host change is numerically inert.
- `poisoned_buffer_leaks_nan False` — the cached buffer was filled with NaN and
  then reused; the returned tensor contains no NaN. This is a direct proof of
  the "cached buffer is fully overwritten by the kernel store on every call"
  guardrail, which is the load-bearing safety property of the whole round.
- `ptr_stable_after_poison True` — reuse survives the poisoning, confirming the
  overwrite came from the kernel store and not from a reallocation.
- `stride_miss_realloc True` — a same-shape, differently-strided query
  (`as_strided` with a different batch stride) is a **miss** and reallocates,
  never a silent reinterpretation.
- `shape_miss_realloc True`, `return_to_original_shape_realloc True` — shape
  changes invalidate in both directions.

## Deviations

None.

Two conformance notes, both preserving all normative semantics:

1. **Private constructor aliases.** `self._num_heads`, `self._head_dim`, and
   `self._block` mirror `num_heads`, `head_size`, and the `block` literal. This
   is the hoist the decision's section 4 calls for. Consequence: if a caller
   mutated `model.num_heads` after construction, the launch would still use the
   constructor value. The public attributes remain readable and unchanged, and
   nothing in the harness mutates them (it calls only `.eval()` and, in the
   profiling path, `.to()`).
2. **Guard ordering preserved.** The `q_len > block` `ValueError` guard keeps
   its round-001 position, after the cache lookup. The unsupported-shape path
   never executes at the campaign shape `S=83`; keeping the original order
   avoids any behavioural-ordering question.

## Compile and Smoke Evidence

- Command:
  ```bash
  cd /workspace/kernelswift-dev-4ff2094
  python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py \
    --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_003.py \
    --warmup 5 --repeat 10 --full-traceback
  ```
- Exit status: `0`
- Observation: `PASS accuracy; v0=0.370530 ms, v1=0.307365 ms, speedup=1.206x`
  followed by `Summary: 1 passed, 0 failed, 1 total.`
- Correctness passes at the harness default tolerance (`atol=1e-2`,
  `rtol=1e-2`, `equal_nan=True`).
- Corroborating: max absolute difference against the accepted round-001 output
  is `0.0`, so the host change cannot have altered numerics.
- **This is a smoke-tier number at `warmup 5 / repeat 10`. It is not the
  adoption measurement.** Verifier owns the `warmup 50 / repeat 100` interleaved
  wall comparison exclusively. Coder did not run that regime and did not run
  `--profile`.
- Ancillary observation: in the same-process probe, round-001's
  `torch.empty_like(query)` emitted the recorded internal-format
  `UserWarning` from `TensorFactories.cpp:340`, while the candidate's
  `torch.empty(...)` did not. This confirms the decision's claim that the
  warning belongs to the per-call allocation path this round removes.

### Local gate (Coder contract step 6)

| Gate | Command / check | Result |
|---|---|---|
| `ast.parse` | `python3 -c "import ast; ast.parse(open(candidate).read())"` | `AST_PARSE=ok` |
| real harness loader | `auto_bench.load_ks_module(Path(candidate))` via `python3 -c` | `HARNESS_LOADER=ok; has ModelNew: True ; get_init_inputs: [8, 64, 8]` |
| warm-up / compile smoke | the `auto_bench.py` command above at warmup 5 / repeat 10 | `PASS accuracy`, exit `0` |
| linter | `read_lints` on the candidate | `totalCount: 0` |
| kernel-body identity | `diff` lines 1-74 vs canonical | exit `0`, no output |

## Attempt Ledger

| Attempt | Command | Exit | Defect | Candidate SHA256 before | Candidate SHA256 after |
|---:|---|---:|---|---|---|
| 1 | write candidate; `validate_decision.py`; kernel-body `diff`; `ast.parse`; harness loader; `auto_bench.py --warmup 5 --repeat 10 --full-traceback` | 0 | none | `not-applicable` | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` |

No repair was needed; the candidate passed every gate on the first submission.
One in-flight refinement (hoisting `_num_heads` / `_head_dim` in addition to
`_block`, to complete decision section 4 item 3) was applied **before** the
first gate run and is therefore covered by attempt 1 rather than recorded as a
repair.

## Risks Noted for Verifier

1. **The smoke speedup is not the result.** `1.206x` at `warmup 5 / repeat 10`
   must not be read as the round's effect. For calibration, round 001's smoke
   read `1.094x` while its adopted `warmup 50 / repeat 100` measurement was
   `1.100x`; the two tiers are not interchangeable. The only adoption test is
   the interleaved paired wall median against `base.py` at `warmup 50 /
   repeat 100`.
2. **The lever may be smaller than 16.3885 us.** The round removes exactly one
   allocator round-trip per call. If the NPU caching allocator serves that
   request from a free block in low single-digit microseconds, the mechanism
   observables can all move as predicted while wall time fails to clear the 5%
   threshold. That is the dominant adoption risk and it is a magnitude risk, not
   a correctness risk.
3. **`output_allocations_per_call` is the observable to instrument first.** It
   is the only mechanism observable that isolates this round's link directly. If
   it goes to 0 and `host_us_per_call` does not move, the residual is elsewhere
   in the host path and the round is `no-improvement` on magnitude.
4. **Control observables must be checked before attributing anything to host.**
   `device_us_per_call` should stay near `13.4064` and `kernel_count_per_call`
   at `1.00`. If wall moves while device time also moves, the attribution is
   broken and the round must not be adopted on a host story.
5. **One extra `torch.device` construction per call.** `query.device` builds a
   new `torch.device` object on every call because the decision names "output
   device" as a cache-key component that must be compared on every call. If
   `host_us_per_call` fails to decrease, this is the first candidate to isolate.
6. **Steady-state assumption.** `auto_bench.time_forward` passes the same
   `inputs` list on every iteration (it does not clone per iteration), so the
   measurement regime is a pure cache-hit stream after the first warm-up call.
   This is the regime the decision targets.
7. **`.to()` does not move the cached buffer.** The buffer is a plain attribute,
   so `nn.Module.to()` in `build_profile_reference` will not carry it. The
   buffer is created lazily on the first forward, which happens after `.to()` in
   that path, and the device component of the cache key would catch any
   mismatch regardless. Relevant only if Verifier runs `--profile`.
8. **The reuse invariant is contingent on full store coverage.** It holds
   because the `(16,)` grid writes every row `0..82` of every head slice. It was
   verified empirically by poisoning the buffer with NaN. If a later round
   introduces a masked or partial store, this Host Plan must be revisited.
9. **Round-001 risk carried forward unchanged:** the second `tl.dot` tile
   `(128,64,128)` compiles and is numerically correct here but was not one of
   the eleven probed tiles.
