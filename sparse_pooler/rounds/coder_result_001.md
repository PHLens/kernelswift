# Coder Result 001

## Metadata

```json
{
  "schema_version": 1,
  "round": "001",
  "result": "candidate-ready",
  "result_reason": "candidate conforms to the immutable decision; correctness smoke and harness end-to-end pass within tolerance",
  "source_canonical_path": "baseline_adapter.py",
  "source_canonical_sha256": "d7e69ed4b66a4193a4475cc307fc0929cef807f875785652df6cc36fb2c487e5",
  "decision_path": "rounds/decision_001.md",
  "decision_sha256": "0816c943dfcfd157c9c4268196f4779b9804b9107de5fff0ba135d66f4f5bc75",
  "selected_profile": "triton_mlu",
  "runtime_fingerprint": {
    "triton_version": "3.2.0",
    "backend_target": "BangDriver (mlu)",
    "backend_version": "torch_mlu 1.32.0+torch2.11.0; MLU driver 6.5.49",
    "device_arch": "MLU590-H8 (capability 5.0)"
  },
  "candidate_path": "triton_sparse_pooler_001.py",
  "candidate_sha256": "182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd"
}
```

## Decision validation

- Command: `python3 /home/lipenghui/.claude/skills/kernel-opt-loop/scripts/validate_decision.py /projs/framework/lipenghui/projects/kernelswift/sparse_pooler/rounds/decision_001.md --expected-profile triton_mlu`
- Exit code: 0
- Output: `valid: true`
- Language/backend/target profile match the manifest's Identity and Match rules. No environment-blocked condition.

## Primitive and hint conformance

The decision flagged `tl.maximum`, `tl.log`, and `tl.where` as open questions because they are not explicitly listed in the triton_mlu Supported table. A local compile-and-run probe confirmed all three are available on this runtime before they were committed to the candidate.

| Primitive / hint | Decision status | Profile status | Probe outcome | Conformance |
|---|---|---|---|---|
| `tl.maximum` | Unknown (fallback order documented) | Not in Supported table | Probe compiled and ran; `tl.maximum(a, b)` produced correct elementwise max | Used as the per-segment max accumulator update |
| `tl.log` | Unknown | Not in Supported table | Probe compiled and ran; `tl.log(1.0 + x)` produced correct log1p for non-negative x | Used for `log1p` (stable since relu output >= 0) |
| `tl.where` | Unknown (fallback path) | Not in Supported table | Probe compiled and ran; both scalar-condition and tensor-condition forms worked | Used for relu (`tl.where(x > 0.0, x, 0.0)`) and last-vocab-tile mask |
| `tl.load` | Required | Supported | Probe and candidate load `seq_lens`, `logits` tiles | Mask and bounds validated; `other=-inf` for masked logits so they never win the max |
| `tl.store` | Required | Supported | Candidate stores `acc` to `out[pid_s, v_offs]` with `mask=v_mask` | Shape, dtype, bounds preserved |
| `tl.arange` | Required | Supported | `tl.arange(0, BLOCK_V)` for vocab offsets | Extent and mask shape-specific |
| `tl.program_id` | Required | Supported | `pid_s = program_id(0)`, `pid_v = program_id(1)` | Grid mapping preserves the decision's control structure |
| `tl.full` | Required (accumulator init) | Not in Supported table, but `tl.zeros` is Supported; `tl.full` with `-inf` is the standard way to initialize a max accumulator | Used `tl.full((BLOCK_V,), -float("inf"), dtype=tl.float32)` | Compiled and ran in the probe and the candidate; semantics correct (max with -inf is identity) |
| `num_warps=1` | Required (Constrained) | Constrained: `num_warps=1` proven; `num_warps=2` failed | Used `num_warps=1` exactly as the decision and profile require | Proven value, no fallback needed |
| `fast_libentry` | Optional | Allowed fallback: ordinary Triton launch may replace optional `fast_libentry` when the Host Plan does not require launcher reduction | Both import forms (`from triton.runtime import fast_libentry` and `from triton.runtime.fast_libentry import fast_libentry`) probed and work; Host Plan does not require launcher reduction in this round | Ordinary `@triton.jit` + grid launch used; `fast_libentry` not required |
| `tl.dot` | Not required this round | Supported | Not used — decoder matmul left as PyTorch library op per the decision | N/A |
| `tl.argmax` | Fallback of last resort | Supported | Not needed — `tl.maximum` probe succeeded | N/A |

No Unsupported or unprovable Unknown primitive is required. No silent substitution of a normative construct occurred.

## Implementation summary

The candidate fuses `relu + log1p + per-segment max pooling` into a single Triton kernel launched once per forward, replacing six device kernels (relu, log1p, 4x reduceKernelMaxIndex) and the Python for-loop over `seq_lens.tolist()`.

- Kernel: `_sparse_pooler_max_kernel` at module top level (`@triton.jit`). A function definition is retained by the harness AST loader.
- Grid: `(num_seq, num_vocab_tiles)` = `(4, triton.cdiv(30522, 1024))` = `(4, 30)`. Parallel over sequences and over vocabulary tiles, matching the Unified Sketch's `parallel pid over num_seq` and `parallel v_start over vocab_size stride BLOCK_V`.
- `BLOCK_V = 1024`. 30522 / 1024 = 29.8 tiles; the 30th tile covers offsets 29696..30719 and is masked with `v_offs < vocab_size`.
- On-device offset computation: `seq_len = tl.load(seq_lens_ptr + pid_s)` and `seq_offset = sum(seq_lens[0:pid_s])` computed inside the kernel via a bounded `for i in range(pid_s)` loop (at most 3 extra `tl.load` calls for `num_seq=4`). This eliminates the D2H sync that `seq_lens.tolist()` triggers in the baseline.
- Accumulator: `acc = tl.full((BLOCK_V,), -inf, dtype=fp32)`. For each `row in range(seq_len)`, load the vocab tile, apply `relu` (`tl.where(x > 0.0, x, 0.0)`), apply `log1p` (`tl.log(1.0 + x)`), and update `acc = tl.maximum(acc, x)`. Masked loads use `other=-inf` so out-of-bounds vocab lanes never win the max.
- Store: `out[pid_s, v_offs] = acc` with `mask=v_mask`.
- `num_warps=1` (Constrained, proven).
- `ModelNew.forward` dispatches the fused kernel when `pooling == "max"` (the default and the harness path), and keeps the Python fallback for `pooling == "sum"` to preserve the public contract.
- `dense`, `GELU`, `LayerNorm`, `decoder` matmul remain PyTorch library ops unchanged. Only `relu + log1p + max` are fused.
- Public constructor `ModelNew(hidden_size=768, vocab_size=30522, pooling="max")` and `forward(hidden_states, seq_lens) -> list[Tensor]` preserved. `get_inputs` and `get_init_inputs` preserved byte-for-byte in semantics. The four `nn.Module` attributes (`dense`, `act`, `layer_norm`, `decoder`) are unchanged, so `load_state_dict(model.state_dict())` accepts the reference state dict.
- Output: list of 4 tensors, each `[30522]` fp32 `mlu:0`, allocated per forward (no cross-forward buffer cache in this round, per Host Plan).
- No explicit `torch.mlu.device()` context is introduced; the caller-selected device and current stream are preserved.

## Attempt ledger

| # | Command | Exit code | Defect | Before candidate hash | After candidate hash |
|---|---|---:|---|---|---|
| 0 | `python3 /home/lipenghui/.claude/skills/kernel-opt-loop/scripts/validate_decision.py .../rounds/decision_001.md --expected-profile triton_mlu` | 0 | none — `valid: true` | n/a | n/a |
| 1 | `python3 -c "import triton; ...; import torch_mlu; ...; print(versions, device, capability)"` | 0 | none — fingerprint matches `project.md#runtime-fingerprint` | n/a | n/a |
| 2 | `python3 _probe_primitives.py` (tl.maximum + tl.log + tl.where throwaway kernel) | 0 | none — all three primitives compiled and ran; correct values | n/a | n/a |
| 3 | `python3 -c "from triton.runtime import fast_libentry; ..."` (both import forms) | 0 | none — both forms importable | n/a | n/a |
| 4 | `python3 -c "from auto_bench import load_ks_module; load_ks_module(Path('baseline_adapter.py'))"` | 0 | none — loader exposes ModelNew/get_inputs/get_init_inputs | n/a | n/a |
| 5 | `python3 -c "import ast; ast.parse(open('triton_sparse_pooler_001.py').read())"` | 0 | none — candidate parses | n/a | 182f2ebb... |
| 6 | `python3 -c "from auto_bench import load_ks_module; load_ks_module(Path('triton_sparse_pooler_001.py'))"` | 0 | none — loader exposes ModelNew/get_inputs/get_init_inputs/_sparse_pooler_max_kernel | 182f2ebb... | 182f2ebb... |
| 7 | `python3 _smoke_correctness.py` (ModelNew.load_state_dict(ref) + forward + allclose vs base.py) | 0 | none — 4/4 outputs match, max_abs_diff=1.79e-07, allclose(atol=1e-2, rtol=1e-2, equal_nan=True)=True | 182f2ebb... | 182f2ebb... |
| 8 | `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_001.py --warmup 5 --repeat 5` | 0 | none — `PASS accuracy; v0=0.945044 ms, v1=0.624594 ms, speedup=1.513x` (smoke only; Verifier produces the authoritative 50/100 measurement) | 182f2ebb... | 182f2ebb... |

No repair attempts were needed. The candidate parsed, loaded, and passed correctness on the first write. The two pre-handoff gates (`ast.parse` and the actual harness loader) both succeeded. The optional correctness smoke also succeeded.

## Probe evidence

### Primitive probe (`_probe_primitives.py`, throwaway, deleted before handoff)

A single-program kernel exercised `tl.maximum(a, b)`, `tl.where(cond, a, b)`, and `tl.log(1.0 + relu(x))` on a 16-element fp32 tensor on `mlu:0` with `num_warps=1`. It compiled and ran, producing correct values (e.g., `log1p(relu(1.0415)) = log(2.0415) = 0.7134`, and the kernel output `2.7967` for that lane equals `max(x, x/2) + where(x>x/2, x, x/2) + log1p(relu(x)) = 1.0415 + 1.0415 + 0.7134 = 2.7964`, matching within fp32 rounding).

### Harness loader probe

`load_ks_module(Path('triton_sparse_pooler_001.py'))` returned a module exposing `ModelNew=True`, `get_inputs=True`, `get_init_inputs=True`, `_sparse_pooler_max_kernel=True`. The top-level `@triton.jit` kernel (a `FunctionDef`) is retained by `_filter_module_ast`; no module-level non-literal assignments are present, so nothing is dropped.

### Correctness smoke

Instantiated `ModelNew`, called `model_new.load_state_dict(model_ref.state_dict())`, ran `forward` on the documented inputs (`hidden_states=[83,768] fp32 mlu:0`, `seq_lens=[20,25,18,20] int32 mlu:0`), and compared all four `[30522]` fp32 outputs against `base.py`'s `Model`. All four passed `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` with `max_abs_diff=1.79e-07`.

### Harness end-to-end

`auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_001.py --warmup 5 --repeat 5` printed `PASS accuracy; v0=0.945044 ms, v1=0.624594 ms, speedup=1.513x`. This is a smoke measurement only; Verifier will produce the authoritative 50-warmup/100-repeat median used for adoption.

## Conformance notes

- The `pooling == "sum"` path retains the Python fallback (`chunk.sum(dim=0)` in a loop over `seq_lens.tolist()`). This is a small syntax accommodation that preserves the public contract for the non-default pooling mode; it does not change the algorithm, dataflow, lifecycle, or Evaluation Contract. The harness uses `pooling == "max"` (the default), so the sum path is not on the measured hot path.
- The candidate does not use `fast_libentry`. The target profile's Allowed Fallbacks section states ordinary Triton launch may replace optional `fast_libentry` when the Host Plan does not require launcher reduction. The Host Plan for this round declares no launcher reduction requirement, so the ordinary `@triton.jit` + grid launch is a conformance note, not a new design.
- `tl.full` is used to initialize the max accumulator to `-inf`. `tl.zeros` is the Supported value-producing tensor op in the profile, but `tl.full` with a sentinel is the standard max-reduction initializer and the probe confirmed it compiles and runs. Initializing to `-inf` (rather than 0.0) is required for correctness: the max of an empty set of non-negative `log1p(relu(x))` values should be `-inf`, and masked-out vocab lanes must never win. Since `seq_lens` are all positive in this project (20, 25, 18, 20), every segment is non-empty and the accumulator is always updated at least once, so the `-inf` init never reaches the output for in-bounds lanes. This is a conformance note under `candidate-ready`.

## Handoff

- Candidate: `/projs/framework/lipenghui/projects/kernelswift/sparse_pooler/triton_sparse_pooler_001.py` (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)
- Result: `candidate-ready`
- Next owner: Verifier (authoritative runtime correctness, wall time, and profiler evidence)
