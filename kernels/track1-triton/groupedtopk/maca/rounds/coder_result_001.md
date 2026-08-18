# Coder Result 001

Result: `candidate-ready`

- reason_code: `all-required-coder-gates-passed`
- round: `001`
- source_canonical: `baseline_adapter.py`
- source_canonical_sha256: `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827`
- decision: `rounds/decision_001.md`
- decision_sha256: `c71c970e3bcf6d7984272611627d711ce64b6f3c18d1a057b2aab440c50c173f`
- candidate: `triton_grouped_topk_001.py`
- candidate_sha256: `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384`
- selected_profile: `triton_maca`
- selected_profile_sha256: `2cfa08c2664f01e70bb43eec7bb998be836a6a719b17535268a8d6ca18c85540`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- runtime_identity: `Python 3.12.11; torch 2.8.0+metax3.5.3.9; Triton 3.0.0+metax3.5.3.9; MACA 3.5.3.26; GPUTarget(backend='maca', arch=80, warp_size=64); MetaX C500`
- measurement_fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`

## Gate Summary

| Gate | Return code | Evidence |
|---|---:|---|
| Decision validation | 0 | `validate_decision.py --expected-profile triton_maca .../rounds/decision_001.md` returned `"valid": true`. |
| Final `ast.parse` | 0 | Parsed the complete 284-line candidate at SHA `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384`. |
| Actual `auto_bench` AST loader | 0 | Covered by the Orchestrator-assisted remote `auto_bench.py` invocation below; local AST-filter precheck also returned 0 with minimal import stubs. |
| Current-regime compile/smoke | 0 | Remote SHA matched exactly; `PASS accuracy`; `Summary: 1 passed, 0 failed, 1 total.` |

The remote `--warmup 1 --repeat 1` output reported
`v0=0.335423 ms, v1=0.097561 ms, speedup=3.438x`. These values are smoke
timing only and are not formal measurement evidence or an adoption claim.

## Primitive and Hint Conformance

- Imports use only `torch`, `torch.nn`, `triton`, and
  `triton.language`; `triton_gcu`, `torch_gcu`, `triton_cuda`, and
  `fast_libentry` are absent.
- The fixed path launches one program per token directly with
  `BLOCK_E=256` and `num_warps=1`; no `num_stages`, block pointer,
  vectorization directive, launcher replacement, or `tl.static_range` is
  present.
- `tl.load`, `tl.store`, 256/8-wide `tl.arange`, `tl.reshape`,
  `tl.broadcast_to`, 8x32 group `tl.max`, 8-wide group `tl.argmax`,
  `tl.where`, 256-wide `tl.sum`, and `tl.exp` follow the immutable
  raw-logit sketch. The four group selections and eight expert selections are
  textually unrolled.
- The required repeated 256-wide expert `tl.argmax` was the explicit compile
  gate. It compiled and executed through the real remote harness at the exact
  fixed regime; it was not silently replaced.
- Selected values are extracted with a 256-wide masked `tl.sum`, and the final
  denominator is the explicit sum of eight shifted scalar exponentials. This
  preserves the decision's exact softmax-denominator cancellation.
- Static inspection found exactly 12 `tl.argmax` calls (4 group + 8 expert),
  and the PyTorch fallback AST after the dispatch branch is identical to
  `baseline_adapter.py`.
- The fast-path guard covers exact shape, dtype, contiguous layout, shared
  CUDA-compatible device, constructor values, and no-required-autograd
  conditions. The token-count assertion precedes dispatch. Output allocation is
  per-forward on `gating_output.device`; no cache, global mutable state,
  device switch, stream switch, or input mutation is introduced.
- Fixed seeded correctness passed in smoke. Targeted equal-logit group-cutoff
  and expert-cutoff ID parity remains an authoritative Verifier check; no
  epsilon, ID perturbation, or approximate tie key was added.

## Attempt Ledger

| Attempt | Command | Exit status | Defect / evidence | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | Scoped initial candidate patch | 0 | Patch hunk line count truncated the file at line 228; non-semantic materialization defect discovered by AST gate. | `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827` | `73f5d2b008bb7a8c0607570b66a506d965c743819892e663d4c9c7dcd58ea37f` |
| 2 | `ast.parse(triton_grouped_topk_001.py)` | 1 | `SyntaxError: '(' was never closed` at the truncated `fast_path` assignment. | `73f5d2b008bb7a8c0607570b66a506d965c743819892e663d4c9c7dcd58ea37f` | `73f5d2b008bb7a8c0607570b66a506d965c743819892e663d4c9c7dcd58ea37f` |
| 3 | Scoped `git apply --recount` tail repair | 0 | Completed the already-designed guard, direct launch, unchanged fallback, and public entry points; no algorithm or dataflow change. | `73f5d2b008bb7a8c0607570b66a506d965c743819892e663d4c9c7dcd58ea37f` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` |
| 4 | Complete `ast.parse(triton_grouped_topk_001.py)` | 0 | None. | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` |
| 5 | Local actual `auto_bench.load_ks_module(candidate)` | 1 | Local WSL Python lacked `torch`; failure occurred while importing `auto_bench.py`, before candidate loading. Environment note, not a candidate defect. | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` |
| 6 | Actual `auto_bench.load_ks_module(candidate)` with minimal local import stubs | 0 | AST-filter/compile/exec precheck only; explicitly not a substitute for the true runtime loader. | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` |
| 7 | Subagent `scp` to the designated C500 path | not-run | Sandbox policy denied external upload before command execution; no remote attempt or smoke was consumed. | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` |
| 8 | Orchestrator-assisted remote SHA check and `c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/triton_grouped_topk_001.py --warmup 1 --repeat 1 --full-traceback` | 0 | Remote SHA matched; real AST loader, Triton-MACA compile, execution, and fixed seeded correctness passed. | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` |

One of the two permitted non-semantic repair attempts was used. No semantic
repair, algorithm substitution, or decision revision was required.
