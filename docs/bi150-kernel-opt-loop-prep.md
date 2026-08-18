# BI150 Kernel Opt Loop Prep

## Local branch/worktree

- base branch: `dev`
- prep branch: `kernel-opt/bi150-prepare-20260818`
- prep worktree: `.worktrees/bi150-prepare-20260818`

## Remote environment findings

The BI150 host is not immediately ready in a fresh shell. The required runtime
bootstrap is:

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Without this bootstrap, `python3` cannot import `torch` or `triton`, and
`ixsmi` fails to resolve CoreX shared libraries. After enabling CoreX, the
runtime probes succeeded.

## Observed runtime fingerprint

Observed directly on the BI150 host after sourcing `/usr/local/corex/enable`:

- Python: `3.10.18`
- Torch: `2.7.1`
- Triton: `3.1.0`
- Device exposure: `torch.cuda`
- Device name: `Iluvatar BI-V150`
- CUDA capability: `7.1`
- SM count: `16`
- Total memory: `17179869184` bytes (`16 GiB`)
- `ixsmi -L`: reports one `Iluvatar BI-V150`
- CoreX toolchain path: `/usr/local/corex-4.4.0`

## Triton smoke result

A minimal Triton vector-add kernel was compiled and executed successfully on the
BI150 host with:

- `triton.jit`
- `tl.program_id`
- `tl.arange`
- `tl.load`
- `tl.store`

Observed result:

```json
{
  "torch_version": "2.7.1",
  "triton_version": "3.1.0",
  "device": "Iluvatar BI-V150",
  "n": 1024,
  "max_err": 0.0,
  "ok": true
}
```

## Implication for this repository

`auto_bench.py` already probes accelerators in this order:

- `gcu`
- `cuda`
- `npu`
- `mlu`

So on BI150 the current harness should bind to the device through
`torch.cuda` once the CoreX environment is enabled.

## Current state for a full kernel-opt-loop run

A `triton_cuda` target profile has now been added in this BI150 worktree using
this document and the smoke script as the initial repository evidence.

The next remaining step is no longer target-profile absence; it is project
selection and Phase 0 initialization on a BI150-backed optimization project.
That means choosing the first operator, syncing this worktree to the BI150
machine, and collecting baseline correctness / benchmark / profiler artifacts
under the matched runtime fingerprint.

## Recommended next steps

1. Decide which operator/project to optimize first on BI150.
2. Sync this worktree to the BI150 host or clone the repository there.
3. Run Phase 0 baseline setup under the CoreX-enabled environment.
4. Expand the `triton_cuda` profile with new matched probes as the first BI150
   project establishes more primitive evidence.
