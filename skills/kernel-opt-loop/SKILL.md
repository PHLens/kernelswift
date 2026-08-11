---
name: kernel-opt-loop
description: Iterative Triton kernel optimization loop on MLU/Cambricon (or any accelerator with an auto_bench-style harness). Use when the user asks to "optimize an operator", "improve a kernel's speedup", or mentions Triton + auto_bench + a base.py reference. Drives the loop: run baseline → write v1 Triton kernel → measure correctness + wall time → update log.md (groupedtopk format) → commit per round on a branch → pick the next single bottleneck. Do NOT use for one-shot bug fixes or non-iterative work.
---

# Kernel Opt Loop

Iterative methodology for taking a PyTorch eager operator (base.py) and producing a sequence of Triton kernels, each round committed with its optimization method + measured speedup, and a log.md that grows one entry per round in the groupedtopk format.

## When to use

- User says "优化 X 算子" / "optimize operator X" + there's a `base.py` reference impl
- An `auto_bench.py` harness exists that takes `--v0_file` / `--v1_file` and reports correctness + wall time
- Triton is the target (MLU/Cambricon Triton, NVIDIA Triton, etc.)
- The work is iterative (multiple rounds, each picking ONE bottleneck)

Do NOT use for: single-shot fixes, pure refactors, or non-iterative one-pass work.

## Required inputs

Before starting, confirm these exist or ask the user:

1. **Operator directory** (e.g. `fused_moe/`, `groupedtopk/`) containing:
   - `base.py` — PyTorch eager reference with `Model`, `get_inputs()`, `get_init_inputs()`
2. **`auto_bench.py`** in the repo root (or path) — takes `--v0_file --v1_file --warmup --repeat`
3. **Python interpreter** that has torch + torch_mlu (or target accelerator) + triton installed
4. **Target device** is set (e.g. `export CUDA_VISIBLE_DEVICES=0` or MLU equivalent)

If `base.py` doesn't exist, ask the user to provide it. Do not write base.py yourself — it's the reference contract.

## Workflow

### Phase 0 — Setup

1. Read `base.py` and identify the operator's shape, dtype, semantic.
2. Read `auto_bench.py` enough to know:
   - How `time_forward` measures (does it call `set_seed` + `sync_devices` per iter?)
   - Whether there's an AST filter (`_filter_module_ast`) that strips non-literal module-level assigns — this affects how you write `fast_libentry` patterns
   - What fields `ModelNew` must expose (`__init__` signature matching `get_init_inputs()`, `forward(hidden_states, router_logits)` matching `get_inputs()`)
3. Pick measurement rules and stick with them for the whole project:
   - **auto_bench wall time is authoritative** — do NOT mix with manual `time.perf_counter()` medians in the same table
   - device time = sum of `dur` for `cat == "kernel"` events in profiler JSON
4. Create a branch: `git checkout -b <op-name>-opt` from master.

### Phase 1 — Round 0: baseline commit

1. Run `auto_bench --v0_file <op>/base.py --v1_file <op>/base.py --warmup 50 --repeat 100` to get the baseline wall time and verify the harness works.
2. Optionally profile: run forward 50 iters under torch.profiler, dump JSON to `<op>/log/<op>_forward_50iter.pt.trace.json` (`**/log/` is gitignored).
3. Write `log.md` in round-0 state — see [log-template.md](references/log-template.md). Round-0 state has:
   - Section 1 (problem + measurement rules)
   - Section 2 (upbound definition — be honest, don't fake a stretch goal)
   - Section 3 (table with ONLY the base.py row)
   - Section 4 (only Entry 000 — "PyTorch eager 起点", no optimization)
   - Section 5 (current bottleneck, post-base state)
   - Section 6 (next direction — P0 only)
   - Section 7 (reproduction command)
   - Section 8 (checkpoint date + "v1 Triton: 待补")
4. Commit: `git add <op>/base.py <op>/log.md && git commit -m "<op>: add eager baseline"`

### Phase 2 — Optimization round N (N ≥ 1)

Each round is ONE bottleneck, ONE .py file, ONE log.md update, ONE commit. Roughly 5 minutes of work.

#### Step 1: Pick the bottleneck

Compute `device_ratio = sum(kernel dur) / wall_time` first (kernel dur = sum of `dur` for all `cat == "kernel"` events in one forward's profiler JSON; wall = auto_bench per-call). This single ratio classifies the round:

- **device-bound** (`device_ratio > 80%`): optimize the kernel itself — loop fusion, `tl.dot` for GEMM, fewer `tl.exp` calls, fewer `tl.load`s.
- **host-bound** (`device_ratio < 20%`): reduce host overhead — `fast_libentry`, cache output buffer, drop `torch.mlu.device()` context, fuse routing PyTorch ops into the kernel.
- **measurement-bound** (host overhead is only `set_seed` + `sync_devices`, the harness's fixed costs): stop, declare done.

For the full procedure — including how to break down host overhead into launcher / `set_seed` / `sync_devices` / harness-state components, how to find the dominant kernel in the trace when device-bound, the compressible-vs-fixed table, and the 5-round fused_moe worked example — see [references/bottleneck-judgment.md](references/bottleneck-judgment.md).

Pick ONE bottleneck class. Don't fix two things in one round — if you do, you can't attribute the speedup. The workflow requires ≥5% stable improvement in the same trace to keep a change; sub-5% is noise.

#### Step 2: Write the new kernel file

- Create `<op>/triton_<op>_<NNN>.py` (zero-padded 3-digit, e.g. `001`, `002`, ...).
- Copy the previous round's file as the starting point. Change only what the chosen bottleneck requires.
- Expose `ModelNew` with the same `__init__` / `forward` / `get_inputs` / `get_init_inputs` contract as `base.py`.
- Keep the kernel body minimal — don't refactor unrelated parts.

#### Step 3: Handle `auto_bench`'s AST filter (if present)

If `auto_bench.py` has a `_filter_module_ast` that strips non-literal module-level assigns, patterns like:

```python
_fast = fast_libentry()(_kernel)  # STRIPPED at filter time → NameError at runtime
```

won't work. Use the class-body `globals()` trick:

```python
@triton.jit
def _kernel(...): ...

class ModelNew(nn.Module):
    if "_fast" not in globals():
        globals()["_fast"] = fast_libentry()(_kernel)

    def forward(self, ...):
        _fast[grid](...)  # resolves via globals()
```

`_filter_module_ast` keeps ClassDef nodes, so the class body runs at import time and populates the module global.

#### Step 4: Run auto_bench

```bash
<python> auto_bench.py \
  --v0_file <op>/base.py \
  --v1_file <op>/triton_<op>_<NNN>.py \
  --warmup 50 --repeat 100
```

Check:
- `PASS accuracy` — if FAIL, fix before logging anything. Don't commit broken kernels.
- `v0=... ms, v1=... ms, speedup=...` — record these exact numbers.

If accuracy fails, the most common causes:
- argmax using a sentinel value that corrupts the sum (e.g. `tl.where(is_best, e_idx, E)` sums to `best + (E-1)*E`). Use `tl.where(is_best, e_idx, 0)`.
- shape mismatch in `tl.dot` (forgot `tl.trans`). `[1, H] @ [2I, H]` is wrong; need `[1, H] @ [H, 2I]`.
- dtype: `tl.dot` may require fp16 or fp32 inputs explicitly cast.

#### Step 5: Optionally profile

If device time matters for this round, run torch.profiler and dump JSON to `<op>/log/<op>_<NNN>_forward_50iter.pt.trace.json`. Summarize kernel time by name:

```bash
jq -r '.traceEvents[] | select(.cat == "kernel") | [.name, .dur] | @tsv' \
  <op>/log/<NNN>.json \
| awk -F'\t' '{a[$1]+=$2; c[$1]++} END {for (n in a) printf "%s\tcount=%d\ttotal=%.2fus\tavg=%.2fus\n", n, c[n], a[n], a[n]/c[n]}' \
| sort -t= -k3 -rn | head
```

#### Step 6: Update log.md (BEFORE next round)

Append to log.md:

1. Add a row to Section 3 table: `| <file> vN | <wall> ms | <device> us / iter | <rel-to-prev>x | <rel-to-base>x |`
2. Append Entry 00N under Section 4 with these subsections (in order):
   - **状态** (state — what the previous round looked like, why this round is needed)
   - **假设** (hypotheses — what you expect the change to do)
   - **优化手段** (what you changed — concrete code/pattern)
   - **踩坑** (pitfalls — argmax sentinel, AST filter, etc. Be honest.)
   - **结果** (auto_bench wall + device time + speedup numbers)
   - **与 upbound 的差距** (gap to upbound — be quantitative)
   - **下一步** (next direction — one sentence)
3. Update Section 5 (current bottleneck) to reflect post-round state
4. Update Section 6 (next directions) — reorder or mark P0 done
5. Update Section 8 checkpoint date if needed

**This step is mandatory before the next round.** The user explicitly required: "每轮跑出结果后先更新log.md再分析瓶颈跑下一轮" (update log.md BEFORE analyzing next bottleneck).

#### Step 7: Commit per round

```bash
git add <op>/triton_<op>_<NNN>.py <op>/log.md
git commit -m "<op>: v<NNN> <short method>, <speedup>x"
```

Commit message rules:
- First line: `<op>: v<NNN> <one-phrase method>, <rel-to-base>x`
- Body: what changed (concrete), wall time before/after, device time, why.
- Footer: `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`
- **DO NOT** put log.md in a single final commit. Each round's log.md diff lives in that round's commit. (User explicitly required this.)

### Phase 3 — Stop criteria

Stop when ONE of:
- Wall time is host-overhead-bound (device time < 20% of wall) AND remaining host overhead is from the harness's fixed costs (set_seed, sync_devices). Further device optimization has no wall payoff.
- 5 rounds done with diminishing returns (each round < 5% improvement). The workflow spec says "不能在同 trace 中稳定改善至少 5% 的方案不进入主实现" (a change that doesn't stably improve ≥5% in the same trace doesn't go into main).
- User says stop.

When stopping, write a final entry noting the stop reason and the cumulative speedup.

## Measurement discipline

This is the single most important thing. Most confusion in kernel optimization comes from mixing measurement regimes.

1. **Pick one wall-time source.** auto_bench's `time_forward` is the default. Do NOT mix it with manual `time.perf_counter()` medians — they will disagree by 2-3x because auto_bench adds `set_seed` + `sync_devices` overhead per iter.
2. **Device time is separate.** It comes from profiler JSON, not wall time. A wall-vs-device gap is a signal (host-bound if large), not a bug.
3. **Don't compare across regimes.** If you change the measurement (e.g. preallocated output, fewer syncs), re-run all previous rounds under the new regime before comparing.
4. **Re-run on noise.** If a round's wall time is within 5% of the previous, it's noise — re-run to confirm or reject. Don't log noise as improvement.
5. **Use the same warmup/repeat for all rounds.** `--warmup 50 --repeat 100` is a good default. Changing it mid-project invalidates comparisons.

## Pitfalls log (project-wide)

Things that have bitten this workflow:

- **`_filter_module_ast` stripping**: auto_bench's filter drops non-literal module-level assigns. `fast_libentry()(_kernel)` at module scope → NameError. Fix: class-body `globals()` trick (see Step 3 above).
- **argmax sentinel**: `tl.where(is_best, e_idx, E)` corrupts the sum. Use `tl.where(is_best, e_idx, 0)`.
- **`tl.dot` shape**: requires 2D inputs and matching inner dims. `[1, H] @ [2I, H]` is wrong; transpose first.
- **`torch.mlu.device()` context manager**: has host enter/exit overhead. If the caller already sets the device, drop it.
- **`torch.empty_like` per forward**: allocator overhead. Cache the output tensor on the ModelNew instance.
- **`torch.cuda.is_available()` on MLU box**: returns True (because CUDA stub is loaded), so `sync_devices()` syncs BOTH cuda and mlu — double sync cost per iter. This is harness overhead, not kernel overhead. Don't try to fix it in the kernel.

## References

- [log-template.md](references/log-template.md) — skeleton for log.md with all 8 sections
- See `groupedtopk/log.md` and `fused_moe/log.md` in this repo for full worked examples
