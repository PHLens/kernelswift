# <Operator Name> Triton Kernel Optimization Log

本文记录 `base.py` 中 <operator> 在 <device> 上的 Triton 优化过程。每次优化独立成 entry，记录当时现状、假设、优化手段、踩坑、结果、与性能上界的差距，以及下一步方向。

## 1. 固定问题与测试口径

### 1.1 算子语义

- 输入：`<name>: <dtype>[<shape>]`
- 当前核心 shape：`<T=...>, <H=...>, <E=...>`
- <routing/scoring/etc.>：`<one-line math>`
- 输出：`<name>: <dtype>[<shape>]`

### 1.2 环境

- Device：<MLU590-H8 / GPU / etc.>
- 可见 <accelerator> core：<N>
- PyTorch：`<version>`
- torch_<accelerator>：`<version>`
- Triton：`<version>`
- Python：`<path>`

### 1.3 测量规则

1. 正确性与 wall time 用 `auto_bench.py`，`--warmup 50 --repeat 100`。所有数据以 auto_bench 为准。
2. device time 以 profiler JSON 中 `cat == "kernel"` 的 `dur` 为准，单位为微秒。
3. wall time 是 `time_forward` 中 `sync_devices` 包裹的中位数。
4. <optional: v0 baseline 在 forward-mode profile 下与 v1 同 trace，可以一并分析 host/device 时间分布。>
5. 优化循环每轮选一个明确瓶颈点；不能在同 trace 中稳定改善至少 5% 的方案不进入主实现。

## 2. Upbound 定义

- **工程上界**：<CNNL `Op<half>` 单 op 平均约 N us。本算子等价于 X 次 op-like work；按 N us 估算 <是否可直接加和>，仅作 stretch goal。>
- **更现实的目标**：<把 wall time 压到 X us 量级即可在 PyTorch eager 下与单 op 竞争。>

## 3. 当前结果总览

| 实现 | Wall time/call (auto_bench) | Kernel device time | 相对上一阶段 | 相对 base |
|---|---:|---:|---:|---:|
| `base.py` eager | <X> ms | ~<Y> ms / 50 iter | - | 1.00x |

## 4. Optimization Entries

### Entry 000 - PyTorch eager 起点

**状态**

`base.py` 由 <op1、op2、...> 组成。每个 <unit> 内部有 <step1、step2、step3>。

**优化手段**

无，记录为基准。

**踩坑**

- <N 个 op 串行调度，共约 M 个 kernel launch。>
- <`x[mask]` 命中 `XKernel`，平均 Y us / kernel。>

**结果**

- `auto_bench.py` wall：`v0=<X> ms / call`。
- 50 次 forward 共触发 <N> 类 kernel，total 约 <Y> ms device work。
- trace：[v0+v1 forward trace](log/<file>.pt.trace.json)

**与 upbound 的差距**

无意义：base 不是上限。只是参考点。

**下一步**

<写一个 Triton kernel 把 <bottleneck op> 整体消灭。>

---

<!-- Each subsequent entry follows this template:

### Entry 00N - <one-phrase method>

**状态**

<previous round's state, why this round is needed>

**假设**

- <what you expect the change to do>

**优化手段**

- <concrete code/pattern change>

**踩坑**

- <pitfall 1>
- <pitfall 2>

**结果**

- `auto_bench.py` wall：`v<NNN>=<X> ms / call`，相对 v<NNN-1> <r>x，相对 base <R>x。
- 50 次 forward 的 kernel device time：<D> us / iter。
- trace：[v<NNN> forward trace](log/<file>.pt.trace.json)

**与 upbound 的差距**

- <quantitative gap analysis>

**下一步**

<one sentence pointing to the next bottleneck>

---

-->

## 5. 当前瓶颈判断

### 5.1 <bottleneck name> 占主导

<2-3 sentence justification with numbers>

## 6. 后续优化方向

按优先级：

### P0 - <next direction>

<one-paragraph description>

### P1 - <secondary direction>

<one-paragraph description>

## 7. 复现命令

```bash
<python> \
  auto_bench.py \
  --v0_file <op>/base.py \
  --v1_file <op>/triton_<op>_<NNN>.py \
  --warmup 50 --repeat 100
```

按 kernel name 汇总 trace：

```bash
jq -r '
  .traceEvents[]
  | select(.cat == "kernel")
  | [.name, .dur]
  | @tsv
' <op>/log/<file>.pt.trace.json \
| awk -F'\t' '{a[$1]+=$2; c[$1]++} END {for (n in a) printf "%s\tcount=%d\ttotal=%.2fus\tavg=%.2fus\n", n, c[n], a[n], a[n]/c[n]}' | sort -t= -k3 -rn | head
```

## 8. Checkpoint

记录生成时：<YYYY-MM-DD>。

- `base.py` 未修改
- v1–v<NNN> Triton：<N> 轮累计 <R>x（auto_bench 口径）
- 所有 trace 文件在 `<op>/log/` 下（gitignored）
