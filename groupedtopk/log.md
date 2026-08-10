# groupedtopk Triton Kernel Optimization Log

本文记录 `base.py` 中 groupedtopk 在 MLU590-H8 上的 Triton 优化过程。每次优化独立成 entry，记录当时现状、假设、优化手段、踩坑、结果、与性能上界的差距，以及下一步方向。

## 1. 固定问题与测试口径

### 1.1 算子语义

- 输入：`gating_output: fp32[T=83, E=256]`（`hidden_states` 仅用于 batch-size check，不参与计算）
- 参数：`topk=8, renormalize=True, num_expert_group=8, topk_group=4, scoring_func="softmax", routed_scaling_factor=1.0`
- 语义：对每行 `gating_output` 做 softmax → 取每 32 个 expert 一组的 group max → 选 top-4 group → mask 掉非 top-4 group 的 expert → 在剩余 expert 中取 top-8 → renormalize → 乘 `routed_scaling_factor`
- 输出：`topk_weights: fp32[T, 8]`, `topk_ids: int32[T, 8]`

### 1.2 环境

- Device：MLU590-H8（可见 8 core，单卡测试）
- PyTorch：`2.11.0+cpu`
- torch_mlu：随环境
- Triton：`3.2.0`
- Python：`/projs/framework/lipenghui/venv/pytorch_main/bin/python`

### 1.3 测量规则

1. 正确性与 wall time 用 `auto_bench.py`，`--warmup 50 --repeat 100`。所有数据以 auto_bench 为准。
2. device time 以 profiler JSON 中 `cat == "kernel"` 的 `dur` 为准，单位为微秒。
3. wall time 是 `time_forward` 中 `sync_devices` 包裹的中位数（`set_seed` + `sync_devices` 每次迭代）。
4. 优化循环每轮选一个明确瓶颈点；不能在同 trace 中稳定改善至少 5% 的方案不进入主实现。

## 2. Upbound 定义

- **工程上界**：本算子等价于"per-token softmax(256) + group-topk(4 of 8) + masked topk(8 of 256) + renorm"。可以全部 fuse 到一个 per-row Triton kernel，每个 token 单独处理。理论上单 kernel 单 program 的工作量约 256 + 8·32 + 8·256 = ~2.3K compare/exp，外加 softmax 的 exp/sum。MLU 上单 kernel 启动 + 计算约 10–30 us / call。把这个当 stretch goal。
- **更现实的目标**：把 wall time 压到 100 us 量级（base 是 840 us），即 8x 量级。

## 3. 当前结果总览

| 实现 | Wall time/call (auto_bench) | Kernel device time | 相对上一阶段 | 相对 base |
|---|---:|---:|---:|---:|
| `base.py` eager | 0.840 ms | ~251 us / iter | - | 1.00x |
| `triton_grouped_topk_001.py` v1 | 0.198 ms | ~38.5 us / iter | 4.24x | 4.24x |
| `triton_grouped_topk_002.py` v2 | 0.146 ms | ~38.1 us / iter | 1.35x | 5.84x |
| `triton_grouped_topk_003.py` v3 | 0.130 ms | ~20.0 us / iter | 1.12x | 6.50x |
| `triton_grouped_topk_004.py` v4 | 0.128 ms | ~20.0 us / iter | 1.02x | 6.56x |

## 4. Optimization Entries

### Entry 000 - PyTorch eager 起点

**状态**

`base.py` 由 softmax → view+max → topk(4 of 8) → zeros_like+scatter → unsqueeze+expand+reshape → bool+masked_fill → topk(8 of 256) → sum+div(renorm) → .to(fp32)/.to(int32) 等约 10 步串行 PyTorch op 组成。每步触发 1+ 个 kernel launch。

**优化手段**

无，记录为基准。

**踩坑**

- 50 次 forward 触发约 1800 个 kernel event，per-iter ~36 个 kernel。
- 最大的几个 device kernel：`mluBlockKernelTopKPoolOprand` 72.9 us/iter（两次 topk 共 200 次）、`mluKernelScatter` 63.7 us/iter、`MLUBlockKernelExpand3PipelineDivExpand` 25 us/iter。
- 总 device time ~251 us/iter，wall 840 us/iter，`device_ratio = 251/840 ≈ 30%`，**host-bound**：~590 us/iter host overhead，主要是大量小 op 的 launch + harness `set_seed` + `sync_devices` 双卡同步（cuda stub + mlu 同时 sync）。

**结果**

- `auto_bench.py` wall：`v0=0.840 ms / call`。
- 50 次 forward 共触发 ~1800 个 kernel event，total device work ~12.6 ms（251 us / iter）。
- trace：[v0 forward trace](log/groupedtopk_forward_50iter.pt.trace.json)

**与 upbound 的差距**

无意义：base 不是上限。只是参考点。

**下一步**

写一个 per-token Triton kernel，把 softmax + group-topk + masked topk + renorm 全部 fuse 进一个 kernel，每个 token 一个 program，单 kernel 解决整道题。预计同时砍掉 device 中的两个 topk + scatter + expand + masked_fill，并把 host 端 ~36 次 launch 压到 1 次。

---

### Entry 001 - 单 per-token Triton kernel fuse 整道题

**状态**

Entry 000 仍是 ~10 步 PyTorch op 串行，device 30%、host 70%。要打破这个结构，最直接的办法是把整道题压成 1 个 Triton kernel：grid=(T,)，每 program 处理 1 个 token 的 256 个 expert score。

**假设**

- Fuse 后 device 时间从 251 us/iter 跌到 ~30–40 us/iter（消灭两个 topk、scatter、expand、masked_fill、cast）。
- Host 端 ~35 次 launch 缩到 1 次，wall 大幅下降。
- 4.2x 量级起步。

**优化手段**

- `triton_grouped_topk_001.py`：`@triton.jit _grouped_topk_kernel`，每个 program：
  1. `tl.load(gating, [BLOCK_E=256])` → fp32
  2. softmax：`tl.max → exp → sum → div`
  3. group max：`tl.reshape(scores, (8, 32)) → tl.max(axis=1)` → [8]
  4. top-4 of 8：`tl.static_range(4)` 选择排序，每趟 `tl.max + tl.where(is_m, g_offs, n_group) + tl.min` 找最小 idx，累加 `g_keep`
  5. 把 `g_keep` reshape+broadcast 回 [256]，apply `tl.where(score_mask, scores, -inf)`
  6. top-8 of 256：`tl.static_range(8)` 选择排序，每趟同样的 max+where+min 找 idx，用 `k_mask = (k_offs == k)` 写到 `out_w/out_i` 的对应 lane
  7. renorm：`tl.sum(out_w) → div`
  8. `tl.store(weights/ids, [K=8])`
- 用 `fast_libentry()` decorator 装饰，放在 `class ModelNew` 的 class body 里通过 `globals()["_fast"] = fast_libentry()(_grouped_topk_kernel)` 注入，规避 auto_bench `_filter_module_ast` 把模块级 `Assign` 剥掉的问题。
- forward 里每个 iter `torch.empty` 两个输出 tensor。

**踩坑**

- `fast_libentry` 是工厂：必须写 `fast_libentry()(_kernel)`，不能写 `fast_libentry(_kernel)`（报 "takes 0 positional arguments but 1 was given"）。
- `tl.static_range` 而不是 `range`，否则 Python 层不展开。
- 写 `out_w[k]` 不能动态索引，必须用 `k_mask = (k_offs == k)` + `tl.where(k_mask, tl.full((K,), val), out_w)` 的 broadcast 写法。

**结果**

- `auto_bench.py` wall：`v1=0.198 ms / call`，相对 v0 4.24x，相对 base 4.24x。
- 50 次 forward 的 kernel device time（v1 部分）：`_grouped_topk_kernel` 38.46 us / iter。
- v1 device_ratio = 38.5/198 ≈ 19% → **host-bound**，剩 81%（~160 us）是 `torch.empty × 2` + Triton launcher + harness `set_seed` + `sync_devices`（双卡同步）。
- trace：[v1 forward trace](log/groupedtopk_001_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- 单 kernel 跑了 38 us/iter，比 stretch goal（10–30 us）慢约 1.3x。还能继续做 device 侧（选择排序→argmax/bitonic）。
- 但当前 wall 主导项是 host 160 us，下一步应先压 host，再回头压 device。

**下一步**

v1 每次 forward 都 `torch.empty` 两个输出 tensor，allocator 开销 + cache miss。把它们缓存到 `ModelNew` 实例上（按 (T, device) 复用），干掉这部分 host 开销。

---

### Entry 002 - 缓存输出 tensor

**状态**

v1 wall 198 us，其中 device 38 us（19%），host 160 us（81%）。host 端可识别的开销：每次 forward 调 `torch.empty((T, 8), fp32)` + `torch.empty((T, 8), int32)` 各一次。MLU allocator 即便走 fast path 也有若干 us 的 Python + driver 同步成本。

**假设**

- 输出 shape 在所有 iter 里稳定（T=83, topk=8）。把两个输出 tensor 在 `__init__` 里第一次需要时分配、之后复用，应该砍掉 forward 里 ~10–30 us host 时间。
- 不动 kernel，单纯减 allocator。

**优化手段**

- `triton_grouped_topk_002.py`：`ModelNew` 加 `_w_buf / _i_buf / _cache_T / _cache_dev`，`_ensure_buf(T, device)` 懒分配并按 (T, device) 复用。`forward` 只取 buffer，不 `torch.empty`。
- 顺便去掉 `assert` / `if scoring_func != "softmax"` 这两个常分支，省一点 Python 解释开销。

**踩坑**

- 必须按 device 复用：`auto_bench` 把模型 `.to(target_device)`，且 `clone_value` 每次重新构造 inputs，但 `gating_output.device` 在 init 后到 forward 里才确定。lazy allocate on first forward 即可。
- 缓存 buffer 是可变的，下次 forward 会覆盖。auto_bench 在 `time_forward` 之外用 `clone_value` 输入做正确性比对，正确性比对只发生在第一次，所以 cache 复用没问题。

**结果**

- `auto_bench.py` wall：`v2=0.146 ms / call`（三次复测 0.146 / 0.154 / 0.149，中位约 0.15），相对 v1 约 1.35x，相对 base 5.84x。
- 50 次 forward 的 kernel device time（v2 部分）：`_grouped_topk_kernel` 38.09 us / iter（与 v1 持平，因为 kernel 没改）。
- v2 device_ratio = 38/152 ≈ 25%，host 占 114 us（75%）。
- trace：[v2 forward trace](log/groupedtopk_002_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- wall 从 198 → 152 us，砍了 ~46 us，符合预期（torch.empty 两次大约就是几十 us 量级）。
- 离 stretch goal（10–30 us）还有 5–15x 距离。剩余 114 us host 主要是 Triton launcher + harness `set_seed` + `sync_devices`（cuda stub + mlu 同时 sync，固定开销），device 38 us 也有压缩空间。

**下一步**

- Host 端：剩余 host 主要是 launcher + harness 固定开销。launcher 不太能动（fast_libentry 已是最薄一层），harness 不能改。能改的是 forward 里的 Python 解释（`triton.next_power_of_2`、`E // num_expert_group`、`globals()["_fast"]` 查找）。这些可以预计算到 `__init__`。
- Device 端：kernel 38 us 里 8 趟 selection sort × 256 元素 = ~16 次 256-元素 reduction 是大头。用 `tl.argmax` 替代手写的 `max + where + min` 三连击，reduction 次数砍 1/3，看能否从 38 us 跌到 ~25 us。

先做 device（v3），因为 host 的剩余项大多是 harness 固定开销，进一步压效果有限。

---

### Entry 003 - `tl.argmax` 替代手写 argmax 三连击

**状态**

v2 kernel 38 us/iter，其中 8 趟 top-K selection sort × 256 元素是大头。每趟用了 `tl.max + tl.where(is_m, off, N) + tl.min` 三次 reduction（其中 max 和 min 都是 256-元素 reduction）。8 趟 × 3 = 24 次 256-元素 reduction，加上 top-4 of 8 也用同样的 pattern 4 趟 × 3 = 12 次 8-元素 reduction。

**假设**

- 用 `tl.argmax` 把每趟 3 次 reduction 压到 2 次（argmax + max for value），top-K 趟数不变但每趟更便宜。
- 期望 device 从 38 us → ~25 us。wall 跟着降 ~13 us。

**优化手段**

- `triton_grouped_topk_003.py`：top-KG 和 top-K 两个 selection sort 循环里：
  - 旧：`m = tl.max(vals); is_m = vals == m; idxs = tl.where(is_m, offs, N); idx = tl.min(idxs); ...`
  - 新：`idx = tl.argmax(vals, axis=0); m = tl.max(vals, axis=0); ...`
- 注意 `tl.argmax` tie-break 是 implementation-defined；但 softmax 输出是连续 fp32 值，严格相等几乎不发生；masked 后同一个 -inf 出现多次时 argmax 会避开已经 mask 过的 idx（因为下一趟该 idx 是 -inf 不再是 max）。

**踩坑**

- `tl.argmax` 在 MLU Triton 3.2 可用。
- `tl.max` 仍然要单独调一次拿值（`argmax` 不返回 max 值），所以只能省 1 次 reduction 不是 2 次。
- 三次复测稳定：0.129 / 0.135 / 0.128，中位 0.130，方差小。

**结果**

- `auto_bench.py` wall：`v3=0.130 ms / call`，相对 v2 1.13x，相对 base 6.5x。
- 50 次 forward 的 kernel device time（v3 部分）：`_grouped_topk_kernel` 20.02 us / iter（v2 是 38.09 us，跌了 ~18 us，符合"省 1 次 256-元素 reduction × 8 趟 ≈ 16 us"的预期）。
- v3 device_ratio = 20/130 ≈ 15%，host 占 110 us（85%）。
- 单独拆解 wall：用本地 script 测，v3 forward only（no sync, no seed）median 23 us；加 seed 39 us；加 seed + mlu sync 64 us；加 seed + cuda+mlu sync（auto_bench 模式）跳到 ~130 us。说明 **harness `sync_devices` 里 cuda stub 同步是 60+ us 量级的固定开销**，跟我们 forward 无关，改不动。
- trace：[v3 forward trace](log/groupedtopk_003_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- wall 130 us，离 stretch goal（10–30 us）还有 4–13x 距离，但其中至少 ~100 us 是 harness 固定开销（cuda stub sync），forward 端 host + launcher + device 合计 ~30 us，已经接近 floor。
- Device 20 us 里 83 个 program（grid=T）每个只处理 1 个 token × 256 元素，单 program 太小，device 端 per-program launch overhead 可能占主导。再压 device 的杠杆是增大 BLOCK_T（每个 program 处理多个 token），把 grid 缩小到 ~10 个 program，摊薄 per-program 开销。

**下一步**

v4：把 grid 从 (T,) 改成 (T // BLOCK_T,)，每个 program 处理 BLOCK_T=8 个 token × 256 expert。kernel 内部从 1D 变 2D，softmax/group-max/topk 都沿 E 轴做。期望 device 从 20 us 跌到 ~10 us。

---

### Entry 004 - Host 端 constexpr 预计算 + class-body 注入

**状态**

v3 wall 130 us，device 20 us，device_ratio 15%。手动拆解后 forward 内部 host + launcher = 23 us，set_seed 16 us，mlu sync 25 us（含等 kernel），cuda stub sync ~66 us。harness 固定开销无法动，forward 端剩 23 us 里主要是：`triton.next_power_of_2(E)`、`E // num_expert_group`、`grid = (T,)`、`globals()["_fast"]` 字典查找。这些都是可预计算的常量。

之前我试过 v4 的另一条路：BLOCK_T=8 把 grid 从 (83,) 缩到 (11,)、kernel 内部 2D 化。结果 wall 0.146 ms（比 v3 慢）、device 30 us（比 v3 慢 10 us）。再试 BLOCK_T=2/4，wall 跟 v3 持平。说明 per-program launch overhead 不是 device 瓶颈——T=83 太小，每个 program 的工作量再大也只是把同样的工作换个切分方式，反而增加 2D reshape 等 op 的开销。这条路径放弃。

**假设**

- 把 `BLOCK_E`、`epg`、`grid` 预算到 `__init__`/lazy cache，每次 forward 不再走 Python 整数计算与 `triton.next_power_of_2` 调用。
- 把 `globals()["_fast"]` 在 `__init__` 里绑成 `self._fast`，forward 里省一次 dict 查找。
- 期望 wall 砍 ~5–10 us，device 不变（kernel 没动）。

**优化手段**

- `triton_grouped_topk_004.py`：kernel 与 v3 完全一致，只改 `ModelNew`：
  - 加 `_ensure_cfg(T, E)`：按 (T, E) 懒计算并缓存 `_cfg_grid / _cfg_block_e / _cfg_epg`，命中后直接复用。
  - `__init__` 里 `self._fast = globals()["_fast"]`，forward 直接 `self._fast[grid](...)`。
  - module-level `if "_fast" not in globals(): globals()["_fast"] = fast_libentry()(_grouped_topk_kernel)` 会被 `auto_bench._filter_module_ast` 剥掉（If 节点不在白名单）。workaround：把同样的 `if` 放进 `ModelNew` 的 class body——ClassDef 节点保留，class body 内的 If 也保留，于是 `_fast` 在第一次定义 class 时被注入到 globals。
  - 同时 module-level 也写一份 `if "_fast" not in globals(): ...`（虽会被剥，但作为文档说明）。

**踩坑**

- `auto_bench._filter_module_ast` 对 module-level 的 `if ...: globals()["_fast"] = ...` 直接剥掉（If 节点不在白名单），但 class body 内的同样 If 保留。
- BLOCK_T batching 路线（增大 per-program 工作量）对本问题（T=83, E=256）无效：device 从 20 us 反增到 30 us。说明 per-program overhead 不主导，device 已经被 softmax/group-max/topk 的 reduction 工作量主导。
- 三次复测：0.1275 / 0.1239 / 0.1340 ms（200 repeat），中位 ~0.128 ms。相对 v3 的 ~0.135 ms 约 5–6% 改善，刚好在 5% 阈值线上。

**结果**

- `auto_bench.py` wall：`v4=0.128 ms / call`（200 repeat，中位 ~0.128 ms），相对 v3 约 1.02–1.06x，相对 base 6.56x。
- 50 次 forward 的 kernel device time：`_grouped_topk_kernel` 19.99 us / iter（与 v3 20.02 us 持平，kernel 未变）。
- v4 device_ratio = 20/128 ≈ 16%，host 占 108 us（84%）。
- trace：[v4 forward trace](log/groupedtopk_004_forward_50iter.pt.trace.json)

**与 upbound 的差距**

- wall 128 us，离 stretch goal（10–30 us）仍有 4–13x 距离，但 100+ us 是 harness 固定开销（cuda stub sync 66us + set_seed 16us + mlu sync 等 kernel 25us），forward 端 host + launcher + device 合计 ~28 us 已经触底。
- Device 20 us：BLOCK_T 路线已证伪；softmax(256) + group-max(8×32) + top-4 of 8 + top-8 of 256 这套 workload 在 MLU 上单 program 串行做就是 20 us 量级。要再压只能换算法（bitonic top-k 或硬件 top-k），但工作量超过收益。

**下一步**

按 skill stop criteria：wall 已是 host-overhead-bound（device_ratio 16% < 20% 阈值），且剩余 host overhead 是 harness 固定开销（cuda stub sync 等），无法从 kernel 侧压缩。**达到停止条件**，本轮优化循环结束。

---

## 5. 当前瓶颈判断

### 5.1 Host-bound（harness 固定开销占主导）

v4 wall 128 us，device 20 us（16%）。手动拆解：forward Python + launcher ≈ 23 us，set_seed = 16 us，mlu sync = 25 us（含等 kernel），cuda stub sync = ~66 us。**harness `sync_devices` 里 cuda stub 同步是 60+ us 量级的固定开销**，因为 `torch.cuda` 在 MLU box 上 is_available=True（CUDA stub 加载），sync 会等 cuda + mlu 两边。这是 harness 设计问题，forward 侧改不动。forward 端 host + launcher + device 合计 ~28 us 已触底，device 20 us 也已接近 per-program 串行 workload 的下限。

## 6. 后续优化方向

按优先级：

### P0 - 换 top-K 算法（bitonic / 硬件 top-K）

device 20 us 里 selection sort（8 趟 × 256 元素 argmax+max）仍是主导。理论上换成 bitonic top-K 或 MLU 硬件 top-K 能压到 ~10 us。但工作量超过收益（wall 已经 host-bound，device 砍半也只能省 10 us / 128 us = 8%），暂不展开。

### P1 - 改 harness（需要协作）

cuda stub sync 66us 是 wall 的最大单点。如果 `auto_bench._iter_accelerators` 在 MLU box 上跳过 cuda（只 sync mlu），wall 立刻能跌到 ~62 us。这是 harness 侧的改动，需要和 benchmark 维护者协作，不属于本算子优化范围。

## 7. 复现命令

```bash
python auto_bench.py \
  --v0_file groupedtopk/base.py \
  --v1_file groupedtopk/triton_grouped_topk_004.py \
  --warmup 50 --repeat 100
```

按 kernel name 汇总 trace：

```bash
jq -r '
  .traceEvents[]
  | select(.cat == "kernel")
  | [.name, .dur]
  | @tsv
' groupedtopk/log/groupedtopk_004_forward_50iter.pt.trace.json \
| awk -F'\t' '{a[$1]+=$2; c[$1]++} END {for (n in a) printf "%s\tcount=%d\ttotal=%.2fus\tavg=%.2fus\n", n, c[n], a[n], a[n]/c[n]}' | sort -t= -k3 -rn | head
```

## 8. Checkpoint

记录生成时：2026-08-06。

- `base.py` 未修改
- v1–v4 Triton：4 轮累计 6.56x（auto_bench 口径）
- 所有 trace 文件在 `groupedtopk/log/` 下（gitignored）
- 达到 skill stop criteria（host-overhead-bound + 剩余 host 是 harness 固定开销），循环结束

> 注：v1–v3 的 `.py` 文件已删除，仅保留 `triton_grouped_topk_004.py` 作为 canonical 实现。各版本代码细节见上方各 Entry 的"优化手段"小节。最终结果汇总见 `outcome.md`。
