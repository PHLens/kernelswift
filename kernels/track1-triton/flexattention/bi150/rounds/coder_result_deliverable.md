# Coder Result Deliverable — flexattention (BI150)

## Classification

**参赛交付物（正确性优先，非优化）**

本算子是参赛交付物任务，产出正确的 naive Triton causal attention
实现作为可提交代码。base 是厂商 Ixmma FlashAttention，已知无法超越，
不追求 speedup，只要求正确性通过 harness。

## Candidate

- path: `kernels/track1-triton/flexattention/bi150/triton_flexattention_001.py`
- sha256: `14c2af71fb8689e79caf53f6222e5e72e0acf027e43d2a8f9582882d097dac56`

## Correctness Result

命令：
```
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py \
  --v1_file kernels/track1-triton/flexattention/bi150/triton_flexattention_001.py \
  --warmup 50 --repeat 100 --full-traceback
```

结果：**PASS accuracy**

- v0 = 0.145599 ms
- v1 = 0.237728 ms
- speedup = 0.612x（naive 实现，慢于 base，符合预期；正确性优先）

## Kernel 结构

单 kernel `_causal_attention_kernel`，grid = `(H,)`（每个 head 一个 program，
pid = head index h）。

输入布局 `[S=83, H=8, D=64]` fp16 连续，元素 `[s,h,d]` 位于偏移
`s*H*D + h*D + d`，per-head 基址 `pid*D`，无需 transpose 物化。

流程（每 head 内）：
1. 加载 Q `[BLOCK_S, D]`、K^T `[D, BLOCK_S]`、V `[BLOCK_S, D]`，
   `BLOCK_S = next_pow2(83) = 128`，pad 行/列用 `mask=..., other=0.0` 填零。
2. `scores = tl.dot(q, kt) * scale`（scale = 0.125 = 1/sqrt(64)），
   fp16 输入 → fp32 累加。
3. **因果掩码**：
   ```python
   causal_mask = offs_s[:, None] >= offs_s[None, :]   # query m, key n, 条件 m >= n
   valid_key   = offs_s[None, :] < S                  # 掩掉 pad 列
   scores = tl.where(causal_mask & valid_key, scores, float("-inf"))
   ```
   query 位置 m 只允许 attend 到 key 位置 n <= m（下三角），j > i 置 -inf。
4. 数值稳定 softmax：`m = tl.max(scores, axis=1)` 后 `exp(scores - m)`，
   `exp(-inf) == 0` 精确处理掩码列（每有效 query 行保留对角 m==n 有限值，
   per-row max 必为有限）。分母 `denom = tl.sum(p, axis=1)`。
5. `acc = tl.dot(p, v)`，`to(tl.float16)` 后按 `row_mask` store 回
   `out [S, H, D]`。
6. forward 返回 `out.reshape(S, H*D)` = `[83, 512]` fp16。

## Tolerance 验证

harness 默认 `atol=1e-2, rtol=1e-2, equal_nan=True`，输出 fp16，
`torch.allclose` 通过（PASS accuracy）。

## 语义一致性说明

- base 通过 `unsqueeze(0).transpose(1,2)` 得到 `[1,8,83,64]`，最终
  `squeeze(0).transpose(0,1).reshape(83,512)`。本实现直接按 `[S,H,D]`
  布局 per-head 计算后 `reshape(S, H*D)`，`[s, h*64+d]` 位置与 base 完全
  一致（等价于 out[h,s,d] 的转置视图）。
- `num_kv_heads == num_heads == 8`，无 GQA repeat_interleave 分支。
- `get_init_inputs()` 返回 `[8, 64, None, 8]`（num_heads, head_size, scale,
  num_kv_heads），scale=None 时 forward 回退到 `1/sqrt(head_size)=0.125`，
  与 base 一致。

## Constraint Compliance

- 未改动 `base.py`、`baseline_adapter.py`、`project.md`、`team-state.md`、
  `auto_bench.py`。
- 使用 `@triton.jit` + `tl.dot`（profile 中 tl.dot 已 Supported）。
- AST loader 保留 `Import`/`FunctionDef`/`ClassDef`；`@triton.jit` 装饰器、
  `_next_pow2` 辅助函数、`ModelNew`、`get_inputs`、`get_init_inputs` 均为
  顶层 FunctionDef/ClassDef，可被 harness 正常加载。
