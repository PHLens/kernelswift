# Coder Result Deliverable — mm_encoder_attention (BI150, task 6)

> **参赛交付物（正确性优先，非优化）。** 本任务产出正确的 naive Triton attention
> 实现作为可提交代码，**不追求超越 base**（base 为厂商 Ixmma FlashAttention，
> 已知无法超越），验收标准仅为 harness 正确性 PASS。

## Result

- `candidate-ready`（正确性通过，作为参赛交付物）

## Candidate

- path: `kernels/track1-triton/mm_encoder_attention/bi150/triton_mm_encoder_attention_001.py`
- sha256: `88ade697da35a51362c2a8643e054a61362a68ff3e9e2e60110bd3e45285e87e`
- source canonical (base): `kernels/track1-triton/mm_encoder_attention/base.py`
- base sha256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`

## Harness Correctness Result

- command:
  `python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/triton_mm_encoder_attention_001.py --warmup 50 --repeat 100 --full-traceback`
- result: **PASS accuracy**
- v0 (base) time: `0.196228 ms`
- v1 (candidate) time: `0.358623 ms`
- speedup: `0.547x`（低于 1，符合预期 —— base 为厂商 Ixmma FlashAttention，
  本交付物不追求超越，仅要求正确性）
- tolerance: `atol=1e-2, rtol=1e-2, equal_nan=True`（harness 默认）
- 本地门：`python3 -m py_compile ...` 通过。

## Implementation Notes

### Kernel structure

单个 `@triton.jit` kernel `_mm_encoder_attention_kernel`，每个 program 处理一个
`(batch, head)` 对，`grid = (B * H,) = (16,)`。三个输入 `[B, S, H*D]` 在 host 端
view/transpose 为 `[B, H, S, D]` 连续布局后传入 kernel。

kernel 内部：
1. **QK^T**：`scores = tl.dot(q, kt) * scale`，其中 `q` 为 `[BLOCK_S, D]`，
   `kt`（K 的转置，直接按转置布局加载）为 `[D, BLOCK_S]`，得 `[BLOCK_S, BLOCK_S]`。
2. **softmax**：先 `tl.where` 将无效 key 列（`offs_s >= S`）置 `-inf`；
   再 `m = tl.max(scores, axis=1)` 减除，`p = exp(scores - m)`，除以行和，
   保证数值稳定（防止 exp 溢出）。
3. **PV**：`acc = tl.dot(p, v)`，`p` 为 `[BLOCK_S, BLOCK_S]`，`v` 为
   `[BLOCK_S, D]`，得 `[BLOCK_S, D]`。
4. 写回 `out` 前 `.to(tl.float16)`。

### 数值处理

- 输入为 fp16，kernel 内 `tl.load(...).to(tl.float32)` 后全部以 fp32 计算。
- `tl.dot` 使用 fp32 输入（该 profile 已证明 fp32 `tl.dot` 精确，
  `(32,32)@(32,32)` 结果为精确），累加为 fp32，保证精度。
- `scale = 1/sqrt(64) = 0.125` 在 softmax 前乘。
- softmax 采用 `exp(x - max)` 稳定形式。
- 输出 cast 回 fp16，与 base 输出 dtype 一致。

### 形状处理

- `S=83` 非 2 的幂，`tl.arange` 要求 2 的幂，故 `BLOCK_S = 128`（next pow2）。
- 无效行（`offs_s >= S`）通过 `row_mask` 用 `other=0.0` 加载，写回时同样 mask。
- `D=64` 本身为 2 的幂，无需 padding。

### 启动配置

- 直接 launch `kernel[(grid,)](...)`，未显式指定 `num_warps`/`num_stages`
  （该 profile 将其标记为 Unknown，不依赖未验证的 launch hint）。

## Conformance Notes

- 无任何算法、数据流或生命周期偏差；为满足 `tl.arange` 的 2 的幂约束而引入的
  `BLOCK_S=128` padding 与 `-inf`/`other=0` 掩码属于数值正确性实现细节，不改变
  softmax 语义。
- 未改动 base.py、baseline_adapter.py、project.md、team-state.md、harness。
