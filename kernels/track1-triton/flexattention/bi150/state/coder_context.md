# Coder Context — flexattention (BI150)

## Contract / Profile

- contract: skills/kernel-opt-loop/prompts/coder.md
- target_profile: triton_cuda (BI150 / CoreX 4.4.0)
- implementation_language: triton
- implementation_backend: cuda

## Task Type

参赛交付物任务（正确性优先，非优化）：产出 naive Triton causal attention。

## Candidate

- path: kernels/track1-triton/flexattention/bi150/triton_flexattention_001.py
- sha256: 14c2af71fb8689e79caf53f6222e5e72e0acf027e43d2a8f9582882d097dac56
- result: candidate-ready（harness PASS accuracy）
- speedup: 0.612x（naive，正确性优先，不追求 speedup）

## Key Implementation Facts

- 单 kernel，grid=(H,)，pid=head。输入 [S=83,H=8,D=64] fp16 连续，per-head
  基址 pid*D，stride H*D，无需 transpose 物化。
- 因果掩码：`tl.where((offs_s[:,None] >= offs_s[None,:]) & (offs_s[None,:] < S), scores, -inf)`。
- softmax 数值稳定：max 减除，exp(-inf)=0 处理掩码列。
- 输出 reshape(S, H*D) = [83,512] fp16，与 base 布局等价。

## Open Checks

无。正确性已通过 harness（atol=1e-2, rtol=1e-2）。
