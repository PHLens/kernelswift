# MM Encoder Attention S60 (GCU) 优化结果

Branch: `kernel-opt/mm-encoder-attn-s60`。目标:`B=2, T=83, H=8, D=64, fp16, non-causal SDPA`。

## 结论:measurement-bound,无优化空间

baseline wall 0.229925 ms,**单 kernel(1 launch/call)**。

eager `F.scaled_dot_product_attention`(non-causal)已被 CNNL 融合成单 kernel,与 flexattention s60 完全同构。flexattention s60 已实测证明:手写 Triton SDPA(per-(token,head) program + tl.dot)在 GCU 上 device 慢 ~100x,无法超越库的 flash-attention。

无 kernel 数可减、无 device 优化空间。canonical 保持 `baseline_adapter.py`,无加速比。

## 停止理由

- stop_reason: `measurement-bound`
- 单 kernel eager,与 flexattention s60 同结论。
