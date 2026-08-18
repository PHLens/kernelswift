# MHC Post Layer Mix S60 (GCU) 优化结果

Branch: `kernel-opt/mhc-post-layer-mix-s60`。目标:`B=1,H=12,T=128,D=1024, fp16, einsum matmul + elementwise mix`。

## 结论:measurement-bound,无优化空间

baseline wall 4.270324 ms,**6 launches/call,launch 仅 67us/call ≈ 1.6% wall**。

瓶颈是 einsum `abmn,abmc->abnc` matmul([1,12,128,1024]×[1,12,128,1024])的 device 计算,已被 CNNL 库优化。launch 优化无意义(1.6%),手写 Triton `tl.dot` 在 GCU 上 Unknown 且无法超越库 matmul。

注:base.py 的 `super(Model, self).__init__()` 已修为 `super().__init__()`(适配 make_baseline_adapter 的类名重命名,语义等价)。

canonical 保持 `baseline_adapter.py`,无加速比。

## 停止理由

- stop_reason: `measurement-bound`
- device matmul bound,launch 占比 1.6%,无融合空间。
