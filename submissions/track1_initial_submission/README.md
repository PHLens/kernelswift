# 赛道一初版提交物（Track 1 Triton Submission Package）

本目录按赛题整理赛道一提交材料。每个 `taskXX_*` 目录包含：

- `base.py`：对应赛题的参考实现副本；
- `submission.py`：正式提交入口，暴露 `ModelNew` / `get_init_inputs` / `get_inputs`；
- `impls/`：按后端整理的 Triton 实现副本；
- `README.md`：赛题说明、后端映射与运行说明；
- `requirements.txt`：环境配置文件；
- `run.sh`：运行脚本；
- `results.md`：当前性能测试结果汇总。

提交说明：

1. 正式发邮件/打包时，请将本目录整体打包，并按官方要求替换压缩包名称中的参赛者/团队名称与 UID。
2. `submission.py` 采用统一入口 + 后端分发方式：
   - `mlu` → 寒武纪 MLU
   - `s60` → 燧原 GCU (`torch.gcu`)
   - `ascend` → 昇腾 NPU (`torch.npu`)
   - `bi150` → 天数智芯 BI150 / CoreX（通过 `COREX_VERSION` 或 device name 判定）
   - `maca` → 沐曦 C500（默认 CUDA-compatible 路径）
3. 若某赛题当前没有该后端的专项优化版本，则 `submission.py` 会退回到该赛题的通用 Triton 版本，而不是直接报错；只有在完全没有 Triton 实现可用时才会失败。
