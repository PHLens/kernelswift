# 赛道一 Triton 提交物（Track 1 Triton Submission Package）

每个 `taskXX_*` 目录包含：

- `base.py`：对应赛题的参考实现；
- `submission.py`：正式提交入口，暴露 `ModelNew` / `get_init_inputs` / `get_inputs`；
- `impls/`：各后端 Triton 实现；
- `README.md`：赛题说明、后端映射与运行说明；
- `requirements.txt`：环境配置文件；
- `run.sh`：运行脚本；
- `results.md`：性能测试结果。

## 后端分发

`submission.py` 采用统一入口 + 后端分发方式：

- `mlu` → 寒武纪 MLU
- `s60` → 燧原 GCU (`torch.gcu`)
- `ascend` → 昇腾 NPU (`torch.npu`)
- `bi150` → 天数智芯 BI150 / CoreX（通过 `COREX_VERSION` 或 device name 判定）
- `maca` → 沐曦 C500（默认 CUDA-compatible 路径）

若某赛题没有对应后端的专项优化版本，`submission.py` 会加载该赛题的通用 Triton 实现。
