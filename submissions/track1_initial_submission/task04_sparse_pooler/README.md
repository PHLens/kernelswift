# 赛题4：sparse_pooler

本目录是赛题 `sparse_pooler` 的正式提交材料。

## 文件说明

- `base.py`：参考实现副本，用于本地/平台复现实测；
- `submission.py`：正式提交入口，暴露 `ModelNew` / `get_init_inputs` / `get_inputs`；
- `impls/`：按后端整理的 Triton 实现；
- `requirements.txt`：环境配置文件；
- `run.sh`：运行脚本；
- `results.md`：当前性能测试结果。

## 后端实现映射

- `mlu` -> `impls/triton_sparse_pooler_004.py`
- `s60` -> `impls/triton_sparse_pooler_001.py`
- `bi150` -> `impls/triton_sparse_pooler_001.py`
- `ascend` -> `impls/triton_sparse_pooler_001.py`

当前未纳入本初版提交的后端：`maca`。

## 分发规则

`submission.py` 会在运行时检测当前后端并加载对应实现：

- `torch.mlu.is_available()` -> `mlu`
- `torch.gcu.is_available()` -> `s60`
- `torch.npu.is_available()` -> `ascend`
- `torch.cuda.is_available()` 且 `COREX_VERSION`/device name 命中 BI150 特征 -> `bi150`
- 其余 CUDA-compatible 情况 -> `maca`

若当前后端没有已验证实现，代码会显式报错，不会回退到纯 PyTorch 内置算子路径规避自定义算子执行。

## 运行方式

```bash
cd task04_sparse_pooler
bash run.sh
```

如需导出 trace，可附加：

```bash
bash run.sh --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output profile_trace.json
```

## 备注

- 推荐 Python 3.10。
- `submission.py` 的 `ModelNew.__init__` 与 `forward` 参数签名保持与参考实现一致。
- 所有文件均为 UTF-8 编码。
