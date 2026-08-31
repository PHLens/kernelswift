# 赛题 4：sparse_pooler

本目录包含 `sparse_pooler` 的参考实现、提交入口、后端 Triton 实现和性能结果。

## 文件

- `base.py`：参考实现；
- `submission.py`：提交入口，提供 `ModelNew`、`get_init_inputs` 和 `get_inputs`；
- `impls/`：后端 Triton 实现；
- `results.md`：性能测试结果。

## 后端实现

- `mlu`：`impls/mlu.py`
- `s60`：`impls/s60.py`
- `bi150`：`impls/bi150.py`
- `ascend`：`impls/ascend.py`
- 通用实现：`impls/generic.py`

## 测试

在本目录执行：

```bash
bash run.sh
```

`run.sh` 调用根目录的 `run_task.sh`。环境配置见根目录 `requirements.txt`，完整测试参数见根目录 `README.md`。
