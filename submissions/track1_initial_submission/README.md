# 赛道一 Triton 提交包

`submissions/track1_initial_submission/` 集中收录赛道一全部 10 个算子的提交材料。公共文档、环境配置和运行入口位于本目录；每个 `taskXX_*` 目录提供该题的 README、测试入口、参考实现、提交入口、后端 Triton 实现和性能结果。

## 目录

```text
track1_initial_submission/
├── README.md
├── requirements.txt
├── auto_bench.py
├── run_task.sh
├── run_all.sh
├── performance_summary.md
└── taskXX_*/
    ├── README.md
    ├── run.sh
    ├── base.py
    ├── submission.py
    ├── impls/
    └── results.md
```

`submission.py` 提供 `ModelNew`、`get_init_inputs` 和 `get_inputs`，并保持与该题参考实现一致的初始化与调用接口。`impls/` 中的文件以实际后端命名，`generic.py` 为通用 Triton 实现。各题的 `README.md` 记录后端实现和结果入口，`run.sh` 调用根目录的统一测试脚本。

## 环境

推荐使用 Python 3.10，并在目标后端环境中安装根目录的依赖：

```bash
pip install -r requirements.txt
```

后端运行时由目标平台提供，例如 S60 的 `torch_gcu` 与 `triton_gcu`、Ascend 的 `torch_npu`、MLU 的 `torch_mlu` 与 `torch_mlu_ops`，以及 BI150、MACA 的厂商运行时。

## 运行

执行单个赛题：

```bash
bash run_task.sh task01_groupedtopk
```

也可以进入对应赛题目录执行：

```bash
cd task01_groupedtopk
bash run.sh
```

执行全部赛题：

```bash
bash run_all.sh
```

`run_task.sh` 和 `run_all.sh` 会将额外参数传给 `auto_bench.py`。预热和重复次数可通过环境变量调整：

```bash
WARMUP=100 REPEAT=500 bash run_task.sh task03_fused_moe --profile --profile-mode forward
```

## 后端分发

`submission.py` 在运行时检测可用后端：`torch.mlu` 对应 MLU，`torch.gcu` 对应 S60，`torch.npu` 对应 Ascend；CUDA 环境中，BI150 / CoreX 特征对应 BI150，其余 CUDA-compatible 环境对应 MACA。没有专项实现的后端加载该题的 `generic.py`，该路径始终执行 Triton kernel。

## 赛题索引

每题均提供 `generic.py`。下表列出可用的专项后端实现和性能结果入口。

| 赛题 | 算子 | 专项后端实现 | 性能结果 |
|---|---|---|---|
| 01 | `groupedtopk` | `mlu.py`、`s60.py`、`maca.py`、`bi150.py`、`ascend.py` | [`results.md`](task01_groupedtopk/results.md) |
| 02 | `flexattention` | `mlu.py`、`s60.py`、`bi150.py`、`ascend.py` | [`results.md`](task02_flexattention/results.md) |
| 03 | `fused_moe` | `mlu.py`、`s60.py`、`bi150.py`、`ascend.py` | [`results.md`](task03_fused_moe/results.md) |
| 04 | `sparse_pooler` | `mlu.py`、`s60.py`、`bi150.py`、`ascend.py` | [`results.md`](task04_sparse_pooler/results.md) |
| 05 | `music_flamingo_rotary_embedding` | `s60.py`、`maca.py`、`bi150.py`、`ascend.py` | [`results.md`](task05_music_flamingo_rotary_embedding/results.md) |
| 06 | `mm_encoder_attention` | `s60.py`、`maca.py`、`bi150.py`、`ascend.py` | [`results.md`](task06_mm_encoder_attention/results.md) |
| 07 | `mhc_post_layer_mix` | `s60.py`、`maca.py`、`bi150.py`、`ascend.py` | [`results.md`](task07_mhc_post_layer_mix/results.md) |
| 08 | `mhc_head_compute_mix` | `s60.py`、`maca.py`、`bi150.py`、`ascend.py` | [`results.md`](task08_mhc_head_compute_mix/results.md) |
| 09 | `centre_random_augmentation` | `s60.py`、`bi150.py`、`ascend.py` | [`results.md`](task09_centre_random_augmentation/results.md) |
| 10 | `mhc_head_compute_mix_backward` | `s60.py`、`bi150.py`、`ascend.py` | [`results.md`](task10_mhc_head_compute_mix_backward/results.md) |
