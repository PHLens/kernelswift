# 后端注册表

后端目录命名沿用仓库历史（`mlu`/`s60`/`maca`/`bi150`/`ascend910b`），
本表给出每个编码对应的芯片、软件栈与 target profile。

| 目录编码 | 芯片 / 设备 | 软件栈 | 运行时标识 | target profile | 已知远端环境 |
|---|---|---|---|---|---|
| `mlu` | 寒武纪 Cambricon MLU590-H8 | Triton-MLU（torch_mlu） | `mlu:0` | `triton_mlu` | `triton 3.x` + `torch_mlu`（MLU590） |
| `s60` | 燧原 Enflame GCU（S60 板卡） | Triton-GCU（torch_gcu / triton_gcu） | `gcu:0` | `triton_gcu` | triton 3.6.0 + triton_gcu 3.6.0+1.0.20260722；device arch major=3 minor=0 |
| `maca` | 沐曦 MetaX C500 | MACA 兼容面（对上层暴露 `cuda:0`） | `cuda:0` | `triton_maca` | triton + MACA 栈（`/data/kernelswift-c500`） |
| `bi150` | 天数智芯 Iluvatar BI-V150 | CoreX 4.4.0（CUDA 兼容运行时，`COREX_VERSION` bootstrap） | `cuda:0` | `triton_cuda` | triton 3.1.0 + torch 2.7.1（CoreX 发行版） |
| `ascend910b` | 华为昇腾 Ascend 910B | （规划中） | （规划中） | （待建 `triton_ascend`） | — |

## 说明

- `skills/kernel-opt-loop/prompts/coder_targets/` 目前只有 `triton_mlu.md` 与
  `triton_gcu.md` 两个 profile。`maca`/`bi150` 的 campaign 需要各自的完整 profile
  （`triton_maca`、`triton_cuda`）才能按技能 Phase 0 正规运行——当前这两端的工作
  在 run 分支/worktree 上进行，尚未建立 dev 上可复现的 profile。
- 同一算子在多后端之间的结果**不可直接外推**（技能 KernelWiki 规则）：加速比、能力
  边界均以各后端自己的测量指纹为准。
