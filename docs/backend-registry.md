# 后端注册表

后端目录命名沿用仓库历史（`mlu`/`s60`/`maca`/`bi150`/`ascend910b`），
本表给出每个编码对应的芯片、软件栈与 target profile。

| 目录编码 | 芯片 / 设备 | 软件栈 | 运行时标识 | target profile | 已知远端环境 |
|---|---|---|---|---|---|
| `mlu` | 寒武纪 Cambricon MLU590-H8 | Triton-MLU（torch_mlu） | `mlu:0` | `triton_mlu` | `triton 3.x` + `torch_mlu`（MLU590） |
| `s60` | 燧原 Enflame GCU（S60 板卡） | Triton-GCU（torch_gcu / triton_gcu） | `gcu:0` | `triton_gcu` | triton 3.6.0 + triton_gcu 3.6.0+1.0.20260722；device arch major=3 minor=0 |
| `maca` | 沐曦 MetaX C500 | MACA 兼容面（对上层暴露 `cuda:0`） | `cuda:0` | `triton_maca` | triton + MACA 栈（`/data/kernelswift-c500`） |
| `bi150` | 天数智芯 Iluvatar BI-V150 | CoreX 4.4.0（CUDA 兼容运行时，`COREX_VERSION` bootstrap） | `cuda:0` | `triton_cuda` | triton 3.1.0 + torch 2.7.1（CoreX 发行版） |
| `ascend910b` | 华为昇腾 Ascend 910B | torch_npu + Triton Ascend backend | `npu:0` | `triton_ascend` | torch_npu 2.7.1.post4 + triton 3.2.0（Ascend910B4 probe） |

## 说明

- `skills/kernel-opt-loop/prompts/coder_targets/` 当前包含 `triton_mlu`、
  `triton_gcu`、`triton_cuda`、`triton_maca` 和 `triton_ascend`。每个 profile 的能力
  证据只适用于其记录的运行时与设备指纹，正式 campaign 仍须在 Phase 0 重新发现并匹配。
- 赛道二 C-like 的首个后端是 Ascend，implementation profile id 为 `ascendc`
  （`language=ascendc`、`backend=ascend`）。它不复用 `triton_ascend` 的语言能力表，
  但可以复用经过重新验证的 Ascend 运行时身份和 profiler 事实。implementation/build/
  runner/profiler 规划见 [`../skills/track2-clike-roadmap.md`](../skills/track2-clike-roadmap.md)。
- 同一算子在多后端之间的结果**不可直接外推**（技能 KernelWiki 规则）：加速比、能力
  边界均以各后端自己的测量指纹为准。
