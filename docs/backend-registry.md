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

- target id 与 implementation profile id 是两个不同的标识，绝不混用：具体部署目标 id（如 `bi150`、`s60`、`ascend910b`）对应 `target_id`，实现能力契约 id（如 `triton_cuda`、`triton_ascend`）对应 `implementation_profile_id`。上表
  「target profile」列为渲染式 Markdown 说明（`triton_mlu` / `triton_gcu` /
  `triton_cuda` / `triton_maca` / `triton_ascend`），机器可读的 canonical
  implementation profile 位于 `skills/kernel-opt-loop/profiles/` 下。
- 当前 `triton_mlu` 与 `triton_gcu` 拥有 vNext canonical implementation profile
  （`partial` 状态、可执行版本化 probe 套件与 reviewed evidence 目录）；
  `triton_cuda`、`triton_maca`、`triton_ascend` 尚未各自获得 reviewed
  `profile.yaml`、可执行 probe 套件与 approved evidence，不能宣称完整。
- `maca`/`bi150`/`ascend910b` 的 campaign 需要各自的完整 profile 才能按技能
  Phase 0 正规运行——当前这部分工作在 run 分支/worktree 上进行。
- 同一算子在多后端之间的结果**不可直接外推**（技能 KernelWiki 规则）：加速比、能力
  边界均以各后端自己的测量指纹为准。
- Pre-campaign profile onboarding 可在 campaign 创建之前完成，且不创建 campaign
  状态；现有 v1/v2 campaigns 保持只读历史。
