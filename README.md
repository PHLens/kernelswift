# KernelSwift 算子优化仓库

KernelSwift 算子创新大赛（上海人工智能实验室 + 沐曦 + DeepLink）参赛工作仓库。
当前主战场：**赛道一（Triton 算子优化）**——在多个国产硬件后端上优化 10 个 Triton 算子；
后续接入 **赛道二（C-like 算子优化）**。

## 目录地图

```
kernels/                  # 全部算子优化 campaign（算子优先：<算子>/<后端>/）
├── track1-triton/        #   赛道一：Triton 算子优化
│   └── README.md         #   10 算子 × 后端 进度/加速比矩阵（先看这里）
└── track2-clike/         #   赛道二：C-like（3 个算子 base 已就绪）
auto_bench.py             # 共享评测 harness（v0/v1 对比 + profiler），所有 campaign 引用它
docs/
├── competition/          #   比赛规则与算子清单（按赛道）
├── backend-registry.md   #   后端编码 ↔ 芯片/软件栈/target profile 映射
└── superpowers/          #   内部设计文档（kernel-opt-loop 技能 specs/plans）
skills/kernel-opt-loop/   # 优化循环技能（Designer/Coder/Verifier 契约 + target profiles）
```

## 赛道

| 赛道 | 内容 | 状态 | 文档 |
|---|---|---|---|
| 赛道一 | Triton 算子优化：10 算子 × 多国产芯片 | 进行中 | [track1-triton](docs/competition/track1-triton.md) |
| 赛道二 | Clike 算子优化：首个后端 Ascend，使用 Ascend C，3 个算子 | 进行中 | [track2-clike](docs/competition/track2-clike.md) |
| 赛道三 | AI4S / 新型模型架构（太初 TECOO） | 未开始 | — |

官方飞书文档（需登录）：[赛道一](https://aicarrier.feishu.cn/wiki/YS3OwsOEGiKz2jkaMh9cyGVvngb) ·
[赛道二](https://aicarrier.feishu.cn/wiki/MLfLwP3pGiBO8kkKhO9cY7mVn3g)

## 如何新增一个 campaign

1. 在 `kernels/track1-triton/<算子>/<后端>/` 下由
   [kernel-opt-loop 技能](skills/kernel-opt-loop/SKILL.md) 的 Phase 0 生成 campaign 根
   （`project.md` + `state/` + `rounds/` + `log/` + `baseline_adapter.py`）。
   共享 `base.py` 已位于 `<算子>/base.py`，**不要复制**。
   创建专属 run 分支 `kernel-opt/<算子>-<后缀>`（worktree 方式运行，避免与 dev 冲突）。
2. 更新 `kernels/track1-triton/README.md` 矩阵表对应格子。
3. 后端首次出现时，确认 `skills/kernel-opt-loop/prompts/coder_targets/` 下存在匹配的
   target profile（`triton_mlu` / `triton_gcu` / `triton_cuda` / `triton_maca` /
   `triton_ascend`）。机器可读的 canonical implementation profile 位于
   `skills/kernel-opt-loop/profiles/<implementation_profile_id>/`；当前 `triton_mlu`
   和 `triton_gcu` 已完成 vNext 迁移（含 reviewed `profile.yaml`、可执行版本化 probe
   套件与 approved evidence），其余 Markdown 页面仍为渲染式说明，直到各自拥有
   reviewed `profile.yaml`、可执行版本化 probe 套件与 approved evidence。

## 约定

- `auto_bench.py` 固定在仓库根路径：所有 campaign 的 `project.md` 记录了它的绝对路径与
  SHA-256，移动会破坏测量指纹，**不要移动**。
- 正在运行的 campaign 在各自 run 分支/worktree 上；dev 主线上放已完结 campaign 的
  canonical 代码与记录。

## vNext 新 run 边界与 profile onboarding

- 具体部署目标 `target_id`（如 `bi150`、`s60`、`ascend910b`）与实现能力契约
  `implementation_profile_id`（如 `triton_cuda`、`triton_ascend`）是两个不同的
  标识：API 兼容性（例如暴露 `cuda:0`）绝不把能力证据迁移到其他厂商、设备、架构或
  工具链。
- Pre-campaign profile onboarding 属于 kernel-opt-loop profile 子系统：运行版本化
  probes、产出哈希化 run-local evidence 与 proposed promotion candidate，并且可以
  不创建 campaign 就结束。它绝不编辑 canonical implementation profile。
- 现有 v1/v2 campaigns 保持只读历史；vNext 激活是新建 campaign 时的选择。
- 每个 Triton submission snapshot 只运行一次离线、有界、config-only 的 finalization
  gate，要求 exact-source 确认与 post-pin 官方验证。最终候选包含一个固定配置，无
  runtime/online autotune、首次使用搜索或缓存依赖的配置选择。
