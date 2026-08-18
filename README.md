# KernelSwift 算子优化仓库

KernelSwift 算子创新大赛（上海人工智能实验室 + 沐曦 + DeepLink）参赛工作仓库。
当前主战场：**赛道一（Triton 算子优化）**——在多个国产硬件后端上优化 10 个 Triton 算子；
后续接入 **赛道二（C-like 算子优化）**。

## 目录地图

```
kernels/                  # 全部算子优化 campaign（算子优先：<算子>/<后端>/）
├── track1-triton/        #   赛道一：Triton 算子优化
│   └── README.md         #   10 算子 × 后端 进度/加速比矩阵（先看这里）
└── track2-clike/         #   赛道二：C-like（预留，含 fused_moe bangc 探针）
auto_bench.py             # 共享评测 harness（v0/v1 对比 + profiler），所有 campaign 引用它
docs/
├── competition/          #   比赛规则与算子清单（按赛道）
├── backend-registry.md   #   后端编码 ↔ 芯片/软件栈/target profile 映射
├── directory-restructure-proposal.md  # 本次目录重构提案（含决策记录）
└── superpowers/          #   内部设计文档（kernel-opt-loop 技能 specs/plans）
skills/kernel-opt-loop/   # 优化循环技能（Designer/Coder/Verifier 契约 + target profiles）
```

## 赛道

| 赛道 | 内容 | 状态 | 文档 |
|---|---|---|---|
| 赛道一 | Triton 算子优化：10 算子 × 多国产芯片 | 进行中 | [track1-triton](docs/competition/track1-triton.md) |
| 赛道二 | Clike（C-like，如 BANG C）算子优化 | 预留 | [track2-clike](docs/competition/track2-clike.md) |
| 赛道三 | AI4S / 新型模型架构（太初 TECOO） | 未开始 | — |

官方飞书文档（需登录）：[赛道一](https://aicarrier.feishu.cn/wiki/YS3OwsOEGiKz2jkaMh9cyGVvngb) ·
[赛道二](https://aicarrier.feishu.cn/wiki/MLfLwP3pGiBO8kkKhO9cY7mVn3g)

## 如何新增一个 campaign

1. 在 `kernels/track1-triton/<算子>/<后端>/` 下由
   [kernel-opt-loop 技能](../skills/kernel-opt-loop/SKILL.md) 的 Phase 0 生成 campaign 根
   （`base.py` + `project.md` + `state/` + `rounds/` + `log/`），并创建专属 run 分支
   `kernel-opt/<算子>-<后缀>`（worktree 方式运行，避免与 dev 冲突）。
2. 更新 `kernels/track1-triton/README.md` 矩阵表对应格子。
3. 后端首次出现时，确认 `skills/kernel-opt-loop/prompts/coder_targets/` 下存在匹配的
   target profile（目前只有 `triton_mlu` / `triton_gcu`）。

## 约定

- `auto_bench.py` 固定在仓库根路径：所有 campaign 的 `project.md` 记录了它的绝对路径与
  SHA-256，移动会破坏测量指纹，**不要移动**。
- 正在运行的 campaign 在各自 run 分支/worktree 上；dev 主线上放已完结 campaign 的
  canonical 代码与记录。
