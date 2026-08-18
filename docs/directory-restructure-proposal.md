# KernelSwift 仓库目录重构（算子优先 · 已决策）

**创建**: 2026-08-18
**状态**: ✅ 已决策并执行于分支 `refactor/operator-first-layout`（未直接改 dev）
**决策记录**:
- 目录结构：**算子优先**（`kernels/track1-triton/<算子>/<后端>/`）
- 后端命名：**沿用现有命名**（`mlu`/`s60`/`maca`/`bi150`/`ascend910b`），映射写进
  `docs/backend-registry.md`
- 分支策略：**新建迁移分支**，重构不在 dev 上直接提交

---

## 1. 背景

- 比赛：KernelSwift 算子创新大赛（上海人工智能实验室 + 沐曦 + DeepLink）。
  - 赛道一：Triton 算子优化 —— 10 个算子 × 多款国产芯片（主赛道）。
  - 赛道二：Clike（C-like，如 BANG C）算子优化 —— 后续接入。
  - 赛道三：AI4S / 新型模型架构（太初 TECOO）—— 可选预留。
- 官方文档（飞书，需登录）：
  - 赛道一：https://aicarrier.feishu.cn/wiki/YS3OwsOEGiKz2jkaMh9cyGVvngb
  - 赛道二：https://aicarrier.feishu.cn/wiki/MLfLwP3pGiBO8kkKhO9cY7mVn3g

## 2. 迁移前现状（dev）

```
kernelswift/
├── auto_bench.py
├── docs/superpowers/          # 内部设计文档
├── mlu/                       # flexattention / fused_moe(+bangc) / groupedtopk / sparse_pooler
├── s60/                       # groupedtopk
└── skills/kernel-opt-loop/
```

maca / bi150 / ascend910b 仅存在于 run worktree / 分支，dev 无对应目录。

### 迁移前问题

1. 顶层语义混杂：`mlu`/`s60`/`maca`/`bi150` 混用厂商、板卡、软件栈命名；后端矩阵在 dev 不可见。
2. 无赛道维度：赛道二（Clike）无处安放，`mlu/fused_moe/bangc/` 混在赛道一 campaign 里。
3. 无矩阵视图、无仓库级导航（无根 README）。

## 3. 约束（迁移中遵守）

| 约束 | 处理 |
|---|---|
| `auto_bench.py` 保持仓库根路径 | ✅ 未移动（所有 project.md 记录其绝对路径与 SHA-256） |
| `skills/kernel-opt-loop/` 布局不动 | ✅ 未移动；内部证据路径字符串（如 `s60/groupedtopk/triton_grouped_topk_001.py`）是历史证据指针，保留原样 |
| campaign 根目录内部结构不动 | ✅ 只改挂载路径，`base.py`/`state/`/`rounds/`/`log/` 原样 |
| run 分支/worktree 不迁移 | ✅ 正在跑的 campaign 留在各自分支/worktree |
| `docs/superpowers/` 保留 | ✅ 未动 |

## 4. 目标结构（算子优先，已执行）

```
kernelswift/
├── README.md                        # 仓库导航 + 赛道说明 + 矩阵入口
├── auto_bench.py                    # 共享 harness（保持根路径）
├── docs/
│   ├── README.md
│   ├── competition/                 # track1-triton.md / track2-clike.md
│   ├── backend-registry.md          # 后端编码 ↔ 芯片/软件栈/target profile
│   ├── directory-restructure-proposal.md   # 本文档
│   └── superpowers/                 # 原样保留
├── kernels/
│   ├── README.md                    # kernels/ 组织约定 + campaign 根结构说明
│   ├── track1-triton/               # ⭐ 赛道一（算子优先）
│   │   ├── README.md                #   算子 × 后端 进度矩阵
│   │   ├── flexattention/mlu/       #   ✅ 完结 7.08x
│   │   ├── fused_moe/mlu/           #   ✅ 完结 50.4x（bangc 已拆到 track2）
│   │   ├── groupedtopk/
│   │   │   ├── mlu/                 #   ✅ 完结 6.56x
│   │   │   └── s60/                 #   🔄 运行中（r001 +39.1%）
│   │   └── sparse_pooler/mlu/       #   ✅ 完结 1.60x
│   └── track2-clike/                # ⭐ 赛道二（预留）
│       ├── README.md
│       └── fused_moe/mlu/bangc/     #   原 mlu/fused_moe/bangc/ 探针迁入
└── skills/
    └── kernel-opt-loop/             # 不动
```

## 5. 备选方案（后端优先，未采用）

`kernels/track1-triton/<后端>/<算子>/`。优点：一个后端 = 一台远端机器 + 一套 profile，
同步/部署更直白。缺点：跨后端对比同一算子需跨目录。已决策采用算子优先，理由：
同算子的多后端 campaign 相邻，跨后端 diff/对比最直观，贴合"10 个算子"的赛道心智模型。

## 6. 迁移执行记录

1. 从 dev 创建分支 `refactor/operator-first-layout`。
2. `git mv`（保留历史）：
   - `mlu/{flexattention,fused_moe,groupedtopk,sparse_pooler}` → `kernels/track1-triton/<算子>/mlu/`
   - `s60/groupedtopk` → `kernels/track1-triton/groupedtopk/s60/`
   - `mlu/fused_moe/bangc/` → `kernels/track2-clike/fused_moe/mlu/bangc/`
3. 新建骨架：根 `README.md`、`docs/README.md`、`docs/competition/*`、
   `docs/backend-registry.md`、`kernels/README.md`、
   `kernels/track1-triton/README.md`（矩阵）、`kernels/track2-clike/README.md`。
4. 未动：`auto_bench.py`、`skills/`、`docs/superpowers/`、各 campaign 内部文件。

## 7. 后续待办

- [ ] 拿到飞书赛道一文档后，把 10 算子完整清单录入 `docs/competition/track1-triton.md` 与矩阵表
- [ ] 赛道二规则确认后，按算子优先结构新建 track2 campaign；`fused_moe/mlu/bangc/` 探针作为既有材料
- [ ] （可选）为 `maca`/`bi150` 建立完整 target profile（`triton_maca`/`triton_cuda`），
      使这两端的 campaign 能在 dev 上正规复现
- [ ] （可选）`scripts/update_matrix.py` 从各 project.md 自动汇总矩阵
- [ ] 评审通过后把本分支合入 dev（由你决定时机）

## 8. 参考

- 赛道一飞书文档：https://aicarrier.feishu.cn/wiki/YS3OwsOEGiKz2jkaMh9cyGVvngb
- 赛道二飞书文档：https://aicarrier.feishu.cn/wiki/MLfLwP3pGiBO8kkKhO9cY7mVn3g
- [算子创新大赛 CompeteHub](https://competehub.dev/zh/competitions/urlscc6a2494e07d8a7433d83987406c82ca)
- [KernelSwift 算子创新大赛启动（中国电子报）](https://www.cena.com.cn/intelligence/20260723/129268.html)
