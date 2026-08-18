# kernels/ — 算子优化 campaign

本目录按 **赛道 → 算子 → 后端** 组织：

```
kernels/
├── track1-triton/     # 赛道一：Triton 算子优化
│   ├── <算子>/
│   │   ├── base.py    # 算子级共享参考（一份，设备无关）
│   │   └── <后端>/    # 各后端优化目录（skill Phase 0 生成）
│   └── README.md      # 算子 × 后端 进度矩阵
└── track2-clike/      # 赛道二：C-like（预留）
```

## base.py：算子级共享参考（不按后端重复）

- 每个算子**只有一份** `base.py`，放在算子目录下；不同后端在 `<算子>/<后端>/`
  目录里做优化，**不复制 base**。
- base 用**设备无关写法**：纯 torch + `device="cuda"` 字符串。
  `auto_bench.py` 会自动处理设备（`'npu'`/`'cuda'` 字符串重写为目标加速器、
  输入/模型自动搬运），因此一份 base 可在 mlu / s60(gcu) / bi150(cuda) 等后端复用。
- 运行环境需预加载后端包（如 MLU 机加载 `torch_mlu` 使 harness 能检测到设备；
  bi150 需 CoreX bootstrap），这属于环境配置，不写进 base。
- 已完结 campaign 的 base 若保留历史内容（如早期带 `import torch_mlu` 的
  flexattention/fused_moe/sparse_pooler），可在未来跨后端复用时再清理。

## 后端优化目录结构（由 kernel-opt-loop 技能维护，勿手改）

每个 `<算子>/<后端>/` 目录是一个完整 campaign，包含：

```
baseline_adapter.py   # Phase 0 生成（共享 base 的 Model → ModelNew 副本）
triton_<op>_NNN.py    # 各轮候选实现
project.md            # 项目身份 / 语义 / 测量指纹 / 轮次总表（base 字段指向 ../base.py）
team-state.md         # 运行状态机
state/                # Designer / Coder / Verifier 上下文与状态
rounds/               # 每轮 decision / report / status
log/                  # 测量日志与 profiler trace（gitignore）
```

正在运行的 campaign 在各自 `kernel-opt/*` run 分支的 worktree 中，
dev 主线只保留已完结 campaign 的 canonical 代码与记录。
