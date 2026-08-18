# kernels/ — 算子优化 campaign

本目录按 **赛道 → 算子 → 后端** 组织：

```
kernels/
├── track1-triton/     # 赛道一：Triton 算子优化
│   ├── <算子>/
│   │   ├── <后端>/    # 一个 campaign 根（skill Phase 0 生成）
│   │   └── ...
│   └── README.md      # 算子 × 后端 进度矩阵
└── track2-clike/      # 赛道二：C-like（预留）
```

## campaign 根结构（由 kernel-opt-loop 技能维护，勿手改）

每个 `<算子>/<后端>/` 目录是一个完整 campaign，包含：

```
base.py               # 不可变 PyTorch 参考实现
baseline_adapter.py   # Phase 0 生成（Model → ModelNew）
triton_<op>_NNN.py    # 各轮候选实现
project.md            # 项目身份 / 语义 / 测量指纹 / 轮次总表
team-state.md         # 运行状态机
state/                # Designer / Coder / Verifier 上下文与状态
rounds/               # 每轮 decision / report / status
log/                  # 测量日志与 profiler trace（gitignore）
```

正在运行的 campaign 在各自 `kernel-opt/*` run 分支的 worktree 中，
dev 主线只保留已完结 campaign 的 canonical 代码与记录。
