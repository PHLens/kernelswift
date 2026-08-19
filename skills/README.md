# Skills

本目录存放仓库内可复用的 Agent 技能。技能目录中的 `SKILL.md`、角色契约、target profile 和辅助脚本共同定义可执行流程；项目 campaign 产生的状态、候选代码和测量记录则保留在 `kernels/` 下。

## kernel-opt-loop

[`kernel-opt-loop/`](kernel-opt-loop/SKILL.md) 用于对具有不可变 `base.py` 和既有 benchmark harness 的 Triton 算子开展有边界的连续优化。它将职责拆分为 Designer、Coder、Verifier 和 Orchestrator，并通过持久化 artifact 和 Git 提交保持每轮可恢复、可审计。

![Kernel Opt Skill 优化闭环](kernel-opt-loop-flow.svg)

流程要点：

- Phase 0 固化运行时、target profile、基线实现与 measurement fingerprint。
- Designer 每轮只提出一个可证伪的优化假设；Coder 从当前 accepted implementation 实现候选；Verifier 独占真实设备进行正确性、wall-time 和 profiler 验证。
- 只有 `accepted` 结果可推进 `last_accepted_kernel` 和 `last_accepted_report`。被拒绝或失败的候选保留为证据，不能作为下一轮基线。
- 是否采用候选由比赛定义的 e2e benchmark wall time 决定；kernel time、kernel count 与 host overhead 用于定位瓶颈和选择下一轮优化。
- 每个终态 round 提交后，由策略评估决定继续、停止或因环境问题进入可恢复的 blocked 状态。

详细的运行契约、角色边界和停止策略见 [`kernel-opt-loop/SKILL.md`](kernel-opt-loop/SKILL.md)。
