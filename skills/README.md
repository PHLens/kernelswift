# Operator Optimization Skills

本目录提供可复用的算子优化流程。`SKILL.md` 定义执行规则，角色契约明确职责边界，implementation profile 描述语言、后端和工具链能力，辅助脚本负责校验、测量与状态推进。

## kernel-opt-loop

[`kernel-opt-loop/`](kernel-opt-loop/SKILL.md) 面向具有参考实现和 benchmark harness 的算子优化任务。流程将工作拆分为四个明确角色：

- **Designer**：分析语义与性能证据，每轮提出一个可验证的优化方案；
- **Coder**：按照已确认的方案和 implementation profile 实现目标语言候选；
- **Verifier**：在目标设备上完成正确性、端到端耗时和 profiler 验证；
- **Orchestrator**：管理流程状态、角色交接、结果采用与恢复。

![算子优化闭环](assets/kernel-opt-loop-flow.png)

### 流程原则

1. 初始化阶段固定参考实现、评测方式、目标后端和性能基线。
2. 每轮只验证一个清晰、可证伪的优化假设。
3. 候选实现必须先通过正确性验证，再比较端到端性能。
4. 只有通过正确性且达到采用阈值的候选才替换当前最佳实现。
5. 未被采用的候选及其测量结果保留为后续优化依据。
6. 环境异常不会被误判为实现失败，修复后可以从安全步骤继续。

## Unified Sketch

Unified Sketch 将优化意图转换为结构化的实现约束，使 Designer、Coder 和 Verifier 对同一方案保持一致理解。它由四个有序部分组成：

- **D — Declarations**：输入输出、shape、dtype、layout、memory 和 tile；
- **O — Operations**：load、compute、store 等数据流与计算步骤；
- **C — Control**：并行映射、循环、条件和边界保护；
- **H — Target Hints**：目标 profile 以及已验证的编译提示。

![Unified Sketch 结构](assets/unified-sketch.png)

Unified Sketch 不是伪代码，也不是性能结果。它是连接优化假设、目标语言实现和验证指标的可检查契约：Coder 据此实现，Verifier 根据对应的 Evaluation Contract 判断机制是否生效。

## 当前支持与赛道二扩展

当前 v1 执行契约面向赛道一：候选是实现 `ModelNew` 的 Python 文件，核心算子使用 Triton，评测由 `auto_bench.py` 完成。赛道二的 C-like 实现以 Ascend 为首个后端，首个 profile id 为 `ascendc`；它复用相同的优化闭环，同时新增多文件构建、原生 ABI、运行适配和设备侧 profiler 能力。

![赛道二 C-like Skill 架构](assets/track2-clike-architecture.png)

具体缺口、建议架构和实施顺序见 [`track2-clike-roadmap.md`](track2-clike-roadmap.md)。

详细的现行执行规则、角色边界、状态转换和停止条件见 [`kernel-opt-loop/SKILL.md`](kernel-opt-loop/SKILL.md)。
