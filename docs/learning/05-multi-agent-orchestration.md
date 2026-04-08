# 05 — 多 Agent 编排模式

> 对应代码: `pipeline/research.py`, 所有 `agents/`

## 编排模式全景

多个 Agent 怎么协作？业界有四种主流模式：

### 1. Sequential Chain（顺序链）

```
A → B → C → D
```

最简单。每个 Agent 的输出是下一个的输入。

- **优点**: 简单可控，调试容易
- **缺点**: 不灵活，无法跳步或并行
- **代表**: LangChain 的 SimpleSequentialChain

### 2. DAG（有向无环图）

```
A → B → D
A → C → D
```

支持分支和合并。

- **优点**: 支持并行、分支逻辑
- **缺点**: 静态结构，不支持循环
- **代表**: LangGraph、Airflow

### 3. Coordinator（协调者）

```
Coordinator ←→ Agent A
Coordinator ←→ Agent B
Coordinator ←→ Agent C
```

一个中心 Agent 动态决定下一步调谁。

- **优点**: 最灵活，支持动态决策
- **缺点**: Coordinator 本身可能成为瓶颈
- **代表**: AutoGen、MetaGPT

### 4. Sequential + Conditional Loop（我们的选择）

```
A → B → C → D → E ─── 满意 → 输出
                  └── 不满意 → 回到 A（策略调整）
```

这是模式 1 + 循环。兼顾简单性和自进化能力。

## 为什么我们选这个模式？

| 考虑因素 | 我们的情况 | 选择 |
|---------|-----------|------|
| Agent 间有数据依赖？ | Retriever 必须等 Planner | 不能并行 → 顺序 |
| 需要动态决策？ | Critic 决定是否继续 | 需要条件分支 → 循环 |
| 复杂度承受度？ | 4 周实习项目 | 越简单越好 → 不用 DAG |

**面试话术**: "我评估了四种编排模式。DAG 适合有并行性的场景，但我们的 Agent 间有严格数据依赖，不适合。Coordinator 模式太灵活，对 4 周项目来说过度设计。Sequential + Conditional Loop 最适合——保持简单的同时支持自进化循环。"

## 我们的编排实现

```python
class ResearchPipeline:
    async def run(self, question: str) -> PipelineResult:
        for iteration in range(max_iterations):
            plan = await self._planner.run(question, feedback)    # 顺序 1
            papers = await self._retriever.run(plan)              # 顺序 2
            notes = await self._reader.run(papers, question)      # 顺序 3
            all_notes.extend(notes)                               # 状态累积
            report = await self._writer.run(all_notes, plan)      # 顺序 4
            feedback = await self._critic.run(report, question)   # 顺序 5
            if feedback.is_satisfactory:
                break                                             # 条件退出
        return PipelineResult(report, evolution_log)
```

关键设计：
- `all_notes.extend(notes)`: 累积而非覆盖，保留历史轮次的论文
- `seen_paper_ids` 去重: 避免同一篇论文被精读两次
- `evolution_log` 记录: 每轮的策略和分数，用于消融实验

## 和 LangGraph 的对比

| 维度 | LangGraph | 我们 |
|------|----------|------|
| 编排定义 | 图结构 (nodes + edges) | 代码逻辑 (for loop + if) |
| 状态管理 | 显式 State 对象 | 局部变量 |
| 条件分支 | conditional_edge | if/break |
| 可视化 | 内置图可视化 | 日志输出 |
| 学习成本 | 中等（需要学 API） | 低（纯 Python） |
| 调试难度 | 较高（抽象层多） | 低（直接读代码） |

## 学习资源

### 必读
- [Anthropic: Building effective agents](https://docs.anthropic.com/en/docs/build-with-claude/agentic) — 最实用的编排指南
- [Andrew Ng: Agentic Design Patterns](https://www.youtube.com/watch?v=sal78ACtGTc) — 四种模式讲解

### 对比阅读
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) — DAG 编排的代表
- [AutoGen Documentation](https://microsoft.github.io/autogen/) — Coordinator 模式的代表
- [CrewAI Documentation](https://docs.crewai.com/) — 角色扮演式多 Agent
