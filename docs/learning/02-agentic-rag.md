# 02 — Agentic RAG vs Traditional RAG

> 对应代码: `agents/planner.py` (动态策略), `pipeline/research.py` (自进化循环)

## RAG 基础回顾

**RAG (Retrieval-Augmented Generation)** = 检索 + 生成。给 LLM 提供外部知识来回答问题。

```
传统 RAG 流程:
  用户问题 → 固定 query → 向量检索 → top-k 文档 → LLM 生成回答
```

问题在于：**检索策略是固定的**。query 怎么写、检索几次、从哪里检索——都是预设的。

## Agentic RAG: 让 Agent 主导检索

Agentic RAG 的核心变化：**把检索决策权交给 Agent**。

```
Agentic RAG 流程:
  用户问题 → Agent 分析问题
           → Agent 决定检索策略（搜什么、从哪搜、搜几次）
           → 执行检索
           → Agent 评估结果质量
           → 不够好？Agent 调整策略再搜
           → 够好了？生成回答
```

### 对比表

| 维度 | Traditional RAG | Agentic RAG |
|------|----------------|-------------|
| 检索策略 | 固定（hardcoded） | Agent 动态决策 |
| 检索次数 | 通常 1 次 | 按需多轮 |
| Query 生成 | 直接用用户问题 | Agent 分解 + 重写 |
| 质量控制 | 无 | Agent 自评估 |
| 失败恢复 | 无（返回低质量结果） | Agent 调整策略重试 |

## 在我们项目中的体现

我们的系统是 **Agentic RAG 的典型实现**：

```
Planner Agent: "这个问题需要搜 3 个方向，每个方向 2 个 query"
    ↓ (动态策略生成)
Retriever Agent: 执行检索
    ↓
Reader Agent: 精读论文
    ↓
Writer Agent: 生成综述
    ↓
Critic Agent: "覆盖度不够，缺少 X 方向的论文"
    ↓ (Agent 主动评估)
Planner Agent: "增加 X 方向的 query，再搜一轮"  ← 策略自适应调整
    ↓
... (循环直到满意)
```

**Agentic 的四个体现**：
1. **问题分解**：Planner 不是直接搜原始问题，而是分解成子问题
2. **多源检索**：Agent 决定用 Semantic Scholar 还是 arXiv
3. **质量自评**：Critic 评估结果质量
4. **策略调整**：根据评估反馈改变检索策略

## 吴恩达的四种 Agentic Design Patterns

Andrew Ng 总结的四种模式，我们的项目用到了其中三种：

| 模式 | 含义 | 我们项目 |
|------|------|---------|
| **Reflection** | Agent 评估自己的输出并改进 | ✅ Critic 评估 → Planner 改进 |
| **Tool Use** | Agent 调用外部工具 | ✅ Retriever 调用搜索 API |
| **Planning** | Agent 分解任务为步骤 | ✅ Planner 分解子问题 |
| **Multi-Agent** | 多个 Agent 协作 | ✅ 5 Agent 编排 |

## 面试要点

1. **"你的系统和传统 RAG 有什么区别？"**
   - 传统 RAG 是 query → retrieve → generate，策略固定
   - 我们是 Agent 主动决策检索策略，支持多轮迭代和策略调整
   - 关键词：adaptive retrieval, query decomposition, self-refinement

2. **"Agentic RAG 的缺点是什么？"（面试官爱问缺点）**
   - 延迟更高（多轮 LLM 调用）
   - 成本更高（每轮都消耗 token）
   - 可控性降低（Agent 可能做出不合理的决策）
   - 调试更难（多 Agent 交互的状态难追踪）

3. **"什么场景适合用 Agentic RAG？"**
   - 复杂问题（需要多步推理）
   - 质量要求高（宁可慢也要准确）
   - 数据源异构（需要从多个来源检索）
   - 简单问题用传统 RAG 就够了，不需要 Agent

## 学习资源

### 必读
- [Andrew Ng: Agentic Design Patterns](https://www.youtube.com/watch?v=sal78ACtGTc) — 四种模式的讲解视频
- [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — Agent 系统综述（含 RAG 部分）
- [Anthropic: Building effective agents](https://docs.anthropic.com/en/docs/build-with-claude/agentic) — 从工程角度讲 Agentic 系统

### 论文
- [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511) — 自适应检索的学术基础
- [Adaptive-RAG](https://arxiv.org/abs/2403.14403) — 根据问题复杂度选择 RAG 策略
- [CRAG: Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) — 检索结果的自动纠错

### 对比阅读
- [GPT Researcher 架构](https://docs.gptr.dev/docs/gpt-researcher/getting-started/how-it-works) — 看看别人怎么做 Agentic RAG
- [STORM (Stanford)](https://arxiv.org/abs/2402.14207) — 对比另一种文献综述生成方式
