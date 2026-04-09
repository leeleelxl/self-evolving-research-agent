# Agent 知识学习路线图

> 边做 ReSearch v2 项目，边学 Agent 核心知识。每个知识点对应项目中的一个实现步骤。

## 知识地图

```
                    ┌─────────────────────┐
                    │  01 Tool Use &      │ ← Sprint 1A: 检索客户端
                    │  Function Calling   │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │  02 Agentic RAG     │ ← Sprint 1B: PlannerAgent
                    │  vs Traditional RAG │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                              ▼
   ┌──────────────────┐          ┌──────────────────┐
   │ 03 Agent-as-Judge│          │ 04 Self-Evolution │ ← Sprint 2
   │                  │          │    Mechanism      │
   └────────┬─────────┘          └────────┬─────────┘
            └──────────────┬──────────────┘
                           ▼
                ┌──────────────────┐
                │ 05 Multi-Agent   │ ← Sprint 3
                │ Orchestration    │
                └────────┬─────────┘
                         │
              ┌──────────┼──────────┐
              ▼                      ▼
   ┌──────────────────┐  ┌──────────────────┐
   │ 06 Prompt        │  │ 07 Testing Agent │ ← Sprint 4
   │ Engineering      │  │ Systems          │
   └──────────────────┘  └──────────────────┘
```

## 学习顺序

| 序号 | 知识点 | 对应代码 | 面试覆盖 |
|------|--------|---------|---------|
| 01 | [Tool Use & Function Calling](01-tool-use-and-function-calling.md) | `core/llm.py`, `retrieval/search.py` | 字节/阿里必问 |
| 02 | [Agentic RAG](02-agentic-rag.md) | `agents/planner.py`, `pipeline/research.py` | 所有大厂 |
| 03 | [Agent-as-Judge](03-agent-as-judge.md) | `agents/critic.py` | 偏研究岗 |
| 04 | [Self-Evolution](04-self-evolution-mechanism.md) | `pipeline/research.py` | 核心差异化 |
| 05 | [Multi-Agent Orchestration](05-multi-agent-orchestration.md) | `pipeline/research.py`, 所有 agents/ | 字节/阿里必问 |
| 06 | [Prompt Engineering](06-prompt-engineering-for-agents.md) | 所有 agents/ 的 prompt | 所有大厂 |
| 07 | [Testing Agent Systems](07-testing-agent-systems.md) | tests/ | 加分项 |
| 08 | [RAG Pipeline 集成](08-rag-pipeline-integration.md) | `retrieval/knowledge_base.py`, `pipeline/research.py` | 阿里/字节必问 |

## 推荐学习资源（通用）

### 必读论文
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Agent 的理论基础
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — Tool Use 原理
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) — 自进化机制的理论依据
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) — Agent-as-Judge 的可靠性分析

### 必读博客
- [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — Agent 全景综述
- [Anthropic: Building effective agents](https://docs.anthropic.com/en/docs/build-with-claude/agentic) — 工业级 Agent 构建指南
- [LangChain: Agentic RAG](https://blog.langchain.dev/agentic-rag-with-langgraph/) — Agentic RAG 概念（我们不用 LangChain 但概念通用）

### 视频
- [Andrew Ng: Agentic Design Patterns](https://www.youtube.com/watch?v=sal78ACtGTc) — 吴恩达讲 Agent 设计模式（4 种核心模式）
- [Harrison Chase: What is an AI Agent?](https://www.youtube.com/watch?v=F8NKVhkZZWI) — LangChain 创始人讲 Agent 概念
