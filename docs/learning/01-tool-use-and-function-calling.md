# 01 — Tool Use & Function Calling

> 对应代码: `core/llm.py` (structured output), `retrieval/search.py` (外部工具)

## 核心概念

**Tool Use** 是让 LLM 能够调用外部工具（API、数据库、代码执行器等）的机制。LLM 本身只能生成文本，通过 Tool Use，它可以：

1. **决定**调用哪个工具
2. **生成**工具所需的参数（结构化 JSON）
3. **接收**工具返回的结果
4. **基于结果**继续推理或生成回复

```
用户问题 → LLM 思考 → "我需要搜索论文"
                          ↓
               生成 tool call: search(query="RAG survey 2024")
                          ↓
               系统执行搜索，返回论文列表
                          ↓
               LLM 基于结果生成回答
```

## Function Calling vs Tool Use

两个术语在不同 provider 中指同一件事：

| Provider | 术语 | API 字段 |
|----------|------|---------|
| OpenAI | Function Calling | `tools`, `tool_choice` |
| Anthropic | Tool Use | `tools`, `tool_choice` |

本质都是：给 LLM 一组工具的 JSON Schema，LLM 返回要调用的工具名和参数。

## Structured Output: Tool Use 的巧妙应用

我们项目的一个关键设计：**用 Tool Use 机制实现 Structured Output**。

传统做法（脆弱）：
```
prompt: "请以 JSON 格式输出: {title: ..., abstract: ...}"
→ LLM 可能输出不合法的 JSON，或者加了多余的文字
```

我们的做法（可靠）：
```python
# 1. 定义 Pydantic 模型
class ResearchPlan(BaseModel):
    sub_questions: list[str]
    search_strategy: SearchStrategy

# 2. 把模型的 JSON Schema 包装成 "tool"
# 3. 用 tool_choice 强制 LLM "调用"这个 tool
# 4. LLM 返回的 tool call 参数就是结构化数据
plan = await client.generate_structured(messages, ResearchPlan)
```

**为什么可靠？** 因为 LLM provider 的 tool calling 实现有 **constrained decoding** — 它在生成 token 时会参考 JSON Schema，确保输出格式合法。

## 在我们项目中的应用

```
                    Tool Use 在系统中的两种用途
                    
  ┌─────────────────────────────┬──────────────────────────────┐
  │  Structured Output          │  External Tool Calling       │
  │  (内部: Agent → 结构化数据)   │  (外部: Agent → API)         │
  ├─────────────────────────────┼──────────────────────────────┤
  │  PlannerAgent → ResearchPlan│  RetrieverAgent → Semantic   │
  │  ReaderAgent → PaperNote    │    Scholar API / arXiv API   │
  │  CriticAgent → CriticFeedback│                             │
  │  WriterAgent → ResearchReport│                             │
  └─────────────────────────────┴──────────────────────────────┘
```

## 面试要点

1. **"Tool Use 和 RAG 有什么关系？"**
   - RAG 中的检索步骤本质就是一次 Tool Use（调用检索工具）
   - 区别在于：传统 RAG 的检索是硬编码的，Agentic RAG 的检索是 LLM 主动决策的

2. **"如何保证 LLM 输出格式正确？"**
   - 用 function calling / tool use 的 constrained decoding
   - 用 Pydantic 做二次校验
   - 失败时重试 1-2 次

3. **"为什么不直接让 LLM 输出 JSON？"**
   - 自由文本中的 JSON 容易格式错误（缺引号、多逗号等）
   - Tool calling 有 schema 约束，格式保证更强
   - 代码层面用 Pydantic `model_validate` 兜底

## 学习资源

### 必读
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) — 最清晰的 Tool Use 教程
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — Claude 的 Tool Use 文档
- [Toolformer 论文](https://arxiv.org/abs/2302.04761) — 理论基础：LLM 如何学会使用工具

### 进阶
- [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334) — 大规模 API 调用的挑战
- [ToolBench](https://arxiv.org/abs/2307.16789) — Tool Use 的评估基准
- [Instructor 库](https://github.com/jxnl/instructor) — Structured Output 的最佳实践库（可参考设计，我们自己实现）

### 动手练习
- 看我们的 `core/llm.py` 的 `generate_structured` 方法，理解 tool calling 流程
- 修改 `tests/test_llm_integration.py`，尝试更复杂的 structured output（嵌套模型）
