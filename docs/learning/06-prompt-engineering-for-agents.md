# 06 — Agent Prompt 设计

> 对应代码: 所有 `agents/*.py` 中的 SYSTEM_PROMPT 和 prompt template

## Prompt 的三层结构

```
┌─────────────────────────┐
│  System Prompt (角色)    │ ← 定义 Agent 是谁，一次设定
├─────────────────────────┤
│  User Prompt (任务)      │ ← 每次调用不同，包含输入数据
├─────────────────────────┤
│  Output Format (格式)    │ ← 通过 structured output 约束
└─────────────────────────┘
```

### Layer 1: System Prompt — 角色定义

```python
# 差的 system prompt（太泛）
"You are a helpful assistant."

# 好的 system prompt（具体、有边界）
"You are the Critic Agent in an academic research system.
Your job is to evaluate the quality of a research survey and provide
structured feedback for improvement. You must be rigorous and specific."
```

**原则**: 告诉 Agent 它是谁、做什么、怎么做。越具体越好。

### Layer 2: User Prompt — 任务描述

```python
# 差的 user prompt（信息不足）
"Evaluate this report."

# 好的 user prompt（结构化输入 + 明确任务）
"## Original Research Question\n{question}\n\n## Survey to Evaluate\n{report}\n\n
## Your Task\n1. Score each dimension...\n2. List missing aspects..."
```

**原则**: 用 Markdown 结构化输入，明确列出任务步骤。

### Layer 3: Output Format — 格式约束

```python
# 差的做法（祈祷 LLM 输出正确 JSON）
"Please output in JSON format: {score: ..., feedback: ...}"

# 好的做法（structured output 强制约束）
await self.generate_structured(prompt, CriticFeedback)
# CriticFeedback 的 schema 自动约束输出格式
```

## 我们项目中的 Prompt 设计案例

### PlannerAgent — 首轮 vs 迭代轮

```python
# 首轮 prompt: 只有研究问题
INITIAL_PROMPT = """
Research question: {question}
Decompose into sub-questions and create a search strategy.
"""

# 迭代轮 prompt: 注入 Critic 反馈
REFINE_PROMPT = """
Research question: {question}
Iteration {iteration}. Critic feedback:
- Coverage: {coverage}/10
- Missing aspects: {missing}
- Suggested queries: {new_queries}

Create an IMPROVED search strategy based on this feedback.
"""
```

**设计要点**: 迭代轮 prompt 把 Critic 的具体反馈注入——不是"做得更好"，而是"覆盖度 4/10，缺少 X，建议搜 Y"。

### CriticAgent — 评分标准锚定

```python
SYSTEM_PROMPT = """
Scoring guidelines (0-10 scale):
- Coverage:
  - 8-10: Comprehensive, covers all key aspects
  - 5-7: Covers most topics but missing some
  - 0-4: Major gaps
"""
```

**设计要点**: 给 LLM 明确的评分锚点。否则它可能总是给 7-8 分（讨好倾向）。

## 常见 Prompt 反模式

| 反模式 | 问题 | 改进 |
|--------|------|------|
| "Be creative" | 学术场景要准确不要创造 | "Be accurate and specific" |
| "Output JSON" | 格式不保证 | 用 structured output |
| "Do your best" | 没有标准 | 给具体评分标准 |
| 一次给太多任务 | 注意力分散 | 一个 prompt 一个任务 |
| 没有示例 | LLM 不知道期望格式 | 给 1-2 个 few-shot 示例 |

## 面试要点

1. **"你的 prompt 是怎么设计的？"**
   - 三层结构：角色 + 任务 + 格式
   - system prompt 固定角色，user prompt 动态注入数据
   - 用 structured output 约束输出格式

2. **"Prompt 调优花了多少时间？"**
   - 诚实回答：Agent prompt 的调优是迭代过程
   - 关键是让 prompt 结构化、具体、可复现
   - 用消融实验对比不同 prompt 的效果

3. **"如果 LLM 输出不符预期怎么办？"**
   - 第一层防线：structured output（格式保证）
   - 第二层防线：Pydantic 校验（类型保证）
   - 第三层防线：代码层重算关键字段（如 is_satisfactory）
   - 第四层防线：重试 1-2 次

## 学习资源

### 必读
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — 最实用的 prompt 指南
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) — Claude 特有技巧

### 进阶
- [DSPy: Programming with Foundation Models](https://arxiv.org/abs/2310.03714) — 用程序化方式优化 prompt
- [PromptBench: Robustness Evaluation](https://arxiv.org/abs/2306.04528) — prompt 鲁棒性评估
