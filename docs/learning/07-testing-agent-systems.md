# 07 — Agent 系统的测试策略

> 对应代码: `tests/` 目录

## 为什么 Agent 系统难测试？

传统软件：输入确定 → 输出确定 → 断言 `assert output == expected`
Agent 系统：输入确定 → LLM 输出**不确定** → 怎么断言？

## 测试金字塔

```
              ╱╲
             ╱  ╲
            ╱ E2E╲          少量，慢，贵（真实 API）
           ╱──────╲
          ╱ 集成测试 ╲        中量，中速（真实 API）
         ╱────────────╲
        ╱   单元测试     ╲     大量，快，免费（Mock）
       ╱──────────────────╲
```

### Layer 1: 单元测试（Mock LLM）

```python
# Mock Agent 的 LLM 调用，验证逻辑
pipeline._planner.run = AsyncMock(return_value=plan)
pipeline._critic.run = AsyncMock(side_effect=[unsatisfied, satisfied])

result = await pipeline.run("test")
assert result.total_iterations == 2  # 验证迭代逻辑
```

测什么：
- Pipeline 编排逻辑（循环、退出条件、去重）
- 数据模型校验（Pydantic）
- 工具函数（normalize_title, f1_score）

### Layer 2: 集成测试（真实 API）

```python
@pytest.mark.integration
async def test_planner_generates_valid_plan():
    planner = PlannerAgent()
    plan = await planner.run("What is RAG?")
    assert len(plan.sub_questions) >= 2  # 结构检查，不检查具体内容
```

测什么：
- API 可用性
- Structured output 格式正确
- 各 Agent 独立功能

### Layer 3: 端到端测试

```python
result = await pipeline.run("What are advances in RAG?")
assert result.total_iterations >= 1
assert len(result.report.sections) > 0
```

测什么：
- 完整流程跑通
- 自进化是否触发
- 最终输出结构完整

## 我们项目的测试组成

| 类型 | 数量 | 特点 |
|------|------|------|
| 数据模型测试 | 9 | 纯逻辑，无 API |
| Retriever 去重测试 | 3 | 纯逻辑，无 API |
| RAG 组件测试 | 7 | chunking + indexing，无 API |
| LLM 集成测试 | 3 | 真实 API |
| Agent 集成测试 | 7 | 真实 API |
| Pipeline Mock 测试 | 3 | Mock Agent |
| Pipeline E2E 测试 | 1 | 完整流程 |

## Agent 测试的特殊技巧

### 1. 检查结构不检查内容

```python
# 差：检查具体内容（LLM 输出不确定）
assert plan.sub_questions[0] == "What is dense retrieval?"

# 好：检查结构（稳定）
assert len(plan.sub_questions) >= 2
assert isinstance(plan, ResearchPlan)
```

### 2. 用 temperature=0 提高可复现性

```python
config = LLMConfig(temperature=0.0)  # 减少随机性
```

### 3. 对 LLM 评分用范围断言

```python
# 差：精确断言
assert feedback.scores.coverage == 7.5

# 好：范围断言
assert 0 <= feedback.scores.coverage <= 10
assert feedback.scores.overall < 7.0  # 空报告应该低分
```

## 面试要点

1. **"你怎么测试 Agent 系统？"**
   - 三层金字塔：单元（Mock）→ 集成（真实 API）→ E2E
   - 单元测试验证逻辑，集成测试验证 API 兼容性
   - 结构断言而非内容断言

2. **"Mock 和真实 API 怎么选择？"**
   - 编排逻辑、去重、排序 → Mock（快、确定性）
   - Structured output 格式、API 可用性 → 真实 API
   - 原则：不需要 LLM 的逻辑不应该依赖 LLM

## 学习资源

- [Testing AI Applications (Anthropic)](https://docs.anthropic.com/en/docs/build-with-claude/develop-tests)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Python AsyncMock](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.AsyncMock)
