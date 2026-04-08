# ReSearch v2

**自进化多 Agent 学术研究系统** — 自动化文献检索、精读、综述生成，通过 Critic 反馈驱动检索策略持续优化。

## 解决什么问题

学术研究者平均花 **40%+ 时间在文献检索和阅读**上。现有 AI 研究工具（GPT Researcher、STORM）能生成报告，但：
- 检索策略固定，无法根据研究问题自适应调整
- 缺乏质量自评估和迭代改进机制
- 无法量化展示不同策略的效果差异

ReSearch v2 通过 **Critic Agent 驱动的自进化机制**，让系统在每轮研究后自动评估质量并改进检索策略。

## 技术亮点

- **自进化机制**：Critic Agent 评分 → 反馈驱动检索策略改进 → 效果可量化提升
- **5 Agent 协作**：Planner / Retriever / Reader / Writer / Critic 各司其职
- **Agentic RAG**：Agent 主动决策何时检索、检索什么、如何验证（非固定 pipeline）
- **完整消融实验**：chunk 策略 × 检索方式 × rerank 方法的全组合对比数据

## 架构

```
研究问题 → Planner → Retriever → Reader → Writer → Critic
                ↑                                      │
                └──────── 自进化反馈循环 ←──────────────┘
```

## Quick Start

```bash
# 安装
uv pip install -e ".[dev]"

# 运行
uv run python -m research "你的研究问题"

# 测试
uv run pytest
```

## Benchmark

| 数据集 | 指标 | Baseline | ReSearch v2 (0轮) | ReSearch v2 (3轮) |
|--------|------|----------|-------------------|-------------------|
| HotpotQA | EM | - | - | - |
| MuSiQue | F1 | - | - | - |

> 数据待填充

## 消融实验

详见 `experiments/` 目录。

## License

MIT
