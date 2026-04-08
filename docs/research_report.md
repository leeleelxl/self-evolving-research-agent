# 调研报告摘要

> 完整报告见原始工作空间：/Users/lxl/.openclaw/code/agent_learning/docs/research_report_2026_04_08.md
> 此处仅保留与本项目直接相关的结论

## 竞品

| 产品 | 差异点 |
|------|--------|
| GPT Researcher | 无自进化机制，无消融数据 |
| STORM (Stanford) | Wikipedia 风格综述，非面向具体学术问题 |
| O-Researcher | Agentic RL，学术性强但工程化不足 |

## 我们的差异化

1. **自进化机制**：Critic → 反馈 → 策略改进 → 效果提升（竞品都没有）
2. **完整消融实验**：chunk × 检索 × rerank 全组合对比（竞品缺乏透明数据）
3. **Agentic RAG**：Agent 主动决策检索策略（非固定 pipeline）
4. **公开 benchmark**：HotpotQA / MuSiQue 可复现结果

## 大厂面试对标

- **字节**：Agent 编排 + 评估 → 讲自进化 + Critic Agent
- **阿里**：多 Agent + Tool Calling → 讲五 Agent 协作 + API 集成
- **小红书/淘天/美团**：垂直场景落地 → 讲学术研究这个垂直场景的工程化
