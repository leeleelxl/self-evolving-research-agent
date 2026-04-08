# 当前状态

> 更新时间：2026-04-08

## 阶段：项目初始化

- 经过系统调研，确认项目方向：自进化多 Agent 学术研究系统
- 项目骨架已搭建，代码尚未开始
- 前序调研报告见 `docs/research_report.md`

## 核心定位

对标 GPT Researcher / STORM，差异化点：
1. **自进化机制**：Critic Agent 评分 → 反馈驱动检索策略改进
2. **完整消融实验**：chunk 策略 × 检索方式 × rerank 方法的全组合对比
3. **Agentic RAG**：Agent 主动决策何时检索、检索什么、如何验证
4. **在公开 benchmark 上跑数据**：HotpotQA / MuSiQue

## 下一步行动

1. 完成架构设计文档 (`docs/architecture.md`)
2. 实现项目骨架代码（Agent 基类、Pipeline 框架）
3. 接入 Semantic Scholar / arXiv 检索
4. 实现基础 RAG pipeline（chunk + 向量检索 + rerank）

## 面试叙事

> "我做了一个自进化的多 Agent 学术研究系统，核心创新是 Critic Agent 驱动的检索策略自适应优化。在 HotpotQA 上，经过 3 轮自进化后准确率提升了 X%。项目包含完整的 RAG 消融实验，对比了 N 种 chunk 策略和 M 种检索方式的组合效果。"
