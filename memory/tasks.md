# 任务看板

> 更新时间：2026-04-09

## P0 — 核心系统 ✅ 全部完成

| # | 任务 | 状态 |
|---|------|------|
| 1 | 架构设计文档 | ✅ |
| 2 | Agent 基类 + Pipeline 框架 | ✅ |
| 3 | Semantic Scholar / arXiv 检索工具 | ✅ |
| 4 | PlannerAgent | ✅ |
| 5 | RetrieverAgent | ✅ |
| 6 | CriticAgent | ✅ |
| 7 | ReaderAgent | ✅ |
| 8 | WriterAgent | ✅ |
| 9 | Pipeline 集成（5 Agent 编排） | ✅ |
| 10 | Chunk 策略（fixed/semantic/recursive） | ✅ |
| 11 | FAISS + BM25 索引 + Hybrid (RRF) | ✅ |
| 12 | LLM Reranker | ✅ |
| 13 | **KnowledgeBase — RAG 接入 Pipeline** | ✅ |
| 14 | **错误处理 + 优雅降级** | ✅ |
| 15 | **Prompt 优化（Critic + Planner）** | ✅ |

## P1 — 实验验证 ✅ 全部完成

| # | 任务 | 状态 |
|---|------|------|
| 16 | 端到端 Pipeline 跑通 | ✅ |
| 17 | 自进化效果验证（7.8 → 8.0, Δ+0.2） | ✅ |
| 18 | RAG 消融实验（5 组 HotpotQA） | ✅ |
| 19 | **KB 消融实验（精读量 -58%，质量持平）** | ✅ |
| 20 | README 填入真实数据 | ✅ |

## P2 — 后续优化（可选）

| # | 任务 | 状态 |
|---|------|------|
| 21 | 接入 sentence-transformers 真实 embedding | 🔲 |
| 22 | 扩大消融样本量（50→200 samples） | 🔲 |
| 23 | PDF 全文处理集成 | 🔲 |
| 24 | 自进化 5 轮对比实验 | 🔲 |
| 25 | GitHub 发布 | 🔲 |
| 26 | CLI 体验提升 | 🔲 |
