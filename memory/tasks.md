# 任务看板

> 更新时间：2026-04-12

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

## P1.5 — PDF 全文集成 ✅ 完成

| # | 任务 | 状态 |
|---|------|------|
| 21 | PDF 下载 + 文本提取模块 | ✅ |
| 22 | Reader/KnowledgeBase 全文接入 | ✅ |
| 23 | PDF 消融实验（Δ+0.7, depth +0.9） | ✅ |

## P2 — 后续优化

| # | 任务 | 状态 |
|---|------|------|
| 24 | **Hybrid 引用验证（Emb+NLI, 4.8% 矛盾检测）** | ✅ |
| 25 | **SurGE benchmark 外部对标（Coverage 4x 最强基线）** | ✅ |
| 26 | 接入 sentence-transformers 真实 embedding | 🔲 |
| 27 | 扩大消融样本量（50→200 samples） | 🔲 |
| 28 | 自进化 5 轮对比实验 | 🔲 |
| 29 | GitHub 发布 | 🔲 |
| 30 | CLI 体验提升 | 🔲 |
