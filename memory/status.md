# 当前状态

> 更新时间：2026-04-09

## 阶段：KnowledgeBase 集成完成，RAG 组件全部接入 Pipeline

- 5 Agent 全部实现，Pipeline 端到端跑通
- **KnowledgeBase 接入 Pipeline**：RAG 组件不再孤立，真正参与检索-精读流程
- 自进化验证通过：7.8 → 8.0（+0.2），queries 3x 增长
- 消融实验完成：5 组 HotpotQA（50 samples），用真实 embedding（BGE-small）
- KB 消融实验完成：精读量 -58%，质量持平
- 55 tests（新增 10 个 KB 测试），8 篇知识文档 → 9 篇
- 错误处理 + 优雅降级：Agent 重试、Pipeline 单轮失败不崩溃
- Prompt 优化：Critic accuracy 校准、Planner query 去重

## 核心数据（面试直接用）

### 自进化
- Round 0: overall=7.8 → Round 1: overall=8.0 (Δ+0.2)
- Queries: 8 → 24（3x 增长，定向补充缺失方向）
- Notes: 45 → 95（累积）

### KnowledgeBase 集成效果
- Without KB: Reader 读 50 篇，overall=8.05
- With KB: Reader 读 21 篇（-58%），overall=8.0，coherence +0.3
- 结论：精读量大幅减少，质量持平，coherence 反而提升

### 消融实验（BGE-small embedding, 50 samples）
- Best: fixed × hybrid = F1 0.740
- Hybrid > Dense > Sparse（RRF 融合效果最好）
- Embedding 质量对 Dense 影响 +36%

## 下一步

- P2 可选优化：CLI 改进、扩大实验样本量、PDF 全文集成
- 面试准备：整理高频问题和答案
