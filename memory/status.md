# 当前状态

> 更新时间：2026-04-08

## 阶段：项目完成，可面试可发布

- 5 Agent 全部实现，Pipeline 端到端跑通
- 自进化验证通过：7.8 → 8.0（+0.2），queries 3x 增长
- 消融实验完成：5 组 HotpotQA（50 samples），用真实 embedding（BGE-small）
- 最佳配置：fixed chunk + hybrid retrieval（F1=0.740）
- 38+ tests，8 篇知识文档
- README 含真实量化数据

## 核心数据（面试直接用）

### 自进化
- Round 0: overall=7.8 → Round 1: overall=8.0 (Δ+0.2)
- Queries: 8 → 24（3x 增长，定向补充缺失方向）
- Notes: 45 → 95（累积）

### 消融实验（BGE-small embedding, 50 samples）
- Best: fixed × hybrid = F1 0.740
- Hybrid > Dense > Sparse（RRF 融合效果最好）
- Embedding 质量对 Dense 影响 +36%
