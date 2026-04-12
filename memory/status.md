# 当前状态

> 更新时间：2026-04-12

## 阶段：SurGE 对标完成，全部核心功能闭环

- 5 Agent Pipeline + KnowledgeBase + Multi-LLM Critic + PDF 全文 + Hybrid 引用验证 + SurGE 对标
- 自进化 3 次重复验证：Δ=+0.30±0.19（全部为正）
- 消融实验：5 组 HotpotQA + KB 消融 + Multi-LLM 验证 + PDF 消融 + 引用验证三方法对比 + SurGE 对标
- 80 tests，9 篇知识文档
- 错误处理 + 优雅降级

## 核心数据（面试直接用）

### Hybrid 引用验证（Embedding + NLI）
- Embedding-only: 100% grounding（过宽松，无法区分话题相关 vs 真实支撑）
- NLI-only: 4.3% grounding（过严格，NLI 区分 entailment 和 paraphrase）
- **Hybrid: 100% grounding + 4.8% contradiction 检测（10/208 引用矛盾）**
- 关键 insight: NLI 不适合替代 embedding 做 grounding，但矛盾检测是独特能力
- 模型: DeBERTa-v3-base cross-encoder，句子级推理

### PDF 全文 vs Abstract-only（消融）
- Abstract-only: overall=7.2（depth=6.9）
- Full-text: overall=7.9（depth=7.8）
- **Δ=+0.7，depth +0.9 提升最大**
- PDF 提取成功率 96.7%（58/60）
- 耗时 3x（435s → 1252s），可优化

### 自进化（3 次重复，Multi-LLM Critic）
- Overall: 7.79 ± 0.07（系统稳定）
- Evolution Δ: +0.30 ± 0.19（3/3 全部为正）
- Cross-model spread: accuracy 最不稳定（1.4），coherence 最稳定（0.3）

### KnowledgeBase 集成
- Without KB: Reader 读 50 篇，overall=8.05
- With KB: Reader 读 21 篇（-58%），overall=8.0
- 精读量大幅减少，质量持平

### Multi-LLM 交叉评估
- GPT-4o vs Claude: overall 差 ~1.0（Claude 更严）
- 同家族模型（gpt-4o vs gpt-4o-mini）分数完全一致 → 同家族 Multi-LLM 无意义
- accuracy 跨模型分歧最大 → 已知局限

### 消融实验（BGE-small, 50 samples）
- Best: fixed × hybrid = F1 0.740
- Hybrid > Dense > Sparse

### SurGE Benchmark 对标
- Coverage: 25.0% vs 最强基线 StepSurvey 6.3%（**4x**）
- Logic: 4.00/5 vs StepSurvey 4.85/5（略低，改进方向）
- 独有能力: NLI 矛盾引用检测（1.1%），SurGE 基线无此维度
- 注意: ground truth 规模不同（我们 20 篇 vs SurGE ~100 篇/综述），绝对值不直接可比

## 下一步
- README 更新（PDF + Hybrid 引用验证 + SurGE 数据）
- PDF 耗时优化（选择性下载/缓存）
- GitHub 发布
