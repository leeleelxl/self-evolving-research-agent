# 03 — Agent-as-Judge: 用 LLM 做质量评估

> 对应代码: `agents/critic.py`

## 核心概念

**Agent-as-Judge** (也叫 LLM-as-Judge) = 用 LLM 代替人类做质量评估。

```
传统评估: 人工审核（慢、贵、不可扩展）
Auto 评估: 自动指标如 BLEU/ROUGE（快但浅，无法评估语义质量）
Agent-as-Judge: LLM 评估（快、深、可扩展，但有偏差）
```

## 为什么需要？

在自进化系统中，每轮迭代都需要评估输出质量。人工评估不可扩展（每轮都找人看？），自动指标太浅（ROUGE 不知道综述是否覆盖了关键方向）。LLM 评估是唯一实际可行的选择。

## 已知偏差（面试必知）

| 偏差类型 | 含义 | 影响 |
|---------|------|------|
| **Self-preference bias** | LLM 更喜欢自己生成的内容 | Critic 对自己系统的输出评分偏高 |
| **Position bias** | 先出现的选项更容易被选中 | 在 A/B 对比时影响排序 |
| **Verbosity bias** | 更长的回答得分更高 | 啰嗦的综述可能被评为高质量 |
| **Anchoring bias** | 评分受参考标准影响 | 如果先看了高质量样本，后续评分偏低 |

## 我们的应对策略

```python
# 1. 多维度评分（不是一个笼统分数）
class CriticScores(BaseModel):
    coverage: float   # 覆盖度
    depth: float       # 深度
    coherence: float   # 连贯性
    accuracy: float    # 准确性

# 2. 代码层面重算 is_satisfactory（不完全信任 LLM）
feedback.is_satisfactory = feedback.scores.overall >= self._threshold

# 3. 要求具体证据（不只是分数）
missing_aspects: list[str]       # "缺少 X 方向的论文"
improvement_suggestions: list[str] # "应该增加对 Y 方法的分析"
new_queries: list[str]           # "search: Y method 2024 survey"
```

## 评估维度设计的学问

我们的 4 个维度参考了学术论文审稿标准：

| 维度 | 对标学术审稿 | 在自进化中的作用 |
|------|------------|----------------|
| Coverage | "文献是否全面" | 低 → Planner 增加新 query |
| Depth | "分析是否深入" | 低 → Planner 增加 focus_areas |
| Coherence | "逻辑是否通顺" | 低 → Writer 需要改进结构 |
| Accuracy | "引用是否准确" | 低 → Reader 需要更仔细提取 |

## 面试要点

1. **"LLM-as-Judge 可靠吗？"**
   - 诚实回答：有偏差，但比纯自动指标强
   - 应对方法：多维度评分、阈值可配置、消融实验交叉验证
   - 论文支撑：MT-Bench 研究显示 GPT-4 与人工评估的一致性 >80%

2. **"如何校准 Critic 的评分？"**
   - 可以用几个人工评估的样本做 calibration
   - 在 prompt 中给出具体的评分标准（我们的 0-10 scale 指南）
   - 对比不同模型做 Critic 的效果（消融实验）

3. **"Critic 和 Planner 之间的反馈闭环是怎么工作的？"**
   - Critic 输出 `CriticFeedback`（分数 + 具体建议 + 新 query）
   - Planner 的 refine prompt 直接注入这些信息
   - 关键是反馈要**结构化且可执行**，不是"做得更好"

## 学习资源

### 必读
- [Judging LLM-as-a-Judge with MT-Bench](https://arxiv.org/abs/2306.05685) — 最重要的 LLM-as-Judge 论文
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) — 自反馈迭代的理论基础
- [Anthropic: Evaluating AI outputs](https://docs.anthropic.com/en/docs/build-with-claude/develop-tests) — 工程实践

### 进阶
- [G-Eval: NLG Evaluation using GPT-4](https://arxiv.org/abs/2303.16634) — 用 LLM 做自然语言生成评估
- [ChatEval: Towards Better LLM-based Evaluators](https://arxiv.org/abs/2308.07201) — 多 Agent 辩论式评估
- [FActScore: Fine-grained Atomic Evaluation of Factual Precision](https://arxiv.org/abs/2305.14251) — 事实性的细粒度评估
