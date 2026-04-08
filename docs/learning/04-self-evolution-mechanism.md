# 04 — 自进化机制：Critic 驱动的策略改进

> 对应代码: `pipeline/research.py` (自进化循环), `agents/critic.py` → `agents/planner.py` (反馈闭环)

## 核心概念

**自进化 (Self-Evolution)** = 系统根据自身输出的评估结果，自动改进行为策略。

```
简单重试:   同样的策略 → 同样的结果（或运气好有些变化）
自进化:     评估弱点 → 定向调整策略 → 针对性改进结果
```

类比：
- 简单重试 = 考试没过，再考一遍同样的题
- 自进化 = 考试没过，分析错在哪，专门练薄弱知识点，再考

## 在我们系统中的实现

```
第 0 轮 (Baseline):
  Planner 生成初始策略: 5 个 query
  → Retriever 搜到 30 篇论文
  → Reader 精读 20 篇
  → Writer 生成综述
  → Critic 评分:
      coverage=5 "缺少 X 方向"
      depth=4    "Y 方法只提了名字，没分析细节"

第 1 轮 (Evolution):
  Planner 收到 Critic 反馈:
    → 增加 query: "X 方向 2024 survey"
    → 增加 focus: "Y 方法的技术细节"
    → 排除已覆盖方向（avoid redundant retrieval）
  → Retriever 搜到 15 篇新论文
  → Reader 精读 12 篇
  → Writer 用 32 篇笔记重新生成综述（累积！）
  → Critic 评分:
      coverage=7 ↑  "X 方向已覆盖"
      depth=6 ↑      "Y 方法有了基础分析，但缺少对比"

第 2 轮 (Evolution):
  → coverage=8, depth=8 → 达标，停止 ✓
```

## 关键设计决策

### 1. 状态累积 vs 无状态

```python
# 我们的做法：累积
all_notes.extend(notes)  # 每轮新笔记累加
report = await self._write(all_notes, plan)  # 用所有笔记写

# 而不是：每轮从头开始
notes = await self._read(papers, question)  # ✗ 丢弃历史
```

为什么累积？因为第 0 轮搜到的论文仍然有价值，不应该丢弃。自进化改进的是"边际增量"。

### 2. 反馈的结构化程度

```python
# 差的反馈（不可执行）
"综述需要改进"

# 好的反馈（可执行）
CriticFeedback(
    scores=CriticScores(coverage=5, depth=4, ...),
    missing_aspects=["dense passage retrieval techniques"],
    new_queries=["dense passage retrieval survey 2024"],
    improvement_suggestions=["Add comparison between BM25 and DPR"],
)
```

反馈越结构化，Planner 就越容易据此改进。这就是为什么 `new_queries` 是直接可执行的搜索关键词。

### 3. 停止条件

```python
# 两个停止条件，取先到达的
if feedback.is_satisfactory:  # overall >= threshold (7.0)
    break
# 或
for iteration in range(max_iterations):  # 最多 N 轮
    ...
```

为什么需要最大轮数？防止 Critic 永远不满意的死循环。实际中，边际收益递减——第 3 轮后改进通常很小。

## 自进化 vs 其他改进范式

| 范式 | 改进时机 | 改进内容 | 代表 |
|------|---------|---------|------|
| **Self-Refine** | 推理时 | 输出文本 | Madaan et al., 2023 |
| **Self-Evolution (我们)** | 推理时 | 检索策略 | 本项目 |
| **RLHF** | 训练时 | 模型权重 | ChatGPT |
| **Self-Play** | 训练时 | 策略网络 | AlphaGo |

我们的自进化和 Self-Refine 最像，但区别在于：
- Self-Refine 只改**输出文本**（让 Writer 改措辞）
- 我们改**检索策略**（让 Planner 搜新方向）

GPT-Researcher 的 Review-Revise 就是 Self-Refine（只改文本），我们的是 Strategy-level Evolution（改策略）。

## 面试要点

1. **"自进化和简单 retry 有什么区别？"**
   - Retry: 同策略重跑，期望随机性带来不同结果
   - 自进化: 分析弱点 → 定向改进策略 → 针对性搜索
   - 类比: Self-Play in RL — 每轮分析弱点，下轮针对性训练

2. **"怎么证明自进化真的有效？"**
   - 画 scores vs iterations 曲线（应递增且收敛）
   - 对比 max_iterations=1 vs 3 vs 5 的最终分数
   - 检查每轮的 query 差异（应该逐轮不同）

3. **"自进化有什么局限？"**
   - 依赖 Critic 的质量——如果 Critic 评估不准，进化方向就错
   - 边际收益递减——第 5 轮后基本没改进了
   - 成本线性增长——每轮都消耗 API token

## 学习资源

### 必读
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) — 最相关的论文
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) — 用语言反馈做强化学习
- [Lilian Weng: LLM Agents - Reflection](https://lilianweng.github.io/posts/2023-06-23-agent/#reflection) — 综述中的反思部分

### 进阶
- [LATS: Language Agent Tree Search](https://arxiv.org/abs/2310.04406) — 用搜索树结构化反思
- [Self-Evolving GPT](https://arxiv.org/abs/2312.02111) — 更广义的自进化框架
- [Constitutional AI](https://arxiv.org/abs/2212.08073) — Anthropic 的自我改进方法（用原则指导修正）
