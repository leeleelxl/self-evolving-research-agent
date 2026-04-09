# ReSearch v2

**自进化多 Agent 学术研究系统** — 自动化文献检索、精读、综述生成，通过 Critic 反馈驱动检索策略持续优化。

## 解决什么问题

学术研究者平均花 **40%+ 时间在文献检索和阅读**上。现有 AI 研究工具（GPT Researcher、STORM）能生成报告，但：
- 检索策略固定，无法根据研究问题自适应调整
- 缺乏质量自评估和迭代改进机制
- 无法量化展示不同策略的效果差异

ReSearch v2 通过 **Critic Agent 驱动的自进化机制**，让系统在每轮研究后自动评估质量并改进检索策略。

## 技术亮点

- **自进化机制**：Critic Agent 评分 → 反馈驱动检索策略改进 → 3 次重复实验 Δ=+0.30±0.19（全部为正）
- **Multi-LLM 交叉评估**：GPT-4o + Claude 双模型 Critic，记录分歧度量化评估置信度（参考 AutoSurvey）
- **5 Agent 协作**：Planner / Retriever / Reader / Writer / Critic 各司其职
- **KnowledgeBase RAG 集成**：chunking → embedding → hybrid index，精读量 -58% 质量持平
- **完整消融实验**：chunk 策略 × 检索方式 × rerank 方法的全组合对比数据
- **不依赖框架**：直接用 Anthropic/OpenAI SDK 构建，每一层可解释

## 架构

```
研究问题 → Planner (问题分解) → Retriever (文献检索)
              ↑                         ↓
              │                  KnowledgeBase (RAG 过滤)
              │                    chunk → embed → index
              │                    按子问题检索 top-k
              │                         ↓
              │                    Reader (精读相关论文)
              │                         ↓
              └──── Critic (质量评估) ←── Writer (综述生成)
                        │
                   自进化反馈循环
                   (策略调整 + 定向补充)
```

## 自进化效果

### 重复实验（n=3, Multi-LLM Critic）

同一问题 "RAG for LLMs" 跑 3 次，使用 GPT-4o + Claude 双模型交叉评估：

| 维度 | Mean ± Std | 3 次结果 |
|------|-----------|---------|
| Coverage | 7.77 ± 0.25 | [8.0, 7.5, 7.8] |
| Depth | 7.57 ± 0.12 | [7.5, 7.7, 7.5] |
| Coherence | 7.97 ± 0.15 | [8.0, 8.1, 7.8] |
| Accuracy | 7.87 ± 0.35 | [7.5, 8.2, 7.9] |
| **Overall** | **7.79 ± 0.07** | — |
| **Evolution Δ** | **+0.30 ± 0.19** | [+0.33, +0.47, +0.10] |

关键发现：
- **自进化 Δ 全部为正**（3/3 runs），mean=+0.30
- Overall 方差极小（±0.07），系统输出稳定
- Accuracy 方差最大（±0.35），与 cross-model 分歧分析一致

### Multi-LLM 交叉评估

用不同 LLM 做 Critic 时，各维度分歧度（spread）揭示评估可靠性：

| 维度 | 平均分歧 | 解读 |
|------|---------|------|
| Coherence | 0.3 | 评估最稳定 — 两个模型高度一致 |
| Depth | 0.4 | 评估稳定 |
| Coverage | 0.6 | 中等分歧 |
| **Accuracy** | **1.4** | **评估最不稳定** — 已知局限，基于 abstract 的 accuracy 难以跨模型对齐 |

> 评估方法参考 AutoSurvey (arXiv:2406.10252) 的 Multi-LLM Judge；分歧度分析参考 LLMs-as-Judges survey (arXiv:2412.05579)。

### 多问题泛化验证

在 3 个不同领域的研究问题上验证自进化的普遍性：

| 问题领域 | Round 0 | Round 1 | Δ | 触发进化？ |
|---------|---------|---------|------|---------|
| NLP / Tool Use | 8.0 | — | — | 否（首轮达标） |
| ML Efficiency | 7.4 | 7.6 | **+0.2** | 是 ✓ |
| AI Safety | 8.0 | — | — | 否（首轮达标） |

> 对比 GPT Researcher 的 Review-Revise 机制：它修改的是*文本*（post-hoc editing），我们改进的是*检索策略*（strategy-level evolution）。

## KnowledgeBase 集成效果

KnowledgeBase 将 RAG 组件（chunking + embedding + indexing）接入 Pipeline，对每个子问题从知识库检索最相关论文：

| 模式 | Reader 精读量 | Overall 评分 | Coherence | 效率 |
|------|-------------|-------------|-----------|------|
| Without KB (全读) | 50 篇 | 8.05 | 8.2 | baseline |
| **With KB (过滤)** | **21 篇** | **8.0** | **8.5** | **-58% 精读量** |

**核心结论**：KB 过滤后 Reader 工作量减少 58%，综述质量持平（Δ-0.05），coherence 反而提升 0.3。

## 消融实验

在 HotpotQA dev set (50 samples) 上对比不同 RAG 配置（embedding: BGE-small-en-v1.5）：

| Chunk 策略 | 检索策略 | EM | F1 |
|-----------|---------|------|------|
| **fixed** | **hybrid** | **0.620** | **0.740** |
| semantic | hybrid | 0.580 | 0.669 |
| recursive | hybrid | 0.600 | 0.696 |
| recursive | dense | 0.580 | 0.707 |
| recursive | sparse | 0.540 | 0.636 |

**发现**：
1. **Hybrid > Dense > Sparse**（F1: 0.740 > 0.707 > 0.636）：RRF 融合了语义理解和关键词匹配，取长补短
2. **Embedding 质量决定 Dense 效果**：使用真实 embedding（BGE-small）后，Dense retrieval F1 从 0.518 → 0.707（+36%）
3. **Fixed chunk 在短文本场景最优**：HotpotQA 段落较短（~100 词），固定切分不会切断语义，overhead 最低

## Quick Start

```bash
# 安装
pip install -e ".[dev]"

# 配置 API Key
cp .env.example .env  # 编辑 .env 填入 OpenAI API Key

# 运行
python -m research "你的研究问题"

# 测试
pytest tests/ -v

# 跑自进化实验
python experiments/self_evolution.py

# 跑消融实验
python experiments/rag_ablation.py --num_samples 50
```

## 项目结构

```
src/research/
├── core/           # 基础设施：数据模型、配置、LLM 封装、Agent 基类
├── agents/         # 5 个 Agent：Planner, Retriever, Reader, Writer, Critic
├── retrieval/      # 检索工具：API 客户端、Chunking、Indexing、Reranker
├── pipeline/       # Pipeline 编排：自进化循环
└── evaluation/     # 评估框架

experiments/        # 可复现的实验脚本
docs/learning/      # Agent 知识学习文档（8 篇）
```

## 技术栈

Python 3.11+ | OpenAI/Anthropic SDK | FAISS + BM25 | Semantic Scholar API | arXiv API | Pydantic v2 | structlog | pytest

## License

MIT
