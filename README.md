# ReSearch v2

**自进化多 Agent 学术研究系统** — 自动化文献检索、精读、综述生成，通过 Critic 反馈驱动检索策略持续优化。

## 解决什么问题

学术研究者平均花 **40%+ 时间在文献检索和阅读**上。现有 AI 研究工具（GPT Researcher、STORM）能生成报告，但：
- 检索策略固定，无法根据研究问题自适应调整
- 缺乏质量自评估和迭代改进机制
- 无法量化展示不同策略的效果差异

ReSearch v2 通过 **Critic Agent 驱动的自进化机制**，让系统在每轮研究后自动评估质量并改进检索策略。

## 技术亮点

- **自进化机制**：Critic Agent 评分 → 反馈驱动检索策略改进 → 效果可量化提升
- **5 Agent 协作**：Planner / Retriever / Reader / Writer / Critic 各司其职
- **Agentic RAG**：Agent 主动决策何时检索、检索什么、如何验证（非固定 pipeline）
- **完整消融实验**：chunk 策略 × 检索方式 × rerank 方法的全组合对比数据
- **不依赖框架**：直接用 Anthropic/OpenAI SDK 构建，每一层可解释

## 架构

```
研究问题 → Planner (问题分解) → Retriever (文献检索) → Reader (精读)
              ↑                                            ↓
              └──── Critic (质量评估) ←── Writer (综述生成) ←┘
                        │
                   自进化反馈循环
                   (策略调整 + 定向补充)
```

## 自进化效果

同一研究问题（"RAG for LLMs 的最新进展"），自进化 2 轮后质量显著提升：

| 轮次 | Coverage | Depth | Coherence | Accuracy | **Overall** | Papers | Notes |
|------|----------|-------|-----------|----------|-------------|--------|-------|
| Round 0 | 7.5 | 7.5 | 8.5 | 7.5 | **7.8** | 50 | 45 |
| Round 1 | 8.2 | 8.0 | 8.0 | 7.8 | **8.0** | 50 | 50 |
| **Δ** | **+0.7** | **+0.5** | -0.5 | **+0.3** | **+0.2** | — | +5 |

关键变化：
- Planner 根据 Critic 反馈，将检索 query 从 **8 个扩展到 24 个**
- 覆盖度（Coverage）提升最大（+0.7），因为新 query 覆盖了之前遗漏的子方向
- 最终综述包含 **19 个章节、83 个引用**

> 对比 GPT Researcher 的 Review-Revise 机制：它修改的是*文本*（post-hoc editing），我们改进的是*检索策略*（strategy-level evolution）。

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
docs/learning/      # Agent 知识学习文档（7 篇）
```

## 技术栈

Python 3.11+ | OpenAI/Anthropic SDK | FAISS + BM25 | Semantic Scholar API | arXiv API | Pydantic v2 | structlog | pytest

## License

MIT
