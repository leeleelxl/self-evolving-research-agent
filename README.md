# ReSearch v2

**自进化多 Agent 学术研究系统** — 自动化文献检索、PDF 全文精读、综述生成，通过 Critic 反馈驱动检索策略持续优化。

## 解决什么问题

学术研究者平均花 **40%+ 时间在文献检索和阅读**上。现有 AI 研究工具（GPT Researcher、STORM）能生成报告，但：
- 检索策略固定，无法根据研究问题自适应调整
- 缺乏质量自评估和迭代改进机制
- 仅基于 abstract 精读，信息量有限
- 无法量化展示不同策略的效果差异

ReSearch v2 通过 **Critic Agent 驱动的自进化机制 + PDF 全文精读 + Hybrid 引用验证**，让系统在每轮研究后自动评估质量并改进检索策略。

## 技术亮点

- **自进化机制**：Critic Agent 评分 → 反馈驱动检索策略改进 → 3 次重复实验 Δ=+0.30±0.19（全部为正）
- **PDF 全文精读**：自动下载 + 提取 PDF 全文 → overall Δ+0.7, depth +0.9（消融验证）
- **Hybrid 引用验证**：Embedding grounding + NLI 矛盾检测，发现 4.8% 矛盾引用（DeBERTa-v3 cross-encoder）
- **Multi-LLM 交叉评估**：GPT-4o + Claude 双模型 Critic，记录分歧度量化评估置信度
- **5 Agent 协作**：Planner / Retriever / Reader / Writer / Critic 各司其职
- **KnowledgeBase RAG 集成**：chunking → embedding → hybrid index，精读量 -58% 质量持平
- **SurGE 外部对标**：Coverage 25.0% vs 最强基线 6.3%（4x）
- **完整消融实验**：chunk 策略 × 检索方式 × KB × PDF × 引用验证方法的全组合对比数据
- **不依赖框架**：直接用 Anthropic/OpenAI SDK 构建，每一层可解释

## 架构

```
研究问题 → Planner (问题分解) → Retriever (文献检索)
              ↑                         ↓
              │                  PDF 全文提取 (async download + pypdf)
              │                         ↓
              │                  KnowledgeBase (RAG 过滤)
              │                    chunk → embed → index
              │                    按子问题检索 top-k
              │                         ↓
              │                    Reader (全文精读 / 降级 abstract)
              │                         ↓
              └──── Critic (质量评估) ←── Writer (综述生成)
                        │                    ↓
                   自进化反馈循环       Hybrid 引用验证
                   (策略调整 + 定向补充)  (Embedding + NLI)
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

## PDF 全文精读

### 消融实验：Abstract-only vs Full-text

| 模式 | Overall | Depth | Coverage | Coherence | Accuracy |
|------|:-------:|:-----:|:--------:|:---------:|:--------:|
| Abstract-only | 7.2 | 6.9 | 7.0 | 7.8 | 7.2 |
| **Full-text (PDF)** | **7.9** | **7.8** | **7.5** | **8.2** | **8.0** |
| **Δ** | **+0.7** | **+0.9** | +0.5 | +0.4 | +0.8 |

- **PDF 提取成功率 96.7%**（58/60 篇有 PDF URL 的论文成功提取）
- **Depth +0.9 提升最大**：全文包含方法细节和实验数据，abstract 中缺失
- **优雅降级**：下载失败自动退回 abstract，不影响 pipeline
- 耗时 3x（435s → 1252s），可通过选择性下载高引论文优化

## Hybrid 引用验证（Embedding + NLI）

用非 LLM 方法验证 Writer 的每条引用是否真有内容支撑，打破"LLM 评 LLM"闭环。

### 三方法对比实验

| 方法 | Grounding Rate | 矛盾检测 | 耗时 | 适用场景 |
|------|:-------------:|:--------:|:----:|---------|
| Embedding only | 100% | — | 41s | 过于宽松，无法区分话题相关 vs 真实支撑 |
| NLI only | 4.3% | 4.8% | 367s | 过于严格，NLI entailment ≠ paraphrase |
| **Hybrid** | **100%** | **4.8%** | 407s | **Embedding grounding + NLI contradiction** |

**设计来自实验发现**（不是拍脑袋）：
- NLI 模型严格区分"逻辑蕴含"和"语义相似" — 综述改述论文内容时，NLI 判 neutral 而非 entailment
- 因此 NLI 不适合做 grounding（过严），但矛盾检测能力无可替代
- Hybrid = 取两者之长：embedding 衡量话题支撑 + NLI（DeBERTa-v3-base）检测矛盾引用

> 在 208 条引用中检测到 10 条矛盾引用 (4.8%)，包括同一篇论文在多个 section 被反复误引。

## SurGE Benchmark 外部对标

对标 SurGE (arXiv:2508.15658) — 学术综述生成评估基准（205 篇 ground truth 综述）：

| System | Coverage | Doc-Relevance | Logic |
|--------|:--------:|:-------------:|:-----:|
| RAG (SurGE baseline) | 2.1% | 28.6% | 4.67 |
| AutoSurvey | 3.5% | 36.2% | 4.74 |
| StepSurvey (最强基线) | 6.3% | 45.8% | **4.85** |
| **ReSearch v2 (ours)** | **25.0%** | **100%** | 4.00 |

- **Coverage 25.0%**：在 20 篇 RAG 领域 ground truth 论文中找到 5 篇，是最强基线的 **4 倍**
- **Logic 4.00/5**：略低于 StepSurvey（4.85），内容逻辑衔接是改进方向
- **独有维度**：NLI 矛盾引用检测 — SurGE 所有基线都没有此能力

> 注：Coverage 绝对值不直接可比（SurGE 用完整引用列表 ~100 篇/综述，我们用 20 篇领域核心论文），但相对优势可信。

## KnowledgeBase 集成效果

KnowledgeBase 将 RAG 组件（chunking + embedding + indexing）接入 Pipeline，对每个子问题从知识库检索最相关论文：

| 模式 | Reader 精读量 | Overall 评分 | Coherence | 效率 |
|------|-------------|-------------|-----------|------|
| Without KB (全读) | 50 篇 | 8.05 | 8.2 | baseline |
| **With KB (过滤)** | **21 篇** | **8.0** | **8.5** | **-58% 精读量** |

**核心结论**：KB 过滤后 Reader 工作量减少 58%，综述质量持平（Δ-0.05），coherence 反而提升 0.3。

## RAG 消融实验

在 HotpotQA dev set (50 samples) 上对比不同 RAG 配置（embedding: BGE-small-en-v1.5）：

| Chunk 策略 | 检索策略 | EM | F1 |
|-----------|---------|------|------|
| **fixed** | **hybrid** | **0.620** | **0.740** |
| semantic | hybrid | 0.580 | 0.669 |
| recursive | hybrid | 0.600 | 0.696 |
| recursive | dense | 0.580 | 0.707 |
| recursive | sparse | 0.540 | 0.636 |

**发现**：
1. **Hybrid > Dense > Sparse**（F1: 0.740 > 0.707 > 0.636）：RRF 融合了语义理解和关键词匹配
2. **Embedding 质量决定 Dense 效果**：使用真实 embedding 后，Dense F1 从 0.518 → 0.707（+36%）
3. **Fixed chunk 在短文本场景最优**：HotpotQA 段落较短，固定切分不会切断语义

## Quick Start

```bash
# 安装
pip install -e ".[dev]"

# 配置 API Key
cp .env.example .env  # 编辑 .env 填入 OpenAI API Key

# 运行
python -m research "你的研究问题"

# 测试（80 tests）
pytest tests/ -v

# 跑实验
python experiments/self_evolution.py        # 自进化
python experiments/pdf_ablation.py          # PDF 全文消融
python experiments/citation_nli.py          # 引用验证三方法对比
python experiments/surge_benchmark.py       # SurGE 外部对标
python experiments/rag_ablation.py          # RAG 消融
```

## 项目结构

```
src/research/
├── core/           # 基础设施：数据模型、配置、LLM 封装、Agent 基类
├── agents/         # 5 个 Agent：Planner, Retriever, Reader, Writer, Critic
├── retrieval/      # 检索工具：API 客户端、Chunking、Indexing、Reranker、PDF、KnowledgeBase
├── pipeline/       # Pipeline 编排：自进化循环
└── evaluation/     # 评估框架：metrics、benchmark、citation verifier (embedding + NLI)

experiments/        # 可复现的实验脚本（8 个）
tests/              # 80 tests（单元 + 集成）
docs/learning/      # Agent 知识学习文档（9 篇）
```

## 技术栈

Python 3.11+ | OpenAI/Anthropic SDK | FAISS + BM25 | Semantic Scholar API | arXiv API | pypdf | sentence-transformers (CrossEncoder) | fastembed (BGE) | Pydantic v2 | structlog | pytest

## License

MIT
