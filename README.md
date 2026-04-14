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

**Bootstrap 95% 置信区间**（post-hoc 分析，paired bootstrap, n_bootstrap=10000, seed=42）：

| 指标 | Mean | 95% CI | 显著性 |
|------|:----:|:------:|:------:|
| Evolution Δ | +0.30 | **[+0.10, +0.47]** | CI 不含 0 |
| Overall | 7.79 | [7.75, 7.87] | — |

关键发现：
- **自进化 Δ 全部为正**（3/3 runs），95% CI **[+0.10, +0.47]** 不含 0
- Overall 方差极小（±0.07），系统输出稳定
- Accuracy 方差最大（±0.35），与 cross-model 分歧分析一致

> _数据源：`experiments/results/self_evolution_repeated.json` + `experiments/results/bootstrap_ci_posthoc.json`。_
> _**统计诚实声明：** n=3 的 bootstrap CI 数学上受限于观测值 [min, max]，"CI 不含 0" 仅说明 3 次实验方向一致，不等价大样本显著性检验。严格显著性验证（n≥30 + paired t-test）是未来工作（见 [已知局限 #2](#已知局限)）。_

### Agent IO 观察：自进化真实性证据

分数只是表象，真正证明"自进化 work"要看 Planner 的 queries 是否真 diverge：

**单次 trace_demo 实证（trace_level=full）**（同一问题跑 2 轮，`experiments/results/trace_demo.json`）：

| | Round 0 | Round 1 | 新增 | 重复 |
|:-:|:-:|:-:|:-:|:-:|
| queries 数 | 6 | 38 | **32 新** | 0 |
| 涵盖方向 | 基础 RAG 架构 | + 多模态 / 代码 / 多语言 / 长上下文 / 安全 / MLOps / 个性化 / 经典论文溯源 | — | — |

**Round 1 的 38 个 queries 零重复**，且**完全响应了 Critic 提出的 25 条 missing_aspects**（如主动搜 "Lewis 2020 / REALM / RETRO / Atlas / FiD" 补经典论文、"KILT / NQ / HotpotQA" 补 benchmark 覆盖）。

查看方式：
```bash
python experiments/inspect_agent_io.py experiments/results/trace_demo.json --diff-queries
python experiments/inspect_agent_io.py experiments/results/trace_demo.json --agent Critic
```

> _自动化回归：`tests/test_pipeline_integration.py::test_self_evolution_actually_diverges` 在 `pytest -m integration` 时真跑 API 验证此行为。_

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

> ⚠️ **Opt-in 功能（默认关闭）**：PDF 全文带来 +0.7 overall / +0.9 depth 提升，但首次运行耗时 ~3x（下载 + 提取）。默认走 abstract-only 路径以保证快速体验。启用方式见下文 [Quick Start](#quick-start)。

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

> _聚合规则：表中数字为最后一轮 (Round 1) 的各维度分数，保留 1 位小数（Python `round()`）。逐轮明细见 `experiments/results/pdf_ablation.json`。_

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

> _数据源：`experiments/results/surge_benchmark.json`。_
> _对比边界：ground truth 规模不一致（我们 20 篇 vs SurGE 每篇综述 ~100 篇），4x 是方向性优势，详见 [已知局限 #3](#已知局限)。_

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
# 安装（核心，含 Pipeline + embedding grounding 引用验证）
pip install -e ".[dev]"

# 可选：启用 NLI 矛盾检测（额外装 sentence-transformers ~500MB）
# pip install -e ".[dev,citation-nli]"

# 配置 API Key
cp .env.example .env  # 编辑 .env 填入 OpenAI API Key

# 运行（默认 abstract-only，快速）
python -m research "你的研究问题"

# 运行启用 PDF 全文（对应 README "PDF 全文精读" 表格的数据）
python -c "
import asyncio
from research.core.config import PipelineConfig, PDFConfig
from research.pipeline.research import ResearchPipeline
cfg = PipelineConfig(pdf=PDFConfig(enabled=True))
asyncio.run(ResearchPipeline(cfg).run('你的研究问题'))
"

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

## 已知局限

> **项目定位：Research-grade prototype，非生产环境验证。** 以下是经过深入思考的已知短板，是项目的真实边界，不是待办列表。

1. **评估依赖 LLM-as-Judge 的固有偏差** — Critic Agent 本身基于 LLM 打分；虽然用 Multi-LLM 交叉（GPT-4o + Claude）+ 非 LLM 引用验证器部分缓解，但评估闭环未完全打破。**改进方向**：引入人类专家评审 sub-sample 做 calibration。

2. **自进化统计强度不足** — n=3 次重复得 Δ=+0.30±0.19，std/mean 比值接近 1，95% CI 可能包含 0。当前样本量只能说明"方向为正"，不能说明"显著为正"。**改进方向**：扩到 n≥10，加 paired bootstrap 置信区间。

3. **SurGE 对标的 ground truth 规模不可直接比** — 我们用 20 篇 curated RAG 核心论文，SurGE 用每篇综述 ~100 篇的完整引用列表。Coverage 25% vs 6.3% 是方向性优势，不是严格对等基准。**改进方向**：下载 SurGE 官方 corpus，在相同 ground truth 上重算。

4. **引用验证阈值敏感性未分析** — Embedding grounding 阈值 0.3、NLI entailment 阈值 0.5 基于经验设定，未做 sweep。100% grounding rate 可能对阈值敏感。**改进方向**：sweep 阈值 0.1-0.7 看指标变化。

5. **NLI 矛盾检测未人工 calibrate** — 4.8% contradiction rate 来自 DeBERTa-v3 模型输出，未抽样人工标注验证 precision/recall。不确定"4.8% 矛盾"里有多少是真矛盾 vs 模型误判。**改进方向**：人工标 30-50 条，对比 NLI 输出。

6. **仅支持英文学术文献** — BGE-small-en 和 SurGE 对标限定英文；Semantic Scholar / arXiv 偏 CS/AI 领域；中文文献、非学术文本、多模态内容均未验证。

7. **RAG 消融样本量小** — HotpotQA dev set 50 samples，F1 差异（如 hybrid 0.740 vs dense 0.707）未做显著性检验。**改进方向**：扩到 200+ samples + paired t-test。

面试可讲：这些局限的每一条我都知道**如何改进 + 为什么现阶段接受**。研究项目的严谨性不是"没有短板"，是"知道自己的短板在哪"。

## License

MIT
