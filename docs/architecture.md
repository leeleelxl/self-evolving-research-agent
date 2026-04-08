# 架构设计

> 状态：v1.0 — 2026-04-08

## 1. 系统全景

```
                         ┌─────────────────────────────────────────────┐
                         │              ReSearch v2 Pipeline           │
                         │                                             │
  研究问题 ──▶ ┌─────────┴───┐    ┌───────────┐    ┌──────────┐       │
               │   Planner   │───▶│ Retriever │───▶│  Reader  │       │
               │ 问题分解     │    │ 文献检索   │    │ 论文精读  │       │
               └─────────────┘    └───────────┘    └────┬─────┘       │
                     ▲                                   │             │
                     │ 自进化                        结构化笔记         │
                     │ 反馈                              │             │
               ┌─────┴─────┐                       ┌────▼─────┐       │
               │   Critic  │◀──────────────────────│  Writer  │       │
               │ 质量评估   │      综述草稿          │ 综述生成  │       │
               └───────────┘                       └──────────┘       │
                     │                                                 │
                     ▼                                                 │
               达标？─── 是 ──▶ 最终综述报告                            │
                    └── 否 ──▶ 回到 Planner（策略改进）                 │
                         └─────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │                        基础设施层                                    │
  │  ┌──────────┐  ┌───────────────┐  ┌──────────┐  ┌───────────────┐  │
  │  │ Claude   │  │ FAISS + BM25  │  │ Semantic │  │ arXiv API     │  │
  │  │ API      │  │ 向量/文本索引   │  │ Scholar  │  │               │  │
  │  └──────────┘  └───────────────┘  └──────────┘  └───────────────┘  │
  └─────────────────────────────────────────────────────────────────────┘
```

## 2. 核心数据模型

所有模型用 Pydantic v2，确保数据流类型安全。

```python
# ── 论文与知识 ──

class Paper(BaseModel):
    """从学术 API 获取的原始论文信息"""
    paper_id: str                          # Semantic Scholar / arXiv ID
    title: str
    abstract: str
    authors: list[str]
    year: int
    url: str
    source: Literal["semantic_scholar", "arxiv"]
    citations: int = 0
    pdf_url: str | None = None

class Chunk(BaseModel):
    """论文的文本片段（检索的基本单位）"""
    chunk_id: str
    paper_id: str
    text: str
    chunk_index: int                       # 在原文中的顺序
    metadata: dict[str, Any] = {}          # section_title, page_num 等
    embedding: list[float] | None = None

class PaperNote(BaseModel):
    """Reader Agent 对单篇论文的结构化精读笔记"""
    paper_id: str
    title: str
    core_contribution: str                 # 一句话总结核心贡献
    methodology: str                       # 方法论
    key_findings: list[str]                # 关键发现
    limitations: list[str]                 # 局限性
    relevance_score: float                 # 与研究问题的相关度 (0-1)
    relevance_reason: str                  # 为什么相关

# ── 研究规划 ──

class SearchQuery(BaseModel):
    """单次检索请求"""
    query: str
    source: Literal["semantic_scholar", "arxiv"]
    max_results: int = 20
    year_range: tuple[int, int] | None = None

class SearchStrategy(BaseModel):
    """一轮检索的完整策略"""
    queries: list[SearchQuery]
    focus_areas: list[str]                 # 应重点关注的方向
    exclude_terms: list[str] = []          # 应排除的内容

class ResearchPlan(BaseModel):
    """Planner Agent 的输出 — 研究计划"""
    original_question: str
    sub_questions: list[str]               # 分解后的子问题
    search_strategy: SearchStrategy
    iteration: int = 0                     # 当前是第几轮迭代

# ── 输出 ──

class ResearchReport(BaseModel):
    """Writer Agent 的输出 — 研究综述"""
    title: str
    abstract: str
    sections: list[ReportSection]
    references: list[str]                  # 引用的论文 ID
    metadata: dict[str, Any] = {}

class ReportSection(BaseModel):
    section_title: str
    content: str
    cited_papers: list[str]                # 该节引用的论文 ID

# ── 评估与反馈 ──

class CriticFeedback(BaseModel):
    """Critic Agent 的输出 — 结构化评估反馈"""
    scores: CriticScores
    missing_aspects: list[str]             # 缺失的研究方向/论文
    improvement_suggestions: list[str]     # 具体改进建议
    new_queries: list[str]                 # 建议的新检索 query
    is_satisfactory: bool                  # 是否达标，Pipeline 据此决定是否继续迭代

class CriticScores(BaseModel):
    coverage: float      # 文献覆盖度 (0-10)
    depth: float          # 分析深度 (0-10)
    coherence: float      # 逻辑连贯性 (0-10)
    accuracy: float       # 事实准确性 (0-10)
    overall: float        # 综合评分 (0-10)
```

## 3. Agent 架构

### 3.1 Agent 基类

```python
class BaseAgent(ABC):
    """
    所有 Agent 的基类。
    
    设计理念：不强制 ReAct 循环，每个 Agent 的工作模式不同。
    - Planner/Writer/Critic 主要做单轮 LLM 调用（structured output）
    - Retriever 主要做工具调用（API 请求 + 索引操作）
    - Reader 做批量处理（每篇论文一次 LLM 调用）
    
    基类只提供：LLM 调用、日志、配置管理。
    """
    name: str
    role: str                              # 写入 system prompt
    model: str = "claude-sonnet-4-6-20250514"  # 默认用 Sonnet，Critic 可覆盖为 Opus
    
    @abstractmethod
    async def run(self, context: AgentContext) -> AgentResult:
        """每个 Agent 自定义的执行逻辑"""
        ...
    
    async def call_llm(
        self,
        messages: list[dict],
        response_model: type[BaseModel] | None = None,  # structured output
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """封装 Claude API 调用，支持 structured output 和 tool use"""
        ...
```

**为什么不用 ReAct 循环作为基类？**

ReAct（Reason + Act）适合通用 Agent，但本项目的 Agent 职责明确，每个 Agent 的执行模式不同。强制统一 `perceive → think → act` 循环会引入不必要的复杂度。基类只抽象公共部分（LLM 调用），让每个 Agent 自由定义执行逻辑。

> **面试话术**：ReAct 范式适合 open-ended 的通用 Agent（如 AutoGPT），但在职责明确的多 Agent 系统中，每个 Agent 应根据自身任务特点设计执行流程。这和微服务架构的理念一致——接口统一，实现各异。

### 3.2 各 Agent 设计

#### Planner Agent

```
输入: 研究问题 + (可选) Critic 反馈
输出: ResearchPlan

执行逻辑:
1. 首轮: 将研究问题分解为 3-5 个子问题，生成初始 SearchStrategy
2. 迭代轮: 分析 Critic 反馈，识别薄弱维度，调整 sub_questions 和 search_strategy
   - coverage 低 → 增加新的检索 query，扩大搜索范围
   - depth 低 → 增加 focus_areas，对关键论文追溯引用链
   - accuracy 低 → 增加交叉验证的 query
```

#### Retriever Agent

```
输入: ResearchPlan (含 SearchStrategy)
输出: list[Paper] + 更新后的知识库索引

执行逻辑:
1. 执行 SearchStrategy 中的每个 SearchQuery（Semantic Scholar / arXiv）
2. 去重（按 title 相似度）
3. 获取论文全文（PDF → 文本 → Chunk）
4. 对 Chunk 做 embedding，写入 FAISS 索引
5. 同时更新 BM25 索引

工具:
- semantic_scholar_search(query, filters) → list[Paper]
- arxiv_search(query, filters) → list[Paper]
- download_pdf(url) → str (文本内容)
- index_chunks(chunks) → None (写入 FAISS + BM25)
```

#### Reader Agent

```
输入: list[Paper] + 研究问题
输出: list[PaperNote]

执行逻辑:
1. 对每篇论文，从知识库检索相关 Chunk
2. 调用 LLM 做结构化信息提取 → PaperNote
3. 按 relevance_score 排序，过滤低相关度论文
4. 批量处理，用 asyncio.gather 并发

关键: 使用 structured output 确保每篇笔记格式统一
```

#### Writer Agent

```
输入: list[PaperNote] + ResearchPlan
输出: ResearchReport

执行逻辑:
1. 根据 sub_questions 规划综述结构（各节标题）
2. 对每个 section，从 PaperNote 中筛选相关笔记
3. 生成综述文本，确保引用准确
4. 生成 abstract
```

#### Critic Agent

```
输入: ResearchReport + 原始研究问题
输出: CriticFeedback

执行逻辑:
1. 评估 4 个维度，每个维度 0-10 分
2. 生成具体的缺失方面和改进建议
3. 建议新的检索 query（直接可执行）
4. 判断是否达标（overall >= 7.0）

关键设计:
- Critic 可用更强的模型（Opus）以确保评估质量
- 评分用 structured output 确保格式
- 每轮评分记录，用于分析自进化效果曲线
```

## 4. Pipeline 编排

```python
class ResearchPipeline:
    """
    多 Agent 研究 Pipeline — 核心编排逻辑
    
    编排模式: 顺序执行 + 条件循环（非 DAG，非并行）
    选择理由: Agent 之间有严格数据依赖，不适合并行
    """
    
    def __init__(self, config: PipelineConfig):
        self.planner = PlannerAgent(config)
        self.retriever = RetrieverAgent(config)
        self.reader = ReaderAgent(config)
        self.writer = WriterAgent(config)
        self.critic = CriticAgent(config)
        self.evolution_log: list[EvolutionRecord] = []
    
    async def run(
        self, 
        question: str, 
        max_iterations: int = 3,
    ) -> PipelineResult:
        """
        执行研究 Pipeline
        
        每轮迭代:
          Planner → Retriever → Reader → Writer → Critic
          如果 Critic 不满意，带着反馈回到 Planner
        """
        feedback = None
        
        for iteration in range(max_iterations):
            # 1. 规划（首轮: 问题分解; 迭代轮: 策略调整）
            plan = await self.planner.run(question, feedback, iteration)
            
            # 2. 检索
            papers = await self.retriever.run(plan)
            
            # 3. 精读
            notes = await self.reader.run(papers, question)
            
            # 4. 写作
            report = await self.writer.run(notes, plan)
            
            # 5. 评估
            feedback = await self.critic.run(report, question)
            
            # 记录本轮进化数据
            self.evolution_log.append(EvolutionRecord(
                iteration=iteration,
                scores=feedback.scores,
                strategy=plan.search_strategy,
                num_papers=len(papers),
                num_notes=len(notes),
            ))
            
            if feedback.is_satisfactory:
                break
        
        return PipelineResult(
            report=report,
            evolution_log=self.evolution_log,
            total_iterations=iteration + 1,
        )
```

## 5. 自进化机制（核心差异化）

### 5.1 进化循环

```
第 0 轮 (Baseline):
  Planner 基于研究问题生成初始策略
  → 检索 → 精读 → 写作
  → Critic 评分: coverage=5, depth=4, coherence=7, accuracy=6

第 1 轮 (Evolution):
  Critic 反馈: "缺少 2024 年后的最新进展，对 X 方法的分析过于表面"
  Planner 调整:
    - 增加 query: "X method 2024 2025 survey"
    - 增加 focus_area: "X 方法的变体和改进"
  → 检索 → 精读 → 写作
  → Critic 评分: coverage=7, depth=6, coherence=7, accuracy=7

第 2 轮 (Evolution):
  Critic 反馈: "缺少与 Y 方法的对比分析"
  Planner 调整:
    - 增加 query: "X vs Y comparison"
    - 增加 sub_question: "X 和 Y 的优劣对比"
  → 检索 → 精读 → 写作
  → Critic 评分: coverage=8, depth=8, coherence=8, accuracy=8 ✓ 达标
```

### 5.2 进化记录 (用于消融实验)

```python
class EvolutionRecord(BaseModel):
    """每轮迭代的快照，用于分析自进化效果"""
    iteration: int
    scores: CriticScores
    strategy: SearchStrategy            # 该轮使用的策略
    num_papers: int                     # 检索到的论文数
    num_notes: int                      # 精读的论文数
    strategy_changes: list[str] = []    # 相比上轮的策略变化描述
```

### 5.3 面试要点

> **自进化 vs 简单重试**：简单重试是用同样的策略再跑一遍，自进化是根据 Critic 的结构化反馈**定向改进**策略。类比：self-play in RL — 每轮对局后分析弱点，下轮针对性训练。
>
> **为什么有效**：研究问题通常有长尾知识需求，首轮检索难以覆盖。Critic 识别缺口后，Planner 能生成更精准的 query。这和人类研究者的工作流一致——先广泛检索，再根据阅读发现补充文献。

## 6. RAG Pipeline 详细设计

### 6.1 文档处理流程

```
PDF → pypdf 提取文本 → 清洗（去页眉页脚、合并断行）
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
           固定大小      语义切分      递归切分
          (baseline)   (by section)  (recursive)
                │           │           │
                └───────────┼───────────┘
                            ▼
                       Chunk 列表
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
             FAISS 索引           BM25 索引
           (dense vector)      (sparse token)
```

### 6.2 Chunk 策略（消融变量 A）

| 策略 | 实现 | 参数 | 适用场景 |
|------|------|------|---------|
| 固定大小 | 按 token 数切分，有 overlap | chunk_size=512, overlap=64 | Baseline |
| 语义切分 | 按 section/paragraph 边界切 | 利用标题和空行 | 结构化论文 |
| 递归切分 | 先按 section，超长再按 paragraph，再按 sentence | max_chunk=1024 | 通用 |

### 6.3 检索策略（消融变量 B）

| 策略 | 实现 | 说明 |
|------|------|------|
| Dense only | FAISS cosine similarity | 语义相似度，擅长同义改写 |
| Sparse only | BM25 | 关键词匹配，擅长精确术语 |
| Hybrid | RRF(dense, sparse) | 融合两者，Reciprocal Rank Fusion |

### 6.4 Rerank 策略（消融变量 C）

| 策略 | 实现 | 说明 |
|------|------|------|
| No rerank | 直接用检索分数 | Baseline |
| LLM rerank | Claude 对 top-k 候选重新排序 | 高质量但慢 |

### 6.5 消融实验矩阵

3 chunk × 3 检索 × 2 rerank = **18 种组合**

每种组合在 HotpotQA dev set (500 样本) 上跑分，记录：
- Exact Match (EM)
- F1 Score
- 检索延迟 (ms/query)
- Token 消耗 (cost)

## 7. 评估框架

### 7.1 Benchmark 评估

```python
class BenchmarkEvaluator:
    """在 HotpotQA / MuSiQue 上评估 Pipeline 效果"""
    
    async def evaluate(
        self,
        dataset: str,               # "hotpotqa" | "musique"
        split: str = "dev",
        num_samples: int = 500,
        pipeline_config: PipelineConfig,
    ) -> BenchmarkResult:
        ...

class BenchmarkResult(BaseModel):
    dataset: str
    num_samples: int
    exact_match: float
    f1_score: float
    avg_iterations: float            # 平均自进化轮数
    avg_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    config: PipelineConfig           # 可复现的配置
```

### 7.2 自进化效果评估

```
对比实验:
- Pipeline(max_iterations=1)  → 无自进化 baseline
- Pipeline(max_iterations=2)
- Pipeline(max_iterations=3)
- Pipeline(max_iterations=5)

期望结果: 分数随迭代轮数递增，但边际收益递减
```

## 8. 目录结构

```
src/research/
├── __init__.py
├── __main__.py              # CLI 入口
├── core/
│   ├── __init__.py
│   ├── config.py            # PipelineConfig, AgentConfig
│   ├── models.py            # 所有 Pydantic 数据模型
│   ├── llm.py               # Claude API 封装
│   └── agent.py             # BaseAgent 基类
├── agents/
│   ├── __init__.py
│   ├── planner.py           # PlannerAgent
│   ├── retriever.py         # RetrieverAgent
│   ├── reader.py            # ReaderAgent
│   ├── writer.py            # WriterAgent
│   └── critic.py            # CriticAgent
├── retrieval/
│   ├── __init__.py
│   ├── search.py            # SemanticScholarClient, ArxivClient
│   ├── chunking.py          # 三种 chunk 策略
│   ├── indexing.py          # FAISS + BM25 索引管理
│   └── reranker.py          # LLM Reranker
├── pipeline/
│   ├── __init__.py
│   └── research.py          # ResearchPipeline 编排逻辑
└── evaluation/
    ├── __init__.py
    ├── benchmark.py          # BenchmarkEvaluator
    └── metrics.py            # EM, F1 等指标计算
```

## 9. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| Agent 通信方式 | 函数调用（直接传参） | 简单可控，本项目不需要消息队列 |
| 编排模式 | 顺序 + 条件循环 | Agent 间有严格数据依赖，无法并行 |
| LLM 调用方式 | Anthropic SDK 原生 | 不引入 LangChain/LlamaIndex，减少依赖，面试时讲得清楚 |
| 向量检索 | FAISS (CPU) | 轻量、无需外部服务、面试常考 |
| 数据校验 | Pydantic v2 | 类型安全 + structured output |
| 日志 | structlog | 结构化日志，方便后续分析自进化过程 |
| Chunk embedding | Claude Embed (或 sentence-transformers) | 先用开源 embedding，后续可切换 |
