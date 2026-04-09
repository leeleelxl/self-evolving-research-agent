# 08 — RAG Pipeline 集成模式

> 对应代码: `retrieval/knowledge_base.py`, `pipeline/research.py`

## 问题：RAG 组件和 Pipeline 脱节

一个常见的新手错误：RAG 的各个组件（chunking、embedding、indexing、reranking）写好了，
测试也过了，但没有接入实际的 Pipeline。这会导致：

1. **面试暴露**：面试官问"你的 RAG 在 Pipeline 里怎么用的？"答不上来
2. **浪费代码**：几百行代码变成了摆设
3. **效果打折**：Pipeline 没有利用 RAG 的能力，检索质量停留在粗糙水平

```
之前的 Pipeline:
  Retriever(API 搜索) → 50 papers → Reader 全部精读 → Writer → Critic
                                     ↑ 暴力全读，效率低，相关度差

RAG 组件:
  chunking.py / indexing.py / embedding.py / reranker.py → 在 tests/ 里测过，但 Pipeline 不用
```

## 解决方案：KnowledgeBase 抽象

KnowledgeBase 是一个**组合层**，把零散的 RAG 组件串联成一个可用的整体：

```
改进后的 Pipeline:
  Retriever → 50 papers
            → KnowledgeBase.add_papers() → chunk abstracts → embed → 建索引
            → 对每个 sub-question → KnowledgeBase.retrieve(sub_q, top_k=10)
            → Reader 只精读相关的论文 → Writer → Critic
```

### 为什么不直接在 Pipeline 里调 chunking/indexing？

**封装的好处：**
- Pipeline 只和 KnowledgeBase 交互，不关心底层用 Dense/Sparse/Hybrid 哪种索引
- 切换索引策略只需改 config，不改 Pipeline 代码
- 消融实验可以直接用 `knowledge_base.enabled=False` 做对比

**这就是"组合模式" (Composition)**：KnowledgeBase 不继承任何 RAG 组件，
而是持有它们的实例，按需编排。这是比继承更灵活的代码复用方式。

## KnowledgeBase 的设计决策

### 1. 以 Abstract 为文本源

当前阶段我们只有论文的 abstract（没有 PDF 全文）。为什么还要过 chunking？

```python
# 虽然 abstract 通常只有 1-2 个 chunk，但：
text = f"{paper.title}\n\n{paper.abstract}"  # title + abstract
chunks = self._chunker.chunk(text, paper.paper_id)
```

**原因：**
- 架构一致性：后续接 PDF 全文时，同一套 KnowledgeBase 直接复用
- 面试叙事："我们的 KnowledgeBase 对 abstract 和全文使用同一套处理流水线"
- chunking 策略的选择（fixed/semantic/recursive）在消融实验中已验证

### 2. 跨迭代累积

```python
# Pipeline 的多轮自进化中，KnowledgeBase 持续扩展
class ResearchPipeline:
    def __init__(self):
        self._kb: KnowledgeBase | None = None  # 延迟初始化，跨轮共享

    def _filter_by_knowledge_base(self, papers, plan):
        if self._kb is None:
            self._kb = KnowledgeBase(self.config)
        self._kb.add_papers(papers)  # 增量索引，已有的自动跳过
        return self._kb.retrieve_for_questions(plan.sub_questions)
```

**为什么跨轮累积而非每轮重建？**
- Round 0 搜到 30 篇，Round 1 搜到 20 篇新的 → 知识库有 50 篇
- Round 1 的子问题可以从所有 50 篇中检索，覆盖度更高
- 重建则丢失 Round 0 的论文，违背"自进化 = 知识累积"的设计

### 3. 延迟初始化 (Lazy Init)

```python
self._kb: KnowledgeBase | None = None  # 不在 __init__ 中创建
```

**为什么？**
- EmbeddingModel 加载要 1-2 秒（下载/加载 ONNX 模型）
- 如果 KB disabled（消融实验），完全不需要加载
- Pipeline 的 mock 测试也不需要加载真实 embedding

### 4. Chunk → Paper 反查

```python
# 索引的粒度是 chunk，但 Pipeline 需要的粒度是 paper
# KnowledgeBase 维护 paper_id → Paper 的映射
self._paper_map: dict[str, Paper] = {}

def retrieve(self, query, top_k):
    results = self._index.search(...)  # 返回 chunks
    # 从 chunk.paper_id 反查 paper，去重
    for chunk, score in results:
        paper = self._paper_map[chunk.paper_id]
```

这是 RAG 系统中常见的"粒度转换"问题：
- 检索粒度 = chunk（精确匹配）
- 业务粒度 = paper（精读单位）
- 需要一个映射层来连接两者

## Hybrid 检索在 Pipeline 中的实际效果

在消融实验中我们已经验证：`hybrid > dense > sparse`（F1: 0.740 > 0.718 > 0.702）。

KnowledgeBase 中 hybrid 检索的实际效果：
- Dense（语义）擅长：query 用同义词时也能匹配（"information retrieval" ↔ "passage ranking"）
- Sparse（BM25）擅长：精确术语匹配（"BM25"、"FAISS"、"Transformer"）
- Hybrid（RRF 融合）：两者互补，论文检索场景尤其适合

```python
# RRF 融合公式
score(paper) = Σ 1/(k + rank_dense) + 1/(k + rank_sparse)
# k=60 是经验参数，平衡 head 和 tail 的权重
```

## 消融开关

```python
config = PipelineConfig(
    knowledge_base=KnowledgeBaseConfig(enabled=False),  # 关闭 KB
)
# 此时 Pipeline 退化为原来的"全读"模式 → 用于消融对比
```

这也是一个好的工程实践：**新功能都要有开关**，方便 A/B 测试和回退。

## 面试要点

1. **"你的 RAG 在 Pipeline 里怎么用的？"**
   - KnowledgeBase 封装了 chunking → embedding → indexing 的完整流水线
   - Pipeline 在 Retriever 和 Reader 之间插入 KB 过滤
   - 对每个子问题检索 top-k 最相关论文，Reader 只精读这些
   - 效率提升 + 质量提升（精读内容和子问题匹配度更高）

2. **"为什么不直接用向量相似度过滤？"**
   - 向量相似度只能捕捉语义，精确术语可能漏掉
   - Hybrid（RRF）融合 dense + sparse，覆盖两种匹配模式
   - 消融实验证明 hybrid 效果最好

3. **"KnowledgeBase 和 Vector DB 的区别？"**
   - 我们的 KnowledgeBase 是 in-memory 的，适合几百篇论文的规模
   - 生产环境用 Pinecone/Weaviate 等 Vector DB，支持持久化和扩展
   - 但核心逻辑（chunk + embed + index + retrieve）是一样的

## 学习资源

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) — 向量检索的工业标准
- [BM25: The Next Generation of Lucene Relevance](https://opensourceconnections.com/blog/2015/10/16/bm25-the-next-generation-of-lucene-relevation/) — BM25 原理
- [Reciprocal Rank Fusion (Cormack+ 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 原论文
- [RAG from Scratch (LangChain)](https://github.com/langchain-ai/rag-from-scratch) — RAG 从零实现教程
