"""
Embedding 模块 — 文本向量化

封装 fastembed，提供统一的 embedding 接口。
上层代码不感知底层使用哪个 embedding 模型。

选择 fastembed 而非 sentence-transformers 的原因:
- 不依赖 PyTorch（fastembed 用 ONNX，轻量）
- 安装快（~50MB vs PyTorch 的 2GB+）
- 默认模型 BAAI/bge-small-en-v1.5 效果好（MTEB 排名靠前）

面试要点:
- embedding 维度 384，适合 FAISS 索引
- 支持批量处理，比逐条调用快
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()


class EmbeddingModel:
    """文本 Embedding 模型

    Usage:
        model = EmbeddingModel()
        vectors = model.embed(["hello world", "another text"])
        # vectors: list[list[float]], 每个 384 维
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        logger.info("loading_embedding_model", model=model_name)
        self._model = TextEmbedding(model_name=model_name)
        self._model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding

        Args:
            texts: 文本列表

        Returns:
            embedding 列表，每个是 float list (384 维)
        """
        if not texts:
            return []

        # fastembed 返回 generator，转 list
        embeddings = list(self._model.embed(texts))
        return [emb.tolist() for emb in embeddings]

    def embed_single(self, text: str) -> list[float]:
        """单条文本 embedding"""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """embedding 维度"""
        return 384  # BGE-small 的维度

    @property
    def model_name(self) -> str:
        return self._model_name
