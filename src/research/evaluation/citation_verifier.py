"""
引用质量验证 — 打破 "LLM 评 LLM" 闭环

三种验证方法:
1. embedding: 余弦相似度 — 衡量话题相关性（轻量，快速）
2. nli: 句子级 NLI — 矛盾引用检测（NLI 独特能力）
3. hybrid: embedding 做 grounding + NLI 做 contradiction 检测（推荐）

实验发现（驱动 hybrid 设计的关键 insight）:
  - NLI 模型严格区分 "entailment"（逻辑蕴含）和 "semantic similarity"（语义相似）
  - 综述 section 改述论文内容时，NLI 判 neutral 而非 entailment
  - 因此: NLI 不适合做 grounding（过于严格），但矛盾检测能力无可替代
  - Hybrid = embedding grounding + NLI contradiction，取两者之长

参考: SurGE benchmark (arXiv:2508.15658)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import structlog

from research.core.models import Paper, PipelineResult, ReportSection
from research.retrieval.embedding import EmbeddingModel

logger = structlog.get_logger()

# NLI 模型的 label 映射（cross-encoder/nli-deberta-v3 系列）
_NLI_LABELS = ["contradiction", "entailment", "neutral"]

# Section 文本截断长度（NLI cross-encoder 输入限制 ~512 tokens）
_NLI_MAX_CHARS = 1500
_EMB_MAX_CHARS = 2000


class CitationVerifier:
    """引用质量验证器

    Usage:
        # Embedding 模式（默认，向后兼容）
        verifier = CitationVerifier(method="embedding")

        # NLI 模式（更精准）
        verifier = CitationVerifier(method="nli")

        report = verifier.verify(pipeline_result)
        print(f"Grounding rate: {report['overall_grounding_rate']:.1%}")
    """

    def __init__(
        self,
        method: Literal["embedding", "nli", "hybrid"] = "embedding",
        embedding_model: EmbeddingModel | None = None,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
        grounding_threshold: float = 0.3,
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
    ) -> None:
        """
        Args:
            method: 验证方法
              - "embedding": 余弦相似度（快速，grounding）
              - "nli": 句子级 NLI（矛盾检测）
              - "hybrid": embedding grounding + NLI contradiction（推荐）
            embedding_model: 复用已有的 embedding 模型
            nli_model_name: NLI cross-encoder 模型名称
            grounding_threshold: embedding 相似度阈值
            entailment_threshold: NLI entailment 概率阈值
            contradiction_threshold: NLI contradiction 概率阈值
        """
        self._method = method
        self._grounding_threshold = grounding_threshold
        self._entailment_threshold = entailment_threshold
        self._contradiction_threshold = contradiction_threshold

        if method in ("embedding", "hybrid"):
            self._embedding = embedding_model or EmbeddingModel()
        if method in ("nli", "hybrid"):
            self._nli_model = self._load_nli_model(nli_model_name)
        if method not in ("embedding", "nli", "hybrid"):
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _load_nli_model(model_name: str):  # noqa: ANN205
        """延迟导入 sentence-transformers，避免未安装时影响其他功能"""
        from sentence_transformers import CrossEncoder

        logger.info("loading_nli_model", model=model_name)
        return CrossEncoder(model_name)

    def verify(self, result: PipelineResult) -> dict:
        """验证 Pipeline 输出的引用质量

        Returns:
            {
                "method": str,
                "sections": [...],
                "overall_grounding_rate": float,
                "overall_avg_score": float,
                "overall_contradiction_rate": float,  # NLI 模式专有
                "num_citations_checked": int,
                "num_citations_grounded": int,
                "num_citations_contradicted": int,  # NLI 模式专有
                "num_citations_missing": int,
            }
        """
        report = result.report
        paper_map = {p.paper_id: p for p in result.papers}

        all_scores: list[float] = []
        num_grounded = 0
        num_contradicted = 0
        num_missing = 0
        section_results = []

        for section in report.sections:
            section_detail = self._verify_section(section, paper_map)
            section_results.append(section_detail)

            for cite in section_detail["citations"]:
                if cite["status"] == "missing":
                    num_missing += 1
                else:
                    all_scores.append(cite["score"])
                    if cite["grounded"]:
                        num_grounded += 1
                    if cite.get("contradicted", False):
                        num_contradicted += 1

        num_checked = len(all_scores)
        overall_rate = num_grounded / num_checked if num_checked > 0 else 0.0
        avg_score = float(np.mean(all_scores)) if all_scores else 0.0
        contradiction_rate = num_contradicted / num_checked if num_checked > 0 else 0.0

        return {
            "method": self._method,
            "sections": section_results,
            "overall_grounding_rate": round(overall_rate, 4),
            "overall_avg_score": round(avg_score, 4),
            "overall_contradiction_rate": round(contradiction_rate, 4),
            "num_citations_checked": num_checked,
            "num_citations_grounded": num_grounded,
            "num_citations_ungrounded": num_checked - num_grounded,
            "num_citations_contradicted": num_contradicted,
            "num_citations_missing": num_missing,
            "threshold": (
                self._grounding_threshold
                if self._method == "embedding"
                else self._entailment_threshold
            ),
        }

    def _verify_section(
        self,
        section: ReportSection,
        paper_map: dict[str, Paper],
    ) -> dict:
        """验证单个 section 的引用"""
        if not section.cited_papers:
            return {
                "section_title": section.section_title,
                "grounding_rate": 0.0,
                "citations": [],
            }

        citations = []
        grounded_count = 0

        for paper_id in section.cited_papers:
            paper = paper_map.get(paper_id)
            if paper is None:
                citations.append({
                    "paper_id": paper_id,
                    "status": "missing",
                    "score": 0.0,
                    "grounded": False,
                })
                continue

            if self._method == "embedding":
                cite_result = self._verify_one_embedding(section, paper)
            elif self._method == "nli":
                cite_result = self._verify_one_nli(section, paper)
            else:  # hybrid
                cite_result = self._verify_one_hybrid(section, paper)

            if cite_result["grounded"]:
                grounded_count += 1
            citations.append(cite_result)

        valid_count = sum(1 for c in citations if c["status"] == "checked")
        rate = grounded_count / valid_count if valid_count > 0 else 0.0

        return {
            "section_title": section.section_title,
            "grounding_rate": round(rate, 4),
            "num_cited": len(section.cited_papers),
            "num_grounded": grounded_count,
            "citations": citations,
        }

    # ── Embedding 模式 ──

    def _verify_one_embedding(self, section: ReportSection, paper: Paper) -> dict:
        """Embedding 余弦相似度验证单条引用"""
        section_emb = self._embedding.embed_single(section.content[:_EMB_MAX_CHARS])
        paper_text = f"{paper.title}. {paper.abstract}"
        paper_emb = self._embedding.embed_single(paper_text[:_EMB_MAX_CHARS])

        sim = self._cosine_similarity(section_emb, paper_emb)
        grounded = sim >= self._grounding_threshold

        return {
            "paper_id": paper.paper_id,
            "title": paper.title[:80],
            "status": "checked",
            "score": round(sim, 4),
            "grounded": grounded,
        }

    # ── NLI 模式 ──

    def _verify_one_nli(self, section: ReportSection, paper: Paper) -> dict:
        """句子级 NLI 验证单条引用

        NLI 模型训练在单句对上，直接传整段 section vs abstract 会全部判 neutral。
        解决: 把 section 拆成句子，逐句和 abstract 做 NLI，取最高 entailment 分。

        premise: 论文 abstract（证据）
        hypothesis: section 中的每个句子（声称）

        聚合策略:
        - entailment: max（至少有一句被支持即可）
        - contradiction: max（任何一句被矛盾即报警）
        """
        premise = f"{paper.title}. {paper.abstract}"[:_NLI_MAX_CHARS]
        sentences = self._split_sentences(section.content)

        if not sentences:
            return {
                "paper_id": paper.paper_id,
                "title": paper.title[:80],
                "status": "checked",
                "score": 0.0,
                "grounded": False,
                "contradicted": False,
                "nli_probs": {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0},
            }

        # 批量推理：所有 (premise, sentence) 对
        pairs = [(premise, sent[:_NLI_MAX_CHARS]) for sent in sentences]
        all_scores = self._nli_model.predict(pairs, apply_softmax=True)

        # 聚合: entailment 取 max, contradiction 取 max
        max_entailment = 0.0
        max_contradiction = 0.0
        avg_entailment = 0.0

        for score in all_scores:
            prob_c, prob_e, prob_n = float(score[0]), float(score[1]), float(score[2])
            max_entailment = max(max_entailment, prob_e)
            max_contradiction = max(max_contradiction, prob_c)
            avg_entailment += prob_e

        avg_entailment /= len(all_scores)

        grounded = max_entailment >= self._entailment_threshold
        contradicted = max_contradiction >= self._contradiction_threshold

        return {
            "paper_id": paper.paper_id,
            "title": paper.title[:80],
            "status": "checked",
            "score": round(max_entailment, 4),
            "grounded": grounded,
            "contradicted": contradicted,
            "nli_probs": {
                "entailment": round(max_entailment, 4),
                "entailment_avg": round(avg_entailment, 4),
                "contradiction": round(max_contradiction, 4),
                "neutral": round(1 - max_entailment - max_contradiction, 4),
            },
            "num_sentences": len(sentences),
        }

    # ── Hybrid 模式 ──

    def _verify_one_hybrid(self, section: ReportSection, paper: Paper) -> dict:
        """Hybrid 验证: embedding grounding + NLI contradiction

        Embedding 擅长: 话题相关性（grounding）
        NLI 擅长: 矛盾检测（contradiction）
        Hybrid = 两者互补
        """
        # 1. Embedding grounding
        emb_result = self._verify_one_embedding(section, paper)

        # 2. NLI contradiction scan
        nli_result = self._verify_one_nli(section, paper)

        return {
            "paper_id": paper.paper_id,
            "title": paper.title[:80],
            "status": "checked",
            # Grounding 由 embedding 决定
            "score": emb_result["score"],
            "grounded": emb_result["grounded"],
            # Contradiction 由 NLI 决定
            "contradicted": nli_result.get("contradicted", False),
            "nli_probs": nli_result.get("nli_probs", {}),
            "embedding_similarity": emb_result["score"],
            "num_sentences": nli_result.get("num_sentences", 0),
        }

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """简单的句子分割（按句号、问号、感叹号分割）

        不引入 nltk/spacy 依赖，对学术文本足够好。
        过滤掉过短的片段（< 20 字符，通常是编号或标题碎片）。
        """
        import re

        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if len(s.strip()) >= 20]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """余弦相似度"""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))
