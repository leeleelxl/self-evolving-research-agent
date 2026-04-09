"""
引用质量验证 — 打破 "LLM 评 LLM" 闭环

核心思路:
  Writer 声称 section X 引用了论文 Y。但论文 Y 的 abstract 真的支持 section X 的内容吗？
  用 embedding 余弦相似度做客观验证，完全不依赖 LLM 判断。

指标:
  - per-citation score: 每条引用的 section-abstract 相似度
  - section grounding rate: 每个 section 中相似度 > 阈值的引用比例
  - overall grounding rate: 全报告的平均引用支撑率

参考: SurGE benchmark (arXiv:2508.15658) 的引用准确性评估思路,
      但用 embedding 替代 NLI 模型，更轻量。
"""

from __future__ import annotations

import numpy as np
import structlog

from research.core.models import Paper, PipelineResult, ReportSection
from research.retrieval.embedding import EmbeddingModel

logger = structlog.get_logger()


class CitationVerifier:
    """引用质量验证器

    Usage:
        verifier = CitationVerifier()
        report = verifier.verify(pipeline_result)
        print(f"Overall grounding rate: {report['overall_grounding_rate']:.1%}")
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        grounding_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            embedding_model: 复用已有的 embedding 模型，避免重复加载
            grounding_threshold: 相似度超过此值认为引用有支撑
        """
        self._embedding = embedding_model or EmbeddingModel()
        self._threshold = grounding_threshold

    def verify(self, result: PipelineResult) -> dict:
        """验证 Pipeline 输出的引用质量

        Returns:
            {
                "sections": [...],  # 每个 section 的详细引用验证
                "overall_grounding_rate": float,  # 总体引用支撑率
                "overall_avg_similarity": float,  # 总体平均相似度
                "num_citations_checked": int,
                "num_citations_grounded": int,
                "num_citations_ungrounded": int,
                "num_citations_missing": int,  # 引用的论文不在 papers 中
            }
        """
        report = result.report
        paper_map = {p.paper_id: p for p in result.papers}

        all_scores: list[float] = []
        num_grounded = 0
        num_missing = 0
        section_results = []

        for section in report.sections:
            section_detail = self._verify_section(section, paper_map)
            section_results.append(section_detail)

            for cite in section_detail["citations"]:
                if cite["status"] == "missing":
                    num_missing += 1
                else:
                    all_scores.append(cite["similarity"])
                    if cite["grounded"]:
                        num_grounded += 1

        num_checked = len(all_scores)
        overall_rate = num_grounded / num_checked if num_checked > 0 else 0.0
        avg_sim = float(np.mean(all_scores)) if all_scores else 0.0

        return {
            "sections": section_results,
            "overall_grounding_rate": round(overall_rate, 4),
            "overall_avg_similarity": round(avg_sim, 4),
            "num_citations_checked": num_checked,
            "num_citations_grounded": num_grounded,
            "num_citations_ungrounded": num_checked - num_grounded,
            "num_citations_missing": num_missing,
            "grounding_threshold": self._threshold,
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

        # Embed section 内容
        section_emb = self._embedding.embed_single(section.content[:2000])

        citations = []
        grounded_count = 0

        for paper_id in section.cited_papers:
            paper = paper_map.get(paper_id)
            if paper is None:
                citations.append({
                    "paper_id": paper_id,
                    "status": "missing",
                    "similarity": 0.0,
                    "grounded": False,
                })
                continue

            # Embed 论文 abstract
            paper_text = f"{paper.title}. {paper.abstract}"
            paper_emb = self._embedding.embed_single(paper_text[:2000])

            # 余弦相似度
            sim = self._cosine_similarity(section_emb, paper_emb)
            grounded = sim >= self._threshold

            if grounded:
                grounded_count += 1

            citations.append({
                "paper_id": paper_id,
                "title": paper.title[:80],
                "status": "checked",
                "similarity": round(sim, 4),
                "grounded": grounded,
            })

        valid_count = sum(1 for c in citations if c["status"] == "checked")
        rate = grounded_count / valid_count if valid_count > 0 else 0.0

        return {
            "section_title": section.section_title,
            "grounding_rate": round(rate, 4),
            "num_cited": len(section.cited_papers),
            "num_grounded": grounded_count,
            "citations": citations,
        }

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
