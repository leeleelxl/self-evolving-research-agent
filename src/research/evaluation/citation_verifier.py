"""
引用质量验证 — 打破部分 "LLM 评 LLM" 闭环

四种验证方法:
1. embedding: 余弦相似度 — 衡量话题相关性（轻量，不依赖 LLM）
2. nli: 句子级 NLI — grounding（DEPRECATED：实验证明过严）
3. attribution: LLM-as-judge 检测 paper_id 错配（主要错误类型）
4. hybrid: embedding grounding + attribution 错配检测（推荐）

## 架构演化历程（真实项目故事）

v1: 纯 embedding → grounding rate 100%，过于宽松
v2: embedding + NLI 矛盾检测 → README 宣传 "4.8% 矛盾检测"
v3 (P4 calibration): 用 gpt-4o 独立 rater 验证，发现 NLI 矛盾检测
    precision=0%, recall=0%。失败模式：NLI 把"section 超出 abstract 范围"
    误判为 contradiction（实为 neutral/support）
v4 (current): pivot 到 paper_id 错配检测（LLM-judge）
    真实错误类型分布:
    - 40% paper_id 错配（section 把别论文特征归给这篇）← 新增 attribution 抓这个
    - 40% scope mismatch （section 过度归纳）
    - 10% 真逻辑矛盾 ← NLI 本该擅长但 precision=0%
    - 10% 幻觉引用（paper 不存在）← embedding 已抓 (missing)

## 诚实说明

attribution 用 LLM-judge 违反了"完全非 LLM 验证"的初衷。但:
- embedding grounding 仍是非 LLM 基线
- LLM-judge 用独立 gpt-4o（和主 Pipeline 的 gpt-4o-mini Critic 有 scale 差异）
- Calibration 仍可用独立 rater 验证

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
        method: Literal["embedding", "nli", "attribution", "hybrid"] = "embedding",
        embedding_model: EmbeddingModel | None = None,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
        grounding_threshold: float = 0.3,
        entailment_threshold: float = 0.5,
        attribution_judge_model: str = "gpt-4o",
    ) -> None:
        """
        Args:
            method: 验证方法
              - "embedding": 余弦相似度（快速，grounding）
              - "nli": 句子级 NLI grounding（DEPRECATED：calibration 证明 precision=0%）
              - "attribution": LLM-judge 检测 paper_id 错配（新增，抓 40% 真实错误）
              - "hybrid": embedding grounding + attribution 错配检测（推荐）
            embedding_model: 复用已有的 embedding 模型
            nli_model_name: NLI cross-encoder 模型（仅 nli 模式用）
            grounding_threshold: embedding 相似度阈值
            entailment_threshold: NLI entailment 阈值（仅 nli 模式用）
            attribution_judge_model: LLM judge 模型名称（默认 gpt-4o，
                建议用和主 Pipeline Critic 不同的模型避免相关性偏差）
        """
        import warnings

        self._method = method
        self._grounding_threshold = grounding_threshold
        self._entailment_threshold = entailment_threshold
        # NLI contradiction threshold 写死 0.5，P4 已证明该路径 precision=0%
        self._attribution_judge_model = attribution_judge_model

        if method not in ("embedding", "nli", "attribution", "hybrid"):
            raise ValueError(f"Unknown method: {method}")

        if method == "nli":
            warnings.warn(
                "method='nli' is DEPRECATED: P4 calibration showed "
                "precision=0% for contradiction detection. Use 'attribution' "
                "or 'hybrid' for better coverage of actual citation errors.",
                DeprecationWarning,
                stacklevel=2,
            )

        if method in ("embedding", "hybrid"):
            self._embedding = embedding_model or EmbeddingModel()
        if method == "nli":
            self._nli_model = self._load_nli_model(nli_model_name)
        if method in ("attribution", "hybrid"):
            self._judge_client = self._load_judge_client(attribution_judge_model)

    @staticmethod
    def _load_nli_model(model_name: str):  # noqa: ANN205
        """延迟导入 sentence-transformers，避免未安装时影响其他功能"""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "NLI citation verification requires sentence-transformers. "
                "Install with: pip install -e '.[citation-nli]'"
            ) from e

        logger.info("loading_nli_model", model=model_name)
        return CrossEncoder(model_name)

    @staticmethod
    def _load_judge_client(model: str):  # noqa: ANN205
        """创建 LLM-judge 客户端（默认走中转站 OpenAI API）"""
        from research.core.llm import create_llm_client

        logger.info("loading_judge_client", model=model)
        return create_llm_client(provider="openai", model=model)

    def verify(self, result: PipelineResult) -> dict:
        """验证 Pipeline 输出的引用质量（sync — attribution/hybrid 内部 asyncio.run）

        Returns:
            {
                "method": str,
                "sections": [...],
                "overall_grounding_rate": float,
                "overall_avg_score": float,
                "overall_mismatch_rate": float,  # attribution/hybrid 模式专有
                "num_citations_checked": int,
                "num_citations_grounded": int,
                "num_citations_mismatched": int,  # attribution/hybrid 模式专有
                "num_citations_missing": int,
            }
        """
        report = result.report
        paper_map = {p.paper_id: p for p in result.papers}

        all_scores: list[float] = []
        num_grounded = 0
        num_mismatched = 0
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
                    if cite.get("mismatched", False):
                        num_mismatched += 1

        num_checked = len(all_scores)
        overall_rate = num_grounded / num_checked if num_checked > 0 else 0.0
        avg_score = float(np.mean(all_scores)) if all_scores else 0.0
        mismatch_rate = num_mismatched / num_checked if num_checked > 0 else 0.0

        return {
            "method": self._method,
            "sections": section_results,
            "overall_grounding_rate": round(overall_rate, 4),
            "overall_avg_score": round(avg_score, 4),
            "overall_mismatch_rate": round(mismatch_rate, 4),
            "num_citations_checked": num_checked,
            "num_citations_grounded": num_grounded,
            "num_citations_ungrounded": num_checked - num_grounded,
            "num_citations_mismatched": num_mismatched,
            "num_citations_missing": num_missing,
            "threshold": (
                self._grounding_threshold
                if self._method in ("embedding", "hybrid", "attribution")
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
            elif self._method == "attribution":
                cite_result = self._verify_one_attribution(section, paper)
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
                "mismatched": False,
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
        # 注：NLI "contradicted" P4 已否证 precision=0%，此字段保留只为向后兼容
        # 语义统一到 mismatched = "有问题的引用"（NLI 版=逻辑矛盾，attribution 版=错配）
        mismatched = max_contradiction >= 0.5

        return {
            "paper_id": paper.paper_id,
            "title": paper.title[:80],
            "status": "checked",
            "score": round(max_entailment, 4),
            "grounded": grounded,
            "mismatched": mismatched,
            "nli_probs": {
                "entailment": round(max_entailment, 4),
                "entailment_avg": round(avg_entailment, 4),
                "contradiction": round(max_contradiction, 4),
                "neutral": round(1 - max_entailment - max_contradiction, 4),
            },
            "num_sentences": len(sentences),
        }

    # ── Attribution 模式（新，pivot 自 NLI）──

    def _verify_one_attribution(self, section: ReportSection, paper: Paper) -> dict:
        """Attribution 验证: LLM-judge 检测 paper_id 错配

        针对的错误类型（~40% 真实错误大头）:
        - Section 把别论文的技术特征归给这篇论文（e.g. 把 FAIR-RAG 特性归给讲 Indonesian QA 的论文）

        NLI 抓不到这种跨论文错配（只看一对文本内部关系），
        LLM-judge 能判断 "section 描述的内容和 abstract 声明的能力是否吻合"。
        """
        import asyncio

        from pydantic import BaseModel, Field

        class AttributionJudgment(BaseModel):
            label: str = Field(
                description=(
                    "'matching': section claims align with abstract; "
                    "'mismatched': section attributes tech/findings NOT in this abstract (wrong paper); "
                    "'partial': scope mismatch, some claims match some don't; "
                    "'unverifiable': abstract too short/vague to judge"
                )
            )
            reasoning: str = Field(description="1-2 sentence justification")
            confidence: float = Field(ge=0, le=1)

        prompt = (
            "Judge whether the SECTION TEXT accurately attributes its claims to the CITED PAPER.\n\n"
            f"SECTION TEXT:\n{section.content[:2000]}\n\n"
            f"CITED PAPER:\nTitle: {paper.title}\n"
            f"Abstract: {paper.abstract[:1500]}\n\n"
            "Labels:\n"
            "- matching: claims align with abstract\n"
            "- mismatched: section attributes tech/findings NOT in this abstract\n"
            "- partial: some match, some don't (scope mismatch)\n"
            "- unverifiable: abstract too short/vague\n"
        )

        try:
            judgment = asyncio.run(
                self._judge_client.generate_structured(
                    messages=[{"role": "user", "content": prompt}],
                    response_model=AttributionJudgment,
                )
            )
            mismatched = judgment.label == "mismatched"
            score = judgment.confidence if judgment.label == "matching" else 0.0

            return {
                "paper_id": paper.paper_id,
                "title": paper.title[:80],
                "status": "checked",
                "score": round(score, 4),
                "grounded": judgment.label in ("matching", "partial"),
                "mismatched": mismatched,
                "attribution_label": judgment.label,
                "attribution_confidence": round(judgment.confidence, 4),
                "attribution_reasoning": judgment.reasoning,
            }
        except Exception as e:
            logger.warning("attribution_judge_failed", paper_id=paper.paper_id, error=str(e)[:200])
            return {
                "paper_id": paper.paper_id,
                "title": paper.title[:80],
                "status": "checked",
                "score": 0.0,
                "grounded": False,
                "mismatched": False,
                "attribution_label": "error",
                "attribution_reasoning": str(e)[:200],
            }

    # ── Hybrid 模式（v4: embedding + attribution，不含 NLI）──

    def _verify_one_hybrid(self, section: ReportSection, paper: Paper) -> dict:
        """Hybrid 验证: embedding grounding + attribution 错配检测

        演化历程:
        - v3: embedding + NLI contradiction
        - v4 (current): embedding + LLM-judge attribution（P4 calibration 否证 NLI）

        Embedding 仍是非 LLM grounding 基线（打破 LLM 评 LLM 的一部分）。
        Attribution 用 LLM-judge 是有意识的 trade-off: 学术文本 NLI 不够用，
        换独立 LLM 做 judge，calibration 可在未来验证此选择（类似 P4）。
        """
        # 1. Embedding grounding（非 LLM 基线）
        emb_result = self._verify_one_embedding(section, paper)

        # 2. Attribution mismatch scan（LLM-judge）
        attr_result = self._verify_one_attribution(section, paper)

        return {
            "paper_id": paper.paper_id,
            "title": paper.title[:80],
            "status": "checked",
            # Grounding 由 embedding 决定
            "score": emb_result["score"],
            "grounded": emb_result["grounded"],
            # Mismatch 由 attribution LLM-judge 决定
            "mismatched": attr_result.get("mismatched", False),
            "embedding_similarity": emb_result["score"],
            "attribution_label": attr_result.get("attribution_label", ""),
            "attribution_confidence": attr_result.get("attribution_confidence", 0.0),
            "attribution_reasoning": attr_result.get("attribution_reasoning", ""),
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
