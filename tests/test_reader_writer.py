"""Reader / Writer Agent 集成测试 — 真实 LLM 调用"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from research.agents.reader import ReaderAgent
from research.agents.writer import WriterAgent
from research.core.models import Paper, PaperNote, ResearchPlan, ResearchReport, SearchQuery, SearchStrategy


@pytest.mark.integration
class TestReaderIntegration:

    @pytest.mark.asyncio
    async def test_read_paper_returns_note(self) -> None:
        """Reader 能对真实论文摘要生成 PaperNote"""
        reader = ReaderAgent()
        papers = [
            Paper(
                paper_id="test_1",
                title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                abstract=(
                    "Large pre-trained language models have been shown to store factual knowledge "
                    "in their parameters. However, their ability to access and precisely manipulate "
                    "knowledge is still limited. We explore a general-purpose fine-tuning recipe for "
                    "retrieval-augmented generation (RAG) — models which combine pre-trained parametric "
                    "and non-parametric memories for language generation."
                ),
                authors=["Patrick Lewis", "Ethan Perez"],
                year=2020,
                url="https://arxiv.org/abs/2005.11401",
                source="arxiv",
            ),
        ]
        notes = await reader.run(papers, "What is retrieval-augmented generation?")

        assert len(notes) >= 1
        note = notes[0]
        assert isinstance(note, PaperNote)
        assert note.paper_id == "test_1"
        assert note.core_contribution  # 不为空
        assert len(note.key_findings) >= 1
        assert 0 <= note.relevance_score <= 1

    @pytest.mark.asyncio
    async def test_reader_filters_low_relevance(self) -> None:
        """Reader 过滤掉不相关的论文"""
        reader = ReaderAgent()
        papers = [
            Paper(
                paper_id="relevant",
                title="Dense Passage Retrieval for Open-Domain QA",
                abstract="We address open-domain question answering using dense representations for retrieval.",
                authors=["Vladimir Karpukhin"],
                year=2020, url="", source="arxiv",
            ),
            Paper(
                paper_id="irrelevant",
                title="Cooking Recipe Generation with GPT-2",
                abstract="We fine-tune GPT-2 to generate cooking recipes from ingredient lists.",
                authors=["Chef Bot"],
                year=2021, url="", source="arxiv",
            ),
        ]
        notes = await reader.run(papers, "What are advances in dense passage retrieval?")

        # 不相关的论文应该被过滤
        paper_ids = [n.paper_id for n in notes]
        assert "relevant" in paper_ids


@pytest.mark.integration
class TestWriterIntegration:

    @pytest.mark.asyncio
    async def test_writer_generates_report(self) -> None:
        """Writer 能从 PaperNote 生成 ResearchReport"""
        writer = WriterAgent()
        notes = [
            PaperNote(
                paper_id="p1", title="RAG Paper",
                core_contribution="Combines retrieval with generation",
                methodology="Dual encoder + seq2seq generator",
                key_findings=["RAG outperforms pure generative models", "Non-parametric memory helps"],
                relevance_score=0.9, relevance_reason="Directly about RAG",
            ),
            PaperNote(
                paper_id="p2", title="DPR Paper",
                core_contribution="Dense passage retrieval for open-domain QA",
                methodology="Dual BERT encoders trained with contrastive learning",
                key_findings=["Dense retrieval outperforms BM25 on NQ"],
                relevance_score=0.8, relevance_reason="Core retrieval method for RAG",
            ),
        ]
        plan = ResearchPlan(
            original_question="What is RAG?",
            sub_questions=["How does retrieval work in RAG?", "How does generation work in RAG?"],
            search_strategy=SearchStrategy(queries=[SearchQuery(query="RAG")], focus_areas=[]),
        )
        report = await writer.run(notes, plan)

        assert isinstance(report, ResearchReport)
        assert report.title
        assert report.abstract
        assert len(report.sections) >= 1
        assert len(report.references) >= 1
