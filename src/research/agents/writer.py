"""
Writer Agent — 综述生成

职责:
1. 将 PaperNote 列表综合成一篇连贯的研究综述
2. 按 ResearchPlan 的 sub_questions 组织章节结构
3. 确保引用准确（cited_papers 对应实际论文）
"""

from __future__ import annotations

from research.core.agent import BaseAgent
from research.core.config import LLMConfig
from research.core.models import PaperNote, ResearchPlan, ResearchReport

SYSTEM_PROMPT = """\
You are the Writer Agent in an academic research system.

Your job is to synthesize structured paper notes into a coherent research survey. \
Write in academic style but keep it readable. Each section should:
1. Cover one sub-question from the research plan
2. Cite specific papers by their paper_id
3. Compare and contrast different approaches
4. Identify trends and research gaps

The survey should be comprehensive yet concise. Aim for quality over quantity.
"""

WRITE_PROMPT_TEMPLATE = """\
## Research Question
{question}

## Sub-questions to Address
{sub_questions}

## Paper Notes (source material)
{notes_text}

## Your Task
Write a structured research survey (ResearchReport):

1. **title**: A descriptive title for this survey
2. **abstract**: 3-5 sentence summary of the survey's scope and key findings
3. **sections**: One section per sub-question. Each section should:
   - Have a clear section_title
   - Synthesize findings from multiple papers (don't just summarize one paper per paragraph)
   - Include cited_papers (list of paper_ids referenced in that section)
4. **references**: Complete list of all paper_ids cited anywhere in the survey

Use the paper_ids exactly as provided in the notes (e.g., "arxiv:2301.00001" or "abc123def").
"""


class WriterAgent(BaseAgent):
    """综述生成 Agent

    Usage:
        writer = WriterAgent()
        report = await writer.run(notes, plan)
    """

    name = "Writer"
    role = SYSTEM_PROMPT

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        super().__init__(llm_config)

    async def run(self, notes: list[PaperNote], plan: ResearchPlan) -> ResearchReport:
        """综合笔记生成研究综述"""
        if not notes:
            # 没有笔记时返回占位报告
            return ResearchReport(
                title=f"Survey: {plan.original_question}",
                abstract="No papers were found or retained after relevance filtering.",
            )

        prompt = self._build_prompt(notes, plan)
        self.logger.info("writing", num_notes=len(notes))

        report = await self.generate_structured(prompt, ResearchReport)

        self.logger.info("writing_done", title=report.title, sections=len(report.sections))
        return report

    def _build_prompt(self, notes: list[PaperNote], plan: ResearchPlan) -> str:
        # 格式化子问题
        sub_q_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(plan.sub_questions))

        # 格式化笔记
        notes_parts: list[str] = []
        for note in notes:
            notes_parts.append(
                f"### [{note.paper_id}] {note.title} (relevance: {note.relevance_score:.1f})\n"
                f"- **Core contribution:** {note.core_contribution}\n"
                f"- **Methodology:** {note.methodology}\n"
                f"- **Key findings:** {'; '.join(note.key_findings)}\n"
                f"- **Limitations:** {'; '.join(note.limitations) if note.limitations else 'N/A'}\n"
            )

        return WRITE_PROMPT_TEMPLATE.format(
            question=plan.original_question,
            sub_questions=sub_q_text,
            notes_text="\n".join(notes_parts),
        )
