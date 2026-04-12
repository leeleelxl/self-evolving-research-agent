"""PDF 处理模块测试"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from research.core.config import PDFConfig
from research.core.models import Paper
from research.retrieval.pdf import (
    _clean_pdf_text,
    download_pdf,
    extract_pdf_text,
    fetch_full_texts,
)


# ── 文本清洗测试 ──


class TestCleanPdfText:
    def test_merge_excessive_newlines(self) -> None:
        text = "paragraph one\n\n\n\n\nparagraph two"
        assert _clean_pdf_text(text) == "paragraph one\n\nparagraph two"

    def test_merge_broken_lines(self) -> None:
        text = "this is a long sentence that was\nbroken by pdf extraction"
        result = _clean_pdf_text(text)
        assert "sentence that was broken" in result

    def test_remove_page_numbers(self) -> None:
        text = "end of page\n42\nstart of next"
        result = _clean_pdf_text(text)
        assert "42" not in result

    def test_strip_trailing_spaces(self) -> None:
        text = "line with spaces   \nnext line"
        result = _clean_pdf_text(text)
        assert "   \n" not in result


# ── PDF 提取测试 ──


class TestExtractPdfText:
    @pytest.mark.asyncio
    async def test_extract_from_valid_pdf_bytes(self) -> None:
        """用 pypdf 生成一个最小 PDF 来测试提取"""
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        # pypdf 的空白页没有文本，所以提取结果为 None
        import io

        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        result = await extract_pdf_text(pdf_bytes)
        # 空白页提取不到文本
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_returns_none_for_invalid_bytes(self) -> None:
        result = await extract_pdf_text(b"not a pdf")
        assert result is None

    @pytest.mark.asyncio
    async def test_truncation(self) -> None:
        """验证 max_text_length 截断"""
        # Mock pypdf reader to return long text
        long_text = "a" * 200
        with patch("research.retrieval.pdf.PdfReader") as mock_reader:
            mock_page = type("Page", (), {"extract_text": lambda self: long_text})()
            mock_reader.return_value.pages = [mock_page]

            result = await extract_pdf_text(b"fake", max_text_length=150)
            assert result is not None
            assert len(result) <= 150


# ── 下载测试 ──


class TestDownloadPdf:
    @pytest.mark.asyncio
    async def test_download_success(self) -> None:
        with patch("research.retrieval.pdf.httpx.AsyncClient") as mock_client_cls:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.content = b"%PDF-1.4 fake pdf content"
            mock_response.headers = {"content-type": "application/pdf"}
            mock_response.raise_for_status = lambda: None

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await download_pdf("https://example.com/paper.pdf")
            assert result == b"%PDF-1.4 fake pdf content"

    @pytest.mark.asyncio
    async def test_download_timeout(self) -> None:
        import httpx

        with patch("research.retrieval.pdf.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("timeout")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await download_pdf("https://example.com/paper.pdf")
            assert result is None


# ── 批量提取测试 ──


def _make_paper(paper_id: str, pdf_url: str | None = None, full_text: str | None = None) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        abstract="Some abstract text.",
        authors=["Author"],
        year=2024,
        url="https://example.com",
        source="semantic_scholar",
        pdf_url=pdf_url,
        full_text=full_text,
    )


class TestFetchFullTexts:
    @pytest.mark.asyncio
    async def test_skips_papers_without_pdf_url(self) -> None:
        papers = [_make_paper("p1"), _make_paper("p2")]
        result = await fetch_full_texts(papers)
        assert all(p.full_text is None for p in result)

    @pytest.mark.asyncio
    async def test_skips_already_extracted(self) -> None:
        papers = [_make_paper("p1", pdf_url="https://x.com/a.pdf", full_text="existing")]
        result = await fetch_full_texts(papers)
        assert result[0].full_text == "existing"

    @pytest.mark.asyncio
    async def test_fills_full_text_on_success(self) -> None:
        papers = [_make_paper("p1", pdf_url="https://x.com/a.pdf")]

        async def mock_download(url: str, timeout: float = 30.0) -> bytes | None:
            return b"fake pdf"

        async def mock_extract(
            pdf_bytes: bytes, max_pages: int = 30, max_text_length: int = 50000,
        ) -> str | None:
            return "Extracted full text content that is long enough to pass the check."

        with (
            patch("research.retrieval.pdf.download_pdf", side_effect=mock_download),
            patch("research.retrieval.pdf.extract_pdf_text", side_effect=mock_extract),
        ):
            result = await fetch_full_texts(papers, PDFConfig(enabled=True))
            assert result[0].full_text is not None
            assert "Extracted" in result[0].full_text
