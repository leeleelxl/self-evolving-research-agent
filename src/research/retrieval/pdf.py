"""
PDF 处理 — 下载 + 全文提取

职责:
1. 异步下载 PDF 文件（从 Semantic Scholar / arXiv 的 openAccessPdf URL）
2. 用 pypdf 提取纯文本
3. 清理和截断（去除乱码、控制长度）

设计决策:
- 优雅降级: 下载失败/提取失败 → 返回 None，Reader 降级到 abstract
- 并发控制: Semaphore 限制同时下载数，避免被 ban
- 文本清洗: 去除多余空白和常见 PDF 提取噪声
"""

from __future__ import annotations

import asyncio
import io
import re

import httpx
import structlog
from pypdf import PdfReader

from research.core.config import PDFConfig
from research.core.models import Paper

logger = structlog.get_logger()


async def extract_pdf_text(
    pdf_bytes: bytes,
    max_pages: int = 30,
    max_text_length: int = 50000,
) -> str | None:
    """从 PDF 字节流提取纯文本

    pypdf 是同步库，放到线程池执行避免阻塞事件循环。
    """
    loop = asyncio.get_running_loop()

    def _extract() -> str | None:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = reader.pages[:max_pages]
            texts: list[str] = []
            for page in pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            if not texts:
                return None
            return "\n\n".join(texts)
        except Exception as e:
            logger.warning("pdf_extract_failed", error=str(e)[:200])
            return None

    raw_text = await loop.run_in_executor(None, _extract)
    if raw_text is None:
        return None

    cleaned = _clean_pdf_text(raw_text)
    if len(cleaned) < 100:
        logger.warning("pdf_text_too_short", length=len(cleaned))
        return None

    if len(cleaned) > max_text_length:
        cleaned = cleaned[:max_text_length]

    return cleaned


def _clean_pdf_text(text: str) -> str:
    """清理 PDF 提取文本中的常见噪声"""
    # 合并连续空白行为单个空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除行尾多余空格
    text = re.sub(r" +\n", "\n", text)
    # 合并被 PDF 换行打断的段落（短行后跟小写字母开头的行）
    text = re.sub(r"(?<=[a-z,])\n(?=[a-z])", " ", text)
    # 去除常见 PDF 页眉/页脚噪声（纯数字行）
    text = re.sub(r"\n\d{1,3}\n", "\n", text)
    return text.strip()


async def download_pdf(
    url: str,
    timeout: float = 30.0,
) -> bytes | None:
    """异步下载 PDF 文件"""
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "ReSearch-v2/0.1 (academic-research-agent)"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type and not url.endswith(".pdf"):
                logger.warning("not_pdf", url=url[:100], content_type=content_type)
                return None

            return response.content

    except httpx.TimeoutException:
        logger.warning("pdf_download_timeout", url=url[:100])
        return None
    except httpx.HTTPStatusError as e:
        logger.warning("pdf_download_http_error", url=url[:100], status=e.response.status_code)
        return None
    except httpx.RequestError as e:
        logger.warning("pdf_download_error", url=url[:100], error=str(e)[:200])
        return None


async def fetch_full_texts(
    papers: list[Paper],
    config: PDFConfig | None = None,
) -> list[Paper]:
    """批量下载 PDF 并提取全文，填充 paper.full_text

    原地修改 papers 列表，返回同一列表。
    下载失败的论文 full_text 保持 None（Reader 降级到 abstract）。
    """
    cfg = config or PDFConfig()

    # 筛选有 pdf_url 且尚未提取全文的论文
    to_fetch = [p for p in papers if p.pdf_url and p.full_text is None]
    if not to_fetch:
        return papers

    logger.info("pdf_fetch_start", total=len(to_fetch))
    semaphore = asyncio.Semaphore(cfg.max_concurrent)

    async def _fetch_one(paper: Paper) -> None:
        async with semaphore:
            assert paper.pdf_url is not None
            pdf_bytes = await download_pdf(paper.pdf_url, timeout=cfg.download_timeout)
            if pdf_bytes is None:
                return
            text = await extract_pdf_text(
                pdf_bytes,
                max_pages=cfg.max_pages,
                max_text_length=cfg.max_text_length,
            )
            if text:
                paper.full_text = text
                logger.debug(
                    "pdf_extracted",
                    paper=paper.title[:60],
                    text_length=len(text),
                )

    await asyncio.gather(
        *[_fetch_one(p) for p in to_fetch],
        return_exceptions=True,
    )

    success = sum(1 for p in to_fetch if p.full_text is not None)
    logger.info("pdf_fetch_done", total=len(to_fetch), success=success)

    return papers
