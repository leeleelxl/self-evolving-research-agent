"""
学术论文检索客户端 — Agent 的"工具"层

两个数据源:
- Semantic Scholar: 学术论文元数据 + 引用关系，结构化 JSON
- arXiv: 预印本，返回 Atom XML

设计原则:
1. 统一接口: search(query) → list[Paper]，上层 Agent 不感知数据源差异
2. async: 用 httpx 异步 HTTP，不阻塞事件循环
3. 容错: 单个请求失败不影响整体，返回空列表 + 日志警告
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

import httpx
import structlog

from research.core.models import Paper

logger = structlog.get_logger()

# arXiv Atom XML 命名空间
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class BaseSearchClient(ABC):
    """检索客户端的统一接口"""

    @abstractmethod
    async def search(self, query: str, max_results: int = 20) -> list[Paper]:
        """搜索论文，返回 Paper 列表"""
        ...


class SemanticScholarClient(BaseSearchClient):
    """Semantic Scholar API 客户端

    API 文档: https://api.semanticscholar.org/api-docs/graph
    速率限制: 无 key 时 ~100 请求/5 分钟
    返回: 结构化 JSON，字段丰富（引用数、开放获取 PDF 等）

    面试知识点:
    - Semantic Scholar 的优势是引用关系图谱，可以做 citation chain 追踪
    - 它的 paperId 是全局唯一的，适合做去重
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "paperId,title,abstract,authors,year,url,citationCount,externalIds,openAccessPdf"

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._headers = {"User-Agent": "ReSearch-v2/0.1 (academic-research-agent)"}

    async def search(self, query: str, max_results: int = 20) -> list[Paper]:
        """搜索 Semantic Scholar"""
        logger.info("semantic_scholar_search", query=query, max_results=max_results)

        try:
            async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client:
                response = await client.get(
                    f"{self.BASE_URL}/paper/search",
                    params={
                        "query": query,
                        "limit": min(max_results, 100),  # API 上限 100
                        "fields": self.FIELDS,
                    },
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429:
                # 速率限制，等待后重试一次
                logger.warning("semantic_scholar_rate_limited", query=query)
                await asyncio.sleep(10)
                try:
                    async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client2:
                        response = await client2.get(
                            f"{self.BASE_URL}/paper/search",
                            params={"query": query, "limit": min(max_results, 100), "fields": self.FIELDS},
                        )
                        response.raise_for_status()
                except Exception:
                    logger.error("semantic_scholar_retry_failed", query=query)
                    return []
            elif status == 403:
                logger.warning("semantic_scholar_forbidden", query=query)
                return []
            else:
                logger.error("semantic_scholar_error", status=status, query=query)
                return []
        except httpx.RequestError as e:
            logger.error("semantic_scholar_request_error", error=str(e), query=query)
            return []

        data = response.json()
        papers: list[Paper] = []

        for item in data.get("data", []):
            # 跳过没有摘要的论文（对我们没用）
            if not item.get("abstract"):
                continue

            # 提取 PDF URL（如果有开放获取）
            pdf_url = None
            if oa_pdf := item.get("openAccessPdf"):
                pdf_url = oa_pdf.get("url")

            # 提取作者姓名列表
            authors = [a.get("name", "") for a in item.get("authors", [])]

            papers.append(
                Paper(
                    paper_id=item["paperId"],
                    title=item.get("title", ""),
                    abstract=item.get("abstract", ""),
                    authors=authors,
                    year=item.get("year", 0) or 0,
                    url=item.get("url", ""),
                    source="semantic_scholar",
                    citations=item.get("citationCount", 0) or 0,
                    pdf_url=pdf_url,
                )
            )

        logger.info("semantic_scholar_results", query=query, count=len(papers))
        return papers


class ArxivClient(BaseSearchClient):
    """arXiv API 客户端

    API 文档: https://info.arxiv.org/help/api/
    速率限制: 建议 3 秒/请求
    返回: Atom XML 格式，需要解析命名空间

    面试知识点:
    - arXiv 是预印本平台，论文可能未经同行评审
    - arXiv ID 格式: 2301.00001 (YYMM.NNNNN)
    - 它的搜索语法支持字段限定: ti:transformer AND abs:attention
    """

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def search(self, query: str, max_results: int = 20) -> list[Paper]:
        """搜索 arXiv"""
        logger.info("arxiv_search", query=query, max_results=max_results)

        # arXiv 搜索语法: 在所有字段中搜索（httpx 自动做 URL 编码）
        search_query = f"all:{query}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                response = await client.get(
                    self.BASE_URL,
                    params={
                        "search_query": search_query,
                        "start": 0,
                        "max_results": min(max_results, 100),
                        "sortBy": "relevance",
                    },
                )
                response.raise_for_status()
        except httpx.RequestError as e:
            logger.error("arxiv_request_error", error=str(e), query=query)
            return []

        papers = self._parse_atom_xml(response.text)
        logger.info("arxiv_results", query=query, count=len(papers))
        return papers

    def _parse_atom_xml(self, xml_text: str) -> list[Paper]:
        """解析 arXiv 返回的 Atom XML

        XML 结构:
        <feed>
          <entry>
            <id>http://arxiv.org/abs/2301.00001v1</id>
            <title>Paper Title</title>
            <summary>Abstract text...</summary>
            <author><name>Author Name</name></author>
            <published>2023-01-01T00:00:00Z</published>
            <link href="http://arxiv.org/pdf/2301.00001v1" type="application/pdf"/>
          </entry>
        </feed>
        """
        papers: list[Paper] = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error("arxiv_xml_parse_error", error=str(e))
            return []

        for entry in root.findall("atom:entry", ARXIV_NS):
            # 提取 arXiv ID (从 URL 中截取)
            id_elem = entry.find("atom:id", ARXIV_NS)
            if id_elem is None or id_elem.text is None:
                continue
            arxiv_url = id_elem.text.strip()
            arxiv_id = arxiv_url.split("/abs/")[-1]

            # 标题（去除换行和多余空格）
            title_elem = entry.find("atom:title", ARXIV_NS)
            title = " ".join((title_elem.text or "").split()) if title_elem is not None else ""

            # 摘要
            summary_elem = entry.find("atom:summary", ARXIV_NS)
            abstract = " ".join((summary_elem.text or "").split()) if summary_elem is not None else ""

            if not abstract:
                continue

            # 作者
            authors = []
            for author in entry.findall("atom:author", ARXIV_NS):
                name_elem = author.find("atom:name", ARXIV_NS)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            # 发表年份
            published_elem = entry.find("atom:published", ARXIV_NS)
            year = 0
            if published_elem is not None and published_elem.text:
                year = int(published_elem.text[:4])

            # PDF URL
            pdf_url = None
            for link in entry.findall("atom:link", ARXIV_NS):
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                    break

            papers.append(
                Paper(
                    paper_id=f"arxiv:{arxiv_id}",
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    year=year,
                    url=arxiv_url,
                    source="arxiv",
                    citations=0,  # arXiv 不提供引用数
                    pdf_url=pdf_url,
                )
            )

        return papers


def create_search_client(source: str) -> BaseSearchClient:
    """工厂函数 — 根据数据源创建检索客户端"""
    if source == "semantic_scholar":
        return SemanticScholarClient()
    elif source == "arxiv":
        return ArxivClient()
    else:
        raise ValueError(f"Unknown search source: {source}")
