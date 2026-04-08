"""
文本 Chunking 策略 — 消融变量 A

三种策略:
1. Fixed: 按固定 token 数切分，有 overlap（baseline，最简单）
2. Semantic: 按段落/章节边界切分（利用文档结构）
3. Recursive: 先按大边界切，超长再按小边界递归切（最灵活）

消融实验对比这三种策略在检索质量上的差异。
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod

from research.core.config import ChunkConfig
from research.core.models import Chunk


class BaseChunker(ABC):
    """Chunker 统一接口"""

    @abstractmethod
    def chunk(self, text: str, paper_id: str) -> list[Chunk]:
        """将文本切分为 Chunk 列表"""
        ...


class FixedChunker(BaseChunker):
    """固定大小切分 — Baseline

    最简单的策略: 按字符数切分，相邻 chunk 有 overlap。
    overlap 的作用: 避免关键信息恰好在边界被切断。

    优点: 实现简单，chunk 大小一致
    缺点: 可能从句子/段落中间切断，语义不完整
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, paper_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    chunk_id=f"{paper_id}_chunk_{idx}",
                    paper_id=paper_id,
                    text=chunk_text,
                    chunk_index=idx,
                    metadata={"strategy": "fixed"},
                ))
                idx += 1

            start = end - self.overlap
            if start >= len(text):
                break

        return chunks


class SemanticChunker(BaseChunker):
    """语义切分 — 按段落/章节边界

    利用文档的自然结构（空行、标题）来切分。
    每个 chunk 是一个完整的段落或章节。

    优点: 语义完整，不会切断段落
    缺点: chunk 大小不一致，某些章节可能过长
    """

    def __init__(self, max_chunk_size: int = 1024) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, paper_id: str) -> list[Chunk]:
        # 按双换行（段落边界）或 Markdown 标题切分
        paragraphs = re.split(r"\n\s*\n|(?=^#{1,4}\s)", text, flags=re.MULTILINE)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: list[Chunk] = []
        current_text = ""
        idx = 0

        for para in paragraphs:
            # 如果当前累积 + 新段落不超限，合并
            if len(current_text) + len(para) + 1 <= self.max_chunk_size:
                current_text = f"{current_text}\n{para}".strip() if current_text else para
            else:
                # 保存当前累积
                if current_text:
                    chunks.append(Chunk(
                        chunk_id=f"{paper_id}_chunk_{idx}",
                        paper_id=paper_id,
                        text=current_text,
                        chunk_index=idx,
                        metadata={"strategy": "semantic"},
                    ))
                    idx += 1
                current_text = para

        # 别忘了最后一个
        if current_text:
            chunks.append(Chunk(
                chunk_id=f"{paper_id}_chunk_{idx}",
                paper_id=paper_id,
                text=current_text,
                chunk_index=idx,
                metadata={"strategy": "semantic"},
            ))

        return chunks


class RecursiveChunker(BaseChunker):
    """递归切分 — 最灵活

    分层切分策略:
    1. 先按章节边界（双换行）切
    2. 如果某个块超过 max_size，按句子边界再切
    3. 如果单个句子还超，按字符切（兜底）

    这是 LangChain 的 RecursiveCharacterTextSplitter 的思路，
    但我们自己实现，面试时能讲清楚每一层。

    优点: 兼顾语义完整性和大小一致性
    缺点: 实现稍复杂
    """

    def __init__(self, max_chunk_size: int = 1024, min_chunk_size: int = 100) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        # 切分分隔符优先级：段落 > 句子 > 字符
        self._separators = ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str, paper_id: str) -> list[Chunk]:
        raw_chunks = self._recursive_split(text, 0)
        chunks: list[Chunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < self.min_chunk_size and chunks:
                # 太短的合并到上一个
                prev = chunks[-1]
                chunks[-1] = Chunk(
                    chunk_id=prev.chunk_id,
                    paper_id=paper_id,
                    text=f"{prev.text}\n{chunk_text}",
                    chunk_index=prev.chunk_index,
                    metadata={"strategy": "recursive"},
                )
            elif chunk_text:
                chunks.append(Chunk(
                    chunk_id=f"{paper_id}_chunk_{idx}",
                    paper_id=paper_id,
                    text=chunk_text,
                    chunk_index=len(chunks),
                    metadata={"strategy": "recursive"},
                ))
        return chunks

    def _recursive_split(self, text: str, sep_idx: int) -> list[str]:
        """递归切分：用当前分隔符切，超大的块用下一级分隔符再切"""
        if len(text) <= self.max_chunk_size:
            return [text]

        if sep_idx >= len(self._separators):
            # 所有分隔符都试过了，硬切
            result = []
            for i in range(0, len(text), self.max_chunk_size):
                result.append(text[i : i + self.max_chunk_size])
            return result

        sep = self._separators[sep_idx]
        parts = text.split(sep)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # 这个 part 本身可能还超，递归用下一级分隔符
                if len(part) > self.max_chunk_size:
                    result.extend(self._recursive_split(part, sep_idx + 1))
                else:
                    current = part

        if current:
            result.append(current)

        return result


def create_chunker(config: ChunkConfig) -> BaseChunker:
    """工厂函数 — 根据配置创建 Chunker"""
    if config.strategy == "fixed":
        return FixedChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap)
    elif config.strategy == "semantic":
        return SemanticChunker(max_chunk_size=config.max_chunk_size)
    elif config.strategy == "recursive":
        return RecursiveChunker(max_chunk_size=config.max_chunk_size)
    else:
        raise ValueError(f"Unknown chunk strategy: {config.strategy}")
