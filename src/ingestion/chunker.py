"""
Parent-child chunker for RAG.

Strategy:
  Parent chunks  (~1024 tokens): large context window, stored in memory/disk.
                                  Returned to LLM for full context.
  Child chunks   (~256 tokens):  small, precise units. Embedded + stored in Qdrant.
                                  Retrieved by semantic search, then expanded to parent.

Why parent-child?
  Small chunks → better embedding precision (less noise per vector).
  Large parent context → LLM gets full surrounding context, not a fragment.
  This is the single highest-impact retrieval improvement after reranking.

Chunking approach:
  We use sentence-aware splitting (not fixed character count) to avoid
  cutting mid-sentence. We respect paragraph boundaries from the markdown text.
"""
import hashlib
import re
import uuid
from dataclasses import dataclass

from src.config import settings
from src.logging_config import get_logger
from src.models import ChildChunk, ChunkType, ParentChunk, RawPage

logger = get_logger(__name__)

# Approximate chars per token (conservative for medical text)
CHARS_PER_TOKEN = 4


def _count_tokens_approx(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving markdown structure."""
    # Split on sentence boundaries but keep paragraph breaks
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Also split on double newlines (paragraph boundaries)
    result: list[str] = []
    for sent in sentences:
        parts = sent.split("\n\n")
        result.extend(p.strip() for p in parts if p.strip())
    return result


def _make_chunk_id(text: str, source_file: str, page: int) -> str:
    """Deterministic chunk ID: stable across re-ingests of same content."""
    content = f"{source_file}:{page}:{text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()


def _build_parent_chunks(pages: list[RawPage], source_file: str) -> list[ParentChunk]:
    """
    Group pages into parent chunks of ~parent_chunk_size tokens.
    Tables are appended to the same parent as their host page.
    """
    max_tokens = settings.parent_chunk_size
    parents: list[ParentChunk] = []

    buffer_lines: list[str] = []
    buffer_tokens = 0
    page_start = pages[0].page_num if pages else 1
    page_end = page_start
    current_section = ""

    def flush_parent() -> None:
        nonlocal buffer_lines, buffer_tokens, page_start
        if not buffer_lines:
            return
        text = "\n\n".join(buffer_lines).strip()
        chunk_id = _make_chunk_id(text, source_file, page_start)
        parents.append(ParentChunk(
            chunk_id=chunk_id,
            text=text,
            source_file=source_file,
            page_start=page_start,
            page_end=page_end,
            section=current_section,
        ))
        buffer_lines.clear()
        buffer_tokens = 0
        page_start = page_end

    for page in pages:
        current_section = page.section or current_section

        # Add page text
        page_text = page.text
        page_tokens = _count_tokens_approx(page_text)

        # Add tables inline after their page text
        table_text = ""
        for table in page.tables:
            table_text += f"\n\n**Table (Page {page.page_num}):**\n{table}"

        full_page_text = page_text + table_text
        full_tokens = _count_tokens_approx(full_page_text)

        if buffer_tokens + full_tokens > max_tokens and buffer_lines:
            flush_parent()
            page_start = page.page_num

        buffer_lines.append(full_page_text)
        buffer_tokens += full_tokens
        page_end = page.page_num

    flush_parent()
    return parents


def _build_child_chunks(parent: ParentChunk) -> list[ChildChunk]:
    """
    Split a parent chunk into child chunks of ~child_chunk_size tokens.
    Sentence-aware: never cuts mid-sentence.
    Overlaps by ~chunk_overlap tokens to preserve context continuity.
    """
    max_tokens = settings.child_chunk_size
    overlap_tokens = settings.chunk_overlap
    children: list[ChildChunk] = []

    sentences = _split_into_sentences(parent.text)
    if not sentences:
        return children

    buffer: list[str] = []
    buffer_tokens = 0

    def flush_child(chunk_type: ChunkType = ChunkType.TEXT) -> None:
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if len(text) < 50:   # skip near-empty children
            return

        chunk_id = _make_chunk_id(text, parent.source_file, parent.page_start)
        children.append(ChildChunk(
            chunk_id=chunk_id,
            parent_id=parent.chunk_id,
            text=text,
            source_file=parent.source_file,
            page=parent.page_start,
            section=parent.section,
            chunk_type=chunk_type,
        ))

    overlap_buffer: list[str] = []

    for sentence in sentences:
        tokens = _count_tokens_approx(sentence)

        # Detect table rows — keep them as a single TABLE chunk
        if sentence.startswith("|") and "---" in sentence:
            flush_child()
            # Collect the whole table block
            table_lines = [sentence]
            buffer.clear()
            buffer_tokens = 0
            # We'll just emit this sentence as a table child
            text = sentence.strip()
            chunk_id = _make_chunk_id(text, parent.source_file, parent.page_start)
            children.append(ChildChunk(
                chunk_id=chunk_id,
                parent_id=parent.chunk_id,
                text=text,
                source_file=parent.source_file,
                page=parent.page_start,
                section=parent.section,
                chunk_type=ChunkType.TABLE,
            ))
            continue

        if buffer_tokens + tokens > max_tokens and buffer:
            flush_child()
            # Carry over overlap sentences
            overlap_buffer = buffer[-(overlap_tokens // max(1, _count_tokens_approx(buffer[-1]))):]
            buffer = list(overlap_buffer)
            buffer_tokens = sum(_count_tokens_approx(s) for s in buffer)

        buffer.append(sentence)
        buffer_tokens += tokens

    flush_child()

    logger.debug(
        "parent_chunked",
        parent_id=parent.chunk_id[:8],
        children=len(children),
    )
    return children


def chunk_pages(pages: list[RawPage], source_file: str) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """
    Main entry point.
    Returns (parent_chunks, child_chunks) for a document.
    """
    if not pages:
        return [], []

    parents = _build_parent_chunks(pages, source_file)
    children: list[ChildChunk] = []
    for parent in parents:
        children.extend(_build_child_chunks(parent))

    logger.info(
        "chunking_complete",
        source=source_file,
        parents=len(parents),
        children=len(children),
        avg_children_per_parent=round(len(children) / max(len(parents), 1), 1),
    )
    return parents, children
