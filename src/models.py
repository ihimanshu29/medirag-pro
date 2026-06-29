"""
Core domain models used across the entire pipeline.
Using dataclasses (not Pydantic) for internal speed — Pydantic is for API boundary only.
"""
from dataclasses import dataclass, field
from enum import Enum


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    HEADER = "header"


@dataclass
class RawPage:
    """A single page extracted from a document before chunking."""
    page_num: int
    text: str
    tables: list[str] = field(default_factory=list)   # tables serialized as markdown
    section: str = ""


@dataclass
class ParentChunk:
    """
    Large context window chunk (1024 tokens).
    Stored in memory/disk — returned to LLM for full context.
    NOT embedded into the vector store.
    """
    chunk_id: str
    text: str
    source_file: str
    page_start: int
    page_end: int
    section: str = ""


@dataclass
class ChildChunk:
    """
    Small, precise chunk (256 tokens) that IS embedded and stored in Qdrant.
    Links back to its parent via parent_id for context expansion at retrieval time.
    """
    chunk_id: str
    parent_id: str
    text: str
    source_file: str
    page: int
    section: str = ""
    chunk_type: ChunkType = ChunkType.TEXT
    # Populated after embedding
    embedding: list[float] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with its score."""
    chunk_id: str
    parent_id: str
    text: str
    source_file: str
    page: int
    section: str
    score: float
    retrieval_method: str = "hybrid"   # dense | bm25 | hybrid
