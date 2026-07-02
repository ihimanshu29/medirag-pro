"""
Document loader: PDF and DOCX → list[RawPage].

Strategy:
- pymupdf4llm  → converts PDF pages to clean Markdown (preserves headers, bold, lists)
- pdfplumber   → detects and extracts tables from each page separately
- python-docx  → handles DOCX files

Why two PDF libraries?
  pymupdf4llm gives the best text quality (layout-aware markdown).
  pdfplumber has the best table detection. We combine both.
"""
from pathlib import Path

from docx import Document

from src.logging_config import get_logger
from src.models import RawPage

logger = get_logger(__name__)


def _serialize_table(table: list[list[str | None]]) -> str:
    """Convert a pdfplumber table (list of rows) to a Markdown table string."""
    if not table or not table[0]:
        return ""

    # Clean None cells
    cleaned = [[cell or "" for cell in row] for row in table]
    header = cleaned[0]
    rows = cleaned[1:]

    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        # Pad row if shorter than header
        padded = row + [""] * (len(header) - len(row))
        md_lines.append("| " + " | ".join(padded[:len(header)]) + " |")

    return "\n".join(md_lines)


def load_pdf(file_path: Path) -> list[RawPage]:
    """
    Load a PDF and return one RawPage per page.
    Text comes from pymupdf4llm (markdown-aware).
    Tables come from pdfplumber (structure-aware).
    """
    import pdfplumber
    import pymupdf4llm

    pages: list[RawPage] = []

    # ── Step 1: Extract full-document markdown via pymupdf4llm ──
    try:
        md_pages: list[dict] = pymupdf4llm.to_markdown(
            str(file_path),
            page_chunks=True,   # returns list, one dict per page
            show_progress=False,
        )
    except Exception as e:
        logger.error("pdf_text_extraction_failed", file=str(file_path), error=str(e))
        raise

    # ── Step 2: Extract tables via pdfplumber ────────────────────
    table_map: dict[int, list[str]] = {}  # page_num → [markdown_table, ...]
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_num = page.page_number   # 1-indexed in pdfplumber
                raw_tables = page.extract_tables()
                if raw_tables:
                    table_map[page_num] = [
                        _serialize_table(t) for t in raw_tables if t
                    ]
                    logger.debug(
                        "tables_found",
                        page=page_num,
                        count=len(raw_tables),
                    )
    except Exception as e:
        # Table extraction failure is non-fatal — continue with text only
        logger.warning("table_extraction_failed", file=str(file_path), error=str(e))

    # ── Step 3: Merge into RawPage objects ───────────────────────
    for idx, page_data in enumerate(md_pages):
        page_num = idx + 1   # convert to 1-indexed
        text = page_data.get("text", "").strip()

        # Infer section from first markdown heading on this page
        section = ""
        for line in text.split("\n"):
            if line.startswith("#"):
                section = line.lstrip("#").strip()
                break

        pages.append(RawPage(
            page_num=page_num,
            text=text,
            tables=table_map.get(page_num, []),
            section=section,
        ))

    logger.info(
        "pdf_loaded",
        file=file_path.name,
        pages=len(pages),
        pages_with_tables=len(table_map),
    )
    return pages


def load_docx(file_path: Path) -> list[RawPage]:
    """
    Load a DOCX file. Each paragraph section becomes a RawPage.
    DOCX doesn't have real "pages" so we group by heading sections.
    """

    doc = Document(str(file_path))
    pages: list[RawPage] = []
    current_section = ""
    current_lines: list[str] = []
    page_num = 1

    def flush() -> None:
        nonlocal page_num
        text = "\n".join(current_lines).strip()
        if text:
            pages.append(RawPage(
                page_num=page_num,
                text=text,
                section=current_section,
            ))
            page_num += 1
        current_lines.clear()

    for para in doc.paragraphs:
        if para.style is not None and para.style.name.startswith("Heading"):
            flush()
            current_section = para.text.strip()
            current_lines.append(f"# {current_section}")
        else:
            if para.text.strip():
                current_lines.append(para.text.strip())

        # Flush every ~800 words to approximate page size
        if len(" ".join(current_lines).split()) > 800:
            flush()

    flush()

    logger.info("docx_loaded", file=file_path.name, sections=len(pages))
    return pages


def load_document(file_path: Path) -> list[RawPage]:
    """Dispatch to the correct loader based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix == ".docx":
        return load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
