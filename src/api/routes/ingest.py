"""
/ingest endpoint.
Accepts a PDF file upload, runs the full ingestion pipeline:
extract → chunk → embed → upsert to Qdrant + BM25.
"""
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from src.api.limiter import limiter
from src.api.schemas import IngestResponse, IngestStatus
from src.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_MB = 50


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
@limiter.limit("5/minute")
async def ingest_document(request: Request, file: UploadFile) -> IngestResponse:
    """
    Upload and ingest a medical document (PDF or DOCX).

    Process:
    1. Validate file type and size
    2. Extract text + tables
    3. Semantic chunking with parent-child pairs
    4. Embed child chunks (BGE)
    5. Upsert to Qdrant with metadata
    6. Update BM25 index
    """
    from src.pipeline.ingest_pipeline import IngestPipeline

    # Validate file
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
        )

    logger.info("ingest_start", filename=file.filename, size_mb=round(size_mb, 2))

    # Write to temp file for processing
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        pipeline = IngestPipeline()
        result = await pipeline.run(file_path=tmp_path, filename=file.filename or "unknown")
    except Exception as e:
        logger.error("ingest_error", filename=file.filename, error=str(e))
        return IngestResponse(
            filename=file.filename or "unknown",
            status=IngestStatus.FAILED,
            chunks_created=0,
            tables_extracted=0,
            message=str(e),
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    logger.info(
        "ingest_complete",
        filename=file.filename,
        chunks=result["chunks_created"],
        tables=result["tables_extracted"],
    )

    return IngestResponse(
        filename=file.filename or "unknown",
        status=IngestStatus.SUCCESS,
        chunks_created=result["chunks_created"],
        tables_extracted=result["tables_extracted"],
        message="Document ingested successfully.",
    )
