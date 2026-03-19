"""
Gajashastra Sanskrit Intelligence Platform -- FastAPI Application.

Entry point for the API server. Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Or from the project root:
    cd r-and-d/gajashastra && uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.base import get_db, init_db, close_db
from app.routes import corpus, search, assistant, zoo, reference

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB on startup, close on shutdown."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    if settings.debug:
        logger.info("Debug mode enabled -- creating tables if needed")
        await init_db()

    yield

    logger.info("Shutting down")
    await close_db()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "API for the Gajashastra Sanskrit Intelligence Platform. "
        "Provides hybrid search, conversational RAG, and zoo management "
        "integration over the ancient Sanskrit treatise on elephant science "
        "by Palakapya Muni."
    ),
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(corpus.router)
app.include_router(search.router)
app.include_router(assistant.router)
app.include_router(zoo.router)
app.include_router(reference.router)


# ---------------------------------------------------------------------------
# Health and utility endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/health/db", tags=["system"])
async def db_health_check(db: AsyncSession = Depends(get_db)):
    """Database connectivity health check."""
    from sqlalchemy import text as sa_text
    try:
        result = await db.execute(sa_text("SELECT 1"))
        row = result.scalar()
        if row == 1:
            return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unhealthy: {e}")


@app.get("/stats", tags=["system"])
async def corpus_stats(db: AsyncSession = Depends(get_db)):
    """Get corpus statistics."""
    from sqlalchemy import text as sa_text
    from app.models.corpus import Text, Chapter, Verse
    from app.models.embedding import VerseEmbedding
    from app.models.knowledge import Insight
    from app.models.integration import Protocol, AssistantSession
    from sqlalchemy import select, func

    async def _count(model):
        result = await db.execute(select(func.count()).select_from(model))
        return result.scalar() or 0

    return {
        "texts": await _count(Text),
        "chapters": await _count(Chapter),
        "verses": await _count(Verse),
        "embeddings": await _count(VerseEmbedding),
        "insights": await _count(Insight),
        "protocols": await _count(Protocol),
        "sessions": await _count(AssistantSession),
    }


# ---------------------------------------------------------------------------
# Admin endpoints (ingestion, embedding, extraction)
# ---------------------------------------------------------------------------

@app.post("/admin/ingest", tags=["admin"])
async def ingest_pages(
    text_id: str,
    pages_dir: str,
    start_page: int = 1,
    end_page: int = 0,
    model: str | None = None,
    batch_size: int = 5,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
):
    """
    Ingest page images from the corpus into the database.

    Runs Claude Vision extraction on each page image and stores
    the extracted verses in the database.
    """
    from app.services.ingestion import IngestionService

    service = IngestionService(db)
    result = await service.ingest_pdf_pages(
        pages_dir=pages_dir,
        text_id=text_id,
        start_page=start_page,
        end_page=end_page,
        model=model,
        batch_size=batch_size,
    )
    return result


@app.post("/admin/embed", tags=["admin"])
async def embed_verses(
    text_id: str | None = None,
    batch_size: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate embeddings for all un-indexed verses.

    Uses gemini-embedding-001 (or local fallback) to create vector
    embeddings for semantic search.
    """
    from app.services.ingestion import IngestionService

    service = IngestionService(db)
    result = await service.embed_verses(text_id=text_id, batch_size=batch_size)
    return result


@app.post("/admin/extract", tags=["admin"])
async def extract_insights(
    text_id: str | None = None,
    verse_ids: list[str] | None = None,
    skip_extracted: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """
    Extract structured insights from verses using Claude.

    Extracts teachings, applications, classifications, remedies,
    and creates Protocol records for treatment-related insights.
    """
    from app.services.extraction import ExtractionService

    service = ExtractionService(db)
    result = await service.extract_batch(
        verse_ids=verse_ids,
        text_id=text_id,
        skip_extracted=skip_extracted,
    )
    return result


# ---------------------------------------------------------------------------
# Transliteration utility endpoint
# ---------------------------------------------------------------------------

@app.post("/api/v1/transliterate", tags=["utilities"])
async def transliterate(
    text: str,
    source: str = "auto",
    target: str = "iast",
):
    """
    Transliterate text between Devanagari and IAST.

    Args:
        text: Input text to transliterate.
        source: Source script ("devanagari", "iast", or "auto" for detection).
        target: Target script ("devanagari" or "iast").
    """
    from app.utils.sanskrit import detect_script
    from app.services.transliteration import devanagari_to_iast, iast_to_devanagari

    if source == "auto":
        source = detect_script(text)

    if source == "devanagari" and target == "iast":
        result = devanagari_to_iast(text)
    elif source == "iast" and target == "devanagari":
        result = iast_to_devanagari(text)
    elif source == target:
        result = text
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported transliteration: {source} -> {target}. "
                   f"Supported: devanagari<->iast",
        )

    return {
        "input": text,
        "output": result,
        "source_script": source,
        "target_script": target,
    }
