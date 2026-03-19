"""
Search routes: hybrid, semantic, keyword search over the corpus.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import get_db
from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    SimilarVersesRequest,
)
from app.services.search import (
    hybrid_search,
    vector_search,
    keyword_search,
    SearchHit,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/search", tags=["search"])


def _hit_to_result(hit: SearchHit) -> SearchResult:
    """Convert internal SearchHit to API response."""
    return SearchResult(
        verse_id=hit.verse_id,
        chapter_id=hit.chapter_id,
        verse_number=hit.verse_number,
        chapter_title=hit.chapter_title,
        sanskrit_devanagari=hit.sanskrit_devanagari,
        sanskrit_iast=hit.sanskrit_iast,
        english_summary=hit.english_summary,
        commentary=hit.commentary,
        verse_type=hit.verse_type,
        page_number=hit.page_number,
        score=hit.score,
        vector_score=hit.vector_score,
        keyword_score=hit.keyword_score,
        rrf_score=hit.rrf_score,
        chunk_text=hit.chunk_text,
        chunk_index=hit.chunk_index,
        highlight=hit.highlight,
    )


@router.post("/", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Search the Gajashastra corpus.

    Modes:
      - hybrid: RRF fusion of pgvector cosine + tsvector keyword (default)
      - semantic: Pure vector similarity search
      - keyword: Pure keyword/full-text search
    """
    try:
        if request.mode == "hybrid":
            hits, elapsed, fallback = await hybrid_search(
                db,
                request.query,
                limit=request.limit,
                offset=request.offset,
                chapter_id=request.chapter_id,
                text_id=request.text_id,
                verse_type=request.verse_type,
                vector_weight=request.vector_weight,
                keyword_weight=request.keyword_weight,
                mmr_lambda=request.mmr_lambda,
            )
        elif request.mode == "semantic":
            hits, elapsed = await vector_search(
                db,
                request.query,
                limit=request.limit,
                chapter_id=request.chapter_id,
                text_id=request.text_id,
                verse_type=request.verse_type,
            )
            fallback = None
        elif request.mode == "keyword":
            hits, elapsed = await keyword_search(
                db,
                request.query,
                limit=request.limit,
                chapter_id=request.chapter_id,
                text_id=request.text_id,
                verse_type=request.verse_type,
            )
            fallback = None
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid search mode: {request.mode}. Use: hybrid, semantic, keyword",
            )

        results = [_hit_to_result(h) for h in hits]

        return SearchResponse(
            query=request.query,
            mode=request.mode,
            results=results,
            total=len(results),
            limit=request.limit,
            offset=request.offset,
            search_time_ms=elapsed,
            fallback_used=fallback,
        )

    except Exception as e:
        logger.error("Search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/similar", response_model=SearchResponse)
async def find_similar_verses(
    request: SimilarVersesRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Find verses similar to a given verse.

    Uses the verse's existing embedding to find nearest neighbors.
    """
    from sqlalchemy import text as sa_text
    from app.models.corpus import Verse

    verse = await db.get(Verse, request.verse_id)
    if not verse:
        raise HTTPException(status_code=404, detail="Verse not found")

    # Get the verse's embedding
    result = await db.execute(
        sa_text("""
            SELECT embedding
            FROM verse_embeddings
            WHERE verse_id = :verse_id
            LIMIT 1
        """),
        {"verse_id": request.verse_id},
    )
    row = result.first()
    if not row:
        raise HTTPException(
            status_code=404,
            detail="Verse has no embedding. Run embedding generation first.",
        )

    embedding = row[0]

    # Find similar via cosine distance
    similar_result = await db.execute(
        sa_text("""
            SELECT
                ve.verse_id,
                ve.chunk_text,
                ve.chunk_index,
                1 - (ve.embedding <=> :emb::vector) AS similarity,
                v.chapter_id,
                v.verse_number,
                v.sanskrit_devanagari,
                v.sanskrit_iast,
                v.english_summary,
                v.commentary,
                v.verse_type,
                v.page_number,
                ch.title AS chapter_title
            FROM verse_embeddings ve
            JOIN verses v ON v.id = ve.verse_id
            LEFT JOIN chapters ch ON ch.id = v.chapter_id
            WHERE ve.verse_id != :verse_id
              AND 1 - (ve.embedding <=> :emb::vector) >= :min_sim
            ORDER BY ve.embedding <=> :emb::vector
            LIMIT :limit
        """),
        {
            "verse_id": request.verse_id,
            "emb": str(embedding),
            "min_sim": request.min_similarity,
            "limit": request.limit,
        },
    )

    rows = similar_result.mappings().all()
    results = []
    for row in rows:
        results.append(SearchResult(
            verse_id=row["verse_id"],
            chapter_id=row.get("chapter_id"),
            verse_number=row.get("verse_number"),
            chapter_title=row.get("chapter_title"),
            sanskrit_devanagari=row.get("sanskrit_devanagari", ""),
            sanskrit_iast=row.get("sanskrit_iast"),
            english_summary=row.get("english_summary"),
            commentary=row.get("commentary"),
            verse_type=row.get("verse_type"),
            page_number=row.get("page_number"),
            score=float(row.get("similarity", 0)),
            vector_score=float(row.get("similarity", 0)),
            chunk_text=row.get("chunk_text"),
            chunk_index=row.get("chunk_index"),
        ))

    return SearchResponse(
        query=f"similar to verse {request.verse_id}",
        mode="similar",
        results=results,
        total=len(results),
        limit=request.limit,
        offset=0,
    )
