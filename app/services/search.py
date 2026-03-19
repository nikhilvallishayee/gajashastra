"""
Hybrid search service.

Implements the three-tier search strategy from second-brain-api:
  1. Hybrid RRF (pgvector cosine + tsvector keyword, fused via Reciprocal Rank Fusion)
  2. Vector-only fallback
  3. Keyword-only fallback

Plus MMR diversity re-ranking (Jaccard-based, lambda=0.7).

Key formulas:
  RRF: score = vector_weight / (k + v_rank) + keyword_weight / (k + k_rank)
  MMR: MMR(d) = lambda * score(d) - (1 - lambda) * max_sim(d, selected)
"""

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services.embedding import generate_embedding_query

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """Internal representation of a search result."""

    verse_id: str
    chapter_id: str | None = None
    verse_number: str | None = None
    chapter_title: str | None = None
    sanskrit_devanagari: str = ""
    sanskrit_iast: str | None = None
    english_summary: str | None = None
    commentary: str | None = None
    verse_type: str | None = None
    page_number: int | None = None
    chunk_text: str | None = None
    chunk_index: int | None = None
    score: float = 0.0
    vector_score: float | None = None
    keyword_score: float | None = None
    rrf_score: float | None = None
    highlight: str | None = None
    _words: set = field(default_factory=set, repr=False)


async def hybrid_search(
    db: AsyncSession,
    query: str,
    *,
    limit: int = 10,
    offset: int = 0,
    chapter_id: str | None = None,
    text_id: str | None = None,
    verse_type: str | None = None,
    vector_weight: float | None = None,
    keyword_weight: float | None = None,
    mmr_lambda: float | None = None,
) -> tuple[list[SearchHit], float, str | None]:
    """
    Execute hybrid RRF search.

    Returns:
        (results, search_time_ms, fallback_used)
    """
    settings = get_settings()
    vw = vector_weight if vector_weight is not None else settings.vector_weight
    kw = keyword_weight if keyword_weight is not None else settings.keyword_weight
    rrf_k = settings.rrf_k
    ml = mmr_lambda if mmr_lambda is not None else settings.mmr_lambda

    start = time.monotonic()
    fallback_used = None

    # Try hybrid first
    try:
        query_embedding = await generate_embedding_query(query)
        results = await _hybrid_rrf_search(
            db, query, query_embedding,
            limit=limit * 3,  # Over-fetch for MMR
            chapter_id=chapter_id,
            text_id=text_id,
            verse_type=verse_type,
            vector_weight=vw,
            keyword_weight=kw,
            rrf_k=rrf_k,
        )
    except Exception as e:
        logger.warning("Hybrid search failed, trying vector-only: %s", e)
        fallback_used = "vector_only"
        try:
            query_embedding = await generate_embedding_query(query)
            results = await _vector_search(
                db, query_embedding,
                limit=limit * 3,
                chapter_id=chapter_id,
                text_id=text_id,
                verse_type=verse_type,
            )
        except Exception as e2:
            logger.warning("Vector search failed, falling back to keyword: %s", e2)
            fallback_used = "keyword_only"
            results = await _keyword_search(
                db, query,
                limit=limit * 3,
                chapter_id=chapter_id,
                text_id=text_id,
                verse_type=verse_type,
            )

    # Apply MMR diversity re-ranking
    if len(results) > 1:
        results = _mmr_rerank(results, lambda_=ml, target=limit + offset)

    # Apply offset/limit
    results = results[offset:offset + limit]

    elapsed = (time.monotonic() - start) * 1000
    return results, elapsed, fallback_used


async def vector_search(
    db: AsyncSession,
    query: str,
    *,
    limit: int = 10,
    chapter_id: str | None = None,
    text_id: str | None = None,
    verse_type: str | None = None,
) -> tuple[list[SearchHit], float]:
    """Pure semantic search."""
    start = time.monotonic()
    query_embedding = await generate_embedding_query(query)
    results = await _vector_search(
        db, query_embedding,
        limit=limit,
        chapter_id=chapter_id,
        text_id=text_id,
        verse_type=verse_type,
    )
    elapsed = (time.monotonic() - start) * 1000
    return results, elapsed


async def keyword_search(
    db: AsyncSession,
    query: str,
    *,
    limit: int = 10,
    chapter_id: str | None = None,
    text_id: str | None = None,
    verse_type: str | None = None,
) -> tuple[list[SearchHit], float]:
    """Pure keyword search."""
    start = time.monotonic()
    results = await _keyword_search(
        db, query,
        limit=limit,
        chapter_id=chapter_id,
        text_id=text_id,
        verse_type=verse_type,
    )
    elapsed = (time.monotonic() - start) * 1000
    return results, elapsed


# ---------------------------------------------------------------------------
# Internal search implementations
# ---------------------------------------------------------------------------

async def _hybrid_rrf_search(
    db: AsyncSession,
    query: str,
    query_embedding: list[float],
    *,
    limit: int,
    chapter_id: str | None,
    text_id: str | None,
    verse_type: str | None,
    vector_weight: float,
    keyword_weight: float,
    rrf_k: int,
) -> list[SearchHit]:
    """
    Hybrid RRF search combining pgvector cosine and keyword matching.

    Uses raw SQL with CTEs for vector and keyword sub-queries,
    joined via FULL OUTER JOIN with RRF scoring.
    """
    # Build filter clause
    filters, params = _build_filters(chapter_id, text_id, verse_type)
    filter_clause = f"AND {filters}" if filters else ""

    emb_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
    params["query_text"] = query
    params["limit"] = limit
    params["rrf_k"] = rrf_k
    params["vector_weight"] = vector_weight
    params["keyword_weight"] = keyword_weight

    sql = f"""
    WITH vector_results AS (
        SELECT
            ve.verse_id,
            ve.chunk_text,
            ve.chunk_index,
            1 - (ve.embedding <=> '{emb_literal}') AS cosine_sim,
            ROW_NUMBER() OVER (ORDER BY ve.embedding <=> '{emb_literal}') AS v_rank
        FROM verse_embeddings ve
        JOIN verses v ON v.id = ve.verse_id
        {"JOIN chapters c ON c.id = v.chapter_id" if text_id else ""}
        WHERE 1=1 {filter_clause}
        ORDER BY ve.embedding <=> '{emb_literal}'
        LIMIT :limit
    ),
    keyword_results AS (
        SELECT
            v.id AS verse_id,
            v.sanskrit_devanagari AS match_text,
            0 AS chunk_index,
            ts_rank_cd(
                to_tsvector('simple', coalesce(v.sanskrit_devanagari, '') || ' ' ||
                            coalesce(v.sanskrit_iast, '') || ' ' ||
                            coalesce(v.english_summary, '') || ' ' ||
                            coalesce(v.commentary, '')),
                plainto_tsquery('simple', :query_text)
            ) AS kw_rank_score,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(
                    to_tsvector('simple', coalesce(v.sanskrit_devanagari, '') || ' ' ||
                                coalesce(v.sanskrit_iast, '') || ' ' ||
                                coalesce(v.english_summary, '') || ' ' ||
                                coalesce(v.commentary, '')),
                    plainto_tsquery('simple', :query_text)
                ) DESC
            ) AS k_rank
        FROM verses v
        {"JOIN chapters c ON c.id = v.chapter_id" if text_id else ""}
        WHERE to_tsvector('simple', coalesce(v.sanskrit_devanagari, '') || ' ' ||
                          coalesce(v.sanskrit_iast, '') || ' ' ||
                          coalesce(v.english_summary, '') || ' ' ||
                          coalesce(v.commentary, ''))
              @@ plainto_tsquery('simple', :query_text)
        {filter_clause}
        ORDER BY kw_rank_score DESC
        LIMIT :limit
    ),
    rrf_fused AS (
        SELECT
            COALESCE(vr.verse_id, kr.verse_id) AS verse_id,
            vr.chunk_text,
            vr.chunk_index,
            vr.cosine_sim AS vector_score,
            kr.kw_rank_score AS keyword_raw_score,
            COALESCE(:vector_weight / (:rrf_k + vr.v_rank), 0) +
            COALESCE(:keyword_weight / (:rrf_k + kr.k_rank), 0) AS rrf_score
        FROM vector_results vr
        FULL OUTER JOIN keyword_results kr ON vr.verse_id = kr.verse_id
    )
    SELECT
        rf.verse_id,
        rf.chunk_text,
        rf.chunk_index,
        rf.vector_score,
        rf.keyword_raw_score,
        rf.rrf_score,
        v.chapter_id,
        v.verse_number,
        v.sanskrit_devanagari,
        v.sanskrit_iast,
        v.english_summary,
        v.commentary,
        v.verse_type,
        v.page_number,
        ch.title AS chapter_title
    FROM rrf_fused rf
    JOIN verses v ON v.id = rf.verse_id
    LEFT JOIN chapters ch ON ch.id = v.chapter_id
    ORDER BY rf.rrf_score DESC
    LIMIT :limit
    """

    result = await db.execute(sa_text(sql), params)
    rows = result.mappings().all()
    return [_row_to_hit(row) for row in rows]


async def _vector_search(
    db: AsyncSession,
    query_embedding: list[float],
    *,
    limit: int,
    chapter_id: str | None,
    text_id: str | None,
    verse_type: str | None,
) -> list[SearchHit]:
    """Pure vector (cosine similarity) search."""
    filters, params = _build_filters(chapter_id, text_id, verse_type)
    filter_clause = f"AND {filters}" if filters else ""
    emb_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
    params["limit"] = limit

    sql = f"""
    SELECT
        ve.verse_id,
        ve.chunk_text,
        ve.chunk_index,
        1 - (ve.embedding <=> '{emb_literal}') AS vector_score,
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
    WHERE 1=1 {filter_clause}
    ORDER BY ve.embedding <=> '{emb_literal}'
    LIMIT :limit
    """

    result = await db.execute(sa_text(sql), params)
    rows = result.mappings().all()
    hits = []
    for row in rows:
        hit = _row_to_hit(row)
        hit.score = hit.vector_score or 0.0
        hits.append(hit)
    return hits


async def _keyword_search(
    db: AsyncSession,
    query: str,
    *,
    limit: int,
    chapter_id: str | None,
    text_id: str | None,
    verse_type: str | None,
) -> list[SearchHit]:
    """
    Keyword search using tsvector or ILIKE fallback.

    Uses 'simple' text search config to avoid language-specific stemming
    issues with Sanskrit.
    """
    filters, params = _build_filters(chapter_id, text_id, verse_type)
    filter_clause = f"AND {filters}" if filters else ""
    params["query_text"] = query
    params["like_query"] = f"%{query}%"
    params["limit"] = limit

    # Try tsvector first, fall back to ILIKE
    sql = f"""
    SELECT
        v.id AS verse_id,
        NULL AS chunk_text,
        0 AS chunk_index,
        ts_rank_cd(
            to_tsvector('simple', coalesce(v.sanskrit_devanagari, '') || ' ' ||
                        coalesce(v.sanskrit_iast, '') || ' ' ||
                        coalesce(v.english_summary, '') || ' ' ||
                        coalesce(v.commentary, '')),
            plainto_tsquery('simple', :query_text)
        ) AS keyword_score,
        v.chapter_id,
        v.verse_number,
        v.sanskrit_devanagari,
        v.sanskrit_iast,
        v.english_summary,
        v.commentary,
        v.verse_type,
        v.page_number,
        ch.title AS chapter_title
    FROM verses v
    LEFT JOIN chapters ch ON ch.id = v.chapter_id
    WHERE (
        to_tsvector('simple', coalesce(v.sanskrit_devanagari, '') || ' ' ||
                    coalesce(v.sanskrit_iast, '') || ' ' ||
                    coalesce(v.english_summary, '') || ' ' ||
                    coalesce(v.commentary, ''))
        @@ plainto_tsquery('simple', :query_text)
        OR v.sanskrit_devanagari ILIKE :like_query
        OR v.sanskrit_iast ILIKE :like_query
        OR v.english_summary ILIKE :like_query
    )
    {filter_clause}
    ORDER BY keyword_score DESC
    LIMIT :limit
    """

    result = await db.execute(sa_text(sql), params)
    rows = result.mappings().all()
    hits = []
    for row in rows:
        hit = _row_to_hit(row)
        hit.keyword_score = row.get("keyword_score", 0.0)
        hit.score = hit.keyword_score or 0.0
        hits.append(hit)
    return hits


# ---------------------------------------------------------------------------
# MMR diversity re-ranking
# ---------------------------------------------------------------------------

def _mmr_rerank(
    hits: list[SearchHit],
    lambda_: float = 0.7,
    target: int = 10,
) -> list[SearchHit]:
    """
    Maximal Marginal Relevance re-ranking.

    Uses Jaccard word overlap for fast similarity (no re-embedding).

    MMR(d) = lambda * score(d) - (1 - lambda) * max_sim(d, selected)
    """
    if len(hits) <= 1:
        return hits

    # Pre-compute word sets for Jaccard
    for hit in hits:
        text = (hit.sanskrit_devanagari or "") + " " + (hit.english_summary or "")
        hit._words = set(text.lower().split())

    selected: list[SearchHit] = []
    remaining = list(hits)

    while remaining and len(selected) < target:
        best_score = -float("inf")
        best_idx = 0

        for i, candidate in enumerate(remaining):
            relevance = candidate.rrf_score if candidate.rrf_score else candidate.score

            # Max similarity to already selected
            max_sim = 0.0
            for sel in selected:
                sim = _jaccard(candidate._words, sel._words)
                max_sim = max(max_sim, sim)

            mmr_score = lambda_ * relevance - (1 - lambda_) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two word sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_filters(
    chapter_id: str | None,
    text_id: str | None,
    verse_type: str | None,
) -> tuple[str, dict]:
    """Build SQL filter clause and params."""
    clauses = []
    params = {}

    if chapter_id:
        clauses.append("v.chapter_id = :chapter_id")
        params["chapter_id"] = chapter_id
    if text_id:
        clauses.append("c.text_id = :text_id")
        params["text_id"] = text_id
    if verse_type:
        clauses.append("v.verse_type = :verse_type")
        params["verse_type"] = verse_type

    return " AND ".join(clauses), params


def _row_to_hit(row) -> SearchHit:
    """Convert a database row mapping to a SearchHit."""
    rrf = row.get("rrf_score")
    vs = row.get("vector_score")
    ks = row.get("keyword_raw_score") or row.get("keyword_score")

    score = rrf if rrf is not None else (vs if vs is not None else (ks or 0.0))

    return SearchHit(
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
        chunk_text=row.get("chunk_text"),
        chunk_index=row.get("chunk_index"),
        score=score or 0.0,
        vector_score=vs,
        keyword_score=ks,
        rrf_score=rrf,
    )
