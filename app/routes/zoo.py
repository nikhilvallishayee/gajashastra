"""
Zoo management integration routes.

Provides endpoints for querying elephant care protocols extracted from
the Gajashastra, designed for zoo/sanctuary veterinary staff.

Query by:
  - Disease category
  - Body part
  - Season
  - Protocol type (treatment, diet, training, etc.)
  - Severity level
  - Free-text semantic search
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import get_db
from app.models.integration import Protocol
from app.schemas.zoo import (
    ProtocolCreate,
    ProtocolUpdate,
    ProtocolResponse,
    ProtocolListResponse,
    ProtocolSearchRequest,
)
from app.services.embedding import generate_embedding_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/zoo", tags=["zoo"])


@router.post("/protocols", response_model=ProtocolResponse, status_code=201)
async def create_protocol(
    payload: ProtocolCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new elephant care protocol."""
    from app.services.embedding import generate_embedding

    protocol = Protocol(
        id=str(uuid.uuid4()),
        **payload.model_dump(exclude_none=True),
    )

    # Generate embedding from protocol text
    embed_text = f"{payload.title} {payload.condition} {payload.treatment or ''}"
    try:
        protocol.content_embedding = await generate_embedding(embed_text)
    except Exception as e:
        logger.warning("Failed to generate protocol embedding: %s", e)

    db.add(protocol)
    await db.flush()
    return protocol


@router.get("/protocols", response_model=ProtocolListResponse)
async def list_protocols(
    protocol_type: str | None = None,
    disease_category: str | None = None,
    body_part: str | None = None,
    season: str | None = None,
    severity: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    List protocols with optional filters.

    Filterable by: protocol_type, disease_category, body_part, season, severity.
    """
    query = select(Protocol)
    count_query = select(func.count()).select_from(Protocol)

    if protocol_type:
        query = query.where(Protocol.protocol_type == protocol_type)
        count_query = count_query.where(Protocol.protocol_type == protocol_type)
    if disease_category:
        query = query.where(Protocol.disease_category == disease_category)
        count_query = count_query.where(Protocol.disease_category == disease_category)
    if body_part:
        query = query.where(Protocol.body_part == body_part)
        count_query = count_query.where(Protocol.body_part == body_part)
    if season:
        query = query.where(Protocol.season == season)
        count_query = count_query.where(Protocol.season == season)
    if severity:
        query = query.where(Protocol.severity == severity)
        count_query = count_query.where(Protocol.severity == severity)

    query = query.order_by(Protocol.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    protocols = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return ProtocolListResponse(protocols=protocols, total=total)


@router.post("/protocols/search", response_model=ProtocolListResponse)
async def search_protocols(
    request: ProtocolSearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Search protocols by semantic query and/or structured filters.

    Combines vector similarity search with SQL filtering.
    """
    from sqlalchemy import text as sa_text

    filters = []
    params = {"limit": request.limit, "offset": request.offset}

    if request.disease_category:
        filters.append("p.disease_category = :disease_category")
        params["disease_category"] = request.disease_category
    if request.body_part:
        filters.append("p.body_part = :body_part")
        params["body_part"] = request.body_part
    if request.season:
        filters.append("p.season = :season")
        params["season"] = request.season
    if request.protocol_type:
        filters.append("p.protocol_type = :protocol_type")
        params["protocol_type"] = request.protocol_type
    if request.severity:
        filters.append("p.severity = :severity")
        params["severity"] = request.severity

    filter_clause = " AND ".join(filters) if filters else "1=1"

    if request.query:
        # Semantic search over protocols
        try:
            query_emb = await generate_embedding_query(request.query)
            params["query_emb"] = str(query_emb)

            sql = f"""
                SELECT p.*,
                       1 - (p.content_embedding <=> :query_emb::vector) AS similarity
                FROM protocols p
                WHERE {filter_clause}
                  AND p.content_embedding IS NOT NULL
                ORDER BY p.content_embedding <=> :query_emb::vector
                LIMIT :limit OFFSET :offset
            """
        except Exception:
            # Fallback to keyword search
            params["like_query"] = f"%{request.query}%"
            sql = f"""
                SELECT p.*, 0.0 AS similarity
                FROM protocols p
                WHERE {filter_clause}
                  AND (p.title ILIKE :like_query
                       OR p.condition ILIKE :like_query
                       OR p.treatment ILIKE :like_query)
                ORDER BY p.created_at DESC
                LIMIT :limit OFFSET :offset
            """
    else:
        sql = f"""
            SELECT p.*, 0.0 AS similarity
            FROM protocols p
            WHERE {filter_clause}
            ORDER BY p.created_at DESC
            LIMIT :limit OFFSET :offset
        """

    result = await db.execute(sa_text(sql), params)
    rows = result.mappings().all()

    protocols = []
    for row in rows:
        protocols.append(ProtocolResponse(
            id=row["id"],
            verse_id=row.get("verse_id"),
            title=row["title"],
            title_sanskrit=row.get("title_sanskrit"),
            protocol_type=row["protocol_type"],
            condition=row["condition"],
            condition_sanskrit=row.get("condition_sanskrit"),
            symptoms=row.get("symptoms"),
            treatment=row.get("treatment"),
            herbs=row.get("herbs"),
            procedure_steps=row.get("procedure_steps"),
            precautions=row.get("precautions"),
            body_part=row.get("body_part"),
            season=row.get("season"),
            severity=row.get("severity"),
            disease_category=row.get("disease_category"),
            is_reviewed=row.get("is_reviewed", False),
            confidence=row.get("confidence", 0.0),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        ))

    return ProtocolListResponse(protocols=protocols, total=len(protocols))


@router.get("/protocols/{protocol_id}", response_model=ProtocolResponse)
async def get_protocol(
    protocol_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific protocol by ID."""
    protocol = await db.get(Protocol, protocol_id)
    if not protocol:
        raise HTTPException(status_code=404, detail="Protocol not found")
    return protocol


@router.patch("/protocols/{protocol_id}", response_model=ProtocolResponse)
async def update_protocol(
    protocol_id: str,
    payload: ProtocolUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a protocol."""
    protocol = await db.get(Protocol, protocol_id)
    if not protocol:
        raise HTTPException(status_code=404, detail="Protocol not found")

    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(protocol, key, value)

    await db.flush()
    return protocol


@router.delete("/protocols/{protocol_id}", status_code=204)
async def delete_protocol(
    protocol_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a protocol."""
    protocol = await db.get(Protocol, protocol_id)
    if not protocol:
        raise HTTPException(status_code=404, detail="Protocol not found")
    await db.delete(protocol)


# ---------------------------------------------------------------------------
# Aggregation endpoints
# ---------------------------------------------------------------------------

@router.get("/categories")
async def list_categories(db: AsyncSession = Depends(get_db)):
    """List all unique disease categories with counts."""
    from sqlalchemy import text as sa_text
    result = await db.execute(sa_text("""
        SELECT disease_category, COUNT(*) AS count
        FROM protocols
        WHERE disease_category IS NOT NULL
        GROUP BY disease_category
        ORDER BY count DESC
    """))
    return [{"category": row[0], "count": row[1]} for row in result.all()]


@router.get("/body-parts")
async def list_body_parts(db: AsyncSession = Depends(get_db)):
    """List all unique body parts with counts."""
    from sqlalchemy import text as sa_text
    result = await db.execute(sa_text("""
        SELECT body_part, COUNT(*) AS count
        FROM protocols
        WHERE body_part IS NOT NULL
        GROUP BY body_part
        ORDER BY count DESC
    """))
    return [{"body_part": row[0], "count": row[1]} for row in result.all()]


@router.get("/seasons")
async def list_seasons(db: AsyncSession = Depends(get_db)):
    """List all seasons referenced in protocols with counts."""
    from sqlalchemy import text as sa_text
    result = await db.execute(sa_text("""
        SELECT season, COUNT(*) AS count
        FROM protocols
        WHERE season IS NOT NULL
        GROUP BY season
        ORDER BY count DESC
    """))
    return [{"season": row[0], "count": row[1]} for row in result.all()]


@router.get("/herbs")
async def list_herbs(db: AsyncSession = Depends(get_db)):
    """List all unique herbs referenced across protocols."""
    from sqlalchemy import text as sa_text
    result = await db.execute(sa_text("""
        SELECT DISTINCT jsonb_array_elements_text(herbs) AS herb
        FROM protocols
        WHERE herbs IS NOT NULL AND jsonb_array_length(herbs) > 0
        ORDER BY herb
    """))
    return [{"herb": row[0]} for row in result.all()]
