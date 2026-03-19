"""
Reference book browsing routes.

Provides a structured reading experience for the Gajashastra:
  - Browse by chapter/section structure
  - Word-by-word breakdown
  - Cross-reference navigation
  - Bookmark/annotation support
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import get_db
from app.models.corpus import Text, Chapter, Verse, Word
from app.models.knowledge import CrossReference, Insight

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/reference", tags=["reference"])


# ---------------------------------------------------------------------------
# Table of contents / Navigation
# ---------------------------------------------------------------------------

@router.get("/texts/{text_id}/toc")
async def get_table_of_contents(
    text_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get full table of contents for a text.

    Returns hierarchical structure: text -> chapters with verse counts.
    """
    text = await db.get(Text, text_id)
    if not text:
        raise HTTPException(status_code=404, detail="Text not found")

    result = await db.execute(
        select(
            Chapter.id,
            Chapter.title,
            Chapter.title_devanagari,
            Chapter.order,
            Chapter.page_start,
            Chapter.page_end,
            Chapter.summary,
            func.count(Verse.id).label("verse_count"),
        )
        .outerjoin(Verse, Verse.chapter_id == Chapter.id)
        .where(Chapter.text_id == text_id)
        .group_by(Chapter.id)
        .order_by(Chapter.order)
    )
    rows = result.all()

    chapters = []
    for row in rows:
        chapters.append({
            "id": row[0],
            "title": row[1],
            "title_devanagari": row[2],
            "order": row[3],
            "page_start": row[4],
            "page_end": row[5],
            "summary": row[6],
            "verse_count": row[7],
        })

    return {
        "text_id": text.id,
        "title": text.title,
        "title_devanagari": text.title_devanagari,
        "author": text.author,
        "total_chapters": len(chapters),
        "total_verses": sum(c["verse_count"] for c in chapters),
        "chapters": chapters,
    }


# ---------------------------------------------------------------------------
# Reading view
# ---------------------------------------------------------------------------

@router.get("/chapters/{chapter_id}/read")
async def read_chapter(
    chapter_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Read a chapter with paginated verses.

    Returns verses in order with their translations and metadata,
    designed for a reading/study interface.
    """
    chapter = await db.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    offset = (page - 1) * page_size

    # Get verses
    result = await db.execute(
        select(Verse)
        .where(Verse.chapter_id == chapter_id)
        .order_by(Verse.order)
        .offset(offset)
        .limit(page_size)
    )
    verses = result.scalars().all()

    # Get total count
    count_result = await db.execute(
        select(func.count()).where(Verse.chapter_id == chapter_id)
    )
    total = count_result.scalar() or 0
    total_pages = (total + page_size - 1) // page_size

    verse_data = []
    for verse in verses:
        verse_data.append({
            "id": verse.id,
            "verse_number": verse.verse_number,
            "order": verse.order,
            "sanskrit_devanagari": verse.sanskrit_devanagari,
            "sanskrit_iast": verse.sanskrit_iast,
            "english_translation": verse.english_translation,
            "english_summary": verse.english_summary,
            "commentary": verse.commentary,
            "verse_type": verse.verse_type,
            "meter": verse.meter,
            "page_number": verse.page_number,
        })

    return {
        "chapter_id": chapter.id,
        "chapter_title": chapter.title,
        "chapter_title_devanagari": chapter.title_devanagari,
        "verses": verse_data,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_verses": total,
            "total_pages": total_pages,
        },
    }


# ---------------------------------------------------------------------------
# Word-by-word breakdown
# ---------------------------------------------------------------------------

@router.get("/verses/{verse_id}/breakdown")
async def verse_breakdown(
    verse_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get word-by-word grammatical breakdown for a verse.

    Returns the verse text with each word analyzed:
      - Root (dhatu)
      - Grammatical form
      - Part of speech
      - English meaning
      - Compound analysis (if applicable)
    """
    verse = await db.get(Verse, verse_id)
    if not verse:
        raise HTTPException(status_code=404, detail="Verse not found")

    # Get word breakdown
    result = await db.execute(
        select(Word)
        .where(Word.verse_id == verse_id)
        .order_by(Word.position)
    )
    words = result.scalars().all()

    word_data = []
    for word in words:
        entry = {
            "position": word.position,
            "word_devanagari": word.word_devanagari,
            "word_iast": word.word_iast,
            "root": word.root,
            "root_meaning": word.root_meaning,
            "grammatical_form": word.grammatical_form,
            "part_of_speech": word.part_of_speech,
            "case": word.case,
            "number": word.number,
            "gender": word.gender,
            "tense": word.tense,
            "english_meaning": word.english_meaning,
        }
        if word.is_compound:
            entry["compound"] = {
                "type": word.compound_type,
                "parts": word.compound_parts,
            }
        word_data.append(entry)

    return {
        "verse_id": verse.id,
        "verse_number": verse.verse_number,
        "sanskrit_devanagari": verse.sanskrit_devanagari,
        "sanskrit_iast": verse.sanskrit_iast,
        "verse_type": verse.verse_type,
        "meter": verse.meter,
        "words": word_data,
        "has_breakdown": len(word_data) > 0,
    }


# ---------------------------------------------------------------------------
# Cross-references
# ---------------------------------------------------------------------------

@router.get("/verses/{verse_id}/cross-references")
async def get_cross_references(
    verse_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get all cross-references for a verse.

    Returns both outgoing (this verse references X) and
    incoming (X references this verse) links.
    """
    verse = await db.get(Verse, verse_id)
    if not verse:
        raise HTTPException(status_code=404, detail="Verse not found")

    # Outgoing references
    outgoing_result = await db.execute(
        select(CrossReference)
        .where(CrossReference.source_verse_id == verse_id)
    )
    outgoing = outgoing_result.scalars().all()

    # Incoming references
    incoming_result = await db.execute(
        select(CrossReference)
        .where(CrossReference.target_verse_id == verse_id)
    )
    incoming = incoming_result.scalars().all()

    def _ref_to_dict(ref, direction: str) -> dict:
        other_verse = ref.target_verse if direction == "outgoing" else ref.source_verse
        return {
            "id": ref.id,
            "direction": direction,
            "reference_type": ref.reference_type,
            "description": ref.description,
            "external_reference": ref.external_reference,
            "similarity_score": ref.similarity_score,
            "linked_verse": {
                "id": other_verse.id if other_verse else None,
                "verse_number": other_verse.verse_number if other_verse else None,
                "sanskrit_devanagari": other_verse.sanskrit_devanagari if other_verse else None,
                "english_summary": other_verse.english_summary if other_verse else None,
            } if other_verse else None,
        }

    return {
        "verse_id": verse_id,
        "outgoing": [_ref_to_dict(r, "outgoing") for r in outgoing],
        "incoming": [_ref_to_dict(r, "incoming") for r in incoming],
        "total": len(outgoing) + len(incoming),
    }


# ---------------------------------------------------------------------------
# Insights for a verse
# ---------------------------------------------------------------------------

@router.get("/verses/{verse_id}/insights")
async def get_verse_insights(
    verse_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get all extracted insights for a specific verse."""
    verse = await db.get(Verse, verse_id)
    if not verse:
        raise HTTPException(status_code=404, detail="Verse not found")

    result = await db.execute(
        select(Insight)
        .where(Insight.verse_id == verse_id)
        .order_by(Insight.confidence.desc())
    )
    insights = result.scalars().all()

    return {
        "verse_id": verse_id,
        "insights": [
            {
                "id": i.id,
                "type": i.insight_type,
                "content": i.content,
                "content_sanskrit": i.content_sanskrit,
                "confidence": i.confidence,
                "category": i.category,
                "subcategory": i.subcategory,
                "tags": i.tags,
                "is_reviewed": i.is_reviewed,
                "is_approved": i.is_approved,
            }
            for i in insights
        ],
        "total": len(insights),
    }


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

@router.get("/verses/{verse_id}/adjacent")
async def get_adjacent_verses(
    verse_id: str,
    context: int = Query(default=2, ge=0, le=10, description="Verses before/after"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get verses adjacent to the given verse (for contextual reading).

    Returns `context` verses before and after the target verse.
    """
    verse = await db.get(Verse, verse_id)
    if not verse:
        raise HTTPException(status_code=404, detail="Verse not found")

    # Get surrounding verses by order in the same chapter
    result = await db.execute(
        select(Verse)
        .where(
            Verse.chapter_id == verse.chapter_id,
            Verse.order >= verse.order - context,
            Verse.order <= verse.order + context,
        )
        .order_by(Verse.order)
    )
    verses = result.scalars().all()

    return {
        "target_verse_id": verse_id,
        "target_order": verse.order,
        "verses": [
            {
                "id": v.id,
                "verse_number": v.verse_number,
                "order": v.order,
                "sanskrit_devanagari": v.sanskrit_devanagari,
                "sanskrit_iast": v.sanskrit_iast,
                "english_summary": v.english_summary,
                "verse_type": v.verse_type,
                "is_target": v.id == verse_id,
            }
            for v in verses
        ],
    }
