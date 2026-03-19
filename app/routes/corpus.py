"""
CRUD routes for corpus entities: texts, chapters, verses, words.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.base import get_db
from app.models.corpus import Text, Chapter, Verse, Word
from app.schemas.corpus import (
    TextCreate, TextUpdate, TextResponse, TextListResponse,
    ChapterCreate, ChapterUpdate, ChapterResponse, ChapterListResponse,
    VerseCreate, VerseUpdate, VerseResponse, VerseListResponse,
    WordCreate, WordResponse, WordListResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/corpus", tags=["corpus"])


# ---------------------------------------------------------------------------
# Texts
# ---------------------------------------------------------------------------

@router.post("/texts", response_model=TextResponse, status_code=201)
async def create_text(
    payload: TextCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new text record."""
    text = Text(id=str(uuid.uuid4()), **payload.model_dump(exclude_none=True))
    db.add(text)
    await db.flush()
    return text


@router.get("/texts", response_model=TextListResponse)
async def list_texts(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List all texts."""
    result = await db.execute(
        select(Text).order_by(Text.created_at.desc()).offset(offset).limit(limit)
    )
    texts = result.scalars().all()

    count_result = await db.execute(select(func.count()).select_from(Text))
    total = count_result.scalar() or 0

    return TextListResponse(texts=texts, total=total)


@router.get("/texts/{text_id}", response_model=TextResponse)
async def get_text(text_id: str, db: AsyncSession = Depends(get_db)):
    """Get a text by ID."""
    text = await db.get(Text, text_id)
    if not text:
        raise HTTPException(status_code=404, detail="Text not found")
    return text


@router.patch("/texts/{text_id}", response_model=TextResponse)
async def update_text(
    text_id: str,
    payload: TextUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a text record."""
    text = await db.get(Text, text_id)
    if not text:
        raise HTTPException(status_code=404, detail="Text not found")

    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(text, key, value)

    await db.flush()
    return text


@router.delete("/texts/{text_id}", status_code=204)
async def delete_text(text_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a text and all its chapters/verses (cascading)."""
    text = await db.get(Text, text_id)
    if not text:
        raise HTTPException(status_code=404, detail="Text not found")
    await db.delete(text)


# ---------------------------------------------------------------------------
# Chapters
# ---------------------------------------------------------------------------

@router.post("/chapters", response_model=ChapterResponse, status_code=201)
async def create_chapter(
    payload: ChapterCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new chapter."""
    # Verify text exists
    text = await db.get(Text, payload.text_id)
    if not text:
        raise HTTPException(status_code=404, detail="Text not found")

    chapter = Chapter(id=str(uuid.uuid4()), **payload.model_dump(exclude_none=True))
    db.add(chapter)
    await db.flush()
    return chapter


@router.get("/texts/{text_id}/chapters", response_model=ChapterListResponse)
async def list_chapters(
    text_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List chapters for a text, ordered by position."""
    result = await db.execute(
        select(Chapter)
        .where(Chapter.text_id == text_id)
        .order_by(Chapter.order)
    )
    chapters = result.scalars().all()

    return ChapterListResponse(chapters=chapters, total=len(chapters))


@router.get("/chapters/{chapter_id}", response_model=ChapterResponse)
async def get_chapter(chapter_id: str, db: AsyncSession = Depends(get_db)):
    """Get a chapter by ID."""
    chapter = await db.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    return chapter


@router.patch("/chapters/{chapter_id}", response_model=ChapterResponse)
async def update_chapter(
    chapter_id: str,
    payload: ChapterUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a chapter."""
    chapter = await db.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(chapter, key, value)

    await db.flush()
    return chapter


# ---------------------------------------------------------------------------
# Verses
# ---------------------------------------------------------------------------

@router.post("/verses", response_model=VerseResponse, status_code=201)
async def create_verse(
    payload: VerseCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new verse."""
    chapter = await db.get(Chapter, payload.chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    verse = Verse(id=str(uuid.uuid4()), **payload.model_dump(exclude_none=True))
    db.add(verse)
    await db.flush()
    return verse


@router.get("/chapters/{chapter_id}/verses", response_model=VerseListResponse)
async def list_verses(
    chapter_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    verse_type: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List verses in a chapter, ordered by position."""
    query = select(Verse).where(Verse.chapter_id == chapter_id)
    count_query = select(func.count()).select_from(Verse).where(
        Verse.chapter_id == chapter_id
    )

    if verse_type:
        query = query.where(Verse.verse_type == verse_type)
        count_query = count_query.where(Verse.verse_type == verse_type)

    query = query.order_by(Verse.order).offset(offset).limit(limit)

    result = await db.execute(query)
    verses = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return VerseListResponse(verses=verses, total=total)


@router.get("/verses/{verse_id}", response_model=VerseResponse)
async def get_verse(verse_id: str, db: AsyncSession = Depends(get_db)):
    """Get a verse by ID."""
    verse = await db.get(Verse, verse_id)
    if not verse:
        raise HTTPException(status_code=404, detail="Verse not found")
    return verse


@router.patch("/verses/{verse_id}", response_model=VerseResponse)
async def update_verse(
    verse_id: str,
    payload: VerseUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a verse."""
    verse = await db.get(Verse, verse_id)
    if not verse:
        raise HTTPException(status_code=404, detail="Verse not found")

    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(verse, key, value)

    await db.flush()
    return verse


# ---------------------------------------------------------------------------
# Words (word-by-word breakdown)
# ---------------------------------------------------------------------------

@router.post("/words", response_model=WordResponse, status_code=201)
async def create_word(
    payload: WordCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a word-by-word breakdown entry."""
    verse = await db.get(Verse, payload.verse_id)
    if not verse:
        raise HTTPException(status_code=404, detail="Verse not found")

    word = Word(id=str(uuid.uuid4()), **payload.model_dump(exclude_none=True))
    db.add(word)
    await db.flush()
    return word


@router.get("/verses/{verse_id}/words", response_model=WordListResponse)
async def list_words(verse_id: str, db: AsyncSession = Depends(get_db)):
    """Get word-by-word breakdown for a verse."""
    result = await db.execute(
        select(Word)
        .where(Word.verse_id == verse_id)
        .order_by(Word.position)
    )
    words = result.scalars().all()
    return WordListResponse(words=words, total=len(words))
