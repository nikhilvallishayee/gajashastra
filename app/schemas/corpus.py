"""
Pydantic schemas for corpus entities: Text, Chapter, Verse, Word.
"""

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

class TextCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    title_devanagari: str | None = None
    author: str | None = None
    author_devanagari: str | None = None
    language: str = "sanskrit"
    description: str | None = None
    source_url: str | None = None
    total_pages: int | None = None
    metadata_json: dict | None = None


class TextUpdate(BaseModel):
    title: str | None = None
    title_devanagari: str | None = None
    author: str | None = None
    description: str | None = None
    source_url: str | None = None
    total_pages: int | None = None
    total_chapters: int | None = None
    total_verses: int | None = None
    metadata_json: dict | None = None


class TextResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    title_devanagari: str | None = None
    author: str | None = None
    author_devanagari: str | None = None
    language: str
    description: str | None = None
    source_url: str | None = None
    total_pages: int | None = None
    total_chapters: int | None = None
    total_verses: int | None = None
    created_at: datetime
    updated_at: datetime


class TextListResponse(BaseModel):
    texts: list[TextResponse]
    total: int


# ---------------------------------------------------------------------------
# Chapter
# ---------------------------------------------------------------------------

class ChapterCreate(BaseModel):
    text_id: str
    title: str = Field(..., min_length=1, max_length=500)
    title_devanagari: str | None = None
    order: int = Field(..., ge=0)
    page_start: int | None = None
    page_end: int | None = None
    summary: str | None = None
    metadata_json: dict | None = None


class ChapterUpdate(BaseModel):
    title: str | None = None
    title_devanagari: str | None = None
    order: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    summary: str | None = None
    metadata_json: dict | None = None


class ChapterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    text_id: str
    title: str
    title_devanagari: str | None = None
    order: int
    page_start: int | None = None
    page_end: int | None = None
    summary: str | None = None
    created_at: datetime
    updated_at: datetime


class ChapterListResponse(BaseModel):
    chapters: list[ChapterResponse]
    total: int


# ---------------------------------------------------------------------------
# Verse
# ---------------------------------------------------------------------------

class VerseCreate(BaseModel):
    chapter_id: str
    verse_number: str | None = None
    order: int = Field(..., ge=0)
    sanskrit_devanagari: str = Field(..., min_length=1)
    sanskrit_iast: str | None = None
    english_translation: str | None = None
    english_summary: str | None = None
    commentary: str | None = None
    verse_type: str = "verse"
    meter: str | None = None
    page_number: int | None = None
    source_file: str | None = None
    extraction_model: str | None = None
    extraction_confidence: float | None = None
    metadata_json: dict | None = None


class VerseUpdate(BaseModel):
    verse_number: str | None = None
    sanskrit_devanagari: str | None = None
    sanskrit_iast: str | None = None
    english_translation: str | None = None
    english_summary: str | None = None
    commentary: str | None = None
    verse_type: str | None = None
    meter: str | None = None
    metadata_json: dict | None = None


class VerseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    chapter_id: str
    verse_number: str | None = None
    order: int
    sanskrit_devanagari: str
    sanskrit_iast: str | None = None
    english_translation: str | None = None
    english_summary: str | None = None
    commentary: str | None = None
    verse_type: str
    meter: str | None = None
    page_number: int | None = None
    is_indexed: bool
    created_at: datetime
    updated_at: datetime


class VerseListResponse(BaseModel):
    verses: list[VerseResponse]
    total: int


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------

class WordCreate(BaseModel):
    verse_id: str
    position: int = Field(..., ge=0)
    word_devanagari: str = Field(..., min_length=1)
    word_iast: str | None = None
    root: str | None = None
    root_meaning: str | None = None
    grammatical_form: str | None = None
    part_of_speech: str | None = None
    case: str | None = None
    number: str | None = None
    gender: str | None = None
    tense: str | None = None
    is_compound: bool = False
    compound_type: str | None = None
    compound_parts: dict | None = None
    english_meaning: str | None = None


class WordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    verse_id: str
    position: int
    word_devanagari: str
    word_iast: str | None = None
    root: str | None = None
    root_meaning: str | None = None
    grammatical_form: str | None = None
    part_of_speech: str | None = None
    case: str | None = None
    number: str | None = None
    gender: str | None = None
    tense: str | None = None
    is_compound: bool
    compound_type: str | None = None
    compound_parts: dict | None = None
    english_meaning: str | None = None


class WordListResponse(BaseModel):
    words: list[WordResponse]
    total: int
