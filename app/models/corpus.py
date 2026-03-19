"""
Corpus models: Text, Chapter, Verse, Word.

Represents the hierarchical structure of the Gajashastra manuscript:
  Text -> Chapter -> Verse -> Word (optional word-by-word breakdown)
"""

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text as SQLText,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class Text(Base, UUIDMixin, TimestampMixin):
    """
    A complete text / manuscript.

    For the Gajashastra project this is typically one text, but the model
    supports multiple texts (e.g., Palakapya's Gajashastra,
    Vyasa's Gajashastra, Brahmananda Purana sections).
    """

    __tablename__ = "texts"

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    title_devanagari: Mapped[str | None] = mapped_column(String(500))
    author: Mapped[str | None] = mapped_column(String(255))
    author_devanagari: Mapped[str | None] = mapped_column(String(255))
    language: Mapped[str] = mapped_column(String(50), default="sanskrit")
    description: Mapped[str | None] = mapped_column(SQLText)
    source_url: Mapped[str | None] = mapped_column(String(1000))
    total_pages: Mapped[int | None] = mapped_column(Integer)
    total_chapters: Mapped[int | None] = mapped_column(Integer)
    total_verses: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    # Relationships
    chapters: Mapped[list["Chapter"]] = relationship(
        back_populates="text", cascade="all, delete-orphan", order_by="Chapter.order"
    )


class Chapter(Base, UUIDMixin, TimestampMixin):
    """
    A chapter (adhyaya / prakarana) within a text.

    Maps to sections like:
      - upoddhata (introduction)
      - prathamam prakaranam (first chapter)
      - gajachikitsa (elephant medicine)
    """

    __tablename__ = "chapters"

    text_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("texts.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    title_devanagari: Mapped[str | None] = mapped_column(String(500))
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    summary: Mapped[str | None] = mapped_column(SQLText)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    # Relationships
    text: Mapped["Text"] = relationship(back_populates="chapters")
    verses: Mapped[list["Verse"]] = relationship(
        back_populates="chapter",
        cascade="all, delete-orphan",
        order_by="Verse.order",
    )


class Verse(Base, UUIDMixin, TimestampMixin):
    """
    A verse (shloka), prose block (gadya), or other text unit.

    Each verse stores both Devanagari and IAST transliteration,
    plus optional English translation and commentary.
    """

    __tablename__ = "verses"

    chapter_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False
    )
    verse_number: Mapped[str | None] = mapped_column(String(50))
    order: Mapped[int] = mapped_column(Integer, nullable=False)

    # Sanskrit text in multiple scripts
    sanskrit_devanagari: Mapped[str] = mapped_column(SQLText, nullable=False)
    sanskrit_iast: Mapped[str | None] = mapped_column(SQLText)

    # Translation and commentary
    english_translation: Mapped[str | None] = mapped_column(SQLText)
    english_summary: Mapped[str | None] = mapped_column(SQLText)
    commentary: Mapped[str | None] = mapped_column(SQLText)

    # Classification
    verse_type: Mapped[str] = mapped_column(
        String(50), default="verse"
    )  # verse, prose, chapter_title, commentary, colophon
    meter: Mapped[str | None] = mapped_column(String(100))  # anushtubh, etc.

    # Source tracking
    page_number: Mapped[int | None] = mapped_column(Integer)
    source_file: Mapped[str | None] = mapped_column(String(255))
    extraction_model: Mapped[str | None] = mapped_column(String(100))
    extraction_confidence: Mapped[float | None] = mapped_column(Float)

    # Flexible metadata (sandhi analysis, grammatical notes, etc.)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    # Full-text search support
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    chapter: Mapped["Chapter"] = relationship(back_populates="verses")
    words: Mapped[list["Word"]] = relationship(
        back_populates="verse",
        cascade="all, delete-orphan",
        order_by="Word.position",
    )


class Word(Base, UUIDMixin, TimestampMixin):
    """
    Word-by-word breakdown of a verse.

    Supports pada-paatha style analysis:
      word -> root (dhatu) -> grammatical form -> meaning
    """

    __tablename__ = "words"

    verse_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("verses.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    # The word itself
    word_devanagari: Mapped[str] = mapped_column(String(500), nullable=False)
    word_iast: Mapped[str | None] = mapped_column(String(500))

    # Grammatical analysis
    root: Mapped[str | None] = mapped_column(String(255))  # dhatu
    root_meaning: Mapped[str | None] = mapped_column(String(500))
    grammatical_form: Mapped[str | None] = mapped_column(
        String(255)
    )  # e.g., "nom. sg. masc."
    part_of_speech: Mapped[str | None] = mapped_column(
        String(100)
    )  # noun, verb, adjective...
    case: Mapped[str | None] = mapped_column(String(50))  # vibhakti
    number: Mapped[str | None] = mapped_column(String(20))  # ekavachana, etc.
    gender: Mapped[str | None] = mapped_column(String(20))  # linga
    tense: Mapped[str | None] = mapped_column(String(50))  # lakaara

    # Compound analysis
    is_compound: Mapped[bool] = mapped_column(Boolean, default=False)
    compound_type: Mapped[str | None] = mapped_column(
        String(100)
    )  # tatpurusha, dvandva, etc.
    compound_parts: Mapped[dict | None] = mapped_column(JSONB)

    # Meaning
    english_meaning: Mapped[str | None] = mapped_column(SQLText)

    # Relationships
    verse: Mapped["Verse"] = relationship(back_populates="words")
