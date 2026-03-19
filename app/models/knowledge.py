"""
Knowledge extraction models: Insight, Pattern, CrossReference.

These represent the "wisdom layer" extracted from raw verses:
  - Insight: a teaching, fact, or principle extracted from a verse
  - Pattern: a recurring theme or practice across multiple verses
  - CrossReference: links between related verses
"""

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    Text as SQLText,
    Boolean,
)
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin
from app.config import get_settings

_dim = get_settings().embedding_dimension


class Insight(Base, UUIDMixin, TimestampMixin):
    """
    A structured insight extracted from one or more verses.

    Types:
      - teaching: a principle or maxim from the text
      - application: a practical procedure (e.g., elephant care protocol)
      - classification: a taxonomy entry (e.g., types of elephants)
      - observation: a factual observation about elephants
      - remedy: a medical treatment or remedy
    """

    __tablename__ = "insights"

    # Source verse(s)
    verse_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("verses.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    insight_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # teaching, application, classification, observation, remedy
    content: Mapped[str] = mapped_column(SQLText, nullable=False)
    content_sanskrit: Mapped[str | None] = mapped_column(SQLText)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Categorization
    category: Mapped[str | None] = mapped_column(
        String(100)
    )  # e.g., "anatomy", "disease", "training", "mythology"
    subcategory: Mapped[str | None] = mapped_column(String(100))
    tags: Mapped[dict | None] = mapped_column(JSONB, default=list)

    # For deduplication and retrieval
    content_embedding: Mapped[list | None] = mapped_column(Vector(_dim))

    # Extraction metadata
    extraction_model: Mapped[str | None] = mapped_column(String(100))
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    is_approved: Mapped[bool | None] = mapped_column(Boolean)

    metadata_json: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    # Relationships
    verse: Mapped["Verse"] = relationship("Verse", lazy="joined")


class Pattern(Base, UUIDMixin, TimestampMixin):
    """
    A recurring pattern or theme across multiple insights.

    Examples:
      - "Seasonal feeding patterns" (appears in multiple chapters)
      - "Musth management protocols" (treatment sequence)
      - "Classification by body type" (jati/kula taxonomy)
    """

    __tablename__ = "patterns"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(SQLText, nullable=False)
    pattern_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # recurring_theme, protocol_sequence, taxonomy, cross_reference
    frequency: Mapped[int] = mapped_column(Integer, default=1)

    # Related insight IDs stored as JSON array
    insight_ids: Mapped[dict | None] = mapped_column(JSONB, default=list)
    # Related verse IDs
    verse_ids: Mapped[dict | None] = mapped_column(JSONB, default=list)

    # For semantic search
    description_embedding: Mapped[list | None] = mapped_column(Vector(_dim))

    metadata_json: Mapped[dict | None] = mapped_column(JSONB, default=dict)


class CrossReference(Base, UUIDMixin, TimestampMixin):
    """
    Explicit link between two verses or between a verse and an external source.

    Types:
      - internal: link between two verses in the same text
      - commentary: link between base text and commentary
      - parallel: link to a parallel passage in another text
      - citation: link to a secondary source or modern reference
    """

    __tablename__ = "cross_references"

    source_verse_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("verses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_verse_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("verses.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    reference_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # internal, commentary, parallel, citation
    description: Mapped[str | None] = mapped_column(SQLText)
    external_reference: Mapped[str | None] = mapped_column(
        SQLText
    )  # URL or bibliographic reference
    similarity_score: Mapped[float | None] = mapped_column(Float)

    metadata_json: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    # Relationships
    source_verse: Mapped["Verse"] = relationship(
        "Verse", foreign_keys=[source_verse_id], lazy="joined"
    )
    target_verse: Mapped["Verse"] = relationship(
        "Verse", foreign_keys=[target_verse_id], lazy="joined"
    )
