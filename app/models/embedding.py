"""
Embedding and search index models.

Stores vector embeddings for verses / chunks for semantic search,
plus a SearchIndex model for tracking batch embedding jobs.
"""

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text as SQLText,
    DateTime,
)
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin
from app.config import get_settings


_dim = get_settings().embedding_dimension


class VerseEmbedding(Base, UUIDMixin, TimestampMixin):
    """
    Vector embedding for a verse or text chunk.

    A single verse may produce one or more embeddings depending on
    chunking strategy (short verses = 1 embedding, long prose = multiple).
    """

    __tablename__ = "verse_embeddings"

    verse_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("verses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # The text that was embedded (may be verse + commentary combined)
    chunk_text: Mapped[str] = mapped_column(SQLText, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)

    # Context metadata injected into the chunk for better retrieval
    context_prefix: Mapped[str | None] = mapped_column(
        SQLText
    )  # e.g., "Chapter: Gajachikitsa, Verse 42"

    # The embedding vector (1536 dimensions from gemini-embedding-001)
    embedding: Mapped[list] = mapped_column(Vector(_dim), nullable=False)

    # Embedding metadata
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)

    # For keyword search fallback
    # tsv column added via migration: to_tsvector('simple', chunk_text)

    metadata_json: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    # Relationships
    verse: Mapped["Verse"] = relationship(
        "Verse", lazy="joined"
    )


class SearchIndex(Base, UUIDMixin, TimestampMixin):
    """
    Tracks batch embedding / indexing jobs.

    Useful for monitoring progress when embedding the full 457-page corpus.
    """

    __tablename__ = "search_indexes"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending, processing, completed, failed
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    processed_items: Mapped[int] = mapped_column(Integer, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(SQLText)
    started_at: Mapped[str | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[str | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, default=dict)
