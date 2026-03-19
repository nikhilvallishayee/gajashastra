"""
Integration models: Protocol (zoo management), AssistantSession, AssistantMessage.

Protocol stores structured elephant care procedures extracted from the text.
AssistantSession/Message track conversational RAG interactions.
"""

from sqlalchemy import (
    ForeignKey,
    Integer,
    String,
    Text as SQLText,
    Boolean,
    Float,
)
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin
from app.config import get_settings

_dim = get_settings().embedding_dimension


class Protocol(Base, UUIDMixin, TimestampMixin):
    """
    A structured elephant care protocol extracted from the Gajashastra.

    Format:
      condition -> diagnosis -> treatment -> herbs -> procedure -> precautions

    Designed for zoo management / veterinary use.
    """

    __tablename__ = "protocols"

    # Source verse(s)
    verse_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("verses.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Protocol structure
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    title_sanskrit: Mapped[str | None] = mapped_column(String(500))
    protocol_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # treatment, diet, training, daily_care, seasonal, emergency

    # Condition / trigger
    condition: Mapped[str] = mapped_column(SQLText, nullable=False)
    condition_sanskrit: Mapped[str | None] = mapped_column(SQLText)
    symptoms: Mapped[dict | None] = mapped_column(JSONB, default=list)

    # Treatment details
    treatment: Mapped[str | None] = mapped_column(SQLText)
    herbs: Mapped[dict | None] = mapped_column(JSONB, default=list)
    procedure_steps: Mapped[dict | None] = mapped_column(JSONB, default=list)
    precautions: Mapped[dict | None] = mapped_column(JSONB, default=list)

    # Classification
    body_part: Mapped[str | None] = mapped_column(String(100))
    season: Mapped[str | None] = mapped_column(
        String(50)
    )  # grishma, varsha, sharad, etc.
    severity: Mapped[str | None] = mapped_column(
        String(50)
    )  # mild, moderate, severe, critical
    disease_category: Mapped[str | None] = mapped_column(String(100))

    # For semantic search
    content_embedding: Mapped[list | None] = mapped_column(Vector(_dim))

    # Validation
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    metadata_json: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    # Relationships
    verse: Mapped["Verse"] = relationship("Verse", lazy="joined")


class AssistantSession(Base, UUIDMixin, TimestampMixin):
    """
    A conversation session with the Gajashastra assistant.

    Tracks multi-turn conversation state for RAG.
    """

    __tablename__ = "assistant_sessions"

    title: Mapped[str | None] = mapped_column(String(500))
    user_id: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    context_summary: Mapped[str | None] = mapped_column(SQLText)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    # Relationships
    messages: Mapped[list["AssistantMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="AssistantMessage.created_at",
    )


class AssistantMessage(Base, UUIDMixin, TimestampMixin):
    """
    A single message in a conversation session.
    """

    __tablename__ = "assistant_messages"

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("assistant_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # user, assistant, system
    content: Mapped[str] = mapped_column(SQLText, nullable=False)

    # Source citations from RAG
    sources: Mapped[dict | None] = mapped_column(JSONB, default=list)

    # Embedding for conversation search
    content_embedding: Mapped[list | None] = mapped_column(Vector(_dim))

    metadata_json: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    # Relationships
    session: Mapped["AssistantSession"] = relationship(back_populates="messages")
