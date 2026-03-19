"""
Pydantic schemas for the assistant (conversational RAG) endpoints.
"""

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ChatRequest(BaseModel):
    """A user message to the assistant."""

    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str | None = None
    stream: bool = False

    # Context controls
    max_sources: int = Field(default=5, ge=1, le=20)
    include_sanskrit: bool = True
    script: str = Field(default="devanagari", description="devanagari or iast")


class SourceCitation(BaseModel):
    """A source verse cited in the assistant's response."""

    verse_id: str
    verse_number: str | None = None
    chapter_title: str | None = None
    sanskrit_snippet: str | None = None
    english_summary: str | None = None
    relevance_score: float = 0.0


class ChatResponse(BaseModel):
    """Assistant response to a chat message."""

    session_id: str
    message: str
    sources: list[SourceCitation] = []
    is_streaming: bool = False
    metadata: dict | None = None


class SessionResponse(BaseModel):
    """Session info."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str | None = None
    is_active: bool
    message_count: int
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int


class MessageResponse(BaseModel):
    """A single message in a session."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    role: str
    content: str
    sources: list | None = None
    created_at: datetime


class SessionMessagesResponse(BaseModel):
    messages: list[MessageResponse]
    total: int
