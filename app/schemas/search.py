"""
Pydantic schemas for search endpoints.
"""

from pydantic import BaseModel, Field, ConfigDict


class SearchRequest(BaseModel):
    """Request for hybrid search."""

    query: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field(
        default="hybrid",
        description="Search mode: hybrid, semantic, keyword",
    )
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    # Filters
    chapter_id: str | None = None
    text_id: str | None = None
    verse_type: str | None = None

    # Search tuning
    vector_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    keyword_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    mmr_lambda: float | None = Field(default=None, ge=0.0, le=1.0)

    # Script preference for results
    script: str = Field(
        default="devanagari",
        description="Preferred script for results: devanagari, iast",
    )


class SearchResult(BaseModel):
    """A single search result."""

    model_config = ConfigDict(from_attributes=True)

    verse_id: str
    chapter_id: str | None = None
    verse_number: str | None = None
    chapter_title: str | None = None

    sanskrit_devanagari: str
    sanskrit_iast: str | None = None
    english_summary: str | None = None
    commentary: str | None = None
    verse_type: str | None = None
    page_number: int | None = None

    # Scores
    score: float = 0.0
    vector_score: float | None = None
    keyword_score: float | None = None
    rrf_score: float | None = None

    # Chunk info (if result came from embedding chunks)
    chunk_text: str | None = None
    chunk_index: int | None = None

    # Highlight for keyword matches
    highlight: str | None = None


class SearchResponse(BaseModel):
    """Search results response."""

    query: str
    mode: str
    results: list[SearchResult]
    total: int
    limit: int
    offset: int
    search_time_ms: float | None = None
    fallback_used: str | None = None


class SimilarVersesRequest(BaseModel):
    """Find verses similar to a given verse."""

    verse_id: str
    limit: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
