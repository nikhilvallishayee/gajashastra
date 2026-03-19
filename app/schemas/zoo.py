"""
Pydantic schemas for zoo management integration endpoints.
"""

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ProtocolCreate(BaseModel):
    """Create a care protocol."""

    verse_id: str | None = None
    title: str = Field(..., min_length=1, max_length=500)
    title_sanskrit: str | None = None
    protocol_type: str = Field(
        ...,
        description="treatment, diet, training, daily_care, seasonal, emergency",
    )
    condition: str = Field(..., min_length=1)
    condition_sanskrit: str | None = None
    symptoms: list[str] | None = None
    treatment: str | None = None
    herbs: list[str] | None = None
    procedure_steps: list[str] | None = None
    precautions: list[str] | None = None
    body_part: str | None = None
    season: str | None = None
    severity: str | None = None
    disease_category: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata_json: dict | None = None


class ProtocolUpdate(BaseModel):
    title: str | None = None
    protocol_type: str | None = None
    condition: str | None = None
    treatment: str | None = None
    herbs: list[str] | None = None
    procedure_steps: list[str] | None = None
    precautions: list[str] | None = None
    body_part: str | None = None
    season: str | None = None
    severity: str | None = None
    disease_category: str | None = None
    is_reviewed: bool | None = None
    confidence: float | None = None


class ProtocolResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    verse_id: str | None = None
    title: str
    title_sanskrit: str | None = None
    protocol_type: str
    condition: str
    condition_sanskrit: str | None = None
    symptoms: list | None = None
    treatment: str | None = None
    herbs: list | None = None
    procedure_steps: list | None = None
    precautions: list | None = None
    body_part: str | None = None
    season: str | None = None
    severity: str | None = None
    disease_category: str | None = None
    is_reviewed: bool
    confidence: float
    created_at: datetime
    updated_at: datetime


class ProtocolListResponse(BaseModel):
    protocols: list[ProtocolResponse]
    total: int


class ProtocolSearchRequest(BaseModel):
    """Search protocols by various criteria."""

    query: str | None = None
    disease_category: str | None = None
    body_part: str | None = None
    season: str | None = None
    protocol_type: str | None = None
    severity: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
