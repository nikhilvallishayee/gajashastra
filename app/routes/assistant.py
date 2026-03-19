"""
Assistant routes: conversational RAG for Q&A over the Gajashastra.

Supports:
  - Non-streaming chat (POST /chat)
  - Streaming chat via SSE (POST /chat/stream)
  - Session management
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import get_db
from app.schemas.assistant import (
    ChatRequest,
    ChatResponse,
    SessionResponse,
    SessionListResponse,
    MessageResponse,
    SessionMessagesResponse,
    SourceCitation,
)
from app.services.assistant import AssistantService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and get a response with source citations.

    The assistant retrieves relevant Gajashastra verses via hybrid search
    and generates a contextual response with citations.
    """
    service = AssistantService(db)
    try:
        result = await service.chat(
            message=request.message,
            session_id=request.session_id,
            max_sources=request.max_sources,
            include_sanskrit=request.include_sanskrit,
        )
        return ChatResponse(
            session_id=result["session_id"],
            message=result["message"],
            sources=result["sources"],
            metadata=result["metadata"],
        )
    except Exception as e:
        logger.error("Chat failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and stream the response via Server-Sent Events.

    Events:
      - {"type": "sources", "data": [...]}  - Source citations
      - {"type": "text", "data": "..."}     - Text chunk
      - {"type": "done", "data": {...}}     - Completion signal
    """
    service = AssistantService(db)

    async def event_generator():
        try:
            async for event in service.chat_stream(
                message=request.message,
                session_id=request.session_id,
                max_sources=request.max_sources,
                include_sanskrit=request.include_sanskrit,
            ):
                yield f"data: {event}\n\n"
        except Exception as e:
            logger.error("Stream failed: %s", e, exc_info=True)
            import json
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List conversation sessions."""
    service = AssistantService(db)
    sessions, total = await service.list_sessions(
        user_id=user_id, limit=limit, offset=offset
    )
    return SessionListResponse(sessions=sessions, total=total)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get session info."""
    from app.models.integration import AssistantSession
    session = await db.get(AssistantSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Get messages for a session."""
    service = AssistantService(db)
    messages = await service.get_session_messages(session_id, limit=limit)
    return SessionMessagesResponse(messages=messages, total=len(messages))


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a session and all its messages."""
    from app.models.integration import AssistantSession
    session = await db.get(AssistantSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
