"""
Conversational RAG assistant service.

Provides multi-turn Q&A over the Gajashastra corpus with:
  - Context retrieval via hybrid search
  - Source citation (verse references)
  - Conversation memory (multi-turn)
  - Streaming responses via SSE
"""

import json
import logging
import uuid
from typing import AsyncIterator

import anthropic
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.integration import AssistantSession, AssistantMessage
from app.services.search import hybrid_search, SearchHit
from app.schemas.assistant import SourceCitation

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a scholarly assistant specializing in the Gajashastra (गजशास्त्र), the ancient Sanskrit treatise on the science of elephants by Palakapya Muni.

Your role:
- Answer questions about elephant care, training, classification, diseases, and treatments as described in the Gajashastra
- Cite specific verses when answering (use verse numbers and chapter references)
- Explain Sanskrit terminology clearly
- Bridge ancient wisdom with modern veterinary understanding where relevant
- Be honest when the text does not address a question

When citing sources, use this format: [Chapter: X, Verse: Y]

Guidelines:
- Always ground answers in the actual text provided in the context
- If the context does not contain relevant information, say so clearly
- Provide Sanskrit terms alongside English translations
- For medical/treatment questions, always include any precautions mentioned in the text

{context}"""


CONTEXT_TEMPLATE = """<gajashastra_context>
The following verses are the most relevant to the user's question:

{sources}

Use these verses to inform your response. Cite verse references when making claims.
</gajashastra_context>"""


class AssistantService:
    """Conversational RAG over the Gajashastra corpus."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key,
            )
        return self._client

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def get_or_create_session(
        self, session_id: str | None = None, user_id: str | None = None
    ) -> AssistantSession:
        """Get an existing session or create a new one."""
        if session_id:
            session = await self.db.get(AssistantSession, session_id)
            if session:
                return session

        session = AssistantSession(
            id=session_id or str(uuid.uuid4()),
            user_id=user_id,
            is_active=True,
        )
        self.db.add(session)
        await self.db.flush()
        return session

    async def list_sessions(
        self,
        user_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[AssistantSession], int]:
        """List sessions, optionally filtered by user."""
        query = select(AssistantSession)
        count_query = select(func.count()).select_from(AssistantSession)

        if user_id:
            query = query.where(AssistantSession.user_id == user_id)
            count_query = count_query.where(AssistantSession.user_id == user_id)

        query = query.order_by(AssistantSession.updated_at.desc())
        query = query.offset(offset).limit(limit)

        result = await self.db.execute(query)
        sessions = result.scalars().all()

        count_result = await self.db.execute(count_query)
        total = count_result.scalar() or 0

        return sessions, total

    async def get_session_messages(
        self, session_id: str, limit: int = 50
    ) -> list[AssistantMessage]:
        """Get messages for a session."""
        result = await self.db.execute(
            select(AssistantMessage)
            .where(AssistantMessage.session_id == session_id)
            .order_by(AssistantMessage.created_at.asc())
            .limit(limit)
        )
        return result.scalars().all()

    # ------------------------------------------------------------------
    # Chat (non-streaming)
    # ------------------------------------------------------------------

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        *,
        max_sources: int = 5,
        include_sanskrit: bool = True,
    ) -> dict:
        """
        Process a chat message and return a response with sources.

        Returns dict with: session_id, message, sources, metadata.
        """
        session = await self.get_or_create_session(session_id)

        # Store user message
        user_msg = AssistantMessage(
            id=str(uuid.uuid4()),
            session_id=session.id,
            role="user",
            content=message,
        )
        self.db.add(user_msg)

        # Retrieve relevant context
        hits, search_time, fallback = await hybrid_search(
            self.db, message, limit=max_sources
        )

        # Build context
        context_str = self._build_context(hits, include_sanskrit)
        system_prompt = SYSTEM_PROMPT.format(context=context_str)

        # Build conversation history
        history = await self._build_history(session.id)
        messages = history + [{"role": "user", "content": message}]

        # Call Claude
        try:
            response = self.client.messages.create(
                model=self.settings.assistant_model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
            )
            assistant_text = response.content[0].text
        except Exception as e:
            logger.error("Assistant LLM call failed: %s", e)
            assistant_text = (
                "I apologize, but I encountered an error processing your question. "
                "Please try again."
            )

        # Build source citations
        sources = self._build_citations(hits)

        # Store assistant message
        assistant_msg = AssistantMessage(
            id=str(uuid.uuid4()),
            session_id=session.id,
            role="assistant",
            content=assistant_text,
            sources=[s.model_dump() for s in sources],
        )
        self.db.add(assistant_msg)

        # Update session
        session.message_count = (session.message_count or 0) + 2
        if not session.title and len(message) > 0:
            session.title = message[:100]

        await self.db.flush()

        return {
            "session_id": session.id,
            "message": assistant_text,
            "sources": sources,
            "metadata": {
                "search_time_ms": search_time,
                "fallback_used": fallback,
                "sources_found": len(hits),
            },
        }

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        message: str,
        session_id: str | None = None,
        *,
        max_sources: int = 5,
        include_sanskrit: bool = True,
    ) -> AsyncIterator[str]:
        """
        Process a chat message and yield SSE events.

        Yields JSON-encoded SSE events:
          - {"type": "sources", "data": [...]}
          - {"type": "text", "data": "..."}
          - {"type": "done", "data": {"session_id": "..."}}
        """
        session = await self.get_or_create_session(session_id)

        # Store user message
        user_msg = AssistantMessage(
            id=str(uuid.uuid4()),
            session_id=session.id,
            role="user",
            content=message,
        )
        self.db.add(user_msg)
        await self.db.flush()

        # Retrieve context
        hits, search_time, fallback = await hybrid_search(
            self.db, message, limit=max_sources
        )

        # Emit sources first
        sources = self._build_citations(hits)
        yield json.dumps({
            "type": "sources",
            "data": [s.model_dump() for s in sources],
        })

        # Build messages
        context_str = self._build_context(hits, include_sanskrit)
        system_prompt = SYSTEM_PROMPT.format(context=context_str)
        history = await self._build_history(session.id)
        messages = history + [{"role": "user", "content": message}]

        # Stream response
        full_text = ""
        try:
            with self.client.messages.stream(
                model=self.settings.assistant_model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    full_text += text
                    yield json.dumps({"type": "text", "data": text})
        except Exception as e:
            logger.error("Streaming assistant call failed: %s", e)
            error_msg = "I apologize, but I encountered an error. Please try again."
            full_text = error_msg
            yield json.dumps({"type": "text", "data": error_msg})

        # Store assistant message
        assistant_msg = AssistantMessage(
            id=str(uuid.uuid4()),
            session_id=session.id,
            role="assistant",
            content=full_text,
            sources=[s.model_dump() for s in sources],
        )
        self.db.add(assistant_msg)
        session.message_count = (session.message_count or 0) + 2
        if not session.title:
            session.title = message[:100]
        await self.db.flush()

        yield json.dumps({
            "type": "done",
            "data": {"session_id": session.id},
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context(self, hits: list[SearchHit], include_sanskrit: bool) -> str:
        """Build context string from search hits."""
        if not hits:
            return ""

        source_blocks = []
        for i, hit in enumerate(hits, 1):
            parts = [f"### Source {i}"]
            if hit.chapter_title:
                parts.append(f"Chapter: {hit.chapter_title}")
            if hit.verse_number:
                parts.append(f"Verse: {hit.verse_number}")

            if include_sanskrit and hit.sanskrit_devanagari:
                parts.append(f"Sanskrit: {hit.sanskrit_devanagari}")
            if hit.sanskrit_iast:
                parts.append(f"IAST: {hit.sanskrit_iast}")
            if hit.english_summary:
                parts.append(f"Summary: {hit.english_summary}")
            if hit.commentary:
                parts.append(f"Commentary: {hit.commentary}")

            relevance = hit.score * 100 if hit.score else 0
            parts.append(f"(relevance: {relevance:.0f}%)")

            source_blocks.append("\n".join(parts))

        sources_text = "\n\n".join(source_blocks)
        return CONTEXT_TEMPLATE.format(sources=sources_text)

    def _build_citations(self, hits: list[SearchHit]) -> list[SourceCitation]:
        """Build source citations from search hits."""
        citations = []
        for hit in hits:
            snippet = hit.sanskrit_devanagari[:200] if hit.sanskrit_devanagari else None
            citations.append(SourceCitation(
                verse_id=hit.verse_id,
                verse_number=hit.verse_number,
                chapter_title=hit.chapter_title,
                sanskrit_snippet=snippet,
                english_summary=hit.english_summary,
                relevance_score=hit.score or 0.0,
            ))
        return citations

    async def _build_history(
        self, session_id: str, max_turns: int = 10
    ) -> list[dict]:
        """
        Build conversation history for the LLM.

        Keeps last max_turns message pairs.
        """
        result = await self.db.execute(
            select(AssistantMessage)
            .where(
                AssistantMessage.session_id == session_id,
                AssistantMessage.role.in_(["user", "assistant"]),
            )
            .order_by(AssistantMessage.created_at.desc())
            .limit(max_turns * 2)
        )
        messages = list(reversed(result.scalars().all()))

        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
