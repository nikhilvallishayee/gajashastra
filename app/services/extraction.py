"""
Wisdom extraction service (InsightExtractor pattern).

Extracts structured insights from verses using Claude:
  - Teachings / principles
  - Practical applications (elephant care)
  - Classifications (taxonomy)
  - Cross-references
  - Medical remedies

Follows the second-brain-api InsightExtractor pattern:
  - Background async extraction
  - Confidence scoring
  - Deduplication via cosine similarity (0.85 threshold)
"""

import json
import logging
import uuid
from typing import Any

import anthropic
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector

from app.config import get_settings
from app.models.corpus import Verse, Chapter
from app.models.knowledge import Insight, Pattern, CrossReference
from app.models.integration import Protocol
from app.services.embedding import generate_embedding

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """You are a scholar of ancient Indian veterinary science (Gajashastra - the science of elephants).

Analyze this Sanskrit verse and extract structured knowledge from it.

**Verse (Devanagari):**
{sanskrit_devanagari}

**Verse (IAST):**
{sanskrit_iast}

**English Summary:**
{english_summary}

**Chapter:** {chapter_title}
**Verse Number:** {verse_number}

Extract ALL applicable insights as a JSON array. Each insight must have:
1. **type**: One of: "teaching", "application", "classification", "observation", "remedy"
2. **content**: The insight in clear English (1-3 sentences)
3. **content_sanskrit**: Key Sanskrit term or phrase related to this insight
4. **confidence**: Your confidence score (0.0-1.0) based on text clarity
5. **category**: Primary category (e.g., "anatomy", "disease", "training", "diet", "mythology", "classification", "seasonal_care", "behavior", "musth", "bathing", "housing")
6. **subcategory**: More specific subcategory if applicable
7. **tags**: Array of relevant tags

For "remedy" type, additionally extract:
8. **protocol**: {{
     "condition": "what condition this treats",
     "symptoms": ["list of symptoms"],
     "treatment": "treatment description",
     "herbs": ["list of herbs/materials used"],
     "procedure_steps": ["step 1", "step 2", ...],
     "precautions": ["any warnings"],
     "body_part": "affected body part if applicable",
     "season": "relevant season if applicable",
     "severity": "mild|moderate|severe|critical"
   }}

For "classification" type, if the verse classifies types of elephants, extract the taxonomy.

If the verse is a chapter title, colophon, or contains no extractable wisdom, return an empty array.

Respond with ONLY valid JSON (no markdown fences):
[
  {{
    "type": "teaching",
    "content": "...",
    "content_sanskrit": "...",
    "confidence": 0.9,
    "category": "...",
    "subcategory": "...",
    "tags": ["..."]
  }}
]"""


class ExtractionService:
    """Extracts structured knowledge from verses."""

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

    async def extract_from_verse(self, verse_id: str) -> list[dict]:
        """
        Extract insights from a single verse.

        Returns list of created insight dicts.
        """
        verse = await self.db.get(Verse, verse_id)
        if not verse:
            raise ValueError(f"Verse not found: {verse_id}")

        # Load chapter for context
        chapter = await self.db.get(Chapter, verse.chapter_id) if verse.chapter_id else None

        # Skip non-content entries
        if verse.verse_type in ("chapter_title", "colophon"):
            logger.debug("Skipping non-content verse %s (type=%s)", verse_id, verse.verse_type)
            return []

        raw_insights = self._call_extraction(
            sanskrit_devanagari=verse.sanskrit_devanagari,
            sanskrit_iast=verse.sanskrit_iast or "",
            english_summary=verse.english_summary or "",
            chapter_title=chapter.title if chapter else "",
            verse_number=verse.verse_number or "",
        )

        created = []
        for raw in raw_insights:
            # Filter low confidence
            confidence = raw.get("confidence", 0.0)
            if confidence < 0.6:
                continue

            # Check for duplicates via cosine similarity
            is_dup = await self._is_duplicate(raw["content"])
            if is_dup:
                logger.debug("Skipping duplicate insight: %s", raw["content"][:80])
                continue

            # Generate embedding
            emb = await generate_embedding(raw["content"])

            # Create insight
            insight = Insight(
                id=str(uuid.uuid4()),
                verse_id=verse_id,
                insight_type=raw.get("type", "observation"),
                content=raw["content"],
                content_sanskrit=raw.get("content_sanskrit"),
                confidence=confidence,
                category=raw.get("category"),
                subcategory=raw.get("subcategory"),
                tags=raw.get("tags", []),
                content_embedding=emb,
                extraction_model=self.settings.extraction_model,
            )
            self.db.add(insight)

            # If it's a remedy, also create a Protocol
            protocol_data = raw.get("protocol")
            if raw.get("type") == "remedy" and protocol_data:
                await self._create_protocol(verse_id, raw, protocol_data, emb)

            created.append({
                "id": insight.id,
                "type": insight.insight_type,
                "content": insight.content,
                "confidence": insight.confidence,
                "category": insight.category,
            })

        await self.db.flush()
        logger.info("Extracted %d insights from verse %s", len(created), verse_id)
        return created

    async def extract_batch(
        self,
        verse_ids: list[str] | None = None,
        *,
        text_id: str | None = None,
        skip_extracted: bool = True,
    ) -> dict:
        """
        Batch extract insights from multiple verses.

        If verse_ids is None, processes all verses (optionally filtered by text_id).
        """
        if verse_ids:
            query = select(Verse).where(Verse.id.in_(verse_ids))
        else:
            query = select(Verse)
            if text_id:
                query = query.join(Chapter).where(Chapter.text_id == text_id)

        result = await self.db.execute(query)
        verses = result.scalars().all()

        total = 0
        errors = 0

        for verse in verses:
            # Skip if already has insights
            if skip_extracted:
                existing = await self.db.execute(
                    select(Insight).where(Insight.verse_id == verse.id).limit(1)
                )
                if existing.scalar_one_or_none():
                    continue

            try:
                created = await self.extract_from_verse(verse.id)
                total += len(created)
            except Exception as e:
                logger.error("Extraction failed for verse %s: %s", verse.id, e)
                errors += 1

        return {
            "verses_processed": len(verses),
            "insights_created": total,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_extraction(
        self,
        sanskrit_devanagari: str,
        sanskrit_iast: str,
        english_summary: str,
        chapter_title: str,
        verse_number: str,
    ) -> list[dict]:
        """Call Claude to extract structured insights."""
        prompt = EXTRACTION_PROMPT.format(
            sanskrit_devanagari=sanskrit_devanagari,
            sanskrit_iast=sanskrit_iast,
            english_summary=english_summary,
            chapter_title=chapter_title,
            verse_number=verse_number,
        )

        try:
            response = self.client.messages.create(
                model=self.settings.extraction_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            parsed = json.loads(text.strip())
            if not isinstance(parsed, list):
                parsed = [parsed]
            return parsed

        except json.JSONDecodeError as e:
            logger.error("JSON parse error in extraction: %s", e)
            return []
        except Exception as e:
            logger.error("Extraction LLM call failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    async def _is_duplicate(self, content: str) -> bool:
        """
        Check if content is a near-duplicate of an existing insight.

        Uses cosine similarity with threshold 0.85.
        """
        try:
            emb = await generate_embedding(content)
            threshold = self.settings.dedup_cosine_threshold

            # Find most similar existing insight
            from sqlalchemy import text as sa_text
            result = await self.db.execute(
                sa_text("""
                    SELECT 1 - (content_embedding <=> :emb::vector) AS similarity
                    FROM insights
                    WHERE content_embedding IS NOT NULL
                    ORDER BY content_embedding <=> :emb::vector
                    LIMIT 1
                """),
                {"emb": str(emb)},
            )
            row = result.first()
            if row and row[0] >= threshold:
                return True
        except Exception as e:
            logger.debug("Dedup check failed (treating as non-duplicate): %s", e)

        return False

    # ------------------------------------------------------------------
    # Protocol creation
    # ------------------------------------------------------------------

    async def _create_protocol(
        self,
        verse_id: str,
        insight_data: dict,
        protocol_data: dict,
        embedding: list[float],
    ) -> Protocol:
        """Create a Protocol record from extraction data."""
        protocol = Protocol(
            id=str(uuid.uuid4()),
            verse_id=verse_id,
            title=insight_data.get("content", "")[:500],
            protocol_type="treatment",
            condition=protocol_data.get("condition", "Unknown"),
            condition_sanskrit=insight_data.get("content_sanskrit"),
            symptoms=protocol_data.get("symptoms", []),
            treatment=protocol_data.get("treatment"),
            herbs=protocol_data.get("herbs", []),
            procedure_steps=protocol_data.get("procedure_steps", []),
            precautions=protocol_data.get("precautions", []),
            body_part=protocol_data.get("body_part"),
            season=protocol_data.get("season"),
            severity=protocol_data.get("severity"),
            disease_category=insight_data.get("category"),
            content_embedding=embedding,
            confidence=insight_data.get("confidence", 0.0),
        )
        self.db.add(protocol)
        return protocol
