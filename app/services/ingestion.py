"""
Ingestion service: PDF -> pages -> Claude Vision -> JSON -> DB.

Improved from the original extract_verses.py script:
  - Async processing with semaphore-based rate limiting
  - Database storage (not just JSON files)
  - Automatic embedding generation after extraction
  - Resume support via database state
  - Structured error handling and logging
"""

import base64
import json
import logging
import uuid
from pathlib import Path

import anthropic
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.corpus import Text, Chapter, Verse
from app.models.embedding import VerseEmbedding, SearchIndex
from app.services.embedding import generate_embeddings_batch
from app.utils.chunking import build_verse_chunk, chunk_sanskrit_text
from app.utils.sanskrit import normalize_devanagari

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """You are a Sanskrit manuscript scholar. Analyze this page image from the Gajashastra (गजशास्त्र) of Palakapya Muni.

Extract ALL Sanskrit verses and text from this page. For each verse or text block found:

1. **sanskrit_devanagari**: The verse in clean Devanagari script. Fix any OCR-like errors you can identify. Use proper danda (।) and double danda (॥) markers.
2. **sanskrit_iast**: Transliteration in IAST (International Alphabet of Sanskrit Transliteration).
3. **verse_number**: If a verse/shloka number is visible, include it as a string. Otherwise null.
4. **section**: Chapter or section name if visible (e.g., "उपोद्धातः", "प्रथमोऽध्यायः"). Otherwise null.
5. **type**: One of: "verse" (shloka/verse), "prose" (gadya), "chapter_title", "commentary", "colophon".
6. **english_summary**: Brief 1-2 sentence summary of the content in English.
7. **meter**: If identifiable, the verse meter (e.g., "anushtubh", "shloka", "arya"). Otherwise null.

If the page contains NO Sanskrit text (title page, illustration, English-only), return entries as an empty array.

Respond with ONLY valid JSON (no markdown code fences):
{
  "page_number": PAGE_NUM,
  "has_sanskrit": true,
  "has_illustrations": false,
  "illustration_description": null,
  "section_name": "current section/chapter name",
  "entries": [
    {
      "sanskrit_devanagari": "...",
      "sanskrit_iast": "...",
      "verse_number": "1",
      "section": "...",
      "type": "verse",
      "english_summary": "...",
      "meter": null
    }
  ]
}"""


class IngestionService:
    """Manages the full PDF -> DB ingestion pipeline."""

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
    # High-level pipeline
    # ------------------------------------------------------------------

    async def ingest_pdf_pages(
        self,
        pages_dir: str,
        text_id: str,
        *,
        start_page: int = 1,
        end_page: int = 0,
        model: str | None = None,
        batch_size: int = 5,
    ) -> dict:
        """
        Ingest page images from a directory into the database.

        Args:
            pages_dir: Path to directory with page-NNN.png files.
            text_id: ID of the Text record to associate verses with.
            start_page: Start from this page number.
            end_page: End at this page (0 = all).
            model: Claude model for vision extraction.
            batch_size: Pages per batch.

        Returns:
            Summary dict with counts.
        """
        model = model or self.settings.vision_model
        pages_path = Path(pages_dir)

        # Find page images
        page_files = sorted(
            list(pages_path.glob("*.png")) + list(pages_path.glob("*.jpg")),
            key=lambda p: self._extract_page_num(p),
        )

        if not page_files:
            raise FileNotFoundError(f"No page images found in {pages_dir}")

        # Filter by page range
        pages = []
        for pf in page_files:
            num = self._extract_page_num(pf)
            if num < start_page:
                continue
            if end_page > 0 and num > end_page:
                continue
            pages.append((str(pf), num))

        logger.info(
            "Starting ingestion: %d pages (range %d-%d), model=%s",
            len(pages), pages[0][1], pages[-1][1], model,
        )

        # Create index tracking record
        index = SearchIndex(
            name=f"ingest-{text_id}-pages-{pages[0][1]}-{pages[-1][1]}",
            status="processing",
            total_items=len(pages),
        )
        self.db.add(index)
        await self.db.flush()

        # Ensure a default chapter exists
        default_chapter = await self._get_or_create_default_chapter(text_id)

        total_entries = 0
        errors = 0
        current_section = None
        chapter_map: dict[str, str] = {}  # section_name -> chapter_id

        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            logger.info("Processing batch %d: pages %d-%d",
                        i // batch_size + 1, batch[0][1], batch[-1][1])

            for page_path, page_num in batch:
                try:
                    page_data = self._extract_page(page_path, page_num, model)

                    section = page_data.get("section_name")
                    if section and section != current_section:
                        current_section = section
                        if section not in chapter_map:
                            chapter = await self._get_or_create_chapter(
                                text_id, section, page_num
                            )
                            chapter_map[section] = chapter.id

                    chapter_id = chapter_map.get(current_section, default_chapter.id)

                    for entry in page_data.get("entries", []):
                        await self._store_entry(entry, chapter_id, page_num, model)
                        total_entries += 1

                    index.processed_items += 1

                except Exception as e:
                    logger.error("Error processing page %d: %s", page_num, e)
                    errors += 1
                    index.failed_items += 1

            await self.db.flush()

        index.status = "completed" if errors == 0 else "completed_with_errors"
        await self.db.flush()

        summary = {
            "pages_processed": len(pages),
            "entries_extracted": total_entries,
            "errors": errors,
            "chapters_created": len(chapter_map),
            "index_id": index.id,
        }
        logger.info("Ingestion complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Page extraction (Claude Vision)
    # ------------------------------------------------------------------

    def _extract_page(self, page_path: str, page_num: int, model: str) -> dict:
        """Extract text from a single page image using Claude Vision."""
        img_data = self._encode_image(page_path)
        ext = Path(page_path).suffix.lower()
        media_type = "image/png" if ext == ".png" else "image/jpeg"

        prompt = EXTRACTION_PROMPT.replace("PAGE_NUM", str(page_num))

        response = self.client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        text = response.content[0].text

        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())

    # ------------------------------------------------------------------
    # Database storage
    # ------------------------------------------------------------------

    async def _store_entry(
        self,
        entry: dict,
        chapter_id: str,
        page_number: int,
        model: str,
    ) -> Verse:
        """Store a single extracted entry as a Verse record."""
        devanagari = entry.get("sanskrit_devanagari", "")
        if devanagari:
            devanagari = normalize_devanagari(devanagari)

        # Count existing verses in chapter for ordering
        count_result = await self.db.execute(
            select(func.count()).where(Verse.chapter_id == chapter_id)
        )
        order = count_result.scalar() or 0

        verse = Verse(
            id=str(uuid.uuid4()),
            chapter_id=chapter_id,
            verse_number=entry.get("verse_number"),
            order=order,
            sanskrit_devanagari=devanagari,
            sanskrit_iast=entry.get("sanskrit_iast"),
            english_summary=entry.get("english_summary"),
            verse_type=entry.get("type", "verse"),
            meter=entry.get("meter"),
            page_number=page_number,
            extraction_model=model,
        )
        self.db.add(verse)
        return verse

    async def _get_or_create_default_chapter(self, text_id: str) -> Chapter:
        """Get or create a default chapter for entries without section info."""
        result = await self.db.execute(
            select(Chapter).where(
                Chapter.text_id == text_id,
                Chapter.title == "Unclassified",
            )
        )
        chapter = result.scalar_one_or_none()
        if chapter is None:
            chapter = Chapter(
                id=str(uuid.uuid4()),
                text_id=text_id,
                title="Unclassified",
                title_devanagari="अवर्गीकृत",
                order=999,
            )
            self.db.add(chapter)
            await self.db.flush()
        return chapter

    async def _get_or_create_chapter(
        self, text_id: str, section_name: str, page_num: int
    ) -> Chapter:
        """Get or create a chapter by section name."""
        result = await self.db.execute(
            select(Chapter).where(
                Chapter.text_id == text_id,
                Chapter.title == section_name,
            )
        )
        chapter = result.scalar_one_or_none()
        if chapter is None:
            count_result = await self.db.execute(
                select(func.count()).where(Chapter.text_id == text_id)
            )
            order = count_result.scalar() or 0

            chapter = Chapter(
                id=str(uuid.uuid4()),
                text_id=text_id,
                title=section_name,
                title_devanagari=section_name,
                order=order,
                page_start=page_num,
            )
            self.db.add(chapter)
            await self.db.flush()
        return chapter

    # ------------------------------------------------------------------
    # Embedding generation for ingested verses
    # ------------------------------------------------------------------

    async def embed_verses(
        self,
        text_id: str | None = None,
        *,
        batch_size: int = 50,
        force_reindex: bool = False,
    ) -> dict:
        """
        Generate embeddings for all un-indexed verses.

        Builds embedding chunks from verse + context, then batch-embeds.
        """
        # Find verses needing embeddings
        query = select(Verse).where(Verse.is_indexed == False)  # noqa: E712
        if text_id:
            query = query.join(Chapter).where(Chapter.text_id == text_id)
        query = query.order_by(Verse.order)

        result = await self.db.execute(query)
        verses = result.scalars().all()

        if not verses:
            return {"embedded": 0, "message": "No verses need embedding"}

        logger.info("Embedding %d verses", len(verses))

        # Build chunks
        texts_to_embed = []
        verse_refs = []

        for verse in verses:
            # Load chapter info for context prefix
            chapter = verse.chapter
            chapter_title = chapter.title if chapter else ""

            chunk_text = build_verse_chunk(
                verse.sanskrit_devanagari,
                chapter_title=chapter_title,
                verse_number=verse.verse_number or "",
                commentary=verse.commentary or "",
                translation=verse.english_translation or verse.english_summary or "",
            )

            # Check if text is too long and needs multi-chunk splitting
            settings = get_settings()
            max_chars = settings.chunk_size_tokens * settings.chars_per_token
            if len(chunk_text) > max_chars:
                chunks = chunk_sanskrit_text(
                    chunk_text,
                    context_prefix=f"Chapter: {chapter_title} | Verse: {verse.verse_number or '?'}",
                )
                for chunk in chunks:
                    texts_to_embed.append(chunk["text"])
                    verse_refs.append((verse.id, chunk["chunk_index"], chunk["text"]))
            else:
                texts_to_embed.append(chunk_text)
                verse_refs.append((verse.id, 0, chunk_text))

        # Batch embed
        embedded_count = 0
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            batch_refs = verse_refs[i:i + batch_size]

            try:
                embeddings = await generate_embeddings_batch(batch_texts)

                for (verse_id, chunk_idx, chunk_text), emb in zip(batch_refs, embeddings):
                    ve = VerseEmbedding(
                        id=str(uuid.uuid4()),
                        verse_id=verse_id,
                        chunk_text=chunk_text,
                        chunk_index=chunk_idx,
                        embedding=emb,
                        embedding_model=settings.embedding_model if not settings.use_local_embeddings else settings.local_embedding_model,
                        embedding_dimension=settings.embedding_dimension,
                    )
                    self.db.add(ve)
                    embedded_count += 1

                # Mark verses as indexed
                for verse_id, _, _ in batch_refs:
                    verse_obj = await self.db.get(Verse, verse_id)
                    if verse_obj:
                        verse_obj.is_indexed = True

                await self.db.flush()
                logger.info("Embedded batch %d: %d chunks", i // batch_size + 1, len(batch_texts))

            except Exception as e:
                logger.error("Embedding batch failed at index %d: %s", i, e)

        return {"embedded": embedded_count, "total_verses": len(verses)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(path: str) -> str:
        """Read image file and return base64 encoded string."""
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _extract_page_num(path: Path) -> int:
        """Extract page number from filename like page-001.png."""
        nums = "".join(c for c in path.stem if c.isdigit())
        return int(nums) if nums else 0
