"""
Shloka-boundary-aware text chunking.

Sanskrit texts have natural boundaries at verse endings (marked by ॥).
This chunker respects those boundaries instead of splitting mid-verse.

Strategies:
  1. Verse-based: each verse + its commentary = one chunk
  2. Sliding window: for prose sections, 512-token chunks with 64-token overlap
     breaking at paragraph or sentence boundaries
"""

import re
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

# Double danda marks verse end
_VERSE_BOUNDARY_RE = re.compile(r"॥[^॥]*?॥")
_DOUBLE_DANDA = "\u0965"  # ॥
_SINGLE_DANDA = "\u0964"  # ।


def chunk_sanskrit_text(
    text: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    max_chunks: int | None = None,
    context_prefix: str = "",
) -> list[dict]:
    """
    Chunk Sanskrit text respecting shloka boundaries.

    Args:
        text: The Sanskrit text to chunk.
        chunk_size: Target chunk size in characters. Defaults from settings.
        chunk_overlap: Overlap between chunks in characters. Defaults from settings.
        max_chunks: Maximum number of chunks. Defaults from settings.
        context_prefix: Metadata prefix to prepend to each chunk
                       (e.g., "Gajashastra, Chapter 3, Verse 42").

    Returns:
        List of dicts with keys: text, chunk_index, context_prefix, char_count.
    """
    settings = get_settings()
    if chunk_size is None:
        chunk_size = settings.chunk_size_tokens * settings.chars_per_token
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap_tokens * settings.chars_per_token
    if max_chunks is None:
        max_chunks = settings.max_chunks_per_document

    if not text or not text.strip():
        return []

    # If short enough, return as single chunk
    if len(text) <= chunk_size:
        return [
            {
                "text": f"{context_prefix}\n\n{text}".strip() if context_prefix else text,
                "chunk_index": 0,
                "context_prefix": context_prefix,
                "char_count": len(text),
            }
        ]

    # Detect if text has verse boundaries
    has_verse_markers = _DOUBLE_DANDA in text
    if has_verse_markers:
        return _chunk_by_verse(text, chunk_size, chunk_overlap, max_chunks, context_prefix)
    else:
        return _chunk_by_paragraph(text, chunk_size, chunk_overlap, max_chunks, context_prefix)


def _chunk_by_verse(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
    context_prefix: str,
) -> list[dict]:
    """
    Chunk by verse boundaries (॥ markers).

    Accumulates verses until chunk_size is exceeded, then starts a new chunk.
    """
    # Split on double danda, keeping the delimiter
    parts = re.split(r"(॥)", text)

    # Reconstruct verse units (text + its closing ॥)
    verse_units = []
    current = ""
    for part in parts:
        current += part
        if part.strip() == _DOUBLE_DANDA:
            verse_units.append(current.strip())
            current = ""
    if current.strip():
        verse_units.append(current.strip())

    # Accumulate verse units into chunks
    chunks = []
    current_chunk = ""
    overlap_text = ""

    for unit in verse_units:
        candidate = (current_chunk + "\n" + unit).strip() if current_chunk else unit

        if len(candidate) > chunk_size and current_chunk:
            # Save current chunk
            full_text = f"{context_prefix}\n\n{current_chunk}".strip() if context_prefix else current_chunk
            chunks.append({
                "text": full_text,
                "chunk_index": len(chunks),
                "context_prefix": context_prefix,
                "char_count": len(current_chunk),
            })

            if len(chunks) >= max_chunks:
                logger.warning(
                    "Hit max_chunks limit (%d) during verse chunking", max_chunks
                )
                break

            # Start new chunk with overlap from end of previous
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = (overlap_text + "\n" + unit).strip() if overlap_text else unit
        else:
            current_chunk = candidate

    # Don't forget the last chunk
    if current_chunk.strip() and len(chunks) < max_chunks:
        full_text = f"{context_prefix}\n\n{current_chunk}".strip() if context_prefix else current_chunk
        chunks.append({
            "text": full_text,
            "chunk_index": len(chunks),
            "context_prefix": context_prefix,
            "char_count": len(current_chunk),
        })

    return chunks


def _chunk_by_paragraph(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
    context_prefix: str,
) -> list[dict]:
    """
    Chunk prose text by paragraph boundaries.

    Falls back to sentence splitting for very long paragraphs.
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current_chunk = ""
    overlap_text = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para

        if len(candidate) > chunk_size and current_chunk:
            # Save current chunk
            full_text = f"{context_prefix}\n\n{current_chunk}".strip() if context_prefix else current_chunk
            chunks.append({
                "text": full_text,
                "chunk_index": len(chunks),
                "context_prefix": context_prefix,
                "char_count": len(current_chunk),
            })

            if len(chunks) >= max_chunks:
                break

            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = (overlap_text + "\n\n" + para).strip() if overlap_text else para
        elif len(para) > chunk_size * 1.5:
            # Very long paragraph: split by sentences
            if current_chunk:
                full_text = f"{context_prefix}\n\n{current_chunk}".strip() if context_prefix else current_chunk
                chunks.append({
                    "text": full_text,
                    "chunk_index": len(chunks),
                    "context_prefix": context_prefix,
                    "char_count": len(current_chunk),
                })
                current_chunk = ""

            sentences = re.split(r"(?<=[।॥.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                test = (current_chunk + " " + sentence).strip() if current_chunk else sentence
                if len(test) > chunk_size and current_chunk:
                    full_text = f"{context_prefix}\n\n{current_chunk}".strip() if context_prefix else current_chunk
                    chunks.append({
                        "text": full_text,
                        "chunk_index": len(chunks),
                        "context_prefix": context_prefix,
                        "char_count": len(current_chunk),
                    })
                    if len(chunks) >= max_chunks:
                        break
                    current_chunk = sentence
                else:
                    current_chunk = test
        else:
            current_chunk = candidate

    # Last chunk
    if current_chunk.strip() and len(chunks) < max_chunks:
        full_text = f"{context_prefix}\n\n{current_chunk}".strip() if context_prefix else current_chunk
        chunks.append({
            "text": full_text,
            "chunk_index": len(chunks),
            "context_prefix": context_prefix,
            "char_count": len(current_chunk),
        })

    return chunks


def build_verse_chunk(
    verse_text: str,
    *,
    chapter_title: str = "",
    verse_number: str = "",
    commentary: str = "",
    translation: str = "",
) -> str:
    """
    Build a single chunk from a verse with its context.

    Combines verse + commentary + translation into one embedding unit,
    with a context prefix for better retrieval.
    """
    parts = []

    # Context prefix
    if chapter_title or verse_number:
        prefix_parts = []
        if chapter_title:
            prefix_parts.append(f"Chapter: {chapter_title}")
        if verse_number:
            prefix_parts.append(f"Verse: {verse_number}")
        parts.append(" | ".join(prefix_parts))

    # The verse itself
    parts.append(verse_text)

    # Commentary
    if commentary:
        parts.append(f"Commentary: {commentary}")

    # Translation
    if translation:
        parts.append(f"Translation: {translation}")

    return "\n\n".join(parts)
