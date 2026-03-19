"""
Embedding generation service.

Uses Google gemini-embedding-001 via Vertex AI + ADC (3072 dimensions)
with local fallback to all-MiniLM-L6-v2 for development.

Patterns from second-brain-api:
  - Singleton model loading
  - LRU cache for dedup within a request
  - 10s timeout on API calls
  - Batch API (up to 100 texts per call)
  - Text truncation to 8000 chars
"""

import logging
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from app.config import get_settings

logger = logging.getLogger(__name__)

# Thread pool for blocking embedding API calls
_executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# Model singletons
# ---------------------------------------------------------------------------

_genai_client = None
_local_model = None


def _get_genai_client():
    """Lazy-load Google GenAI client. Fails gracefully if not installed (Vercel)."""
    global _genai_client
    if _genai_client is None:
        try:
            from google import genai
            settings = get_settings()
            _genai_client = genai.Client(
                vertexai=True,
                project=settings.gcp_project,
                location=settings.gcp_location,
            )
            logger.info("Google GenAI client initialized (Vertex AI + ADC, project=%s)", settings.gcp_project)
        except ImportError:
            logger.warning("google-genai not installed - embedding generation disabled (pre-computed embeddings still work)")
            return None
        except Exception as e:
            logger.error("Failed to initialize Google GenAI client: %s", e)
            return None
    return _genai_client


def _get_local_model():
    """Lazy-load local sentence-transformers model."""
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            settings = get_settings()
            _local_model = SentenceTransformer(settings.local_embedding_model)
            logger.info("Local embedding model loaded: %s", settings.local_embedding_model)
        except Exception as e:
            logger.error("Failed to load local embedding model: %s", e)
            raise
    return _local_model


# ---------------------------------------------------------------------------
# Core embedding functions
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int | None = None) -> str:
    """Truncate text to max_chars."""
    if max_chars is None:
        max_chars = get_settings().embedding_max_chars
    return text[:max_chars]


@lru_cache(maxsize=64)
def _embed_single_cached(text: str) -> tuple:
    """Cache wrapper (returns tuple for hashability)."""
    return tuple(_embed_single(text))


def _embed_single(text: str) -> list[float]:
    """
    Generate embedding for a single text string.

    Uses Google GenAI or local model based on settings.
    """
    settings = get_settings()
    text = _truncate(text)

    if settings.use_local_embeddings:
        return _embed_local(text)
    else:
        return _embed_google(text)


def _embed_google(text: str) -> list[float]:
    """Generate embedding via Google gemini-embedding-001."""
    import concurrent.futures

    settings = get_settings()
    client = _get_genai_client()

    def _call():
        result = client.models.embed_content(
            model=settings.embedding_model,
            contents=text,
            config={
                "task_type": settings.embedding_task_type,
                "output_dimensionality": settings.embedding_dimension,
            },
        )
        return list(result.embeddings[0].values)

    # 10-second timeout to prevent pipeline hangs
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_call)
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            logger.error("Google embedding API timed out after 10s")
            raise TimeoutError("Embedding API timed out")


def _embed_local(text: str) -> list[float]:
    """Generate embedding via local model, zero-padded to target dimension."""
    settings = get_settings()
    model = _get_local_model()
    embedding = model.encode(text).tolist()

    # Zero-pad to target dimension
    target_dim = settings.embedding_dimension
    if len(embedding) < target_dim:
        embedding.extend([0.0] * (target_dim - len(embedding)))

    return embedding[:target_dim]


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------

def _embed_batch_google(texts: list[str]) -> list[list[float]]:
    """
    Batch embed via Google API (up to 100 texts per call).
    """
    settings = get_settings()
    client = _get_genai_client()
    batch_size = settings.embedding_batch_size
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = [_truncate(t) for t in texts[i:i + batch_size]]
        try:
            result = client.models.embed_content(
                model=settings.embedding_model,
                contents=batch,
                config={
                    "task_type": settings.embedding_task_type,
                    "output_dimensionality": settings.embedding_dimension,
                },
            )
            for emb in result.embeddings:
                all_embeddings.append(list(emb.values))
        except Exception as e:
            logger.error("Batch embedding failed at index %d: %s", i, e)
            # Fall back to individual embedding for this batch
            for text in batch:
                try:
                    emb = _embed_single(text)
                    all_embeddings.append(emb)
                except Exception as inner_e:
                    logger.error("Individual embedding also failed: %s", inner_e)
                    all_embeddings.append([0.0] * settings.embedding_dimension)

    return all_embeddings


def _embed_batch_local(texts: list[str]) -> list[list[float]]:
    """Batch embed via local model."""
    settings = get_settings()
    model = _get_local_model()
    truncated = [_truncate(t) for t in texts]
    raw = model.encode(truncated).tolist()

    target_dim = settings.embedding_dimension
    result = []
    for emb in raw:
        if len(emb) < target_dim:
            emb.extend([0.0] * (target_dim - len(emb)))
        result.append(emb[:target_dim])
    return result


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def generate_embedding(text: str) -> list[float]:
    """
    Generate a single embedding vector (async wrapper).

    Returns a list of floats of length embedding_dimension.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _embed_single, text)


async def generate_embedding_query(text: str) -> list[float]:
    """
    Generate an embedding for a search query.

    Uses RETRIEVAL_QUERY task type for better search accuracy.
    """
    settings = get_settings()

    if settings.use_local_embeddings:
        return await generate_embedding(text)

    def _call():
        client = _get_genai_client()
        truncated = _truncate(text)
        result = client.models.embed_content(
            model=settings.embedding_model,
            contents=truncated,
            config={
                "task_type": settings.embedding_query_task_type,
                "output_dimensionality": settings.embedding_dimension,
            },
        )
        return list(result.embeddings[0].values)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _call)


async def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts (async wrapper).

    Uses batch API for efficiency (up to 100 texts per API call).
    """
    settings = get_settings()
    loop = asyncio.get_event_loop()

    if settings.use_local_embeddings:
        return await loop.run_in_executor(_executor, _embed_batch_local, texts)
    else:
        return await loop.run_in_executor(_executor, _embed_batch_google, texts)
