"""
Sanskrit text utilities.

Provides:
  - Script detection (Devanagari vs IAST vs SLP1)
  - Basic sandhi splitting hints
  - Text normalization
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Unicode ranges
# ---------------------------------------------------------------------------

# Devanagari block: U+0900 - U+097F, Extended: U+A8E0 - U+A8FF
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF]")

# IAST diacritics: macrons, underdots, tildes used in romanized Sanskrit
_IAST_DIACRITICS = set("aaiIuUReEoOSsNnMHtdTDlL")
_IAST_DIACRITIC_CHARS = set("\u0101\u012B\u016B\u1E5B\u1E5D\u015B\u1E63\u1E47"
                            "\u1E45\u1E6D\u1E0D\u1E25\u0100\u012A\u016A")

# SLP1 uses uppercase for retroflex/aspirated
_SLP1_SPECIAL = set("TBDNGMHYRVSJ")


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

def detect_script(text: str) -> str:
    """
    Detect the script of a Sanskrit text.

    Returns:
        "devanagari", "iast", "slp1", or "unknown"
    """
    if not text or not text.strip():
        return "unknown"

    devanagari_count = len(_DEVANAGARI_RE.findall(text))
    total_alpha = sum(1 for c in text if unicodedata.category(c).startswith("L"))

    if total_alpha == 0:
        return "unknown"

    devanagari_ratio = devanagari_count / total_alpha if total_alpha else 0

    if devanagari_ratio > 0.5:
        return "devanagari"

    # Check for IAST diacritics
    has_iast_diacritics = any(c in _IAST_DIACRITIC_CHARS for c in text)
    if has_iast_diacritics:
        return "iast"

    # Check for SLP1 conventions (uppercase in word-internal positions)
    words = text.split()
    slp1_indicators = 0
    for word in words:
        if len(word) > 1:
            # SLP1 uses uppercase mid-word for retroflexes
            for i, c in enumerate(word[1:], 1):
                if c in _SLP1_SPECIAL and word[i - 1].islower():
                    slp1_indicators += 1
    if slp1_indicators > 2:
        return "slp1"

    # Default: if it has Latin characters without IAST diacritics, could be
    # either plain transliteration or English
    return "iast" if any(c.isalpha() for c in text) else "unknown"


def is_devanagari(text: str) -> bool:
    """Check if text is primarily in Devanagari script."""
    return detect_script(text) == "devanagari"


def is_iast(text: str) -> bool:
    """Check if text is in IAST transliteration."""
    return detect_script(text) == "iast"


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_devanagari(text: str) -> str:
    """
    Normalize Devanagari text:
      - NFC normalization
      - Normalize dandas
      - Clean whitespace
    """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Normalize various danda forms to standard
    text = text.replace("\u0964", "\u0964")  # already standard purna virama
    text = text.replace("|", "\u0964")  # ASCII pipe to danda

    # Normalize double danda
    text = re.sub(r"\u0964\s*\u0964", "\u0965", text)  # ।। -> ॥

    # Clean excess whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_iast(text: str) -> str:
    """
    Normalize IAST transliteration:
      - NFC normalization
      - Standardize diacritic forms
    """
    text = unicodedata.normalize("NFC", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Verse boundary detection
# ---------------------------------------------------------------------------

# Double danda ॥ marks verse endings
_VERSE_END_RE = re.compile(r"॥\s*(?:\d+\s*॥)?")
# Verse number patterns: ॥ 42 ॥ or || 42 ||
_VERSE_NUM_RE = re.compile(r"॥\s*(\d+)\s*॥")


def split_verses(text: str) -> list[dict]:
    """
    Split a block of Sanskrit text into individual verses at ॥ boundaries.

    Returns list of dicts with keys: text, verse_number (if found).
    """
    if not text.strip():
        return []

    # Split on double danda
    parts = _VERSE_END_RE.split(text)
    verses = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Try to extract verse number from the boundary marker
        verse_num = None
        num_match = _VERSE_NUM_RE.search(part)
        if num_match:
            verse_num = num_match.group(1)

        verses.append({
            "text": part,
            "verse_number": verse_num,
        })

    return verses


# ---------------------------------------------------------------------------
# Sandhi hints
# ---------------------------------------------------------------------------

# Common sandhi junction patterns in Devanagari
_SANDHI_VOWEL_PAIRS = {
    "ा + इ": "े",
    "ा + उ": "ो",
    "इ + अ": "य",
    "उ + अ": "व",
}


def detect_compound_boundaries(word: str) -> list[str]:
    """
    Heuristic compound splitting.

    This is a best-effort heuristic. Proper sandhi analysis requires
    a morphological analyzer. Returns the original word if no split found.
    """
    if not word or len(word) < 4:
        return [word]

    # For now, return as-is. A proper implementation would use
    # the Sanskrit Heritage Engine or similar tool.
    return [word]
