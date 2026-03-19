"""
Devanagari <-> IAST bidirectional transliteration.

Uses a mapping table approach. Handles:
  - Consonants (ka-ha) with virama
  - Vowels (independent and dependent forms)
  - Numerals
  - Punctuation (danda, double danda)
  - Anusvara, visarga, chandrabindu
"""

import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

# Vowels: (Devanagari independent, Devanagari dependent/matra, IAST)
_VOWELS = [
    ("\u0910", "\u0948", "ai"),  # ai must come before a/i
    ("\u0914", "\u094C", "au"),  # au must come before a/u
    ("\u0906", "\u093E", "\u0101"),  # aa
    ("\u0908", "\u0940", "\u012B"),  # ii
    ("\u090A", "\u0942", "\u016B"),  # uu
    ("\u090B", "\u0943", "\u1E5B"),  # r (vocalic)
    ("\u0960", "\u0944", "\u1E5D"),  # rr (vocalic long)
    ("\u090F", "\u0947", "e"),
    ("\u0913", "\u094B", "o"),
    ("\u0905", "", "a"),  # short a (inherent, no matra)
    ("\u0907", "\u093F", "i"),
    ("\u0909", "\u0941", "u"),
]

# Consonants: (Devanagari, IAST)
_CONSONANTS = [
    # Gutturals
    ("\u0916", "kh"),  # kha before ka
    ("\u0918", "gh"),  # gha before ga
    ("\u0915", "k"),
    ("\u0917", "g"),
    ("\u0919", "\u1E45"),  # nga

    # Palatals
    ("\u091B", "ch"),  # cha before ca
    ("\u091D", "jh"),  # jha before ja
    ("\u091A", "c"),
    ("\u091C", "j"),
    ("\u091E", "\u00F1"),  # nya (using tilde-n)

    # Retroflexes
    ("\u0920", "\u1E6Dh"),  # Tha before Ta
    ("\u0922", "\u1E0Dh"),  # Dha before Da
    ("\u091F", "\u1E6D"),  # Ta
    ("\u0921", "\u1E0D"),  # Da
    ("\u0923", "\u1E47"),  # Na

    # Dentals
    ("\u0925", "th"),  # tha before ta
    ("\u0927", "dh"),  # dha before da
    ("\u0924", "t"),
    ("\u0926", "d"),
    ("\u0928", "n"),

    # Labials
    ("\u092B", "ph"),  # pha before pa
    ("\u092D", "bh"),  # bha before ba
    ("\u092A", "p"),
    ("\u092C", "b"),
    ("\u092E", "m"),

    # Semivowels
    ("\u092F", "y"),
    ("\u0930", "r"),
    ("\u0932", "l"),
    ("\u0935", "v"),

    # Sibilants
    ("\u0936", "\u015B"),  # sha (palatal)
    ("\u0937", "\u1E63"),  # Sha (retroflex)
    ("\u0938", "s"),

    # Aspirate
    ("\u0939", "h"),
]

# Special marks
_ANUSVARA = ("\u0902", "\u1E43")  # m with underdot
_VISARGA = ("\u0903", "\u1E25")  # h with underdot
_CHANDRABINDU = ("\u0901", "\u0303")  # combining tilde (approximation)
_VIRAMA = "\u094D"  # halant
_AVAGRAHA = ("\u093D", "'")  # avagraha

# Numerals
_NUMERALS = [
    ("\u0966", "0"), ("\u0967", "1"), ("\u0968", "2"), ("\u0969", "3"),
    ("\u096A", "4"), ("\u096B", "5"), ("\u096C", "6"), ("\u096D", "7"),
    ("\u096E", "8"), ("\u096F", "9"),
]

# Punctuation
_DANDA = ("\u0964", "|")
_DOUBLE_DANDA = ("\u0965", "||")


# ---------------------------------------------------------------------------
# Build lookup dictionaries
# ---------------------------------------------------------------------------

def _build_dev_to_iast() -> dict[str, str]:
    """Build Devanagari -> IAST lookup."""
    d = {}
    for dev, _matra, iast in _VOWELS:
        d[dev] = iast
    for dev, iast in _CONSONANTS:
        d[dev] = iast
    d[_ANUSVARA[0]] = _ANUSVARA[1]
    d[_VISARGA[0]] = _VISARGA[1]
    d[_AVAGRAHA[0]] = _AVAGRAHA[1]
    d[_DOUBLE_DANDA[0]] = _DOUBLE_DANDA[1]
    d[_DANDA[0]] = _DANDA[1]
    for dev, latin in _NUMERALS:
        d[dev] = latin
    return d


def _build_iast_to_dev() -> list[tuple[str, str]]:
    """
    Build IAST -> Devanagari mapping as ordered pairs.

    Order matters: longer sequences must be matched first (e.g., "kh" before "k").
    """
    pairs = []
    for dev, iast in _CONSONANTS:
        pairs.append((iast, dev))
    for dev, _matra, iast in _VOWELS:
        pairs.append((iast, dev))
    pairs.append((_ANUSVARA[1], _ANUSVARA[0]))
    pairs.append((_VISARGA[1], _VISARGA[0]))
    pairs.append((_DOUBLE_DANDA[1], _DOUBLE_DANDA[0]))
    pairs.append((_DANDA[1], _DANDA[0]))
    for dev, latin in _NUMERALS:
        pairs.append((latin, dev))
    # Sort by IAST length descending so longer matches are tried first
    pairs.sort(key=lambda x: -len(x[0]))
    return pairs


_DEV_TO_IAST = _build_dev_to_iast()
_IAST_TO_DEV = _build_iast_to_dev()

# Vowel matra lookup (consonant + vowel -> consonant + matra)
_MATRA_MAP = {}
for _dev_indep, _matra, _iast in _VOWELS:
    if _matra:
        _MATRA_MAP[_iast] = _matra


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def devanagari_to_iast(text: str) -> str:
    """
    Convert Devanagari Sanskrit text to IAST transliteration.

    This is a character-by-character conversion. It handles:
      - Inherent 'a' after consonants (added when no virama or matra follows)
      - Virama (halant) suppresses inherent 'a'
      - Vowel matras replace inherent 'a'
    """
    if not text:
        return ""

    result = []
    i = 0
    chars = list(text)
    length = len(chars)

    while i < length:
        c = chars[i]

        # Check for virama (halant) -- suppress inherent 'a'
        if c == _VIRAMA:
            i += 1
            continue

        # Check if this is a consonant
        is_consonant = c in _DEV_TO_IAST and any(
            c == dev for dev, _iast in _CONSONANTS
        )

        if is_consonant:
            result.append(_DEV_TO_IAST[c])
            # Check what follows
            if i + 1 < length:
                next_c = chars[i + 1]
                if next_c == _VIRAMA:
                    # Virama: no inherent 'a', skip it
                    i += 2
                    continue
                # Check for vowel matra
                matra_found = False
                for _dev_indep, matra, iast in _VOWELS:
                    if matra and next_c == matra:
                        result.append(iast)
                        matra_found = True
                        i += 2
                        break
                if matra_found:
                    continue
                # No virama, no matra: add inherent 'a'
                result.append("a")
            else:
                # End of string: add inherent 'a'
                result.append("a")
            i += 1
        elif c in _DEV_TO_IAST:
            result.append(_DEV_TO_IAST[c])
            i += 1
        else:
            # Pass through (whitespace, punctuation, etc.)
            result.append(c)
            i += 1

    return "".join(result)


def iast_to_devanagari(text: str) -> str:
    """
    Convert IAST transliteration to Devanagari script.

    This handles:
      - Multi-character IAST sequences (kh, gh, etc.)
      - Vowel matras after consonants
      - Virama for consonant clusters
    """
    if not text:
        return ""

    result = []
    i = 0
    length = len(text)
    prev_was_consonant = False

    while i < length:
        matched = False

        # Try matching longest IAST sequences first
        for iast_seq, dev_char in _IAST_TO_DEV:
            end = i + len(iast_seq)
            if end <= length and text[i:end].lower() == iast_seq.lower():
                # Determine if this is a consonant or vowel
                is_consonant = any(dev_char == d for d, _iast in _CONSONANTS)
                is_vowel = any(
                    dev_char == d for d, _m, _iast in _VOWELS
                )

                if is_vowel and prev_was_consonant:
                    # Use matra form instead of independent vowel
                    iast_key = iast_seq.lower()
                    if iast_key in _MATRA_MAP:
                        result.append(_MATRA_MAP[iast_key])
                    elif iast_key == "a":
                        pass  # inherent 'a', no matra needed
                    else:
                        result.append(dev_char)
                    prev_was_consonant = False
                elif is_consonant:
                    if prev_was_consonant:
                        # Add virama before this consonant
                        result.append(_VIRAMA)
                    result.append(dev_char)
                    prev_was_consonant = True
                else:
                    if prev_was_consonant:
                        # Inherent 'a' was expected but we got something else
                        pass
                    prev_was_consonant = False
                    result.append(dev_char)

                i = end
                matched = True
                break

        if not matched:
            if prev_was_consonant and text[i] in " \n\t.,;:!?":
                # End of word: consonant keeps inherent 'a'
                prev_was_consonant = False
            elif prev_was_consonant:
                prev_was_consonant = False
            result.append(text[i])
            i += 1

    return "".join(result)
