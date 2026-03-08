"""
keywords.py
-----------
Keyword extraction using RAKE algorithm.

Improvements over v1:
  - Rake instance cached at module level (not reinstantiated on every call)
  - Deduplication: removes phrases that are subsets of a higher-ranked phrase
  - Minimum score threshold to filter out low-quality phrases
  - Returns scored keywords for richer downstream use
"""

from rake_nltk import Rake  # type: ignore
from typing import Dict, Any

# ── Instantiate RAKE once at module load — not on every call ──────────────────
_RAKE = Rake(min_length=1, max_length=4)

# Only keep phrases scoring above this threshold
_MIN_SCORE = 4.0


def extract_keywords(text: str, max_keywords: int = 15) -> list[str]:
    """
    Extract important keyword phrases from text using RAKE.

    Returns top `max_keywords` deduplicated, scored phrases.
    """
    if not text or not text.strip():
        return []

    _RAKE.extract_keywords_from_text(text)
    scored = _RAKE.get_ranked_phrases_with_scores()

    # Filter by minimum score
    filtered = [(score, phrase) for score, phrase in scored if score >= _MIN_SCORE]

    # Deduplicate: remove a phrase if a higher-scored phrase already contains it
    deduped: list[tuple[float, str]] = []
    seen_words: set[str] = set()
    for score, phrase in sorted(filtered, reverse=True):
        phrase_words = set(phrase.lower().split())
        if not phrase_words.issubset(seen_words):
            deduped.append((score, phrase))
            seen_words.update(phrase_words)

    return [phrase for _, phrase in deduped[:max_keywords]]


def extract_keyword_data(text: str) -> Dict[str, Any]:
    """
    Package keyword extraction results with metadata.

    Returns:
        { "keywords": [...], "top_phrase": "...", "count": N }
    """
    keywords = extract_keywords(text)
    return {
        "keywords": keywords,
        "top_phrase": keywords[0] if keywords else None,
        "count": len(keywords),
    }
