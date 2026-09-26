"""Tokenizer shared by ingestion (BM25 index) and retrieval (BM25 queries)."""
import re

# A section ID like "r302.1" or "404.2.3" is one token; otherwise split on word characters.
TOKEN_PATTERN = re.compile(r"[a-z]?\d+(?:\.\d+)*|\w+")

# Small English stopword list for BM25. Deliberately NOT included:
# - single letters ("a", "b", ...): "35.151(a)" tokenizes to "35.151", "a" and the letter is part of the ID;
# - normative words ("shall", "must", "may", "not", "no"): they carry meaning in code text;
# - numbers: never stopwords (the list is alphabetic only).
STOPWORDS = frozenset("""
about above after again all also am an and any are as at be because been before being between both but by
can could did do does doing during each either for from further had has have having he her here hers him his
how into is it its itself just me more most my of off on once only or other our ours out over own same she
should so some such than that the their theirs them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your yours
""".split())


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """Lowercase and split; BM25 (ingestion and query time) passes remove_stopwords=True."""
    tokens = TOKEN_PATTERN.findall(text.lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens
