"""Tokenizer shared by ingestion (BM25 index) and retrieval (BM25 queries)."""
import re

# A section ID like "r302.1" or "404.2.3" is one token; otherwise split on word characters.
TOKEN_PATTERN = re.compile(r"[a-z]?\d+(?:\.\d+)*|\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())
