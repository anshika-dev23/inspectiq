"""Pins down how rank_bm25's BM25Okapi handles negative IDF, because retrieval relies on it.

BM25Okapi computes idf = ln(N - n + 0.5) - ln(n + 0.5)   (N documents, n containing the term)
and replaces every NEGATIVE idf (term in more than half the documents) with
eps = epsilon * average_idf, where epsilon defaults to 0.25.
"""
import math

import pytest
from rank_bm25 import BM25Okapi


def raw_idf(corpus_size: int, doc_freq: int) -> float:
    return math.log(corpus_size - doc_freq + 0.5) - math.log(doc_freq + 0.5)


def test_negative_idf_is_replaced_by_epsilon_times_average_idf():
    # "a" in 3/4 docs (negative idf), "b" in 2/4 (idf exactly 0), "c" and "d" in 1/4 each.
    bm25 = BM25Okapi([["a", "b", "c"], ["a", "b", "d"], ["a", "x"], ["y"]])
    raw = {"a": raw_idf(4, 3), "b": raw_idf(4, 2), "c": raw_idf(4, 1)}
    assert raw["a"] < 0 and raw["b"] == 0 and raw["c"] > 0

    average_idf = sum(raw_idf(4, bm25_df) for bm25_df in (3, 2, 1, 1, 1, 1)) / 6  # a b c d x y
    assert bm25.idf["a"] == pytest.approx(0.25 * average_idf)
    assert bm25.idf["b"] == 0          # not negative, so NOT floored
    assert bm25.idf["c"] == pytest.approx(raw["c"])


def test_floor_is_not_monotonic():
    """A term in MORE documents ("a", 3/4) outweighs one in fewer ("b", 2/4): a quirk of the floor."""
    bm25 = BM25Okapi([["a", "b", "c"], ["a", "b", "d"], ["a", "x"], ["y"]])
    assert bm25.idf["a"] > bm25.idf["b"]


def test_floor_itself_is_negative_when_average_idf_is_negative():
    """Tiny corpus where every term is everywhere: eps < 0, so matching scores are negative."""
    bm25 = BM25Okapi([["signs"]])
    assert bm25.idf["signs"] < 0
    assert bm25.get_scores(["signs"])[0] < 0


def test_unknown_query_terms_score_zero():
    bm25 = BM25Okapi([["door", "width"], ["ramp", "slope"]])
    assert list(bm25.get_scores(["france"])) == [0.0, 0.0]
