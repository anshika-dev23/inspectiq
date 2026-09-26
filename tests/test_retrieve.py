import json
import math
import pickle
from dataclasses import replace

import chromadb
import pytest
from rank_bm25 import BM25Okapi

from src.config import CorpusFile, RetrievalConfig, load_settings
from src.ingest import Page, build_parents, split_sections, write_stores
from src.retrieve import (
    Candidate,
    ChildHit,
    ParentHit,
    RetrievalResources,
    bm25_search,
    choose_window,
    chroma_where,
    detect_section_refs,
    expand_abbreviations,
    expand_with_glossary,
    format_citation,
    matches_filters,
    normalize_query,
    order_candidates,
    parents_from_children,
    reciprocal_rank_fusion,
    resolve_section_ref,
    retrieve,
    snap_to_lines,
)
from src.text import tokenize


# --- pure functions ---------------------------------------------------------

def test_rrf_math():
    rankings = {
        "child_vector": [ParentHit("p1", 1, 0.9), ParentHit("p2", 2, 0.8)],
        "bm25": [ParentHit("p2", 1, 7.0), ParentHit("p3", 2, 5.0)],
    }
    fused = reciprocal_rank_fusion(rankings, k=60)
    assert [c.parent_id for c in fused] == ["p2", "p1", "p3"]
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].rrf_score == pytest.approx(1 / 61)
    assert fused[2].rrf_score == pytest.approx(1 / 62)
    assert fused[0].ranks == {"child_vector": 2, "bm25": 1}


def test_rrf_ties_are_deterministic():
    rankings = {"a": [ParentHit("x", 1, 0)], "b": [ParentHit("w", 1, 0)]}
    assert [c.parent_id for c in reciprocal_rank_fusion(rankings, k=60)] == ["w", "x"]


def test_parents_from_children_dedupes_and_keeps_best_rank():
    hits = [ChildHit("c1", "pA", 1, 0.9), ChildHit("c2", "pB", 2, 0.8), ChildHit("c3", "pA", 3, 0.7),
            ChildHit("c4", "pC", 4, 0.6)]
    assert parents_from_children(hits) == [ParentHit("pA", 1, 0.9), ParentHit("pB", 2, 0.8), ParentHit("pC", 3, 0.6)]


def test_bm25_filters_before_taking_top_k():
    # Five Standards children mention "door" a lot; two Guidance children mention it once.
    corpus = [["door"] * 5 + [f"s{i}"] for i in range(5)] + [["door", f"g{i}", "x", "y"] for i in range(2)] + [["ramp"]] * 3
    metadatas = [{"code_name": "S"}] * 5 + [{"code_name": "G"}] * 2 + [{"code_name": "S"}] * 3
    index = {
        "bm25": BM25Okapi(corpus),
        "child_ids": [f"c{i}" for i in range(10)],
        "parent_ids": [f"p{i}" for i in range(10)],
        "metadatas": metadatas,
    }
    # Top 2 over everything would be Standards children only, and filtering them would leave nothing.
    hits = bm25_search(index, ["door"], {"code_name": "G"}, top_k=2)
    assert [h.child_id for h in hits] == ["c5", "c6"]
    assert [h.rank for h in hits] == [1, 2]


def test_bm25_skips_children_without_query_terms():
    index = {"bm25": BM25Okapi([["door"], ["ramp"], ["slope"]]), "child_ids": ["a", "b", "c"],
             "parent_ids": ["a", "b", "c"], "metadatas": [{}] * 3}
    assert [h.child_id for h in bm25_search(index, ["door"], {}, top_k=20)] == ["a"]


def test_chroma_where():
    assert chroma_where({}) is None
    assert chroma_where({"code_name": "X"}) == {"code_name": {"$eq": "X"}}
    assert chroma_where({"code_name": "X", "edition_year": [2010, 2012]}) == {
        "$and": [{"code_name": {"$eq": "X"}}, {"edition_year": {"$in": [2010, 2012]}}]
    }


def test_matches_filters():
    metadata = {"code_name": "X", "edition_year": 2010, "section_type": "code"}
    assert matches_filters(metadata, {})
    assert matches_filters(metadata, {"code_name": "X", "edition_year": [2010, 2012]})
    assert not matches_filters(metadata, {"section_type": "regulation"})


def test_expand_abbreviations():
    abbreviations = {"gfci": "ground-fault circuit-interrupter", "atm": "automatic teller machine"}
    assert expand_abbreviations("Are GFCI outlets needed near an ATM?", abbreviations) == (
        "Are GFCI (ground-fault circuit-interrupter) outlets needed near an ATM (automatic teller machine)?"
    )
    assert expand_abbreviations("atmosphere", abbreviations) == "atmosphere"  # whole words only


@pytest.mark.parametrize("query, expected", [
    ("What does 404.2.3 require?", ["404.2.3"]),
    ("Explain 35.151(B) and § 36.406(f)", ["35.151(b)", "36.406(f)"]),
    ("R302.1 fire separation", ["r302.1"]),
    ("a force of 5 pounds (22.2 N) and 1.5 inches", ["22.2"]),  # not a section; dropped later if not in docstore
    ("minimum clear width of a door", []),
])
def test_detect_section_refs(query, expected):
    assert detect_section_refs(query) == expected


def test_normalize_query_can_be_switched_off():
    on = normalize_query("TTY height", RetrievalConfig())
    off = normalize_query("TTY height", RetrievalConfig(use_query_normalization=False))
    assert "teletypewriter" in on.text and "teletypewriter" in on.tokens
    assert off.text == "TTY height"


GLOSSARY_SAMPLE = {"light switch": "operable parts, controls", "toilet": "water closet", "bathroom": "toilet room"}


def test_glossary_appends_code_terms():
    assert expand_with_glossary("How high can a light switch be?", GLOSSARY_SAMPLE) == (
        "How high can a light switch be? (operable parts, controls)"
    )


def test_glossary_matches_plurals_and_ignores_case():
    assert expand_with_glossary("Light switches in Bathrooms", GLOSSARY_SAMPLE) == (
        "Light switches in Bathrooms (operable parts, controls, toilet room)"
    )


def test_glossary_keeps_phrases_intact_and_skips_terms_already_present():
    # "toilet" maps to "water closet", but the query text itself is never rewritten.
    assert expand_with_glossary("toilet room grab bars", GLOSSARY_SAMPLE) == "toilet room grab bars (water closet)"
    assert expand_with_glossary("water closet toilet", GLOSSARY_SAMPLE) == "water closet toilet"


def test_glossary_whole_words_only_and_no_match_unchanged():
    assert expand_with_glossary("toiletries", GLOSSARY_SAMPLE) == "toiletries"
    assert expand_with_glossary("ramp slope", GLOSSARY_SAMPLE) == "ramp slope"


def test_glossary_is_off_unless_enabled():
    query = "How high can a light switch be?"
    assert normalize_query(query, RetrievalConfig(use_glossary=False)).text == query
    assert "operable parts" in normalize_query(query, RetrievalConfig(use_glossary=True)).text


def test_query_tokens_drop_stopwords_but_keep_refs():
    assert normalize_query("What does 604.5 require?", RetrievalConfig()).tokens == ["604.5", "require"]


def candidate(parent_id, rrf, rerank_score, pinned=False):
    return Candidate(parent_id, rrf, {}, pinned=pinned, rerank_score=rerank_score)


def test_order_candidates_replace_uses_rerank_score_only():
    ordered = order_candidates([candidate("rrf_top", 0.05, 6.5), candidate("rerank_top", 0.03, 8.3)], "replace", 0.5, 60)
    assert [c.parent_id for c in ordered] == ["rerank_top", "rrf_top"]


def test_order_candidates_blend_combines_normalized_scores():
    candidates = [candidate("rrf_top", 0.05, 6.5), candidate("rerank_top", 0.03, 8.3)]
    ordered = order_candidates(candidates, "blend", 0.5, 60)
    # rrf_top:    0.5 * sigmoid(6.5) + 0.5 * 1.0  = 0.99924
    # rerank_top: 0.5 * sigmoid(8.3) + 0.5 * 0.6  = 0.79988
    assert [c.parent_id for c in ordered] == ["rrf_top", "rerank_top"]
    assert ordered[0].order_score == pytest.approx(0.5 / (1 + math.exp(-6.5)) + 0.5)
    assert ordered[1].order_score == pytest.approx(0.5 / (1 + math.exp(-8.3)) + 0.5 * 0.6)


def test_order_candidates_blend_weight_extremes():
    candidates = [candidate("rrf_top", 0.05, 6.5), candidate("rerank_top", 0.03, 8.3)]
    assert order_candidates(candidates, "blend", 1.0, 60)[0].parent_id == "rerank_top"   # rerank only
    assert order_candidates(candidates, "blend", 0.0, 60)[0].parent_id == "rrf_top"      # RRF only


def test_order_candidates_rrf_adds_the_rerank_rank_as_a_fourth_list():
    # rrf_top is 1st by RRF, 2nd by rerank; rerank_top is 2nd by RRF, 1st by rerank.
    candidates = [candidate("rrf_top", 1 / 61 + 1 / 61 + 1 / 61, 6.5), candidate("rerank_top", 1 / 62 + 1 / 62 + 1 / 62, 8.3)]
    ordered = order_candidates(candidates, "rrf", 0.5, 60)
    by_id = {c.parent_id: c for c in ordered}
    assert by_id["rerank_top"].rerank_rank == 1 and by_id["rrf_top"].rerank_rank == 2
    assert by_id["rrf_top"].order_score == pytest.approx(3 / 61 + 1 / 62)
    assert by_id["rerank_top"].order_score == pytest.approx(3 / 62 + 1 / 61)
    assert [c.parent_id for c in ordered] == ["rrf_top", "rerank_top"]  # 3 of 4 lists prefer rrf_top


def test_order_candidates_rrf_lets_the_reranker_break_a_tie():
    candidates = [candidate("a", 1 / 61 + 1 / 62, 1.0), candidate("b", 1 / 61 + 1 / 62, 5.0)]
    assert [c.parent_id for c in order_candidates(candidates, "rrf", 0.5, 60)] == ["b", "a"]


def test_order_candidates_keeps_pinned_first_in_all_modes():
    candidates = [candidate("other", 0.05, 9.0), candidate("pinned", 0.0, -5.0, pinned=True)]
    for mode in ("replace", "blend", "rrf"):
        assert order_candidates(candidates, mode, 0.5, 60)[0].parent_id == "pinned"


def test_order_candidates_rejects_unknown_mode():
    with pytest.raises(ValueError, match="rerank_mode"):
        order_candidates([], "average", 0.5, 60)


def test_resolve_section_ref_falls_back_to_paragraph():
    lookup = {"35.151(b)": ["pB"], "604.5": ["p6"]}
    assert resolve_section_ref("35.151(b)(4)", lookup) == ["pB"]
    assert resolve_section_ref("604.5", lookup) == ["p6"]
    assert resolve_section_ref("604.5.1", lookup) == []   # dotted parts are not stripped
    assert resolve_section_ref("22.2", lookup) == []


def test_choose_window_centres_on_best_child():
    assert choose_window(10_000, [(5_000, 5_400)], 4_000) == (3_200, 7_200)


def test_choose_window_includes_nearby_children_and_ignores_far_ones():
    # best child 5000-5400, second 6000-6400 fits (span 1400), third at 9500 would not fit.
    assert choose_window(10_000, [(5_000, 5_400), (6_000, 6_400), (9_500, 9_900)], 4_000) == (3_700, 7_700)


def test_choose_window_is_shifted_inside_the_parent():
    assert choose_window(10_000, [(0, 400)], 4_000) == (0, 4_000)
    assert choose_window(10_000, [(9_600, 10_000)], 4_000) == (6_000, 10_000)


def test_snap_to_lines_widens_to_whole_lines():
    text = "line one\nline two is here\nline three"
    #       0       8 9              25 26
    assert snap_to_lines(text, 12, 20) == (9, 25)       # inside "line two is here"
    assert snap_to_lines(text, 9, 25) == (9, 25)        # already on boundaries
    assert snap_to_lines(text, 3, 30) == (0, len(text))  # first and last line


def test_snap_to_lines_does_not_move_edges_far():
    text = "x" * 1_000 + "\n" + "y" * 1_000
    assert snap_to_lines(text, 500, 1_500) == (500, 1_500)  # nearest breaks are > 200 chars away


def test_format_citation():
    assert format_citation({"source": "s.pdf", "page_start": 123, "page_end": 123, "section_id": "404.2.3"}) == (
        "[s.pdf p.123 §404.2.3]"
    )
    assert format_citation({"source": "s.pdf", "page_start": 10, "page_end": 12, "section_id": "35.151(b)"}) == (
        "[s.pdf p.10-12 §35.151(b)]"
    )


# --- small fake stores ------------------------------------------------------

VOCABULARY = ["door", "width", "clear", "grab", "bar", "toilet", "water", "closet", "ramp", "slope", "alteration"]
STOPWORDS = {"what", "is", "the", "of", "a", "does", "for", "require", "minimum"}


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Bag of words over a tiny vocabulary: similar words -> similar vectors (plus a constant so no zero vector)."""
    vectors = []
    for text in texts:
        tokens = [t.rstrip("s") for t in tokenize(text)]
        vectors.append([float(tokens.count(word)) for word in VOCABULARY] + [0.1])
    return vectors


def fake_rerank(pairs: list[tuple[str, str]]) -> list[float]:
    """Number of shared content words; stands in for the cross-encoder."""
    return [
        float(len((set(tokenize(query)) - STOPWORDS) & set(tokenize(passage))))
        for query, passage in pairs
    ]


STANDARDS_PAGES = [
    Page(1, ["404 Doors, Doorways, and Gates",
             "404.2.3 Clear Width.  Door openings shall provide a clear width of 32 inches minimum."]),
    Page(2, ["604.5 Grab Bars.  Grab bars for water closets shall comply with 609.",
             "The side wall grab bar shall be 42 inches long minimum."]),
    Page(3, ["405.2 Slope.  Ramp runs shall have a running slope not steeper than 1:12."]),
    Page(4, ["§ 35.151 New construction and alterations.",
             "(a) Design and construction.  Each facility shall be designed and constructed.",
             "(b) Alterations.  Each facility or part of a facility altered shall be altered so that",
             "the altered portion is readily accessible. Alteration of a door is an alteration."]),
    Page(5, ["206.2.3 Multi-Story Buildings.  " + "Filler about stories and levels. " * 150
             + "The door clear width of every door on the route matters here. " + "More filler text. " * 100]),
]
GUIDANCE_PAGES = [
    Page(1, ["404 Doors",
             "Commentary: the Department received comments about door clear width at entrances."]),
]


@pytest.fixture(scope="module")
def resources(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stores")
    settings = replace(load_settings(), store_dir=tmp, chroma_dir=tmp / "chroma",
                       bm25_path=tmp / "bm25.pkl", docstore_path=tmp / "docstore.json")
    parents = (
        build_parents(split_sections(STANDARDS_PAGES, "s.pdf"), "s.pdf", CorpusFile("ADA 2010 Standards", 2010))
        + build_parents(split_sections(GUIDANCE_PAGES, "g.pdf"), "g.pdf", CorpusFile("ADA 2010 Guidance", 2010))
    )
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    write_stores(parents, settings, fake_embed, client)
    with settings.bm25_path.open("rb") as f:
        bm25_index = pickle.load(f)
    return RetrievalResources(
        docstore=json.loads(settings.docstore_path.read_text()),
        bm25_index=bm25_index,
        children_collection=client.get_collection("children"),
        sections_collection=client.get_collection("sections"),
        embed_query=lambda query: fake_embed([query])[0],
        rerank=fake_rerank,
    )


# Explicit settings so these tests exercise every stage, whatever the production defaults are.
CONFIG = RetrievalConfig(rerank_threshold=1.0, use_section_vector=True, use_glossary=False)


def section_ids(result) -> list[str]:
    return [context.section_id for context in result.contexts]


# --- retrieve() end to end on fake stores -----------------------------------

def test_retrieve_finds_the_clear_width_section(resources):
    result = retrieve("What is the minimum clear width of a door?", config=CONFIG, resources=resources)
    assert result.contexts[0].section_id == "404.2.3"
    assert result.contexts[0].citation == "[s.pdf p.1 §404.2.3]"
    assert result.contexts[0].window is None                      # small parent: returned whole
    assert set(result.debug.fused[0].ranks) <= {"child_vector", "bm25", "section_vector"}
    assert {"normalize", "embed_query", "child_vector", "bm25", "section_vector", "fusion", "rerank",
            "context", "total"} <= set(result.debug.timings_ms)


def test_unrelated_question_returns_empty(resources):
    result = retrieve("What is the capital of France?", config=CONFIG, resources=resources)
    assert result.is_empty
    assert result.debug.fused                                     # retrievers did return something ...
    assert all(c.rerank_score < CONFIG.rerank_threshold for c in result.debug.reranked)  # ... none passed


def test_exact_ref_is_pinned_at_rank_1_and_survives_the_threshold(resources):
    config = replace(CONFIG, rerank_threshold=100.0)              # nothing can pass on score alone
    result = retrieve("What does 604.5 require?", config=config, resources=resources)
    assert result.debug.fused[0].pinned
    assert section_ids(result) == ["604.5"]
    assert result.contexts[0].pinned


def test_exact_ref_to_regulation_paragraph(resources):
    result = retrieve("35.151(b)", config=CONFIG, resources=resources)
    assert result.contexts[0].section_id == "35.151(b)"
    assert result.contexts[0].pinned


def test_exact_ref_respects_filters(resources):
    result = retrieve("What does 604.5 require?", {"code_name": "ADA 2010 Guidance"}, CONFIG, resources)
    assert result.debug.pinned_parent_ids == []
    assert all(context.code_name == "ADA 2010 Guidance" for context in result.contexts)


def test_exact_ref_boost_can_be_switched_off(resources):
    config = replace(CONFIG, use_exact_ref_boost=False, rerank_threshold=100.0)
    result = retrieve("What does 604.5 require?", config=config, resources=resources)
    assert result.debug.pinned_parent_ids == []
    assert result.is_empty


def test_filters_apply_to_every_retriever(resources):
    result = retrieve("door clear width", {"code_name": "ADA 2010 Guidance"}, CONFIG, resources)
    docstore = resources.docstore
    for hits in (result.debug.child_vector, result.debug.bm25, result.debug.section_vector):
        assert hits and all(docstore[h.parent_id]["metadata"]["code_name"] == "ADA 2010 Guidance" for h in hits)


def test_section_type_filter(resources):
    result = retrieve("alteration of a door", {"section_type": "regulation"}, CONFIG, resources)
    assert result.contexts and all(context.section_id.startswith("35.151") for context in result.contexts)


def test_big_parent_is_returned_as_a_window_around_the_best_child(resources):
    config = replace(CONFIG, parent_full_text_max_chars=3_000, context_window_chars=1_000)
    result = retrieve("door clear width route", {"code_name": "ADA 2010 Standards"}, config, resources)
    [context] = [c for c in result.contexts if c.section_id == "206.2.3"]
    start, end = context.window
    assert 1_000 <= end - start <= 1_000 + 2 * 200            # widened to line boundaries at most 200 each side
    assert "door clear width" in context.text                   # the matching sentence is inside the window
    parent_text = resources.docstore[context.parent_id]["text"]
    assert context.text == parent_text[start:end]
    assert context.parent_length > 3_000


def test_pinned_big_parent_window_starts_at_the_section_start(resources):
    config = replace(CONFIG, parent_full_text_max_chars=3_000, context_window_chars=1_000)
    result = retrieve("What does 206.2.3 require?", config=config, resources=resources)
    context = result.contexts[0]
    assert context.pinned and context.section_id == "206.2.3"
    assert context.window[0] == 0
    assert context.text.startswith("206.2.3 Multi-Story Buildings.")


def test_final_k_limits_the_number_of_contexts(resources):
    config = replace(CONFIG, use_rerank=False)
    assert len(retrieve("door clear width", config=replace(config, final_k=3), resources=resources).contexts) == 3
    assert len(retrieve("door clear width", config=replace(config, final_k=5), resources=resources).contexts) == 5


def test_blend_mode_runs_end_to_end(resources):
    result = retrieve("grab bar for water closets", config=replace(CONFIG, rerank_mode="blend"), resources=resources)
    assert result.contexts[0].section_id == "604.5"
    assert all(c.order_score is not None for c in result.debug.reranked)


def test_context_window_can_be_switched_off(resources):
    config = replace(CONFIG, parent_full_text_max_chars=3_000, use_context_window=False)
    result = retrieve("door clear width route", {"code_name": "ADA 2010 Standards"}, config, resources)
    [context] = [c for c in result.contexts if c.section_id == "206.2.3"]
    assert context.window is None and len(context.text) == context.parent_length


@pytest.mark.parametrize("disabled", ["use_child_vector", "use_bm25", "use_section_vector"])
def test_each_retriever_can_be_switched_off(resources, disabled):
    config = replace(CONFIG, **{disabled: False})
    result = retrieve("grab bar for water closets", config=config, resources=resources)
    name = {"use_child_vector": "child_vector", "use_bm25": "bm25", "use_section_vector": "section_vector"}[disabled]
    assert name not in result.debug.timings_ms
    assert getattr(result.debug, name) == []
    assert all(name not in candidate.ranks for candidate in result.debug.fused)
    assert result.contexts[0].section_id == "604.5"


def test_rerank_off_returns_fused_order_without_threshold(resources):
    config = replace(CONFIG, use_rerank=False)
    result = retrieve("What is the capital of France?", config=config, resources=resources)
    assert result.debug.reranked == []
    assert "rerank" not in result.debug.timings_ms
    assert [c.parent_id for c in result.contexts] == [c.parent_id for c in result.debug.fused[:3]]
    assert all(context.rerank_score is None for context in result.contexts)


def test_all_retrievers_off_returns_empty(resources):
    config = replace(CONFIG, use_child_vector=False, use_bm25=False, use_section_vector=False)
    result = retrieve("door clear width", config=config, resources=resources)
    assert result.is_empty and result.debug.fused == []


def test_unknown_filter_key_is_rejected(resources):
    with pytest.raises(ValueError, match="Unknown filter keys"):
        retrieve("door", {"chapter": 4}, CONFIG, resources)


def test_rrf_scores_in_result_match_debug_ranks(resources):
    result = retrieve("grab bar for water closets", config=CONFIG, resources=resources)
    for candidate in result.debug.fused:
        expected = sum(1 / (CONFIG.rrf_k + rank) for rank in candidate.ranks.values())
        assert math.isclose(candidate.rrf_score, expected)
