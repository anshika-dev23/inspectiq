import json
import math
from pathlib import Path

import pytest

from eval.retrieval_eval import QUESTION_TYPES, QuestionResult, answer_rank, percentile, summarize, summarize_by_type

S = "ada_2010_standards.pdf"
G = "ada_2010_guidance.pdf"


def item(source, section_id):
    return {"source": source, "section_id": section_id}


def test_answer_rank_single_group_any_member():
    expected = [[item(S, "404.2.3"), item(S, "404.3.1")]]
    assert answer_rank([(S, "405.2"), (S, "404.3.1"), (S, "404.2.3")], expected) == 2


def test_answer_rank_all_groups_must_be_covered():
    expected = [[item(S, "604.5.1")], [item(S, "604.5.2")]]
    assert answer_rank([(S, "604.5.2"), (S, "609.4"), (S, "604.5.1")], expected) == 3
    assert answer_rank([(S, "604.5.2"), (S, "609.4")], expected) is None


def test_answer_rank_source_must_match():
    assert answer_rank([(G, "404.2.3")], [[item(S, "404.2.3")]]) is None


def test_answer_rank_out_of_corpus_is_none():
    assert answer_rank([(S, "404.2.3")], []) is None


def test_percentile_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 100.0
    assert percentile([7.0], 95) == 7.0
    assert math.isnan(percentile([], 50))


def result(qid, qtype, found, rank, latency=100.0):
    return QuestionResult(qid, qtype, found, rank, latency)


def test_summarize_metrics():
    results = [
        result("p1", "paraphrase", [(S, "a")], 1),
        result("p2", "paraphrase", [(S, "x"), (S, "y"), (S, "b")], 3),
        result("p3", "paraphrase", [(S, "x")], None),
        result("p4", "paraphrase", [], None),               # false refusal
        result("o1", "out_of_corpus", [], None),             # correct refusal
        result("o2", "out_of_corpus", [(S, "x")], None),     # should have been empty
    ]
    s = summarize(results)
    assert s["hit@1"] == pytest.approx(1 / 4)
    assert s["hit@3"] == pytest.approx(2 / 4)
    assert s["hit@5"] == pytest.approx(2 / 4)
    assert s["mrr"] == pytest.approx((1 + 1 / 3) / 4)
    assert s["refusal_accuracy"] == pytest.approx(1 / 2)
    assert s["false_refusals"] == pytest.approx(1 / 4)


def test_summarize_by_type_covers_every_type():
    by_type = summarize_by_type([result("p1", "paraphrase", [(S, "a")], 1), result("o1", "out_of_corpus", [], None)])
    assert set(by_type) == set(QUESTION_TYPES)
    assert by_type["paraphrase"]["hit@1"] == 1.0
    assert by_type["out_of_corpus"]["refusal_accuracy"] == 1.0
    assert math.isnan(by_type["exact_id"]["hit@1"])  # no questions of that type


def test_golden_set_is_well_formed():
    questions = json.loads((Path(__file__).parent.parent / "eval" / "golden.json").read_text())["questions"]
    assert len({q["id"] for q in questions}) == len(questions)
    for q in questions:
        assert q["type"] in QUESTION_TYPES and q["question"]
        assert (q["expected"] == []) == (q["type"] == "out_of_corpus")
        if q["type"] == "two_sections":
            assert len(q["expected"]) == 2
        for group in q["expected"]:
            assert group and all(set(e) == {"source", "section_id"} for e in group)


# --- answer eval helpers ------------------------------------------------------

from eval.answer_eval import citation_is_valid, contains_expected  # noqa: E402


@pytest.mark.parametrize("text, expected, ok", [
    ("a clear width of 32 inches [S1]", ["32"], True),
    ("17 inches minimum and 19 inches maximum", ["17", "19"], True),
    ("17 inches minimum", ["17", "19"], False),
    ("15 pounds", ["5"], False),                 # whole numbers only
    ("5.5 pounds", ["5"], False),
    ("5 pounds (22.2 N)", ["5"], True),
    ("a slope of 1:12 [S1]", ["1:12"], True),
    ("1:120", ["1:12"], False),
    ("over 1/2 inch", ["½|1/2|0.5"], True),
    ("over ½ inch", ["½|1/2|0.5"], True),
])
def test_contains_expected(text, expected, ok):
    assert contains_expected(text, expected) is ok


def test_citation_is_valid_any_expected_section():
    expected = [[item(S, "604.5.1")], [item(S, "604.5.2")]]
    assert citation_is_valid([(S, "609.4"), (S, "604.5.2")], expected)
    assert not citation_is_valid([(S, "609.4")], expected)


# --- setup comparison helpers -------------------------------------------------

from eval.answer_eval import AnswerResult, graph_decisions_table, outcome, outcome_diff_table  # noqa: E402


def answer_result(qid, qtype="paraphrase", refused=False, reason=None, contains_ok=True, status="verified", agent=None):
    return AnswerResult(setup="s", question_id=qid, question_type=qtype, question=f"question {qid}", refused=refused,
                        refusal_reason=reason, text="t", cited=[], invalid_labels=[], contains_ok=contains_ok,
                        citation_valid=None, total_ms=1.0, llm_calls=1, input_tokens=1, output_tokens=1,
                        shadow_cost_usd={}, grounding_status=status, agent=agent)


def test_outcome_labels():
    assert outcome(answer_result("p1")) == "correct"
    assert outcome(answer_result("p1", contains_ok=False)) == "WRONG"
    assert outcome(answer_result("p1", status="needs_review")) == "correct [review]"
    assert outcome(answer_result("p1", refused=True, reason="graded_not_relevant")) == "refused (graded_not_relevant)"
    assert outcome(answer_result("o1", qtype="out_of_corpus", refused=True, reason="no_relevant_sources")) == (
        "refused (no_relevant_sources) ✓")
    assert outcome(answer_result("o1", qtype="out_of_corpus", contains_ok=None)) == "ANSWERED (out of corpus)"


def test_outcome_diff_table_lists_only_questions_that_differ():
    by_setup = {"baseline": [answer_result("p1"), answer_result("p2", contains_ok=False)],
                "graph": [answer_result("p1"), answer_result("p2", refused=True, reason="model_not_in_sources")]}
    table = outcome_diff_table(by_setup)
    assert len(table) == 3 and "| p2 |" in table[2] and "WRONG" in table[2] and "model_not_in_sources" in table[2]


def test_graph_decisions_table_shows_rejections_and_rewrites():
    agent = {"queries": ["q", "q2"], "retries": 1,
             "grades": [{"section": "502.2 [ada_2010_standards.pdf]", "verdict": "no", "parsed": True},
                        {"section": "502.3.1 [ada_2010_standards.pdf]", "verdict": "yes", "parsed": False}]}
    quiet = {"queries": ["q"], "retries": 0, "grades": [{"section": "x", "verdict": "yes", "parsed": True}]}
    table = graph_decisions_table([answer_result("p10", agent=agent), answer_result("p1", agent=quiet)])
    assert len(table) == 3
    assert "`q` → `q2`" in table[2] and "502.2 [standards]: no" in table[2] and "502.3.1 [standards]: yes?" in table[2]
