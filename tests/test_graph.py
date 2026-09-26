from types import SimpleNamespace

import pytest

from src.graph import MAX_RETRIES, mermaid, parse_grade, parse_rewrite, route_after_grading, graph_answer
from src.llm import LLMResponse
from tests.test_answer import make_context

DOOR = make_context("404.2.3", "404.2.3 Clear Width. Door openings shall provide a clear width of 32 inches minimum.")
SIGNS = make_context("216.2", "216.2 Designations. Interior and exterior signs identifying permanent rooms.")
AUTO = make_context("404.3.1", "404.3.1 Clear Width. Doorways shall provide a clear opening of 32 inches.")


class ScriptedLLM:
    """Replies by prompt type: grading -> grade(section text), rewriting -> a fixed query, answering -> answer."""

    def __init__(self, grade=lambda text: "yes", rewrite="door clear width", answer="Doors need 32 inches [S1]."):
        self.grade, self.rewrite, self.answer = grade, rewrite, answer
        self.calls: list[str] = []

    def complete(self, messages) -> LLMResponse:
        system, user = messages[0].content, messages[-1].content
        if "Reply with one word: yes or no" in system:
            kind, text = "grade", self.grade(user)
        elif "search queries" in system:
            kind, text = "rewrite", self.rewrite
        else:
            kind, text = "answer", self.answer
        self.calls.append(kind)
        return LLMResponse(text=text, provider="fake", model="fake", input_tokens=10, output_tokens=2, latency_ms=5.0)


def fake_retrieve(results_by_query: dict[str, list]):
    """retrieve() stand-in: contexts by query (default: nothing); records the queries it was asked."""
    asked = []

    def retrieve_fn(query, filters, config, resources):
        asked.append(query)
        contexts = results_by_query.get(query, [])
        debug = SimpleNamespace(normalized=SimpleNamespace(text=query), timings_ms={"total": 7.0})
        return SimpleNamespace(contexts=contexts, debug=debug)

    retrieve_fn.asked = asked
    return retrieve_fn


QUESTION = "How wide must a doorway be?"


# --- parsing and routing ------------------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    ("yes", (True, True)),
    ("No.", (False, True)),
    ("YES - it gives the clear width", (True, True)),
    ("The source is relevant: yes", (True, True)),
    ("no, it is about signs", (False, True)),
    ("", (True, False)),                        # parse failure counts as relevant
    ("The passage discusses doors.", (True, False)),
    ("Nope", (True, False)),                    # not the word "no"
])
def test_parse_grade(reply, expected):
    assert parse_grade(reply) == expected


@pytest.mark.parametrize("reply, expected", [
    ("door clear width", "door clear width"),
    ('"door clear width"', "door clear width"),
    ("Search query: door clear width\nexplanation", "door clear width"),
    ("\n\n  operable parts reach range  ", "operable parts reach range"),
    ("", "original question"),
])
def test_parse_rewrite(reply, expected):
    assert parse_rewrite(reply, fallback="original question") == expected


def test_route_after_grading():
    assert route_after_grading({"relevant": [DOOR]}, rewrite=True) == "generate"
    assert route_after_grading({"relevant": [], "retries": 0}, rewrite=True) == "rewrite_query"
    assert route_after_grading({"relevant": [], "retries": MAX_RETRIES}, rewrite=True) == "refuse"
    assert route_after_grading({"relevant": [], "retries": 0}, rewrite=False) == "refuse"


# --- the graph with a scripted LLM and a fake retrieve ------------------------

def test_grade_only_filters_out_irrelevant_sources():
    llm = ScriptedLLM(grade=lambda prompt: "no" if "216.2" in prompt else "yes")
    retrieve_fn = fake_retrieve({QUESTION: [SIGNS, DOOR]})
    answer = graph_answer(QUESTION, rewrite=False, llm=llm, retrieve_fn=retrieve_fn)
    assert not answer.refused
    assert [c.section_id for c in answer.citations] == ["404.2.3"]   # S1 is now the door section
    assert llm.calls == ["grade", "grade", "answer"] and answer.llm_calls == 3
    assert [g["verdict"] for g in answer.agent["grades"]] == ["no", "yes"]


def test_grade_only_refuses_when_nothing_is_relevant():
    llm = ScriptedLLM(grade=lambda prompt: "no")
    answer = graph_answer(QUESTION, rewrite=False, llm=llm, retrieve_fn=fake_retrieve({QUESTION: [SIGNS]}))
    assert answer.refused and answer.refusal_reason == "graded_not_relevant"
    assert llm.calls == ["grade"] and answer.llm_calls == 1


def test_empty_retrieval_refuses_without_llm_calls_in_grade_only():
    llm = ScriptedLLM()
    answer = graph_answer(QUESTION, rewrite=False, llm=llm, retrieve_fn=fake_retrieve({}))
    assert answer.refused and answer.refusal_reason == "no_relevant_sources" and llm.calls == []


def test_rewrite_finds_a_relevant_source_on_retry():
    llm = ScriptedLLM(grade=lambda prompt: "no" if "216.2" in prompt else "yes", rewrite="door clear width")
    retrieve_fn = fake_retrieve({QUESTION: [SIGNS], "door clear width": [DOOR]})
    answer = graph_answer(QUESTION, rewrite=True, llm=llm, retrieve_fn=retrieve_fn)
    assert not answer.refused and [c.section_id for c in answer.citations] == ["404.2.3"]
    assert retrieve_fn.asked == [QUESTION, "door clear width"]
    assert llm.calls == ["grade", "rewrite", "grade", "answer"] and answer.llm_calls == 4
    assert answer.agent["retries"] == 1 and answer.agent["queries"] == [QUESTION, "door clear width"]


def test_rewrite_also_runs_when_retrieval_is_empty():
    llm = ScriptedLLM(rewrite="door clear width")
    retrieve_fn = fake_retrieve({"door clear width": [DOOR]})
    answer = graph_answer(QUESTION, rewrite=True, llm=llm, retrieve_fn=retrieve_fn)
    assert not answer.refused and llm.calls == ["rewrite", "grade", "answer"]


def test_rewrite_gives_up_after_max_retries():
    llm = ScriptedLLM(grade=lambda prompt: "no", rewrite="still wrong")
    retrieve_fn = fake_retrieve({QUESTION: [SIGNS], "still wrong": [SIGNS]})
    answer = graph_answer(QUESTION, rewrite=True, llm=llm, retrieve_fn=retrieve_fn)
    assert answer.refused and answer.refusal_reason == "graded_not_relevant"
    assert len(retrieve_fn.asked) == 1 + MAX_RETRIES
    assert llm.calls == ["grade", "rewrite", "grade", "rewrite", "grade"] and answer.llm_calls == 5


def test_unparseable_grade_keeps_the_source():
    llm = ScriptedLLM(grade=lambda prompt: "The passage discusses door widths.")
    answer = graph_answer(QUESTION, rewrite=False, llm=llm, retrieve_fn=fake_retrieve({QUESTION: [DOOR]}))
    assert not answer.refused and answer.agent["grades"][0] == {
        "query": QUESTION, "section": "404.2.3 [ada_2010_standards.pdf]", "verdict": "yes", "parsed": False,
    }


def test_generate_reuses_the_chain_checks():
    # The answer node is answer_from_contexts: an uncited reply is still refused by the citation check.
    llm = ScriptedLLM(answer="Doors need 32 inches.")
    answer = graph_answer(QUESTION, rewrite=False, llm=llm, retrieve_fn=fake_retrieve({QUESTION: [DOOR, AUTO]}))
    assert answer.refused and answer.refusal_reason == "no_valid_citation"
    assert answer.llm_calls == 3   # 2 grades + 1 answer


def test_timings_include_every_stage():
    llm = ScriptedLLM(grade=lambda prompt: "no" if "216.2" in prompt else "yes")
    retrieve_fn = fake_retrieve({QUESTION: [SIGNS], "door clear width": [DOOR]})
    answer = graph_answer(QUESTION, rewrite=True, llm=llm, retrieve_fn=retrieve_fn)
    assert answer.timings_ms["retrieval"] == 14.0      # two retrievals of 7 ms
    assert answer.timings_ms["grading"] == 10.0 and answer.timings_ms["rewrite"] == 5.0
    assert answer.timings_ms["llm"] == 5.0 and answer.timings_ms["total"] >= 0


def test_mermaid_shows_both_variants():
    with_loop, without = mermaid(rewrite=True), mermaid(rewrite=False)
    assert "rewrite_query --> retrieve" in with_loop and "grade_documents -.-> rewrite_query" in with_loop
    assert "rewrite_query" not in without and "grade_documents -.-> refuse" in without


def test_usage_sums_every_llm_call():
    llm = ScriptedLLM(grade=lambda prompt: "no" if "216.2" in prompt else "yes")
    retrieve_fn = fake_retrieve({QUESTION: [SIGNS], "door clear width": [DOOR]})
    answer = graph_answer(QUESTION, rewrite=True, llm=llm, retrieve_fn=retrieve_fn)
    # 4 calls (grade, rewrite, grade, answer) x 10 input / 2 output tokens each
    assert answer.usage["input_tokens"] == 40 and answer.usage["output_tokens"] == 8
