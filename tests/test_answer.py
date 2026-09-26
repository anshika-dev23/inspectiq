import pytest
from langchain_core.messages import AIMessage

from src.answer import (
    NOT_IN_SOURCES,
    REFUSAL_TEXT,
    allocate_budget,
    answer_from_contexts,
    build_messages,
    build_sources,
    parse_citations,
    shorten,
)
from src.llm import LLM, LLMResponse, token_counts
from src.retrieve import RetrievedContext


def make_context(section_id: str, text: str, pages: str = "123", focus=None,
                 source: str = "ada_2010_standards.pdf") -> RetrievedContext:
    start, end = int(pages.split(",")[0]), int(pages.split(",")[-1])
    page_label = f"{start}" if start == end else f"{start}-{end}"
    return RetrievedContext(
        parent_id=f"id-{section_id}", citation=f"[{source} p.{page_label} §{section_id}]", source=source,
        code_name="ADA 2010 Standards", section_id=section_id, section_title="Title",
        breadcrumb=f"ADA 2010 Standards > {section_id} Title", pages=pages, text=text, window=None,
        parent_length=len(text), rrf_score=0.03, rerank_score=5.0, pinned=False, ranks={}, focus=focus,
    )


class FakeLLM:
    """Stands in for LLM: returns a canned reply and remembers the prompts it was given."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list = []

    def complete(self, messages) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(text=self.reply, provider="fake", model="fake", input_tokens=10, output_tokens=5,
                           latency_ms=12.0)


DOOR = make_context("404.2.3", "404.2.3 Clear Width. Door openings shall provide a clear width of 32 inches minimum.",
                    pages="123,124")
AUTO = make_context("404.3.1", "404.3.1 Clear Width. Doorways shall provide a clear opening of 32 inches.", pages="131")


# --- context budget ---------------------------------------------------------

@pytest.mark.parametrize("lengths, budget, expected", [
    ([1000, 1000, 1000], 6000, [1000, 1000, 1000]),   # everything fits
    ([6000, 6000, 6000], 6000, [2000, 2000, 2000]),   # equal shares
    ([500, 6000, 6000], 6000, [500, 2750, 2750]),     # the short source's unused share goes to the others
    ([6000, 300], 6000, [5700, 300]),                 # rank order does not matter, length does
    ([], 6000, []),
])
def test_allocate_budget(lengths, budget, expected):
    assert allocate_budget(lengths, budget) == expected
    assert sum(allocate_budget(lengths, budget)) <= budget


def test_shorten_keeps_the_start_without_focus():
    text, truncated = shorten("a" * 50 + "b" * 50, None, 50)
    assert truncated and text.startswith("a" * 50) and "b" not in text


def test_shorten_keeps_the_part_around_the_focus():
    text = "x" * 1000 + "THE ANSWER" + "y" * 1000
    shortened, truncated = shorten(text, (1000, 1010), 200)
    assert truncated and "THE ANSWER" in shortened
    assert shortened.startswith("[…]") and shortened.endswith("[…]")


def test_shorten_leaves_short_text_alone():
    assert shorten("short", (0, 5), 100) == ("short", False)


def test_build_sources_respects_the_cap_and_labels_in_rank_order():
    contexts = [make_context(f"40{i}.1", f"{i}" * 5000, focus=(0, 400)) for i in range(3)]
    sources = build_sources(contexts, 6000)
    assert [s.label for s in sources] == ["S1", "S2", "S3"]
    assert [s.context.section_id for s in sources] == ["400.1", "401.1", "402.1"]
    source_chars = sum(len(s.text.replace("[…]", "").strip()) for s in sources)
    assert source_chars <= 6000
    assert all(s.truncated for s in sources)


def test_build_messages_contains_labels_citations_rules_and_question():
    system, user = build_messages("How wide must a door be?", build_sources([DOOR, AUTO], 6000))
    assert NOT_IN_SOURCES in system.content and "[S1]" in system.content
    assert "[S1] [ada_2010_standards.pdf p.123-124 §404.2.3]" in user.content
    assert "[S2] [ada_2010_standards.pdf p.131 §404.3.1]" in user.content
    assert "32 inches minimum" in user.content
    assert "Question: How wide must a door be?" in user.content


# --- citation parsing -------------------------------------------------------

@pytest.mark.parametrize("text, valid, invalid", [
    ("Doors need 32 inches [S1].", ["S1"], []),
    ("32 inches [S1, S2] and again [S1].", ["S1", "S2"], []),
    ("32 inches [S2][S1]; see [S1; S2].", ["S2", "S1"], []),
    ("32 inches [S7].", [], ["S7"]),
    ("32 inches [S1] and [S9].", ["S1"], ["S9"]),
    ("32 inches per S1.", [], []),            # a bare label is not a citation
    ("See [1] or [Source 1].", [], []),
])
def test_parse_citations(text, valid, invalid):
    assert parse_citations(text, {"S1", "S2", "S3"}) == (valid, invalid)


# --- the chain with a fake LLM ----------------------------------------------

def test_no_contexts_refuses_without_calling_the_llm():
    llm = FakeLLM("should not be called")
    result = answer_from_contexts("capital of France?", [], llm, 6000, retrieval_ms=40.0)
    assert result.refused and result.refusal_reason == "no_relevant_sources"
    assert result.text == REFUSAL_TEXT
    assert llm.calls == []
    assert result.timings_ms == {"retrieval": 40.0, "llm": 0.0, "total": 40.0}


def test_model_not_in_sources_is_a_refusal():
    result = answer_from_contexts("q", [DOOR], FakeLLM(NOT_IN_SOURCES), 6000)
    assert result.refused and result.refusal_reason == "model_not_in_sources"


def test_not_in_sources_anywhere_in_the_reply_is_a_refusal():
    result = answer_from_contexts("q", [DOOR], FakeLLM(f"Doors are 32 inches [S1]. {NOT_IN_SOURCES}"), 6000)
    assert result.refused and result.refusal_reason == "model_not_in_sources"


def test_answer_without_citation_is_a_refusal():
    result = answer_from_contexts("q", [DOOR], FakeLLM("Doors must be 32 inches wide."), 6000)
    assert result.refused and result.refusal_reason == "no_valid_citation"
    assert result.raw_llm_text == "Doors must be 32 inches wide."


def test_answer_citing_only_unknown_labels_is_a_refusal():
    result = answer_from_contexts("q", [DOOR], FakeLLM("Doors must be 32 inches wide [S4]."), 6000)
    assert result.refused and result.refusal_reason == "no_valid_citation"
    assert result.invalid_labels == ["S4"]


def test_valid_answer_resolves_citations():
    llm = FakeLLM("Door openings need a 32 inch clear width [S1]; automatic doors too [S2].")
    result = answer_from_contexts("How wide?", [DOOR, AUTO], llm, 6000, retrieval_ms=50.0)
    assert not result.refused and result.refusal_reason is None
    assert [(c.label, c.section_id, c.page_start, c.page_end) for c in result.citations] == [
        ("S1", "404.2.3", 123, 124), ("S2", "404.3.1", 131, 131),
    ]
    assert result.citations[0].citation == "[ada_2010_standards.pdf p.123-124 §404.2.3]"
    assert result.timings_ms == {"retrieval": 50.0, "llm": 12.0, "total": 62.0}
    assert len(llm.calls) == 1


def test_valid_answer_keeps_track_of_invalid_labels():
    result = answer_from_contexts("q", [DOOR], FakeLLM("32 inches [S1], see also [S5]."), 6000)
    assert not result.refused
    assert [c.label for c in result.citations] == ["S1"] and result.invalid_labels == ["S5"]


# --- LLM wrapper ------------------------------------------------------------

def test_token_counts_ollama_and_anthropic():
    assert token_counts("ollama", {"prompt_eval_count": 812, "eval_count": 40}, None) == (812, 40)
    assert token_counts("anthropic", {}, {"input_tokens": 900, "output_tokens": 55}) == (900, 55)
    assert token_counts("ollama", {}, None) == (None, None)


class FakeChatModel:
    def invoke(self, messages):
        return AIMessage("ok [S1]", response_metadata={"prompt_eval_count": 30, "eval_count": 3})


def test_llm_wrapper_records_tokens_and_latency():
    from src.config import load_settings
    from dataclasses import replace

    llm = LLM(replace(load_settings(), llm_provider="ollama"), chat_model=FakeChatModel())
    response = llm.complete([])
    assert (response.text, response.input_tokens, response.output_tokens, response.cost_usd) == ("ok [S1]", 30, 3, 0.0)
    assert response.latency_ms >= 0 and response.provider == "ollama"
