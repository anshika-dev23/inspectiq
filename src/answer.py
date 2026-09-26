"""Step 3 — baseline answer chain (no LangGraph): retrieve -> prompt -> LLM -> cited answer.

The rules that matter are enforced in code, not trusted to the model:
- no retrieved sources            -> refusal, the LLM is not called;
- the model says NOT_IN_SOURCES   -> refusal;
- no valid [S#] citation          -> refusal (every label is checked against the sources in the prompt).
"""
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.config import RetrievalConfig, Settings, load_settings
from src.llm import LLM, LLMResponse
from src.retrieve import RetrievalResult, RetrievedContext, choose_window, retrieve

NOT_IN_SOURCES = "NOT_IN_SOURCES"
REFUSAL_TEXT = "I don't know: the indexed code documents do not answer this question."
TRUNCATION_MARK = " […] "

# "[S1]", "[S1, S2]", "[S1; S3]"; each label inside is checked separately.
CITATION_GROUP = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
LABEL = re.compile(r"S\d+")

SYSTEM_PROMPT = f"""You answer questions about accessibility building codes using ONLY the numbered sources given.
Rules:
1. Use only facts stated in the sources. Do not use outside knowledge.
2. Cite every claim with the label of its source in square brackets, for example [S1] or [S2].
3. Give numbers and units exactly as the source states them.
4. If the sources do not answer the question, reply exactly: {NOT_IN_SOURCES}
Keep the answer short."""


# =============================================================================
# Types
# =============================================================================

@dataclass(frozen=True)
class Source:
    """A retrieved context as it appears in the prompt."""

    label: str                 # "S1", "S2", ...
    context: RetrievedContext
    text: str                  # possibly shortened to fit the context budget
    truncated: bool


@dataclass(frozen=True)
class Citation:
    label: str
    source: str
    page_start: int
    page_end: int
    section_id: str
    section_title: str
    citation: str              # "[ada_2010_standards.pdf p.123-124 §404.2.3]"


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    refusal_reason: str | None     # "no_relevant_sources" | "model_not_in_sources" | "no_valid_citation"
    citations: list[Citation] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)   # cited labels that were not in the prompt
    raw_llm_text: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    llm: LLMResponse | None = None
    sources: list[Source] = field(default_factory=list)
    retrieval: RetrievalResult | None = None


# =============================================================================
# Prompt building with a context budget
# =============================================================================

def allocate_budget(lengths: list[int], budget: int) -> list[int]:
    """Characters each source may use, in total at most `budget` ("water-filling").

    Going from the shortest source to the longest, each takes min(its length, an equal share of what is
    left). Short sources keep all their text; long ones split the rest equally.
    """
    shares = [0] * len(lengths)
    remaining = budget
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    for position, i in enumerate(order):
        equal_share = remaining // (len(lengths) - position)
        shares[i] = min(lengths[i], equal_share)
        remaining -= shares[i]
    return shares


def shorten(text: str, focus: tuple[int, int] | None, max_chars: int) -> tuple[str, bool]:
    """Fit text into max_chars: keep the part around `focus` (the best-matching chunk), or the start
    when there is no focus. Returns (text, truncated)."""
    if len(text) <= max_chars:
        return text, False
    start, end = choose_window(len(text), [focus], max_chars) if focus else (0, max_chars)
    shortened = text[start:end]
    if start > 0:
        shortened = TRUNCATION_MARK.lstrip() + shortened
    if end < len(text):
        shortened = shortened + TRUNCATION_MARK.rstrip()
    return shortened, True


def build_sources(contexts: list[RetrievedContext], max_context_chars: int) -> list[Source]:
    """Label contexts S1, S2, ... in rank order; their texts together use at most max_context_chars
    (plus the short truncation marks)."""
    shares = allocate_budget([len(c.text) for c in contexts], max_context_chars)
    sources = []
    for context, share in zip(contexts, shares):
        if share == 0:
            continue
        text, truncated = shorten(context.text, context.focus, share)
        sources.append(Source(f"S{len(sources) + 1}", context, text, truncated))
    return sources


def format_source(source: Source) -> str:
    return f"[{source.label}] {source.context.citation} {source.context.breadcrumb}\n{source.text}"


def build_messages(question: str, sources: list[Source]) -> list[BaseMessage]:
    body = "\n\n".join(format_source(source) for source in sources)
    user = (f"Sources:\n\n{body}\n\n"
            f"Question: {question}\n"
            f"Answer using only the sources, citing every claim as [S#], or reply {NOT_IN_SOURCES}.")
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage(user)]


# =============================================================================
# Citation checking
# =============================================================================

def parse_citations(text: str, valid_labels: set[str]) -> tuple[list[str], list[str]]:
    """([valid labels], [invalid labels]) cited in the text, each in order of first appearance."""
    valid, invalid = [], []
    for group in CITATION_GROUP.finditer(text):
        for label in LABEL.findall(group[1]):
            target = valid if label in valid_labels else invalid
            if label not in target:
                target.append(label)
    return valid, invalid


def resolve_citation(source: Source) -> Citation:
    context = source.context
    pages = [int(p) for p in context.pages.split(",")]
    return Citation(source.label, context.source, pages[0], pages[-1], context.section_id,
                    context.section_title, context.citation)


# =============================================================================
# The chain
# =============================================================================

def refusal(question: str, reason: str, **fields) -> Answer:
    return Answer(question=question, text=REFUSAL_TEXT, refused=True, refusal_reason=reason, **fields)


def answer_from_contexts(question: str, contexts: list[RetrievedContext], llm: LLM,
                         max_context_chars: int, retrieval_ms: float = 0.0) -> Answer:
    """Prompt -> LLM -> checked, cited answer. Separate from retrieval so it can be tested with a fake LLM."""
    timings = {"retrieval": retrieval_ms}
    if not contexts:
        return refusal(question, "no_relevant_sources", timings_ms={**timings, "llm": 0.0, "total": retrieval_ms})

    sources = build_sources(contexts, max_context_chars)
    response = llm.complete(build_messages(question, sources))
    timings.update(llm=response.latency_ms, total=round(retrieval_ms + response.latency_ms, 1))
    common = dict(raw_llm_text=response.text, timings_ms=timings, llm=response, sources=sources)

    if NOT_IN_SOURCES in response.text:
        return refusal(question, "model_not_in_sources", **common)

    by_label = {source.label: source for source in sources}
    valid, invalid = parse_citations(response.text, set(by_label))
    if not valid:
        return refusal(question, "no_valid_citation", invalid_labels=invalid, **common)

    return Answer(question=question, text=response.text.strip(), refused=False, refusal_reason=None,
                  citations=[resolve_citation(by_label[label]) for label in valid], invalid_labels=invalid, **common)


@lru_cache(maxsize=1)
def get_default_llm() -> LLM:
    return LLM(load_settings())


def answer(question: str, filters: dict | None = None, retrieval_config: RetrievalConfig | None = None,
           settings: Settings | None = None, llm: LLM | None = None, resources=None) -> Answer:
    """Answer a question from the indexed codes, with [S#] citations, or refuse."""
    settings = settings or load_settings()
    start = time.perf_counter()
    retrieval = retrieve(question, filters, retrieval_config, resources)
    retrieval_ms = round((time.perf_counter() - start) * 1000, 1)
    result = answer_from_contexts(question, retrieval.contexts, llm or get_default_llm(),
                                  settings.answer_max_context_chars, retrieval_ms)
    result.retrieval = retrieval
    return result
