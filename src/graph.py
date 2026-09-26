"""Step 6 — LangGraph agent: retrieve -> grade_documents -> generate, with an optional rewrite loop.

The graph only orchestrates existing pieces; it does not re-implement them:
- retrieve        -> src.retrieve.retrieve()
- generate        -> src.answer.answer_from_contexts() (prompt builder, citation check, number grounding)
New here:
- grade_documents: one LLM call per source asking only "relevant: yes/no". A reply without a yes/no counts
  as "yes", so a small model's formatting slips never throw away a good source.
- rewrite_query:   when no source is graded relevant, the LLM rewrites the search query in code vocabulary and
  retrieval runs again (max 2 retries), then the graph refuses.

Two variants, so the evals can measure what the loop adds (build_graph(rewrite=...)):
  grade only:       START -> retrieve -> grade_documents -> generate | refuse -> END
  grade + rewrite:  ... grade_documents -> rewrite_query -> retrieve -> grade_documents ... (at most 2 loops)
Tools, human-in-the-loop and the checklist flow come in step 7.
"""
import re
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src import tracing
from src.answer import Answer, answer_from_contexts, build_sources, get_default_llm, refusal
from src.config import RetrievalConfig, load_settings
from src.llm import LLM, add_usage
from src.retrieve import RetrievedContext, retrieve

MAX_RETRIES = 2

GRADE_SYSTEM = ("You judge whether a source passage from an accessibility building code is relevant to a question. "
                "Reply with one word: yes or no.")
REWRITE_SYSTEM = ("You rewrite questions about accessibility building codes into short search queries. Use the "
                  "vocabulary of the 2010 ADA Standards, for example: clear width, clear floor space, operable parts, "
                  "reach range, water closet, lavatory, running slope, turning space, accessible route. "
                  "Reply with the search query only.")
YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


# =============================================================================
# State and dependencies
# =============================================================================

class GraphState(TypedDict, total=False):
    question: str                        # the user's question: grading and answering always use it
    filters: dict | None
    rewritten: str | None                # current search query after a rewrite (None: search with the question)
    normalized: str                      # the last query as retrieve() normalized it (abbreviations, glossary)
    documents: list[RetrievedContext]    # the last retrieval's contexts
    relevant: list[RetrievedContext]     # those graded relevant
    queries: list[str]                   # every search query tried, in order
    grades: list[dict]                   # every grade: query, section, verdict, parsed
    retries: int
    llm_calls: int
    usage: dict                          # tokens and shadow cost of the grading and rewrite calls
    timings_ms: dict[str, float]         # retrieval / grading / rewrite, summed over loops
    answer: Answer


@dataclass
class GraphDeps:
    llm: LLM
    max_context_chars: int
    retrieval_config: RetrievalConfig | None = None
    resources: object = None
    retrieve_fn: Callable = field(default=retrieve)   # tests pass a fake


def add_time(state: GraphState, stage: str, ms: float) -> dict[str, float]:
    timings = dict(state.get("timings_ms", {}))
    timings[stage] = round(timings.get(stage, 0.0) + ms, 1)
    return timings


# =============================================================================
# Parsing the small model's replies
# =============================================================================

def parse_grade(reply: str) -> tuple[bool, bool]:
    """(relevant, parsed). The first "yes" or "no" in the reply decides; neither -> (True, False)."""
    match = YES_NO.search(reply)
    if match is None:
        return True, False
    return match[1].lower() == "yes", True


def parse_rewrite(reply: str, fallback: str) -> str:
    """First non-empty line, without quotes or a 'Search query:' prefix; empty -> fallback."""
    for line in reply.splitlines():
        line = re.sub(r"^\s*(?:search\s+query\s*:)?\s*", "", line, flags=re.IGNORECASE).strip().strip('"\'`').strip()
        if line:
            return line
    return fallback


# =============================================================================
# Nodes
# =============================================================================

def retrieve_node(state: GraphState, deps: GraphDeps) -> dict:
    query = state.get("rewritten") or state["question"]
    with tracing.observe("retrieve", as_type="retriever", input=query) as span:
        result = deps.retrieve_fn(query, state.get("filters"), deps.retrieval_config, deps.resources)
        tracing.update(span, output=[c.citation for c in result.contexts],
                       metadata={"stage_timings_ms": result.debug.timings_ms})
    return {
        "documents": result.contexts,
        "normalized": result.debug.normalized.text,
        "queries": state.get("queries", []) + [query],
        "timings_ms": add_time(state, "retrieval", result.debug.timings_ms["total"]),
    }


def grade_messages(question: str, citation: str, breadcrumb: str, text: str) -> list:
    return [SystemMessage(GRADE_SYSTEM),
            HumanMessage(f"Question: {question}\n\nSource {citation} {breadcrumb}\n{text}\n\n"
                         "Does this source contain information that helps answer the question? "
                         "Reply with one word: yes or no.")]


def grade_documents_node(state: GraphState, deps: GraphDeps) -> dict:
    """One call per source, graded on the text the generator would see (same context budget)."""
    relevant, grades, grading_ms, usage = [], list(state.get("grades", [])), 0.0, state.get("usage")
    query = state.get("rewritten") or state["question"]
    with tracing.observe("grade_documents", as_type="evaluator", input={"question": state["question"]}) as span:
        for source in build_sources(state["documents"], deps.max_context_chars):
            context = source.context
            response = deps.llm.complete(grade_messages(state["question"], context.citation, context.breadcrumb,
                                                        source.text))
            is_relevant, parsed = parse_grade(response.text)
            grading_ms += response.latency_ms
            usage = add_usage(usage, response)
            grades.append({"query": query, "section": f"{context.section_id} [{context.source}]",
                           "verdict": "yes" if is_relevant else "no", "parsed": parsed})
            if is_relevant:
                relevant.append(context)
        tracing.update(span, output=grades[len(state.get("grades", [])):])
    return {
        "relevant": relevant,
        "grades": grades,
        "llm_calls": state.get("llm_calls", 0) + len(state["documents"]),
        "usage": usage,
        "timings_ms": add_time(state, "grading", grading_ms),
    }


def rewrite_query_node(state: GraphState, deps: GraphDeps) -> dict:
    tried = "; ".join(state.get("queries", []))
    response = deps.llm.complete([
        SystemMessage(REWRITE_SYSTEM),
        HumanMessage(f"Question: {state['question']}\n"
                     f"Queries already tried without finding a relevant source: {tried}\nSearch query:"),
    ])
    return {
        "rewritten": parse_rewrite(response.text, fallback=state["question"]),
        "retries": state.get("retries", 0) + 1,
        "llm_calls": state.get("llm_calls", 0) + 1,
        "usage": add_usage(state.get("usage"), response),
        "timings_ms": add_time(state, "rewrite", response.latency_ms),
    }


def generate_node(state: GraphState, deps: GraphDeps) -> dict:
    """The step-3 chain on the relevant sources only (prompt, citation check, number grounding)."""
    timings = state.get("timings_ms", {})
    answer = answer_from_contexts(state["question"], state["relevant"], deps.llm, deps.max_context_chars,
                                  retrieval_ms=timings.get("retrieval", 0.0))
    answer.llm_calls += state.get("llm_calls", 0)
    if answer.llm is not None:   # add the answer call to the grading / rewrite totals
        answer.usage = add_usage(state.get("usage"), answer.llm)
    return {"answer": answer}


def refuse_node(state: GraphState, deps: GraphDeps) -> dict:
    """Nothing relevant: "no_relevant_sources" if retrieval found nothing, "graded_not_relevant" if the grader
    rejected everything it found."""
    reason = "graded_not_relevant" if state.get("documents") else "no_relevant_sources"
    return {"answer": refusal(state["question"], reason, llm_calls=state.get("llm_calls", 0),
                              usage=state.get("usage"))}


def route_after_grading(state: GraphState, rewrite: bool) -> str:
    if state.get("relevant"):
        return "generate"
    if rewrite and state.get("retries", 0) < MAX_RETRIES:
        return "rewrite_query"
    return "refuse"


# =============================================================================
# Graph
# =============================================================================

def build_graph(deps: GraphDeps, rewrite: bool = True):
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", partial(retrieve_node, deps=deps))
    graph.add_node("grade_documents", partial(grade_documents_node, deps=deps))
    graph.add_node("generate", partial(generate_node, deps=deps))
    graph.add_node("refuse", partial(refuse_node, deps=deps))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    targets = ["generate", "refuse"]
    if rewrite:
        graph.add_node("rewrite_query", partial(rewrite_query_node, deps=deps))
        graph.add_edge("rewrite_query", "retrieve")
        targets.append("rewrite_query")
    graph.add_conditional_edges("grade_documents", partial(route_after_grading, rewrite=rewrite), targets)
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def graph_answer(question: str, filters: dict | None = None, rewrite: bool = True, llm: LLM | None = None,
                 retrieval_config: RetrievalConfig | None = None, resources=None,
                 retrieve_fn: Callable = retrieve) -> Answer:
    """Answer with the agent graph. Returns the same Answer as the chain, plus answer.agent (queries, grades)."""
    settings = load_settings()
    deps = GraphDeps(llm=llm or get_default_llm(), max_context_chars=settings.answer_max_context_chars,
                     retrieval_config=retrieval_config, resources=resources, retrieve_fn=retrieve_fn)
    variant = "grade + rewrite" if rewrite else "grade only"
    start = time.perf_counter()
    with tracing.observe("graph_answer", as_type="agent", input={"question": question, "variant": variant}) as root:
        state = build_graph(deps, rewrite=rewrite).invoke({"question": question, "filters": filters})
        answer = state["answer"]
        answer.timings_ms = {**answer.timings_ms, **state.get("timings_ms", {}),
                             "total": round((time.perf_counter() - start) * 1000, 1)}
        answer.agent = {"variant": variant, "queries": state.get("queries", []), "grades": state.get("grades", []),
                        "retries": state.get("retries", 0)}
        output = {"answer": answer.text, "refused": answer.refused, "refusal_reason": answer.refusal_reason,
                  "grounding_status": answer.grounding_status, "llm_calls": answer.llm_calls}
        tracing.update(root, output=output, metadata=answer.agent)
        tracing.set_trace_io(root, input={"question": question}, output=output)
        answer.trace_url = tracing.current_trace_url()
    return answer


def mermaid(rewrite: bool = True) -> str:
    """The graph as a Mermaid diagram (for the README); no model is loaded or called."""
    deps = GraphDeps(llm=None, max_context_chars=0)
    return build_graph(deps, rewrite=rewrite).get_graph().draw_mermaid()
