"""Step 7 — checklist flow: measured items -> grounded requirement -> deterministic outcome -> human review -> report.

The LLM only finds and cites the rule: each item's question goes through answer() (the step-3 chain with the
citation check and number grounding). Code does the arithmetic: compute_outcome() reads the requirement's
minimum / maximum / range and compares the measured value -> pass / fail / needs_review.

The orchestration is a fixed LangGraph graph, not model-chosen tool calls (llama3.2:3b is unreliable at tool
calling):  START -> draft_findings -> human_review (interrupt) -> write_report -> END
The graph pauses at human_review; scripts/review.py shows the drafted findings, collects approve / edit
decisions and resumes the graph from its checkpoint.
"""
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.answer import Answer
from src.grounding import NumberMention, extract_numbers_with_spans

OUTCOMES = ("pass", "fail", "needs_review")

# Words around a number that make it a limit: "32 inches (815 mm) minimum", "at least 32 inches",
# "36 inches (915 mm) wide minimum", "not steeper than 1:12".
QUALIFIER_AFTER = r"^\s*(?:\([^)]*\)\s*)?(?:(?:high|wide|long|deep|clear)\s+)?"
MIN_AFTER = re.compile(QUALIFIER_AFTER + r"(?:minimum|min)\b", re.IGNORECASE)
MAX_AFTER = re.compile(QUALIFIER_AFTER + r"(?:maximum|max)\b", re.IGNORECASE)
# Before a number, the NEAREST qualifier decides ("... not a minimum height, but a maximum height of 17 inches":
# maximum). Nothing but words may stand between it and the number: no digit, no sentence break.
MIN_WORDS = re.compile(r"at least|no less than|not less than|\bminimum\b", re.IGNORECASE)
MAX_WORDS = re.compile(r"no more than|not more than|not (?:to )?exceed(?:ing)?|at most|no higher than|not higher than|"
                       r"(?:no|not) steeper than|up to|\bmaximum\b", re.IGNORECASE)
WORDS_ONLY = re.compile(r"^[^\d.;:]{0,50}$")
# "between 33 and 36 inches", "33 to 36 inches", "33-36 inches"
RANGE_JOIN_TO = re.compile(r"^\s*(?:\([^)]*\)\s*)?(?:to|-|–)\s*$", re.IGNORECASE)
RANGE_JOIN_AND = re.compile(r"^\s*(?:\([^)]*\)\s*)?and\s*$", re.IGNORECASE)
BETWEEN_BEFORE = re.compile(r"(?:between|from)\s*$", re.IGNORECASE)


# =============================================================================
# compute_outcome: the deterministic comparison
# =============================================================================

@dataclass(frozen=True)
class Limit:
    kind: str       # "min" or "max"
    value: float    # inches, or rise/run for a slope ratio
    raw: str        # as written: "32 inches", "1:12"


@dataclass(frozen=True)
class Outcome:
    status: str                 # pass | fail | needs_review
    reason: str
    minimum: str | None = None  # as written in the requirement
    maximum: str | None = None


def mention_unit(mention: NumberMention) -> str | None:
    """A slope ratio ("1:12") has no unit in the grounding extractor; here it is the unit "ratio"."""
    return "ratio" if ":" in mention.value else mention.unit


def numeric_value(value: str) -> float:
    """ "1:12" -> 1/12 (rise over run, so a steeper slope is a bigger number); "32" -> 32.0."""
    if ":" in value:
        rise, run = value.split(":")
        return float(Fraction(int(rise), int(run)))
    return float(value)


def qualifier(after: str, before: str) -> str | None:
    """"min", "max" or None for a number, from the words right after it ("32 inches (815 mm) minimum") or else
    the nearest qualifier before it ("at least 32 inches", "the maximum running slope is 1:12")."""
    if MIN_AFTER.match(after):
        return "min"
    if MAX_AFTER.match(after):
        return "max"
    candidates = [(match.end(), kind) for kind, words in (("min", MIN_WORDS), ("max", MAX_WORDS))
                  for match in words.finditer(before) if WORDS_ONLY.match(before[match.end():])]
    return max(candidates)[1] if candidates else None


def parse_limits(requirement: str, unit: str) -> list[Limit]:
    """Minimums and maximums in the requirement, in the measured unit only (the mm conversions are ignored)."""
    spans = extract_numbers_with_spans(requirement)
    limits: list[Limit] = []
    for mention, start, end in spans:
        if mention_unit(mention) != unit:
            continue
        kind = qualifier(requirement[end:end + 40], requirement[max(0, start - 70):start])
        if kind is not None:
            limits.append(Limit(kind, numeric_value(mention.value), mention.raw))

    # Ranges without the words minimum / maximum: "33 to 36 inches", "between 33 and 36 inches".
    for (low, low_start, low_end), (high, high_start, _) in zip(spans, spans[1:]):
        if mention_unit(high) != unit or mention_unit(low) not in (None, unit):
            continue
        join = requirement[low_end:high_start]
        before = requirement[max(0, low_start - 10):low_start]
        if RANGE_JOIN_TO.match(join) or (RANGE_JOIN_AND.match(join) and BETWEEN_BEFORE.search(before)):
            limits.append(Limit("min", numeric_value(low.value), low.raw))
            limits.append(Limit("max", numeric_value(high.value), high.raw))
    return limits


def format_measured(measured: dict) -> str:
    return str(measured["value"]) if measured["unit"] == "ratio" else f"{measured['value']} {measured['unit']}"


def compute_outcome(requirement: str | None, grounding_status: str | None, measured: dict,
                    refusal_reason: str | None = None) -> Outcome:
    """pass / fail / needs_review for one measurement against one grounded requirement.

    needs_review when there is no verified requirement, or it does not state exactly one minimum and/or one
    maximum in the measured unit (several different limits are ambiguous: code does not guess which applies).
    """
    if requirement is None:
        return Outcome("needs_review", f"no grounded requirement (refused: {refusal_reason})")
    if grounding_status != "verified":
        return Outcome("needs_review", f"requirement not verified (number grounding: {grounding_status})")

    unit = measured["unit"]
    limits = parse_limits(requirement, unit)
    minimums = {limit.value: limit.raw for limit in limits if limit.kind == "min"}
    maximums = {limit.value: limit.raw for limit in limits if limit.kind == "max"}
    if not minimums and not maximums:
        return Outcome("needs_review", f"no minimum or maximum in {unit} found in the requirement")
    if len(minimums) > 1 or len(maximums) > 1:
        return Outcome("needs_review", f"several different limits: minimums {sorted(minimums.values())}, "
                                       f"maximums {sorted(maximums.values())}")

    minimum = next(iter(minimums.items()), None)   # (value, raw)
    maximum = next(iter(maximums.items()), None)
    if minimum and maximum and minimum[0] > maximum[0]:
        return Outcome("needs_review", f"minimum {minimum[1]} is above maximum {maximum[1]}")

    value = numeric_value(str(measured["value"]))
    shown = format_measured(measured)
    limits_text = dict(minimum=minimum[1] if minimum else None, maximum=maximum[1] if maximum else None)
    if minimum and value < minimum[0]:
        return Outcome("fail", f"{shown} is below the minimum {minimum[1]}", **limits_text)
    if maximum and value > maximum[0]:
        steeper = " (steeper)" if unit == "ratio" else ""
        return Outcome("fail", f"{shown} is above the maximum {maximum[1]}{steeper}", **limits_text)
    return Outcome("pass", f"{shown} is within the limits", **limits_text)


# =============================================================================
# Findings
# =============================================================================

def draft_finding(item: dict, answer_fn: Callable[[str], Answer]) -> dict:
    """Ask the chain for the grounded requirement, then compare the measurement in code."""
    result = answer_fn(item["question"])
    requirement = None if result.refused else result.text
    outcome = compute_outcome(requirement, result.grounding_status, item["measured"], result.refusal_reason)
    return {
        "item_id": item["id"],
        "element": item["element"],
        "measured": item["measured"],
        "question": item["question"],
        "requirement": requirement,
        "citations": [c.citation for c in result.citations],
        "grounding_status": result.grounding_status,
        "flagged_numbers": [f.raw for f in result.flagged_numbers],
        "refusal_reason": result.refusal_reason,
        "outcome": outcome.status,
        "outcome_reason": outcome.reason,
        "limits": {"minimum": outcome.minimum, "maximum": outcome.maximum},
        "trace_url": result.trace_url,
    }


def apply_decisions(findings: list[dict], decisions: list[dict]) -> list[dict]:
    """Merge the reviewer's decisions: {"item_id", "action": "approve" | "edit", "outcome"?, "note"?}.
    Every finding needs exactly one decision; an edit must give a valid outcome."""
    by_item = {decision["item_id"]: decision for decision in decisions}
    unknown = set(by_item) - {finding["item_id"] for finding in findings}
    missing = {finding["item_id"] for finding in findings} - set(by_item)
    if unknown or missing:
        raise ValueError(f"decisions do not match the findings: unknown {sorted(unknown)}, missing {sorted(missing)}")

    reviewed = []
    for finding in findings:
        decision = by_item[finding["item_id"]]
        if decision["action"] == "approve":
            final = finding["outcome"]
        elif decision["action"] == "edit":
            final = decision.get("outcome")
            if final not in OUTCOMES:
                raise ValueError(f"{finding['item_id']}: edited outcome must be one of {OUTCOMES}, got {final!r}")
        else:
            raise ValueError(f"{finding['item_id']}: action must be approve or edit, got {decision['action']!r}")
        reviewed.append({**finding, "final_outcome": final,
                         "review": {"action": decision["action"], "note": decision.get("note", ""),
                                    "changed": final != finding["outcome"]}})
    return reviewed


def report_markdown(checklist: dict, findings: list[dict], reviewed_at: str) -> str:
    lines = [f"# Findings: {checklist['title']}", "",
             f"Checklist `{checklist['id']}`, reviewed {reviewed_at}. Outcomes: drafted by code from the grounded "
             "requirement, then approved or edited by the inspector.", "",
             "| item | measured | requirement limits | citation | drafted | final | reviewer note |",
             "|---|---|---|---|---|---|---|"]
    for f in findings:
        limits = ", ".join(f"{k} {v}" for k, v in f["limits"].items() if v) or "–"
        citation = "<br>".join(f["citations"]) or "–"
        final = f["final_outcome"] + (" (edited)" if f["review"]["changed"] else "")
        lines.append(f"| {f['element']} | {format_measured(f['measured'])} | {limits} | {citation} | "
                     f"{f['outcome']} | **{final}** | {f['review']['note'] or '–'} |")
    lines += ["", "## Requirements as found", ""]
    for f in findings:
        lines += [f"### {f['element']}", "",
                  f"- question: {f['question']}",
                  f"- drafted outcome: {f['outcome']} ({f['outcome_reason']})",
                  f"- number grounding: {f['grounding_status'] or 'n/a'}"
                  + (f", flagged: {f['flagged_numbers']}" if f["flagged_numbers"] else ""),
                  "", "> " + (f["requirement"] or f"(no grounded requirement: {f['refusal_reason']})").replace("\n", "\n> "),
                  ""]
    return "\n".join(lines)


def write_report(checklist: dict, findings: list[dict], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = output_dir / f"{checklist['id']}_findings"
    report = {"checklist_id": checklist["id"], "title": checklist["title"], "reviewed_at": reviewed_at,
              "findings": findings}
    base.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    base.with_suffix(".md").write_text(report_markdown(checklist, findings, reviewed_at), encoding="utf-8")
    return {"json": str(base.with_suffix(".json")), "markdown": str(base.with_suffix(".md"))}


# =============================================================================
# The review graph: draft -> interrupt for the inspector -> report
# =============================================================================

class ReviewState(TypedDict, total=False):
    checklist: dict
    findings: list[dict]
    report: dict[str, str]


def draft_findings_node(state: ReviewState, answer_fn: Callable[[str], Answer]) -> dict:
    return {"findings": [draft_finding(item, answer_fn) for item in state["checklist"]["items"]]}


def human_review_node(state: ReviewState) -> dict:
    """Pause the graph. interrupt() saves the state in the checkpoint and hands the findings to the caller;
    the graph resumes here with the caller's decisions (Command(resume=decisions)).
    On resume this node runs again from its start, so it only interrupts and applies: no side effects before."""
    decisions = interrupt({"checklist_id": state["checklist"]["id"], "findings": state["findings"]})
    return {"findings": apply_decisions(state["findings"], decisions)}


def write_report_node(state: ReviewState, output_dir: Path) -> dict:
    return {"report": write_report(state["checklist"], state["findings"], output_dir)}


def build_review_graph(answer_fn: Callable[[str], Answer], output_dir: Path, checkpointer):
    """A checkpointer is required: interrupt() can only pause a graph that saves its state."""
    graph = StateGraph(ReviewState)
    graph.add_node("draft_findings", partial(draft_findings_node, answer_fn=answer_fn))
    graph.add_node("human_review", human_review_node)
    graph.add_node("write_report", partial(write_report_node, output_dir=output_dir))
    graph.add_edge(START, "draft_findings")
    graph.add_edge("draft_findings", "human_review")
    graph.add_edge("human_review", "write_report")
    graph.add_edge("write_report", END)
    return graph.compile(checkpointer=checkpointer)


def pending_findings(graph, config: dict) -> list[dict] | None:
    """The findings waiting for review in this thread's checkpoint, or None if the graph is not paused there."""
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        for pending in task.interrupts:
            return pending.value["findings"]
    return None
