import json
import sqlite3

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.answer import Answer, Citation
from src.checklist import (
    apply_decisions,
    build_review_graph,
    compute_outcome,
    draft_finding,
    parse_limits,
    pending_findings,
)

INCHES = lambda value: {"value": value, "unit": "in"}  # noqa: E731
RATIO = lambda value: {"value": value, "unit": "ratio"}  # noqa: E731


# --- compute_outcome ------------------------------------------------------------

@pytest.mark.parametrize("requirement, measured, status", [
    ("Door openings shall provide a clear width of 32 inches (815 mm) minimum [S1].", INCHES(30), "fail"),
    ("Door openings shall provide a clear width of 32 inches (815 mm) minimum [S1].", INCHES(32), "pass"),
    ("Grab bars shall be 33 inches (840 mm) minimum and 36 inches (915 mm) maximum above the floor [S1].", INCHES(35), "pass"),
    ("Grab bars shall be 33 inches (840 mm) minimum and 36 inches (915 mm) maximum above the floor [S1].", INCHES(37), "fail"),
    ("The seat height shall be 17 inches (430 mm) minimum and 19 inches (485 mm) maximum [S1].", INCHES(20), "fail"),
    ("Knee clearance shall be 30 inches (760 mm) wide minimum [S1].", INCHES(27), "fail"),
    ("Ramp runs shall have a running slope not steeper than 1:12 [S1].", RATIO("1:10"), "fail"),
    ("Ramp runs shall have a running slope not steeper than 1:12 [S1].", RATIO("1:12"), "pass"),
    ("Ramp runs shall have a running slope not steeper than 1:12 [S1].", RATIO("1:16"), "pass"),
    ("The maximum running slope is 1:12 [S1].", RATIO("1:10"), "fail"),
    ("Grab bars go between 33 and 36 inches above the floor [S1].", INCHES(35), "pass"),
    ("Grab bars go 33 to 36 inches above the floor [S1].", INCHES(32), "fail"),
    ("The opening must be at least 32 inches wide [S1].", INCHES(31), "fail"),
    ("The seat must be no more than 19 inches high [S1].", INCHES(18), "pass"),
    ("The minimum clear width of a door opening is 32 inches (815 mm) [S1].", INCHES(30), "fail"),
    ("The minimum width of knee clearance is 30 inches [S1].", INCHES(27), "fail"),
])
def test_compute_outcome(requirement, measured, status):
    assert compute_outcome(requirement, "verified", measured).status == status


def test_fail_reason_names_the_limit():
    outcome = compute_outcome("A clear width of 32 inches (815 mm) minimum [S1].", "verified", INCHES(30))
    assert outcome.reason == "30 in is below the minimum 32 inches" and outcome.minimum == "32 inches"
    slope = compute_outcome("not steeper than 1:12 [S1].", "verified", RATIO("1:10"))
    assert slope.reason == "1:10 is above the maximum 1:12 (steeper)"


def test_mm_conversions_are_ignored():
    assert parse_limits("32 inches (815 mm) minimum", "in")[0].raw == "32 inches"
    assert all(limit.raw != "815 mm" for limit in parse_limits("32 inches (815 mm) minimum", "in"))


@pytest.mark.parametrize("requirement, status_text", [
    # 306.3.3 has two different depth minimums: code does not guess which applies
    ("Knee clearance shall be 11 inches (280 mm) deep minimum at 9 inches above the floor, and 8 inches (205 mm) "
     "deep minimum at 27 inches above the floor [S1].", "several different limits"),
    ("Door openings shall be wide enough for a wheelchair [S1].", "no minimum or maximum"),
    ("Openings shall be 32 inches minimum; openings over 24 inches deep shall be 36 inches minimum [S1].",
     "several different limits"),
])
def test_ambiguous_or_missing_limits_need_review(requirement, status_text):
    outcome = compute_outcome(requirement, "verified", INCHES(30))
    assert outcome.status == "needs_review" and status_text in outcome.reason


def test_nearest_qualifier_wins():
    # from the live run: the model added a children's-seat remark with a second maximum
    requirement = ("The seat height shall be 17 inches (430 mm) minimum and 19 inches (485 mm) maximum [S1]. "
                   "[S2] does not provide a different minimum height, but rather a maximum height of 17 inches (430 mm) "
                   "for water closets for children's use.")
    kinds = [(limit.kind, limit.raw) for limit in parse_limits(requirement, "in")]
    assert kinds == [("min", "17 inches"), ("max", "19 inches"), ("max", "17 inches")]
    assert compute_outcome(requirement, "verified", INCHES(20)).status == "needs_review"


def test_qualifier_does_not_reach_across_another_number():
    assert parse_limits("a minimum of 9 inches from the wall and 32 inches wide", "in") == [
        parse_limits("a minimum of 9 inches", "in")[0]]


def test_unverified_or_refused_requirements_need_review():
    requirement = "A clear width of 32 inches minimum [S1]."
    assert compute_outcome(requirement, "needs_review", INCHES(30)).status == "needs_review"
    refused = compute_outcome(None, None, INCHES(30), refusal_reason="model_not_in_sources")
    assert refused.status == "needs_review" and "model_not_in_sources" in refused.reason


# --- drafting and review --------------------------------------------------------

def fake_answer(text: str, section: str = "404.2.3", status: str = "verified", refused: bool = False):
    def answer_fn(question: str) -> Answer:
        citations = [] if refused else [Citation("S1", "ada_2010_standards.pdf", 123, 123, section, "Clear Width",
                                                 f"[ada_2010_standards.pdf p.123 §{section}]")]
        return Answer(question=question, text=text, refused=refused, refusal_reason="model_not_in_sources" if refused else None,
                      citations=citations, grounding_status=None if refused else status)
    return answer_fn


DOOR_ITEM = {"id": "door", "element": "Entry door: clear width", "measured": INCHES(30),
             "question": "What is the minimum clear width of a door opening?"}
SEAT_ITEM = {"id": "seat", "element": "Water closet: seat height", "measured": INCHES(18),
             "question": "How high must the seat be?"}
CHECKLIST = {"id": "test-001", "title": "Test restroom", "items": [DOOR_ITEM, SEAT_ITEM]}


def test_draft_finding_uses_the_chain_answer_and_code_for_the_outcome():
    finding = draft_finding(DOOR_ITEM, fake_answer("Door openings shall be 32 inches (815 mm) minimum [S1]."))
    assert finding["outcome"] == "fail" and finding["limits"] == {"minimum": "32 inches", "maximum": None}
    assert finding["citations"] == ["[ada_2010_standards.pdf p.123 §404.2.3]"]


def test_draft_finding_refused_answer_needs_review():
    finding = draft_finding(DOOR_ITEM, fake_answer("I don't know", refused=True))
    assert finding["outcome"] == "needs_review" and finding["requirement"] is None


def test_apply_decisions_approve_and_edit():
    findings = [{"item_id": "door", "outcome": "fail"}, {"item_id": "seat", "outcome": "needs_review"}]
    reviewed = apply_decisions(findings, [
        {"item_id": "door", "action": "approve"},
        {"item_id": "seat", "action": "edit", "outcome": "pass", "note": "measured 18 in, 604.4: 17-19"},
    ])
    assert [f["final_outcome"] for f in reviewed] == ["fail", "pass"]
    assert reviewed[0]["review"] == {"action": "approve", "note": "", "changed": False}
    assert reviewed[1]["review"]["changed"] and reviewed[1]["review"]["note"].startswith("measured")


@pytest.mark.parametrize("decisions, message", [
    ([{"item_id": "door", "action": "approve"}], "missing"),
    ([{"item_id": "door", "action": "approve"}, {"item_id": "seat", "action": "approve"},
      {"item_id": "other", "action": "approve"}], "unknown"),
    ([{"item_id": "door", "action": "edit", "outcome": "maybe"}, {"item_id": "seat", "action": "approve"}], "edited outcome"),
    ([{"item_id": "door", "action": "reject"}, {"item_id": "seat", "action": "approve"}], "approve or edit"),
])
def test_apply_decisions_rejects_bad_input(decisions, message):
    findings = [{"item_id": "door", "outcome": "fail"}, {"item_id": "seat", "outcome": "pass"}]
    with pytest.raises(ValueError, match=message):
        apply_decisions(findings, decisions)


# --- the review graph -----------------------------------------------------------

ANSWER = fake_answer("The limit is 32 inches (815 mm) minimum and 36 inches (915 mm) maximum [S1].")


def test_graph_pauses_for_review_then_writes_the_report(tmp_path):
    graph = build_review_graph(ANSWER, tmp_path, MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"checklist": CHECKLIST}, config)

    findings = pending_findings(graph, config)                # paused at human_review, nothing written yet
    assert [f["outcome"] for f in findings] == ["fail", "fail"]  # 30 < 32; 18 < 32
    assert graph.get_state(config).next == ("human_review",) and not list(tmp_path.iterdir())

    state = graph.invoke(Command(resume=[{"item_id": "door", "action": "approve"},
                                         {"item_id": "seat", "action": "edit", "outcome": "needs_review",
                                          "note": "wrong rule cited"}]), config)
    assert [f["final_outcome"] for f in state["findings"]] == ["fail", "needs_review"]
    report = json.loads((tmp_path / "test-001_findings.json").read_text())
    assert report["findings"][0]["citations"] == ["[ada_2010_standards.pdf p.123 §404.2.3]"]
    markdown = (tmp_path / "test-001_findings.md").read_text()
    assert "[ada_2010_standards.pdf p.123 §404.2.3]" in markdown and "needs_review (edited)" in markdown
    assert pending_findings(graph, config) is None


def test_review_resumes_from_the_sqlite_checkpoint_in_a_new_graph(tmp_path):
    database = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "t2"}}
    calls = []

    def counting_answer(question):
        calls.append(question)
        return ANSWER(question)

    # Process 1: draft and pause.
    first = build_review_graph(counting_answer, tmp_path / "out", SqliteSaver(sqlite3.connect(database, check_same_thread=False)))
    first.invoke({"checklist": CHECKLIST}, config)
    assert len(calls) == 2

    # Process 2: a new graph on the same checkpoint file resumes without drafting again.
    second = build_review_graph(counting_answer, tmp_path / "out", SqliteSaver(sqlite3.connect(database, check_same_thread=False)))
    assert len(pending_findings(second, config)) == 2
    second.invoke(Command(resume=[{"item_id": "door", "action": "approve"}, {"item_id": "seat", "action": "approve"}]), config)
    assert len(calls) == 2                                      # the LLM was not called again
    assert (tmp_path / "out" / "test-001_findings.md").exists()
