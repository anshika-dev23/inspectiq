"""Inspect a checklist: draft findings with the grounded chain, review them, write the findings report.

The review graph (src/checklist.py) drafts the findings, then pauses (LangGraph interrupt) with its state in a
SQLite checkpoint. This script shows the drafted findings, asks you to approve or edit each one, and resumes the
graph with your decisions; the graph then writes the report (JSON + markdown).

Usage:
    .venv/bin/python scripts/review.py eval/checklists/restroom.json                  # interactive
    .venv/bin/python scripts/review.py eval/checklists/restroom.json --thread r1      # quit ([q]) ...
    .venv/bin/python scripts/review.py eval/checklists/restroom.json --thread r1 --resume   # ... and resume later
    .venv/bin/python scripts/review.py eval/checklists/restroom.json --approve-all    # non-interactive
    .venv/bin/python scripts/review.py eval/checklists/restroom.json --decisions decisions.json
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make "src" importable when run as a script
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import src  # noqa: E402,F401  (loads .env before any model library is imported)
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src import tracing  # noqa: E402
from src.answer import answer  # noqa: E402
from src.checklist import OUTCOMES, build_review_graph, format_measured, pending_findings  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

CHECKPOINTS = PROJECT_ROOT / "store" / "checkpoints.sqlite"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports"


def show(finding: dict, number: int, total: int) -> None:
    print(f"\n[{number}/{total}] {finding['element']}: measured {format_measured(finding['measured'])}")
    print(f"  question:    {finding['question']}")
    requirement = finding["requirement"] or f"(no grounded requirement: {finding['refusal_reason']})"
    print("  requirement: " + requirement.replace("\n", "\n               "))
    print(f"  citations:   {', '.join(finding['citations']) or '-'}")
    grounding = finding["grounding_status"] or "n/a"
    flagged = f" (flagged: {finding['flagged_numbers']})" if finding["flagged_numbers"] else ""
    print(f"  grounding:   {grounding}{flagged}")
    print(f"  DRAFT OUTCOME: {finding['outcome'].upper()}  ({finding['outcome_reason']})")


def ask_decision(finding: dict) -> dict | None:
    """Approve or edit one finding; None means quit (the checkpoint keeps the drafted findings)."""
    while True:
        choice = input("  [a]pprove, [e]dit, [q]uit and resume later: ").strip().lower()
        if choice in ("a", "approve"):
            return {"item_id": finding["item_id"], "action": "approve", "note": input("  note (optional): ").strip()}
        if choice in ("e", "edit"):
            outcome = ""
            while outcome not in OUTCOMES:
                outcome = input(f"  outcome {OUTCOMES}: ").strip().lower()
            return {"item_id": finding["item_id"], "action": "edit", "outcome": outcome,
                    "note": input("  note (why): ").strip()}
        if choice in ("q", "quit"):
            return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checklist", type=Path)
    parser.add_argument("--thread", help="review thread id (default: checklist id + time)")
    parser.add_argument("--resume", action="store_true", help="resume a paused review of this thread")
    parser.add_argument("--approve-all", action="store_true", help="approve every drafted finding (non-interactive)")
    parser.add_argument("--decisions", type=Path, help="JSON list of decisions (non-interactive)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="report directory")
    args = parser.parse_args()

    checklist = json.loads(args.checklist.read_text(encoding="utf-8"))
    thread = args.thread or f"{checklist['id']}-{datetime.now():%Y%m%d-%H%M%S}"
    config = {"configurable": {"thread_id": thread}}
    CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
    graph = build_review_graph(answer, args.out, SqliteSaver(sqlite3.connect(CHECKPOINTS, check_same_thread=False)))

    if not args.resume:
        print(f"drafting {len(checklist['items'])} findings (thread {thread}) ...")
        graph.invoke({"checklist": checklist}, config)
    findings = pending_findings(graph, config)
    if findings is None:
        sys.exit(f"thread {thread} is not waiting for review")

    if args.decisions:
        decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    else:
        decisions = []
        for number, finding in enumerate(findings, start=1):
            show(finding, number, len(findings))
            if args.approve_all:
                decisions.append({"item_id": finding["item_id"], "action": "approve",
                                  "note": "auto-approved (--approve-all)"})
                continue
            decision = ask_decision(finding)
            if decision is None:
                print(f"\nreview paused; resume with: --thread {thread} --resume")
                return
            decisions.append(decision)

    state = graph.invoke(Command(resume=decisions), config)
    tracing.flush()
    print("\nfinal outcomes:")
    for finding in state["findings"]:
        edited = " (edited)" if finding["review"]["changed"] else ""
        print(f"  {finding['element']}: {finding['final_outcome']}{edited}")
    print(f"report: {state['report']['markdown']}\n        {state['report']['json']}")


if __name__ == "__main__":
    main()
