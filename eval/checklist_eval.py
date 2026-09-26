"""Step 7 — checklist eval: drafted outcomes vs the expected outcomes in the checklist file.

Drafts every item with the same code the review flow uses (src/checklist.draft_finding: the answer() chain finds
and cites the rule, compute_outcome() compares), then compares with `expected_outcome` and `expected_section`.
No human review here: this measures the draft. Writes eval/checklist_results.md.

Run:  .venv/bin/python eval/checklist_eval.py [eval/checklists/restroom.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make "src" and "eval" importable

import src  # noqa: E402,F401  (loads .env before any model library is imported)
from eval.retrieval_eval import EVAL_DIR  # noqa: E402
from src import tracing  # noqa: E402
from src.answer import answer  # noqa: E402
from src.checklist import draft_finding, format_measured  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL_DIR / "checklists" / "restroom.json"
    checklist = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in checklist["items"]:
        finding = draft_finding(item, answer)
        cited_sections = [c.split("§")[-1].rstrip("]") for c in finding["citations"]]
        rows.append({**finding, "expected_outcome": item["expected_outcome"],
                     "expected_section": item["expected_section"],
                     "outcome_ok": finding["outcome"] == item["expected_outcome"],
                     "cites_expected": item["expected_section"] in cited_sections})

    decided = [r for r in rows if r["outcome"] != "needs_review"]
    lines = [f"# Checklist eval: {checklist['title']}", "",
             "Drafted outcomes (before human review) vs expected. The chain finds and cites the rule; compute_outcome()",
             "compares the measurement. needs_review is never a wrong pass/fail: it goes to the inspector.", "",
             f"- outcome as expected: **{sum(r['outcome_ok'] for r in rows)}/{len(rows)}**",
             f"- wrong pass/fail (the dangerous error): **{sum(not r['outcome_ok'] for r in decided)}**",
             f"- needs_review: {len(rows) - len(decided)}/{len(rows)}",
             f"- expected section cited: {sum(r['cites_expected'] for r in rows)}/{len(rows)}", "",
             "| item | measured | expected | drafted | reason | cited | grounding |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = "✓" if r["outcome_ok"] else ("review" if r["outcome"] == "needs_review" else "✗")
        cited = ", ".join(c.split("§")[-1].rstrip("]") for c in r["citations"]) or "–"
        lines.append(f"| {r['element']} | {format_measured(r['measured'])} | {r['expected_outcome']} "
                     f"(§{r['expected_section']}) | {r['outcome']} {mark} | {r['outcome_reason']} | {cited} | "
                     f"{r['grounding_status'] or '–'} |")
    lines += ["", "## Requirements as drafted", ""]
    for r in rows:
        lines += [f"### {r['element']}", "", "> " + (r["requirement"] or f"(refused: {r['refusal_reason']})").replace("\n", "\n> "), ""]
    (EVAL_DIR / "checklist_results.md").write_text("\n".join(lines), encoding="utf-8")
    tracing.flush()
    for r in rows:
        print(f"{r['item_id']:22} expected {r['expected_outcome']:5} drafted {r['outcome']:12} {r['outcome_reason']}")
    print("wrote eval/checklist_results.md")


if __name__ == "__main__":
    main()
