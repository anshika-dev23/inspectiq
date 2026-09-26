"""Step 6 — deterministic answer eval of three setups (no LLM judge: a 3B model is too weak to judge).

Setups: the step-3 baseline chain, the LangGraph agent with grading only, and the agent with grading + query
rewrite (src/graph.py). Runs every golden and held-out question through each and reports, per set and setup:
- wrong-but-verified (the deciding metric, target 0): verified answers missing an answer_contains number;
- answer correct among verified, and among needs_review (questions with answer_contains);
- needs_review rate: share of given answers (verified + needs_review) flagged for review;
- citation validity: at least one cited section is in `expected` (answered in-corpus questions);
- refusal accuracy (out-of-corpus refused) and false refusals (in-corpus refused, by reason);
- LLM calls per question, p50/p95 latency, tokens, shadow cost per 1,000 questions (estimate).
Lists the questions whose outcome differs between setups, the graph's grading decisions, and every failure.
Writes eval/answer_results.md and eval/answer_results.json.

Run:  .venv/bin/python eval/answer_eval.py [--setups baseline,grade,rewrite]
"""
import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make "src" and "eval" importable

import src  # noqa: E402,F401  (loads .env before any model library is imported)
from eval.retrieval_eval import EVAL_DIR, load_golden, pct, percentile  # noqa: E402
from src import tracing  # noqa: E402
from src.answer import Answer, answer, get_default_llm  # noqa: E402
from src.config import SHADOW_COST_NOTE, SHADOW_PRICES_USD_PER_MTOK, load_settings  # noqa: E402
from src.graph import graph_answer  # noqa: E402

SETUPS: dict[str, tuple[str, Callable[[str], Answer]]] = {
    "baseline": ("baseline chain", lambda question: answer(question)),
    "grade": ("graph (grade only)", lambda question: graph_answer(question, rewrite=False)),
    "rewrite": ("graph (grade + rewrite)", lambda question: graph_answer(question, rewrite=True)),
}


@dataclass
class AnswerResult:
    setup: str
    question_id: str
    question_type: str
    question: str
    refused: bool
    refusal_reason: str | None
    text: str
    cited: list[tuple[str, str]]          # (source, section_id) of each valid citation
    invalid_labels: list[str]
    contains_ok: bool | None              # None: the question has no answer_contains
    citation_valid: bool | None           # None: out of corpus, or refused
    total_ms: float
    llm_calls: int
    input_tokens: int                     # summed over all LLM calls of the question
    output_tokens: int
    shadow_cost_usd: dict                 # summed over all LLM calls of the question
    grounding_status: str | None = None   # verified | needs_review | refused; None: no answer reached the check
    flagged: list[dict] | None = None     # [{"number", "cited", "found_in": [section ids]}]
    raw_contains_ok: bool | None = None   # the model's own text had the expected numbers (even if refused)
    agent: dict | None = None             # graph only: queries tried, grades, retries
    trace_url: str | None = None

    @property
    def in_corpus(self) -> bool:
        return self.question_type != "out_of_corpus"


def contains_expected(text: str, expected: list[str]) -> bool:
    """Every expected string appears; 'a|b' accepts either; numbers only match whole ("5" not in "15" or "5.5")."""
    def appears(alternative: str) -> bool:
        pattern = re.escape(alternative)
        if alternative[:1].isdigit():
            pattern = r"(?<![\d./])" + pattern
        if alternative[-1:].isdigit():
            pattern = pattern + r"(?![\d/]|\.\d)"
        return re.search(pattern, text) is not None

    return all(any(appears(alternative) for alternative in item.split("|")) for item in expected)


def citation_is_valid(cited: list[tuple[str, str]], expected: list[list[dict]]) -> bool:
    acceptable = {(item["source"], item["section_id"]) for group in expected for item in group}
    return any(section in acceptable for section in cited)


def source_tag(context) -> str:
    """ "404.2.3 [standards]": the source matters when Standards and Guidance share a section ID (35.151(b))."""
    return f"{context.section_id} [{context.source.split('_')[2].split('.')[0]}]"


def evaluate(question: dict, setup: str, answer_fn: Callable[[str], Answer]) -> AnswerResult:
    result = answer_fn(question["question"])
    cited = [(c.source, c.section_id) for c in result.citations]
    section_of = {s.label: source_tag(s.context) for s in result.sources}
    expected_strings = question.get("answer_contains")
    in_corpus = bool(question["expected"])
    usage = result.usage or {"input_tokens": 0, "output_tokens": 0, "shadow_cost_usd": {}}
    return AnswerResult(
        setup=setup,
        question_id=question["id"],
        question_type=question["type"],
        question=question["question"],
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        text=result.text if not result.refused else (result.raw_llm_text or result.text),
        cited=cited,
        invalid_labels=result.invalid_labels,
        contains_ok=None if not expected_strings else (not result.refused and contains_expected(result.text, expected_strings)),
        citation_valid=None if (not in_corpus or result.refused) else citation_is_valid(cited, question["expected"]),
        total_ms=result.timings_ms["total"],
        llm_calls=result.llm_calls,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        shadow_cost_usd=usage["shadow_cost_usd"],
        grounding_status=result.grounding_status,
        flagged=[{"number": f.raw, "cited": [section_of[label] for label in f.cited_labels],
                  "found_in": [section_of[label] for label in f.found_in]} for f in result.flagged_numbers],
        raw_contains_ok=(contains_expected(result.raw_llm_text, expected_strings)
                         if expected_strings and result.raw_llm_text else None),
        agent=result.agent,
        trace_url=result.trace_url,
    )


def summarize(results: list[AnswerResult]) -> dict:
    with_contains = [r for r in results if r.contains_ok is not None]
    verified = [r for r in results if r.grounding_status == "verified"]
    review = [r for r in results if r.grounding_status == "needs_review"]
    verified_checkable = [r for r in verified if r.contains_ok is not None]
    review_checkable = [r for r in review if r.contains_ok is not None]
    answered_in_corpus = [r for r in results if r.citation_valid is not None]
    in_corpus = [r for r in results if r.in_corpus]
    out_of_corpus = [r for r in results if not r.in_corpus]

    def ratio(part, whole):
        return part / whole if whole else float("nan")

    return {
        "n": len(results),
        "correct": ratio(sum(r.contains_ok for r in with_contains), len(with_contains)),
        "correct_n": f"{sum(r.contains_ok for r in with_contains)}/{len(with_contains)}",
        "wrong_but_verified_n": sum(not r.contains_ok for r in verified_checkable),
        "correct_verified": ratio(sum(r.contains_ok for r in verified_checkable), len(verified_checkable)),
        "correct_verified_n": f"{sum(r.contains_ok for r in verified_checkable)}/{len(verified_checkable)}",
        "correct_review_n": f"{sum(r.contains_ok for r in review_checkable)}/{len(review_checkable)}",
        "needs_review": ratio(len(review), len(verified) + len(review)),
        "needs_review_n": f"{len(review)}/{len(verified) + len(review)}",
        "citation_valid": ratio(sum(r.citation_valid for r in answered_in_corpus), len(answered_in_corpus)),
        "citation_valid_n": f"{sum(r.citation_valid for r in answered_in_corpus)}/{len(answered_in_corpus)}",
        "refusal_accuracy": ratio(sum(r.refused for r in out_of_corpus), len(out_of_corpus)),
        "refusal_n": f"{sum(r.refused for r in out_of_corpus)}/{len(out_of_corpus)}",
        "false_refusals": ratio(sum(r.refused for r in in_corpus), len(in_corpus)),
        "false_refusal_n": f"{sum(r.refused for r in in_corpus)}/{len(in_corpus)}",
        "false_refusal_reasons": dict(Counter(r.refusal_reason for r in in_corpus if r.refused)),
        "llm_calls_mean": ratio(sum(r.llm_calls for r in results), len(results)),
        "llm_calls_max": max((r.llm_calls for r in results), default=0),
        "p50_ms": percentile([r.total_ms for r in results], 50),
        "p95_ms": percentile([r.total_ms for r in results], 95),
        "mean_input_tokens": ratio(sum(r.input_tokens for r in results), len(results)),
        "mean_output_tokens": ratio(sum(r.output_tokens for r in results), len(results)),
        # shadow cost per 1,000 questions: questions refused before any LLM call cost nothing and are included
        "shadow_per_1000": {model: 1000 * sum(r.shadow_cost_usd.get(model, 0.0) for r in results) / len(results)
                            for model in SHADOW_PRICES_USD_PER_MTOK},
    }


def failures(results: list[AnswerResult], questions: dict[str, dict]) -> list[str]:
    lines = ["| id | failure | question | expected | answer / model output | cited |", "|---|---|---|---|---|---|"]
    for r in results:
        problems = []
        if r.in_corpus and r.refused:
            problems.append(f"false refusal ({r.refusal_reason})")
        if not r.in_corpus and not r.refused:
            problems.append("answered out-of-corpus")
        if r.contains_ok is False and not r.refused:
            problems.append("wrong or missing number")
        if r.citation_valid is False:
            problems.append("cited section not in expected")
        if r.invalid_labels:
            problems.append(f"invalid labels {r.invalid_labels}")
        if not problems:
            continue
        q = questions[r.question_id]
        expected = " + ".join("/".join(i["section_id"] for i in g) for g in q["expected"]) or "(refuse)"
        if q.get("answer_contains"):
            expected += f"; contains {q['answer_contains']}"
        text = r.text.replace("\n", " ").replace("|", "/")
        text = text if len(text) <= 300 else text[:300] + " […]"
        cited = ", ".join(sid for _, sid in r.cited) or "–"
        lines.append(f"| {r.question_id} | {'; '.join(problems)} | {r.question} | {expected} | {text} | {cited} |")
    return lines


def short(text: str, limit: int = 300) -> str:
    text = text.replace("\n", " ").replace("|", "/")
    return text if len(text) <= limit else text[:limit] + " […]"


def grounding_table(results: list[AnswerResult], questions: dict[str, dict]) -> list[str]:
    """Every answer that is needs_review or refused by grounding, with each flagged number and where it was found."""
    lines = ["| id | status | flagged number: cited → found in | expected numbers | answer correct? | model output |",
             "|---|---|---|---|---|---|"]
    for r in results:
        if r.grounding_status not in ("needs_review", "refused"):
            continue
        flags = "; ".join(f"{f['number']}: {'/'.join(f['cited']) or '–'} → {'/'.join(f['found_in']) or 'nowhere'}"
                          for f in r.flagged)
        expected = questions[r.question_id].get("answer_contains") or "–"
        correct = {True: "yes", False: "no", None: "n/a"}[r.raw_contains_ok]
        lines.append(f"| {r.question_id} | {r.grounding_status} | {flags} | {expected} | {correct} | {short(r.text)} |")
    return lines if len(lines) > 2 else ["Every answer was verified."]


def wrong_but_verified_table(results: list[AnswerResult], questions: dict[str, dict]) -> list[str]:
    lines = ["| id | expected numbers | cited | answer |", "|---|---|---|---|"]
    for r in results:
        if r.grounding_status == "verified" and r.contains_ok is False:
            cited = ", ".join(sid for _, sid in r.cited)
            lines.append(f"| {r.question_id} | {questions[r.question_id]['answer_contains']} | {cited} | {short(r.text)} |")
    return lines if len(lines) > 2 else ["None."]


def outcome(r: AnswerResult) -> str:
    if r.refused:
        label = f"refused ({r.refusal_reason})"
        return label + (" ✓" if not r.in_corpus else "")
    if not r.in_corpus:
        return "ANSWERED (out of corpus)"
    label = {True: "correct", False: "WRONG", None: "answered"}[r.contains_ok]
    return label + (" [review]" if r.grounding_status == "needs_review" else "")


def comparison_table(summaries: dict[str, dict]) -> list[str]:
    lines = ["| setup | wrong but verified | correct among verified | needs_review | correct overall | citation valid | "
             "refusal acc. | false refusals | LLM calls / q (mean, max) | p50 ms | p95 ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | **{s['wrong_but_verified_n']}** | {pct(s['correct_verified'])} ({s['correct_verified_n']}) | "
            f"{pct(s['needs_review'])} ({s['needs_review_n']}) | {pct(s['correct'])} ({s['correct_n']}) | "
            f"{pct(s['citation_valid'])} ({s['citation_valid_n']}) | {pct(s['refusal_accuracy'])} ({s['refusal_n']}) | "
            f"{pct(s['false_refusals'])} ({s['false_refusal_n']}) | {s['llm_calls_mean']:.1f}, {s['llm_calls_max']} | "
            f"{s['p50_ms']:.0f} | {s['p95_ms']:.0f} |")
    return lines


def outcome_diff_table(by_setup: dict[str, list[AnswerResult]]) -> list[str]:
    """Questions whose outcome is not the same in every setup."""
    names = list(by_setup)
    lines = ["| id | question | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for i, first in enumerate(by_setup[names[0]]):
        outcomes = [outcome(by_setup[name][i]) for name in names]
        if len(set(outcomes)) > 1:
            lines.append(f"| {first.question_id} | {first.question} | " + " | ".join(outcomes) + " |")
    return lines if len(lines) > 2 else ["Same outcome in every setup."]


def graph_decisions_table(results: list[AnswerResult]) -> list[str]:
    """For a graph setup: questions where the grader rejected a source or the query was rewritten."""
    lines = ["| id | queries tried | grades (section: verdict, ? = unparsed → yes) | outcome |", "|---|---|---|---|"]
    for r in results:
        agent = r.agent or {}
        grades = agent.get("grades", [])
        if not (agent.get("retries") or any(g["verdict"] == "no" or not g["parsed"] for g in grades)):
            continue
        queries = " → ".join(f"`{q}`" for q in agent.get("queries", []))
        verdicts = "; ".join(f"{g['section'].replace('ada_2010_', '').replace('.pdf', '')}: "
                             f"{g['verdict']}{'' if g['parsed'] else '?'}" for g in grades)
        lines.append(f"| {r.question_id} | {queries} | {verdicts} | {outcome(r)} |")
    return lines if len(lines) > 2 else ["Every source was graded relevant; no rewrite."]


def cost_table(summaries: dict[tuple[str, str], dict]) -> list[str]:
    lines = ["| setup | set | " + " | ".join(SHADOW_PRICES_USD_PER_MTOK) + " | tokens in/out per question |",
             "|---|---|" + "---:|" * (len(SHADOW_PRICES_USD_PER_MTOK) + 1)]
    for (setup, set_name), s in summaries.items():
        lines.append(f"| {setup} | {set_name} | "
                     + " | ".join(f"${s['shadow_per_1000'][m]:.2f}" for m in SHADOW_PRICES_USD_PER_MTOK)
                     + f" | {s['mean_input_tokens']:.0f}/{s['mean_output_tokens']:.0f} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setups", default="baseline,grade,rewrite", help=f"comma list of {list(SETUPS)}")
    args = parser.parse_args()
    setups = {SETUPS[key][0]: SETUPS[key][1] for key in args.setups.split(",")}

    settings = load_settings()
    model = settings.ollama_model if settings.llm_provider == "ollama" else settings.anthropic_model
    sets = {"golden": load_golden(EVAL_DIR / "golden.json"), "held-out": load_golden(EVAL_DIR / "heldout.json")}
    answer("warm-up: how wide must a door be?")  # load the embedding, reranking and LLM models before timing
    get_default_llm()

    results: dict[str, dict[str, list[AnswerResult]]] = {}
    for setup, answer_fn in setups.items():
        results[setup] = {}
        for set_name, questions in sets.items():
            print(f"{setup}: {len(questions)} {set_name} questions ...", flush=True)
            results[setup][set_name] = [evaluate(q, setup, answer_fn) for q in questions]
    summaries = {(setup, set_name): summarize(rs) for setup, by_set in results.items() for set_name, rs in by_set.items()}

    report = [
        "# Answer eval: baseline chain vs LangGraph agent (deterministic, no LLM judge)", "",
        f"Model: {settings.llm_provider} `{model}`, temperature {settings.llm_temperature}, "
        f"num_ctx {settings.ollama_num_ctx}; context cap {settings.answer_max_context_chars:,} chars; "
        "retrieval: current RetrievalConfig defaults. Setups: " + ", ".join(setups) + ".", "",
        "Number grounding (src/grounding.py): **verified** = every number is in a source cited in its sentence;",
        "**needs_review** = some number is only in another source of the prompt (answered, flagged); **refused** =",
        "some number is in no source of the prompt. Graph: one yes/no grading call per source (unparseable → yes);",
        "the rewrite loop runs only when no source is graded relevant (max 2 retries).", "",
        "- **wrong but verified** (deciding metric, target 0): verified answers missing an `answer_contains` number;",
        "- correct: every `answer_contains` string appears in a given answer (questions that have one);",
        "- needs_review: share of given answers (verified + needs_review) flagged for review;",
        "- citation valid: at least one cited section is in `expected` (answered in-corpus questions);",
        "- refusal acc.: out-of-corpus refused; false refusals: in-corpus refused;",
        "- LLM calls per question and latency (whole call, warm models, this machine).", "",
    ]
    for set_name, questions in sets.items():
        by_setup = {setup: results[setup][set_name] for setup in setups}
        report += [f"## {set_name}", "", *comparison_table({setup: summaries[(setup, set_name)] for setup in setups}), "",
                   "False refusals by reason: " + "; ".join(
                       f"{setup}: {summaries[(setup, set_name)]['false_refusal_reasons'] or 'none'}" for setup in setups),
                   "", f"### {set_name}: questions whose outcome differs between setups", "",
                   *outcome_diff_table(by_setup), ""]
    report += [f"## Shadow cost per 1,000 questions ({SHADOW_COST_NOTE})", "",
               "All LLM calls of a question (grading, rewrites, answer) at the prices in config.SHADOW_PRICES_USD_PER_MTOK "
               "(USD per million tokens, input/output: "
               + ", ".join(f"{m} ${i:g}/${o:g}" for m, (i, o) in SHADOW_PRICES_USD_PER_MTOK.items())
               + "). Questions refused before any LLM call count as $0.", "", *cost_table(summaries), ""]
    for setup in setups:
        report += [f"## Details: {setup}", ""]
        for set_name, questions in sets.items():
            by_id = {q["id"]: q for q in questions}
            rs = results[setup][set_name]
            report += [f"### {set_name}: wrong but verified", "", *wrong_but_verified_table(rs, by_id), "",
                       f"### {set_name}: needs_review and grounding refusals", "", *grounding_table(rs, by_id), ""]
            if any(r.agent for r in rs):
                report += [f"### {set_name}: grading and rewrite decisions", "", *graph_decisions_table(rs), ""]
            report += [f"### {set_name}: every failure", "", *failures(rs, by_id), ""]

    (EVAL_DIR / "answer_results.md").write_text("\n".join(report), encoding="utf-8")
    raw = {setup: {set_name: [asdict(r) for r in rs] for set_name, rs in by_set.items()} for setup, by_set in results.items()}
    (EVAL_DIR / "answer_results.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
    tracing.flush()
    print("wrote eval/answer_results.md and eval/answer_results.json")


if __name__ == "__main__":
    main()
