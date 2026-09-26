"""Step 3/4b — deterministic answer eval (no LLM judge: a 3B model is too weak to judge).

Runs every golden and held-out question through answer() and reports, per set:
- wrong-but-verified (target 0): answers the grounding check verified whose answer_contains numbers are missing;
- answer correct among verified, and among needs_review (questions with answer_contains);
- needs_review rate: share of given answers (verified + needs_review) flagged for review;
- citation validity: at least one cited section is in `expected` (answered in-corpus questions);
- refusal accuracy (out-of-corpus refused) and false refusals (in-corpus refused, by reason);
- p50/p95 latency and mean tokens; shadow cost per 1,000 questions (estimate).
Lists every needs_review answer with its flagged numbers, every wrong-but-verified answer and every failure. Writes eval/answer_results.md and eval/answer_results.json.

Run:  .venv/bin/python eval/answer_eval.py
"""
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make "src" and "eval" importable

import src  # noqa: E402,F401  (loads .env before any model library is imported)
from eval.retrieval_eval import EVAL_DIR, load_golden, pct, percentile  # noqa: E402
from src import tracing  # noqa: E402
from src.answer import answer, get_default_llm  # noqa: E402
from src.config import SHADOW_COST_NOTE, SHADOW_PRICES_USD_PER_MTOK, load_settings  # noqa: E402


@dataclass
class AnswerResult:
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
    retrieval_ms: float
    llm_ms: float
    input_tokens: int | None
    output_tokens: int | None
    grounding_status: str | None = None   # verified | needs_review | refused; None: no answer reached the check
    flagged: list[dict] | None = None     # [{"number", "cited", "found_in": [section ids]}]
    raw_contains_ok: bool | None = None   # the model's own text had the expected numbers (even if refused)
    shadow_cost_usd: dict | None = None
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


def evaluate(question: dict) -> AnswerResult:
    result = answer(question["question"])
    cited = [(c.source, c.section_id) for c in result.citations]
    # "404.2.3 [standards]": the source matters when Standards and Guidance share a section ID (35.151(b))
    section_of = {s.label: f"{s.context.section_id} [{s.context.source.split('_')[2].split('.')[0]}]" for s in result.sources}
    expected_strings = question.get("answer_contains")
    in_corpus = bool(question["expected"])
    return AnswerResult(
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
        retrieval_ms=result.timings_ms["retrieval"],
        llm_ms=result.timings_ms["llm"],
        input_tokens=result.llm.input_tokens if result.llm else None,
        output_tokens=result.llm.output_tokens if result.llm else None,
        grounding_status=result.grounding_status,
        flagged=[{"number": f.raw, "cited": [section_of[label] for label in f.cited_labels],
                  "found_in": [section_of[label] for label in f.found_in]} for f in result.flagged_numbers],
        raw_contains_ok=(contains_expected(result.raw_llm_text, expected_strings)
                         if expected_strings and result.raw_llm_text else None),
        shadow_cost_usd=result.llm.shadow_cost_usd if result.llm else {},
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
    called_llm = [r for r in results if r.input_tokens is not None]

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
        "p50_ms": percentile([r.total_ms for r in results], 50),
        "p95_ms": percentile([r.total_ms for r in results], 95),
        "retrieval_p50_ms": percentile([r.retrieval_ms for r in results], 50),
        "llm_p50_ms": percentile([r.llm_ms for r in called_llm], 50),
        "llm_p95_ms": percentile([r.llm_ms for r in called_llm], 95),
        "mean_input_tokens": ratio(sum(r.input_tokens for r in called_llm), len(called_llm)),
        "mean_output_tokens": ratio(sum(r.output_tokens for r in called_llm), len(called_llm)),
        "blocked_ungrounded": sum(r.refusal_reason == "ungrounded_number" for r in results),
        # shadow cost per 1,000 questions: questions refused before the LLM cost nothing and are included
        "shadow_per_1000": {model: 1000 * sum((r.shadow_cost_usd or {}).get(model, 0.0) for r in results) / len(results)
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


def cost_table(summaries: dict[str, dict]) -> list[str]:
    lines = ["| set | " + " | ".join(SHADOW_PRICES_USD_PER_MTOK) + " |", "|---|" + "---:|" * len(SHADOW_PRICES_USD_PER_MTOK)]
    for name, s in summaries.items():
        lines.append(f"| {name} | " + " | ".join(f"${s['shadow_per_1000'][m]:.2f}" for m in SHADOW_PRICES_USD_PER_MTOK) + " |")
    return lines


def summary_table(summaries: dict[str, dict]) -> list[str]:
    lines = ["| set | wrong but verified | correct among verified | needs_review | correct among needs_review | "
             "correct overall | citation valid | refusal acc. | false refusals |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | **{s['wrong_but_verified_n']}** | {pct(s['correct_verified'])} ({s['correct_verified_n']}) | "
            f"{pct(s['needs_review'])} ({s['needs_review_n']}) | {s['correct_review_n']} | "
            f"{pct(s['correct'])} ({s['correct_n']}) | {pct(s['citation_valid'])} ({s['citation_valid_n']}) | "
            f"{pct(s['refusal_accuracy'])} ({s['refusal_n']}) | {pct(s['false_refusals'])} ({s['false_refusal_n']}) |")
    return lines


def latency_table(summaries: dict[str, dict]) -> list[str]:
    lines = ["| set | p50 ms | p95 ms | retrieval p50 | LLM p50 | LLM p95 | tokens in/out (mean) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['p50_ms']:.0f} | {s['p95_ms']:.0f} | {s['retrieval_p50_ms']:.0f} | "
                     f"{s['llm_p50_ms']:.0f} | {s['llm_p95_ms']:.0f} | "
                     f"{s['mean_input_tokens']:.0f}/{s['mean_output_tokens']:.0f} |")
    return lines


def main() -> None:
    settings = load_settings()
    model = settings.ollama_model if settings.llm_provider == "ollama" else settings.anthropic_model
    sets = {"golden": load_golden(EVAL_DIR / "golden.json"), "held-out": load_golden(EVAL_DIR / "heldout.json")}
    answer("warm-up: how wide must a door be?")  # load the embedding, reranking and LLM models before timing
    get_default_llm()

    results, summaries = {}, {}
    for name, questions in sets.items():
        print(f"answering {len(questions)} {name} questions ...", flush=True)
        results[name] = [evaluate(q) for q in questions]
        summaries[name] = summarize(results[name])

    report = [
        "# Answer eval (baseline chain + three-state number grounding; deterministic, no LLM judge)", "",
        f"Model: {settings.llm_provider} `{model}`, temperature {settings.llm_temperature}, "
        f"num_ctx {settings.ollama_num_ctx}; context cap {settings.answer_max_context_chars:,} chars; "
        "retrieval: current RetrievalConfig defaults.", "",
        "Three-state number grounding (src/grounding.py): **verified** = every number is in a source cited in its",
        "sentence; **needs_review** = some number is only in another source of the prompt (answered, flagged);",
        "**refused** = some number is in no source of the prompt (refusal `ungrounded_number`).", "",
        "- wrong but verified (target 0): verified answers missing an `answer_contains` number;",
        "- correct: every `answer_contains` string appears in a given answer (questions that have one);",
        "- needs_review: share of given answers (verified + needs_review) flagged for review;",
        "- citation valid: at least one cited section is in `expected` (answered in-corpus questions);",
        "- refusal acc.: out-of-corpus questions refused; false refusals: in-corpus questions refused.", "",
        *summary_table(summaries), "",
        "False refusals by reason: " + "; ".join(f"{n}: {s['false_refusal_reasons'] or 'none'}" for n, s in summaries.items()),
        "",
        "Latency (whole answer() call, warm models, this machine) and tokens:", "",
        *latency_table(summaries), "",
        "## Wrong but verified", "",
        "Grounding only checks that a number comes from a cited source; these came from the wrong section.", "",
    ]
    for name, questions in sets.items():
        report += [f"### {name}", "", *wrong_but_verified_table(results[name], {q["id"]: q for q in questions}), ""]
    report += ["## needs_review and grounding refusals, with each flagged number", "",
               "`cited → found in`: the sections the sentence cited, and the prompt sections that contain the number.", ""]
    for name, questions in sets.items():
        report += [f"### {name}", "", *grounding_table(results[name], {q["id"]: q for q in questions}), ""]
    report += [
        f"## Shadow cost per 1,000 questions ({SHADOW_COST_NOTE})", "",
        "What the LLM calls of this run would cost on each model, from Ollama's token counts and the prices in",
        "config.SHADOW_PRICES_USD_PER_MTOK (USD per million tokens, input/output: "
        + ", ".join(f"{m} ${i:g}/${o:g}" for m, (i, o) in SHADOW_PRICES_USD_PER_MTOK.items())
        + "). Questions refused before the LLM count as $0.", "",
        *cost_table(summaries), "",
    ]
    for name, questions in sets.items():
        report += [f"## Every failure: {name}", "", *failures(results[name], {q["id"]: q for q in questions}), ""]

    (EVAL_DIR / "answer_results.md").write_text("\n".join(report), encoding="utf-8")
    raw = {name: [asdict(r) for r in rs] for name, rs in results.items()}
    (EVAL_DIR / "answer_results.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")
    tracing.flush()
    print("wrote eval/answer_results.md and eval/answer_results.json")


if __name__ == "__main__":
    main()
