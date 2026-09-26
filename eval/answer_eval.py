"""Step 3/4b — deterministic answer eval (no LLM judge: a 3B model is too weak to judge).

Runs every golden and held-out question through answer() and reports, per set:
- answer correctness: every answer_contains string appears in a non-refused answer (questions that have one);
- citation validity: at least one cited section is in `expected` (in-corpus questions that were answered);
- refusal accuracy: out-of-corpus questions refused;
- false refusals: in-corpus questions refused (by reason);
- p50/p95 latency (total, retrieval, LLM) and mean tokens;
- number grounding: answers blocked as "ungrounded_number", and whether each would otherwise have been correct;
- shadow cost per 1,000 questions on the Anthropic models in config.SHADOW_PRICES_USD_PER_MTOK (estimate).
Lists every failure with the answer text. Writes eval/answer_results.md and eval/answer_results.json.

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
    verified: bool | None = None          # number grounding; None: no answer to check
    ungrounded_numbers: list[str] | None = None
    raw_contains_ok: bool | None = None   # the model's own text had the expected numbers (even if blocked)
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
        verified=result.verified,
        ungrounded_numbers=result.ungrounded_numbers,
        raw_contains_ok=(contains_expected(result.raw_llm_text, expected_strings)
                         if expected_strings and result.raw_llm_text else None),
        shadow_cost_usd=result.llm.shadow_cost_usd if result.llm else {},
        trace_url=result.trace_url,
    )


def summarize(results: list[AnswerResult]) -> dict:
    with_contains = [r for r in results if r.contains_ok is not None]
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


def grounding_table(results: list[AnswerResult], questions: dict[str, dict]) -> list[str]:
    """Every answer the number-grounding check blocked, and whether it would otherwise have been correct."""
    lines = ["| id | ungrounded numbers | expected numbers | model's text had them? | model output |",
             "|---|---|---|---|---|"]
    for r in results:
        if r.refusal_reason != "ungrounded_number":
            continue
        expected = questions[r.question_id].get("answer_contains") or "–"
        verdict = {True: "yes: right number, not in the cited source (false block for the user)",
                   False: "no: correct block", None: "n/a (no expected number)"}[r.raw_contains_ok]
        text = r.text.replace("\n", " ").replace("|", "/")
        lines.append(f"| {r.question_id} | {r.ungrounded_numbers} | {expected} | {verdict} | "
                     f"{text[:300] + (' […]' if len(text) > 300 else '')} |")
    return lines if len(lines) > 2 else ["No answer was blocked."]


def cost_table(summaries: dict[str, dict]) -> list[str]:
    lines = ["| set | " + " | ".join(SHADOW_PRICES_USD_PER_MTOK) + " |", "|---|" + "---:|" * len(SHADOW_PRICES_USD_PER_MTOK)]
    for name, s in summaries.items():
        lines.append(f"| {name} | " + " | ".join(f"${s['shadow_per_1000'][m]:.2f}" for m in SHADOW_PRICES_USD_PER_MTOK) + " |")
    return lines


def summary_table(summaries: dict[str, dict]) -> list[str]:
    lines = ["| set | answer correct | citation valid | refusal acc. | false refusals | p50 ms | p95 ms | "
             "retrieval p50 | LLM p50 | LLM p95 | tokens in/out (mean) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | {pct(s['correct'])} ({s['correct_n']}) | {pct(s['citation_valid'])} ({s['citation_valid_n']}) | "
            f"{pct(s['refusal_accuracy'])} ({s['refusal_n']}) | {pct(s['false_refusals'])} ({s['false_refusal_n']}) | "
            f"{s['p50_ms']:.0f} | {s['p95_ms']:.0f} | {s['retrieval_p50_ms']:.0f} | {s['llm_p50_ms']:.0f} | "
            f"{s['llm_p95_ms']:.0f} | {s['mean_input_tokens']:.0f}/{s['mean_output_tokens']:.0f} |")
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
        "# Answer eval (step 3 baseline chain, deterministic, no LLM judge)", "",
        f"Model: {settings.llm_provider} `{model}`, temperature {settings.llm_temperature}, "
        f"num_ctx {settings.ollama_num_ctx}; context cap {settings.answer_max_context_chars:,} chars; "
        "retrieval: current RetrievalConfig defaults.", "",
        "- answer correct: all `answer_contains` strings appear in a non-refused answer (only questions that have one);",
        "- citation valid: at least one cited section is in `expected` (answered in-corpus questions);",
        "- refusal acc.: out-of-corpus questions refused; false refusals: in-corpus questions refused;",
        "- latency: the whole answer() call, warm models, on this machine.", "",
        *summary_table(summaries), "",
        "False refusals by reason: " + "; ".join(f"{n}: {s['false_refusal_reasons'] or 'none'}" for n, s in summaries.items()),
        "",
        f"## Number grounding (strict: {settings.strict_number_grounding})", "",
        "Answers blocked because a number is not in a source cited in its sentence (src/grounding.py).", "",
    ]
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
