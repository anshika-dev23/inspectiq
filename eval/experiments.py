"""Step 4a follow-up experiments, one change at a time, on eval/golden.json (final_k = 3 throughout).

  (a) rerank depth 25 instead of 10          kept only if it answers more questions (it costs latency)
  (b) no section-vector path, on top of (a)  kept if it is not worse (it is a simplification)
  (c) replace vs blend vs rrf on the winner  the simplest mode within one question of the best
  (d) query glossary on the winner of (c)    kept only if it answers more questions

"correct" = in-corpus questions answered in the top 3 + out-of-corpus questions correctly returned empty.
The final configuration is then reported on eval/heldout.json, which is never used for a decision.
Writes eval/experiments.md.

Run:  .venv/bin/python eval/experiments.py
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make "src" and "eval" importable

from eval.retrieval_eval import (  # noqa: E402
    BASE,
    EVAL_DIR,
    QUESTION_TYPES,
    QuestionResult,
    load_golden,
    misses_table,
    pct,
    run_config,
    summarize,
    summarize_by_type,
)
from src.config import RetrievalConfig  # noqa: E402
from src.retrieve import get_default_resources, retrieve  # noqa: E402

K = 3
MODES_SIMPLEST_FIRST = ("replace", "rrf", "blend")  # replace: one score; rrf: rank-based, no parameter; blend: a weight


def correct_count(results: list[QuestionResult]) -> int:
    return sum(
        1 for r in results
        if (r.in_corpus and r.answer_rank is not None and r.answer_rank <= K) or (not r.in_corpus and r.empty)
    )


def describe(config: RetrievalConfig) -> str:
    return (f"depth {config.rerank_top_n}, sections {'on' if config.use_section_vector else 'off'}, "
            f"{config.rerank_mode}, glossary {'on' if config.use_glossary else 'off'}")


def row(label: str, config: RetrievalConfig, results: list[QuestionResult], decision: str = "") -> str:
    s = summarize(results)
    return (f"| {label} | {describe(config)} | {correct_count(results)}/{len(results)} | {pct(s['hit@1'])} | "
            f"{pct(s['hit@3'])} | {s['mrr']:.3f} | {pct(s['refusal_accuracy'])} | {pct(s['false_refusals'])} | "
            f"{s['p50_ms']:.0f} | {s['p95_ms']:.0f} | {decision} |")


HEADER = ["| run | configuration | correct | hit@1 | hit@3 | MRR | refusal acc. | false refusals | p50 ms | p95 ms | decision |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]


def by_type_rows(named_results: dict[str, list[QuestionResult]]) -> list[str]:
    in_corpus = [t for t in QUESTION_TYPES if t != "out_of_corpus"]
    lines = ["| run | " + " | ".join(in_corpus) + " | out_of_corpus (refusal) |", "|---|" + "---:|" * (len(in_corpus) + 1)]
    for label, results in named_results.items():
        types = summarize_by_type(results)
        lines.append(f"| {label} | " + " | ".join(pct(types[t]["hit@3"]) for t in in_corpus)
                     + f" | {pct(types['out_of_corpus']['refusal_accuracy'])} |")
    return lines


def main() -> None:
    golden = load_golden(EVAL_DIR / "golden.json")
    heldout = load_golden(EVAL_DIR / "heldout.json")
    resources = get_default_resources()
    retrieve("warm-up query", None, BASE, resources)

    def run(config: RetrievalConfig, questions=golden) -> list[QuestionResult]:
        return run_config(questions, config, resources)

    report = ["# Retrieval experiments (after step 4a)", "",
              "One change at a time on eval/golden.json (34 questions), final_k = 3, threshold 0.0, exact-ref boost on.",
              "correct = in-corpus answered in the top 3 + out-of-corpus correctly empty.", ""]

    # (a) rerank depth
    base_config = BASE
    base = run(base_config)
    deep_config = replace(BASE, rerank_top_n=25)
    deep = run(deep_config)
    keep_deep = correct_count(deep) > correct_count(base)
    winner_config, winner = (deep_config, deep) if keep_deep else (base_config, base)
    base_p50, deep_p50 = summarize(base)["p50_ms"], summarize(deep)["p50_ms"]
    report += ["## (a) Rerank depth 25 instead of 10", "", *HEADER,
               row("step-4a baseline", base_config, base),
               row("(a) depth 25", deep_config, deep, "kept" if keep_deep else "rejected: no more questions answered"),
               "", f"Latency cost of depth 25: p50 {base_p50:.0f} → {deep_p50:.0f} ms "
               f"({deep_p50 - base_p50:+.0f} ms: 25 instead of 10 cross-encoder pairs).", ""]

    # (b) no section-vector path
    no_sections_config = replace(winner_config, use_section_vector=False)
    no_sections = run(no_sections_config)
    keep_no_sections = correct_count(no_sections) >= correct_count(winner)
    if keep_no_sections:
        winner_config, winner = no_sections_config, no_sections
    report += ["## (b) Without the section-vector path, on top of (a)", "", *HEADER,
               row("winner of (a)", replace(no_sections_config, use_section_vector=True), run(replace(no_sections_config, use_section_vector=True))),
               row("(b) no sections", no_sections_config, no_sections,
                   "kept (not worse, simpler)" if keep_no_sections else "rejected: worse"), ""]

    # (c) rerank modes on the winner
    by_mode = {mode: run(replace(winner_config, rerank_mode=mode)) for mode in MODES_SIMPLEST_FIRST}
    best_correct = max(correct_count(results) for results in by_mode.values())
    chosen_mode = next(m for m in MODES_SIMPLEST_FIRST if correct_count(by_mode[m]) >= best_correct - 1)
    winner_config, winner = replace(winner_config, rerank_mode=chosen_mode), by_mode[chosen_mode]
    report += ["## (c) Rerank modes on the winner of (a)/(b)", "",
               f"Rule: the simplest mode (order {', '.join(MODES_SIMPLEST_FIRST)}) within one question of the best "
               f"(best: {best_correct} correct).", "", *HEADER,
               *[row(f"(c) {mode}", replace(winner_config, rerank_mode=mode), results,
                     "chosen" if mode == chosen_mode else "") for mode, results in by_mode.items()], ""]

    # (d) glossary
    glossary_config = replace(winner_config, use_glossary=True)
    glossary = run(glossary_config)
    keep_glossary = correct_count(glossary) > correct_count(winner)
    final_config, final = (glossary_config, glossary) if keep_glossary else (winner_config, winner)
    report += ["## (d) Query glossary (everyday words → code terms), on the winner of (c)", "", *HEADER,
               row("winner of (c)", winner_config, winner),
               row("(d) + glossary", glossary_config, glossary,
                   "kept" if keep_glossary else "rejected: no more questions answered"), ""]

    # Final configuration on golden, by type and misses
    report += ["## Final configuration", "", f"`{describe(final_config)}`, final_k {K}, threshold "
               f"{final_config.rerank_threshold:+.1f}.", "",
               "### Hit@3 by question type (golden)", "",
               *by_type_rows({"step-4a baseline": base, "final": final}), "",
               "### Every miss of the final configuration on golden, with its top 3", "",
               *misses_table(final, {q["id"]: q for q in golden}, K), ""]

    # Held-out: report only
    heldout_base = run(base_config, heldout)
    heldout_final = run(final_config, heldout)
    report += ["## Held-out set (eval/heldout.json: 10 new paraphrase + 2 negatives), report only", "",
               "Written before the glossary and these experiments; never used for a decision.", "", *HEADER,
               row("step-4a baseline", base_config, heldout_base),
               row("final", final_config, heldout_final), "",
               "### Every miss of the final configuration on held-out, with its top 3", "",
               *misses_table(heldout_final, {q["id"]: q for q in heldout}, K), "",
               "### Every miss of the step-4a baseline on held-out, with its top 3", "",
               *misses_table(heldout_base, {q["id"]: q for q in heldout}, K), ""]

    out = EVAL_DIR / "experiments.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"final configuration: {describe(final_config)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
