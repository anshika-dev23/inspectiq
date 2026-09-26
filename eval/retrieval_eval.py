"""Step 4a — retrieval evals (no LLM).

Runs every golden question through retrieve() under several configurations and reports hit@1/3/5, MRR,
refusal accuracy, false refusals and latency, overall and by question type. Then sweeps the rerank
threshold on the best rerank configuration. Writes eval/results.md and eval/results.json.

Run:  .venv/bin/python eval/retrieval_eval.py
"""
import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make "src" importable when run as a script

import src  # noqa: E402,F401  (loads .env before any model library is imported)
from src.config import RetrievalConfig  # noqa: E402
from src.retrieve import RetrievalResources, get_default_resources, retrieve  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
QUESTION_TYPES = ("paraphrase", "exact_id", "numeric_keyword", "guidance_only", "two_sections", "out_of_corpus")
FINAL_KS = (3, 5)
THRESHOLDS = [float(t) for t in range(-4, 5)]  # -4 .. +4, step 1

# The step-4a settings, pinned so this report stays reproducible when the defaults in RetrievalConfig change.
BASE = RetrievalConfig(rerank_top_n=10, use_section_vector=True, use_glossary=False, final_k=3,
                       rerank_threshold=0.0, use_exact_ref_boost=True, rerank_mode="replace")
# Retriever-comparison rows measure the raw retrievers: exact-ref boost OFF.
# Rerank rows are production behaviour: boost ON. One ablation turns it off on the best rerank row.
RAW = replace(BASE, use_exact_ref_boost=False, use_rerank=False)
CONFIGS: dict[str, RetrievalConfig] = {
    "vector-only": replace(RAW, use_bm25=False, use_section_vector=False),
    "bm25-only": replace(RAW, use_child_vector=False, use_section_vector=False),
    "hybrid (vector+bm25)": replace(RAW, use_section_vector=False),
    "+sections": RAW,
    "+rerank replace": replace(BASE, rerank_mode="replace"),
    "+rerank blend": replace(BASE, rerank_mode="blend"),
    "+rerank rrf": replace(BASE, rerank_mode="rrf"),
}


# =============================================================================
# Scoring one question
# =============================================================================

@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    found: list[tuple[str, str]]   # (source, section_id) of the returned contexts, in order
    answer_rank: int | None        # rank at which every expected group is covered; None = not answered
    latency_ms: float

    @property
    def in_corpus(self) -> bool:
        return self.question_type != "out_of_corpus"

    @property
    def empty(self) -> bool:
        return not self.found


def answer_rank(found: list[tuple[str, str]], expected: list[list[dict]]) -> int | None:
    """1-based rank at which the question is answered, or None.

    expected is a list of groups; a group is covered by its first (best-ranked) hit; the question is
    answered once every group is covered, i.e. at the largest of those ranks. No groups (out of corpus): None.
    """
    if not expected:
        return None
    group_ranks = []
    for group in expected:
        acceptable = {(item["source"], item["section_id"]) for item in group}
        ranks = [rank for rank, hit in enumerate(found, start=1) if hit in acceptable]
        if not ranks:
            return None
        group_ranks.append(min(ranks))
    return max(group_ranks)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least p% of the values at or below it."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[index]


def summarize(results: list[QuestionResult]) -> dict[str, float]:
    """Metrics over a list of question results. Hit and MRR are over in-corpus questions only."""
    in_corpus = [r for r in results if r.in_corpus]
    out_of_corpus = [r for r in results if not r.in_corpus]

    def hit_at(k: int) -> float:
        return sum(1 for r in in_corpus if r.answer_rank is not None and r.answer_rank <= k) / len(in_corpus)

    latencies = [r.latency_ms for r in results]
    return {
        "n": len(results),
        "hit@1": hit_at(1) if in_corpus else float("nan"),
        "hit@3": hit_at(3) if in_corpus else float("nan"),
        "hit@5": hit_at(5) if in_corpus else float("nan"),
        "mrr": sum(1 / r.answer_rank for r in in_corpus if r.answer_rank) / len(in_corpus) if in_corpus else float("nan"),
        # out-of-corpus questions correctly answered with an empty result
        "refusal_accuracy": (sum(1 for r in out_of_corpus if r.empty) / len(out_of_corpus)) if out_of_corpus else float("nan"),
        # in-corpus questions wrongly answered with an empty result
        "false_refusals": (sum(1 for r in in_corpus if r.empty) / len(in_corpus)) if in_corpus else float("nan"),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
    }


def summarize_by_type(results: list[QuestionResult]) -> dict[str, dict[str, float]]:
    return {t: summarize([r for r in results if r.question_type == t]) for t in QUESTION_TYPES}


# =============================================================================
# Running configurations
# =============================================================================

def load_golden(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def run_config(questions: list[dict], config: RetrievalConfig, resources: RetrievalResources) -> list[QuestionResult]:
    results = []
    for question in questions:
        result = retrieve(question["question"], None, config, resources)
        found = [(context.source, context.section_id) for context in result.contexts]
        results.append(QuestionResult(
            question_id=question["id"],
            question_type=question["type"],
            found=found,
            answer_rank=answer_rank(found, question["expected"]),
            latency_ms=result.debug.timings_ms["total"],
        ))
    return results


def best_rerank_run(summaries: dict[tuple[str, int], dict]) -> tuple[str, int]:
    """Best configuration that uses the reranker (the threshold only exists there): MRR, then hit@3."""
    rerank_runs = [key for key in summaries if CONFIGS[key[0]].use_rerank]
    return max(rerank_runs, key=lambda key: (summaries[key]["mrr"], summaries[key]["hit@3"]))


# =============================================================================
# Report
# =============================================================================

def pct(value: float) -> str:
    return "–" if math.isnan(value) else f"{100 * value:.0f}%"


def main_table(summaries: dict[tuple[str, int], dict]) -> list[str]:
    lines = ["| config | final_k | hit@1 | hit@3 | hit@5 | MRR | refusal acc. | false refusals | p50 ms | p95 ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (name, k), s in summaries.items():
        lines.append(f"| {name} | {k} | {pct(s['hit@1'])} | {pct(s['hit@3'])} | {pct(s['hit@5'])} | {s['mrr']:.3f} | "
                     f"{pct(s['refusal_accuracy'])} | {pct(s['false_refusals'])} | {s['p50_ms']:.0f} | {s['p95_ms']:.0f} |")
    return lines


def by_type_table(by_type: dict[tuple[str, int], dict], metric: str) -> list[str]:
    in_corpus_types = [t for t in QUESTION_TYPES if t != "out_of_corpus"]
    header = "| config | final_k | " + " | ".join(in_corpus_types) + " | out_of_corpus (refusal acc.) |"
    lines = [header, "|---|---:|" + "---:|" * (len(in_corpus_types) + 1)]
    for (name, k), types in by_type.items():
        cells = [pct(types[t][metric]) if metric != "mrr" else f"{types[t]['mrr']:.2f}" for t in in_corpus_types]
        lines.append(f"| {name} | {k} | " + " | ".join(cells) + f" | {pct(types['out_of_corpus']['refusal_accuracy'])} |")
    return lines


def sweep_table(sweep: list[tuple[float, dict]]) -> list[str]:
    lines = ["| threshold | refusal acc. | false refusals | hit@1 | hit@3 | MRR |", "|---:|---:|---:|---:|---:|---:|"]
    for threshold, s in sweep:
        lines.append(f"| {threshold:+.0f} | {pct(s['refusal_accuracy'])} | {pct(s['false_refusals'])} | "
                     f"{pct(s['hit@1'])} | {pct(s['hit@3'])} | {s['mrr']:.3f} |")
    return lines


def misses_table(results: list[QuestionResult], questions: dict[str, dict], k: int) -> list[str]:
    lines = ["| id | type | question | expected | returned |", "|---|---|---|---|---|"]
    for r in results:
        missed = (r.answer_rank is None or r.answer_rank > k) if r.in_corpus else not r.empty
        if not missed:
            continue
        q = questions[r.question_id]
        expected = " + ".join("/".join(item["section_id"] for item in group) for group in q["expected"]) or "(empty)"
        returned = ", ".join(f"{sid} [{src.split('_')[2].split('.')[0]}]" for src, sid in r.found[:k]) or "(empty)"
        lines.append(f"| {r.question_id} | {r.question_type} | {q['question']} | {expected} | {returned} |")
    return lines


def write_report(path: Path, questions: list[dict], summaries, by_type, best, sweep, ablation, misses_k3) -> None:
    name, k = best
    lines = [
        "# Retrieval eval results (step 4a, no LLM)", "",
        f"Golden set: {len(questions)} questions ({', '.join(f'{sum(q['type'] == t for q in questions)} {t}' for t in QUESTION_TYPES)}).",
        "Hit@k and MRR are over in-corpus questions; a two-section question counts only when both sections are in the",
        "top k. Refusal accuracy = out-of-corpus questions returned empty; false refusals = in-corpus questions returned",
        "empty. Latency is the whole retrieve() call on this machine, after one warm-up query. With final_k = 3 only 3",
        "contexts are returned, so hit@5 equals hit@3. Configurations without the reranker have no threshold and never",
        "return empty. Exact-ref boost: OFF in the four retriever-comparison rows (raw retrievers), ON in the three",
        "rerank rows (production behaviour); the ablation below turns it off on the best rerank row.", "",
        "## All configurations", "", *main_table(summaries), "",
        "## Hit@3 by question type", "", *by_type_table(by_type, "hit@3"), "",
        "## MRR by question type", "", *by_type_table(by_type, "mrr"), "",
        f"## Threshold sweep on the best rerank configuration: {name}, final_k {k}", "",
        f"Tuned on the same {len(questions)} questions it is reported on: expect it to look better here than on new questions.", "",
        *sweep_table(sweep), "",
        f"## Ablation: {name}, final_k {k}, without the exact-ref boost", "",
        *main_table({(f"{name} (no exact-ref boost)", k): ablation}), "",
        f"## Every miss of {name} at final_k 3 (threshold {CONFIGS[name].rerank_threshold:+.0f}), with its top 3", "",
        "In-corpus: the expected section(s) not all in the top 3. Out-of-corpus: anything returned at all.", "",
        *misses_table(misses_k3, {q["id"]: q for q in questions}, 3), "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=EVAL_DIR / "golden.json")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results.md")
    args = parser.parse_args()

    questions = load_golden(args.golden)
    resources = get_default_resources()
    retrieve("warm-up query", None, BASE, resources)  # load models and caches before timing

    summaries, by_type, raw = {}, {}, {}
    for name, config in CONFIGS.items():
        for k in FINAL_KS:
            print(f"running {name}, final_k={k} ...", flush=True)
            results = run_config(questions, replace(config, final_k=k), resources)
            summaries[(name, k)] = summarize(results)
            by_type[(name, k)] = summarize_by_type(results)
            raw[f"{name} | k={k}"] = [asdict(r) for r in results]

    best = best_rerank_run(summaries)
    best_config = replace(CONFIGS[best[0]], final_k=best[1])
    print(f"best rerank configuration: {best}; sweeping the threshold ...", flush=True)
    sweep = [(t, summarize(run_config(questions, replace(best_config, rerank_threshold=t), resources)))
             for t in THRESHOLDS]
    ablation = summarize(run_config(questions, replace(best_config, use_exact_ref_boost=False), resources))
    misses_k3 = run_config(questions, replace(CONFIGS[best[0]], final_k=3), resources)

    write_report(args.out, questions, summaries, by_type, best, sweep, ablation, misses_k3)
    args.out.with_suffix(".json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
