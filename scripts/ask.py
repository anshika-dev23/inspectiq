"""Debug CLI for retrieval: prints each stage's top 5 and the final contexts.

Usage:
    .venv/bin/python scripts/ask.py "What does 604.5 require?"
    .venv/bin/python scripts/ask.py "grab bar height" --code "ADA 2010 Standards" --type code
    .venv/bin/python scripts/ask.py "35.151(b)" --no-rerank --full
    .venv/bin/python scripts/ask.py "grab bar height for toilets" --rerank-mode blend --final-k 5
"""
import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make "src" importable when run as a script
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import src  # noqa: E402,F401  (loads .env before any model library is imported)
from src.config import RetrievalConfig  # noqa: E402
from src.retrieve import get_default_resources, retrieve  # noqa: E402

TOP = 5
PREVIEW_CHARS = 700


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--code", help='code_name filter, e.g. "ADA 2010 Standards"')
    parser.add_argument("--year", type=int, help="edition_year filter")
    parser.add_argument("--type", dest="section_type", help='section_type filter: "code" or "regulation"')
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder and its threshold")
    parser.add_argument("--rerank-mode", choices=["replace", "blend", "rrf"], default=RetrievalConfig.rerank_mode)
    parser.add_argument("--final-k", type=int, default=RetrievalConfig.final_k)
    parser.add_argument("--full", action="store_true", help=f"print whole contexts, not {PREVIEW_CHARS}-char previews")
    return parser.parse_args()


def label(parent_id: str, docstore: dict) -> str:
    metadata = docstore[parent_id]["metadata"]
    return f"§{metadata['section_id']} {metadata['section_title'][:40]}  [{metadata['code_name']}]"


def print_child_hits(name: str, hits, docstore: dict) -> None:
    print(f"\n-- {name}: {len(hits)} hits")
    for hit in hits[:TOP]:
        print(f"  {hit.rank:>2}. {hit.score:7.3f}  {label(hit.parent_id, docstore)}")


def main() -> None:
    args = parse_args()
    config = replace(RetrievalConfig(), use_rerank=not args.no_rerank, rerank_mode=args.rerank_mode,
                     final_k=args.final_k)
    filters = {"code_name": args.code, "edition_year": args.year, "section_type": args.section_type}

    resources = get_default_resources()
    docstore = resources.docstore
    result = retrieve(args.question, filters, config, resources)
    debug = result.debug

    print(f"== {args.question!r}")
    print(f"normalized: {debug.normalized.text!r}")
    print(f"section refs: {debug.normalized.section_refs}   pinned: {[label(p, docstore) for p in debug.pinned_parent_ids]}")
    print(f"bm25 tokens: {debug.normalized.tokens}")

    print_child_hits("child vector (cosine)", debug.child_vector, docstore)
    print_child_hits("bm25", debug.bm25, docstore)
    print(f"\n-- section vector (cosine): {len(debug.section_vector)} hits")
    for hit in debug.section_vector[:TOP]:
        print(f"  {hit.rank:>2}. {hit.score:7.3f}  {label(hit.parent_id, docstore)}")

    print(f"\n-- fused (RRF k={config.rrf_k}): {len(debug.fused)} parents")
    for rank, candidate in enumerate(debug.fused[:TOP], start=1):
        pin = "  PINNED" if candidate.pinned else ""
        print(f"  {rank:>2}. {candidate.rrf_score:.4f}  {label(candidate.parent_id, docstore)}  ranks={candidate.ranks}{pin}")

    if config.use_rerank:
        print(f"\n-- reranked (cross-encoder, mode {config.rerank_mode}, threshold {config.rerank_threshold}): "
              f"top {len(debug.reranked)}")
        for rank, candidate in enumerate(debug.reranked[:TOP], start=1):
            verdict = "pinned" if candidate.pinned else ("pass" if candidate.rerank_score >= config.rerank_threshold else "FAIL")
            blend = "" if candidate.order_score is None else f"  {config.rerank_mode}={candidate.order_score:.4f}"
            print(f"  {rank:>2}. {candidate.rerank_score:7.2f} {verdict:<6}  {label(candidate.parent_id, docstore)}{blend}")

    print(f"\n-- contexts: {len(result.contexts)}" + ("   -> EMPTY: answer \"I don't know\"" if result.is_empty else ""))
    for context in result.contexts:
        shown = "whole parent" if context.window is None else f"window {context.window[0]}-{context.window[1]}"
        rerank_score = "-" if context.rerank_score is None else f"{context.rerank_score:.2f}"
        print(f"\n  {context.citation}  {context.breadcrumb}")
        print(f"  rerank={rerank_score}  rrf={context.rrf_score:.4f}  pinned={context.pinned}  "
              f"{shown} of {context.parent_length:,} chars")
        text = context.text if args.full or len(context.text) <= PREVIEW_CHARS else context.text[:PREVIEW_CHARS] + " […]"
        print("  | " + text.replace("\n", "\n  | "))

    print(f"\ntimings (ms): {debug.timings_ms}")


if __name__ == "__main__":
    main()
