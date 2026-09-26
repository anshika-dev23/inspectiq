"""Answer a question from the indexed codes and print the answer, its citations and timings.

Usage:
    .venv/bin/python scripts/answer.py "How wide must a door be?"
    .venv/bin/python scripts/answer.py "grab bar height" --code "ADA 2010 Standards" --show-prompt
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make "src" importable when run as a script
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import src  # noqa: E402,F401  (loads .env before any model library is imported)
from src import tracing  # noqa: E402
from src.answer import answer, build_messages  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--code", help='code_name filter, e.g. "ADA 2010 Standards"')
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt sent to the LLM")
    args = parser.parse_args()

    result = answer(args.question, {"code_name": args.code})

    print(f"== {args.question}")
    if result.sources:
        print("sources in the prompt:")
        for source in result.sources:
            cut = f", shortened to {len(source.text):,}" if source.truncated else ""
            print(f"  [{source.label}] {source.context.citation} ({len(source.context.text):,} chars{cut})")
    if args.show_prompt and result.sources:
        for message in build_messages(args.question, result.sources):
            print(f"\n--- {message.type} ---\n{message.content}")

    print(f"\nanswer{' (REFUSED: ' + result.refusal_reason + ')' if result.refused else ''}:")
    print("  " + result.text.replace("\n", "\n  "))
    if result.refused and result.raw_llm_text is not None:
        print(f"model said: {result.raw_llm_text!r}")
    if result.citations:
        print("citations:")
        for citation in result.citations:
            print(f"  [{citation.label}] {citation.citation}  {citation.section_title}")
    if result.invalid_labels:
        print(f"invalid labels (not in the prompt): {result.invalid_labels}")

    llm = result.llm
    tokens = f", tokens in/out {llm.input_tokens}/{llm.output_tokens}, cost ${llm.cost_usd:.4f}" if llm else ""
    print(f"timings (ms): {result.timings_ms}{tokens}")
    if result.verified is not None:
        print(f"number grounding: {'verified' if result.verified else 'UNVERIFIED'}"
              + (f", ungrounded: {result.ungrounded_numbers}" if result.ungrounded_numbers else ""))
    if llm and llm.shadow_cost_usd:
        from src.config import SHADOW_COST_NOTE
        costs = ", ".join(f"{model} ${usd:.5f}" for model, usd in llm.shadow_cost_usd.items())
        print(f"shadow cost ({SHADOW_COST_NOTE}): {costs}")
    tracing.flush()
    if result.trace_url:
        print(f"trace: {result.trace_url}")


if __name__ == "__main__":
    main()
