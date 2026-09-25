# InspectIQ — inspection compliance copilot

## Purpose
Learning project for an Applied AI Engineer interview. The developer must be able to
explain every line, so prefer clear, explicit code over clever abstractions.

## What it does
- RAG over public building-code PDFs in data/ with cited answers ([file p.X])
- LangGraph agent: retrieve → grade → rewrite (max 2) → generate; tools: search_code,
  load_inspection(json), compute_outcome (pass / fail / partial-pass)
- Human-in-the-loop approval of drafted findings (LangGraph interrupt)
- Evals in eval/: golden set, retrieval hit-rate@k, faithfulness (LLM judge), refusal correctness
- LangFuse tracing; log tokens and cost per request; cheap model for grading, stronger for final answer
- FastAPI (POST /ask, POST /review, GET /health), Docker, pytest

## Stack
Python 3.12 (uv venv), langgraph, langchain-anthropic, Chroma, bge-small-en-v1.5 embeddings,
FastAPI, LangFuse. Config and secrets from .env via python-dotenv.

## Rules
- Build in small steps; after each step, explain what the code does and why.
- Never hard-code keys; never commit .env.
- Deterministic chunk IDs so re-ingestion doesn't duplicate.
- Every LLM call goes through one wrapper that records tokens, latency and cost.
- Add a pytest for each non-LLM function.
