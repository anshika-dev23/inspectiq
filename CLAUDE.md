# InspectIQ — inspection compliance copilot

## Purpose
Learning project for an Applied AI Engineer interview. The developer must be able to
explain every line, so prefer clear, explicit code over clever abstractions.
Build in small steps; after each step, explain what the code does and why, then STOP.

## What it does
- Answers questions about public building codes (files in data/) with citations [file p.X §section]
- Checks a completed inspection checklist (JSON) against the code and drafts findings
- Inspector approves or edits drafted findings before they are final (human-in-the-loop)

## Stack
Python 3.12 (uv venv), langchain, langchain-anthropic, langgraph (step 6+), chromadb,
rank-bm25, sentence-transformers (BAAI/bge-small-en-v1.5 embeddings,
cross-encoder/ms-marco-MiniLM-L-6-v2 reranker), FastAPI, LangFuse, pytest, Docker.
Config and secrets from .env via python-dotenv. Settings in src/config.py.

## Ingestion (src/ingest.py) — multi-granularity
Part 1: files -> parent sections
- Load PDFs/Markdown page by page; keep page numbers.
- Clean text: join words broken by end-of-line hyphens, drop repeated page headers/footers.
- Section splitter: regex on code headings (e.g. "R302.1 Exterior walls.", "1604.5 Risk category.");
  each heading starts a PARENT section. Fallback: one section per page if no headings found.
- Parent metadata: parent_id, source, code_name, edition_year, section_id, section_title, pages.
  parent_id = sha1(source|section_id).

Part 2: parent sections -> four stores
- Section path (coarse): section title + opening ~1500 chars -> embed -> Chroma collection "sections".
- Child path (fine): RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60) within each
  section (children never cross section boundaries). child_id = sha1(parent_id|index).
  Child metadata: parent metadata + child_id, level="child".
  - Vector: embed -> Chroma collection "children".
  - Keyword: tokenize -> BM25Okapi -> pickle file.
- Docstore: docstore.json, parent_id -> full section text + metadata (lookup only, not searched).
- Same embedding model for both vector paths and for queries.
- Re-ingestion must not duplicate (deterministic IDs; BM25 rebuilt in full each run).

Tokenizer (src/text.py), used at ingestion AND query time:
- lowercase; re.findall(r"[a-z]?\d+(?:\.\d+)*|\w+", text)
- must keep section IDs whole: "R302.1" -> "r302.1" (unit test this).

## Retrieval (src/retrieve.py) — pure function retrieve(query, filters, config) -> results
- Query normalization first (no LLM): expand common abbreviations (GFCI, AFCI...), detect section refs.
- Metadata pre-filter (code_name, edition_year) passed inside each search (Chroma where=...).
- vector(children) top 20, BM25(children) top 20, vector(sections) top 5.
- Map child hits to parent_id (dedupe, keep best rank); fuse the three lists with
  Reciprocal Rank Fusion (k=60).
- Rerank top 10 parents with the cross-encoder, scoring (query, best-matching child text).
- Relevance threshold on the rerank score; if nothing passes, return empty -> caller says "I don't know".
- Return top 3 parents' full text (from docstore) + citations.
- Every stage toggleable in config so evals can compare: vector-only / +bm25 / +sections / +rerank.

## Answering
- Step 4 baseline: plain LangChain chain (retriever | prompt | llm | parser), no LangGraph.
- Answer only from retrieved text; cite every claim; refuse without a citation.

## Agent (src/graph.py) — LangGraph, step 6 only
- State: question, normalized, rewritten, filters, documents, answer, retries, messages.
- Flow: retrieve -> grade_documents (structured output) -> generate.
  If no relevant docs: LLM rewrite_query and retry, max 2 retries, then "I don't know".
- Tools: search_code, load_inspection(json), compute_outcome (pass / fail / partial-pass).
- Findings flow pauses with LangGraph interrupt for inspector approval.
- Checkpointed memory per thread_id.

## Evals (eval/)
- golden.json: 20+ questions with expected source + section, including 4 not in the corpus
  and some exact-ID questions (e.g. "What does R302.1 require?").
- Metrics: hit-rate@k and refusal correctness (step 3, retrieval only, no LLM), faithfulness (LLM judge,
  step 4), tool-choice accuracy (step 6).
- Report a table per retrieval config and baseline chain vs graph.

## Observability & cost (src/llm.py)
- Every LLM call goes through one wrapper that records model, input/output tokens, latency, cost,
  and sends traces to LangFuse.
- Cheaper model for grading/rewriting, stronger model for final answers; report cost per 1,000
  requests before and after.

## Safety
- Refuse to answer without a citation.
- A test document with a planted prompt injection must not change behaviour.
- Redact obvious PII from logs.

## API & delivery (step 8)
- FastAPI: POST /ask {question, thread_id, filters}, POST /review, GET /health.
- Multi-stage Dockerfile. GitHub Actions: pytest + evals; fail if faithfulness drops below threshold.

## Rules
- Never hard-code keys; never commit .env.
- A pytest for each non-LLM function (cleaning, splitters, tokenizer, ID hashing, RRF, outcome logic).
- Keep functions small and named for what they do.

## Build order (do ONLY the current step; stop and wait after each)
1. Ingestion part 1 + part 2 (four stores) + tests
2. Retrieval as a pure function + tests
3. Retrieval evals (no LLM): golden set; compare vector-only / +bm25 / +sections / +rerank,
   rerank_mode, final_k and the rerank threshold
4. Baseline answer chain (no LangGraph): retrieve -> LLM (provider switch, Ollama first) -> cited answer;
   add faithfulness to the evals
5. LLM wrapper + LangFuse + cost tracking
6. LangGraph agent (grade, rewrite, tools); re-run evals vs baseline
7. Human-in-the-loop, safety tests
8. FastAPI, Docker, CI
Current step: 3