# InspectIQ — findings and decisions

A running log, appended at every build step. Each entry: what we saw (with evidence), what we decided, and
what is still open. Numbers are from the ADA corpus: `ada_2010_standards.pdf` (279 pages) and
`ada_2010_guidance.pdf` (170 pages).

---

## Step 1 — ingestion

### 1.1 Page numbers in citations
- **Decision:** `p.X` in a citation is the 1-based PDF page index, so a citation opens the right page in any
  PDF viewer. The printed page number (from the DOJ footer) is stored as `printed_pages` metadata, best effort.
- **Status:** after the PyMuPDF switch (2.9), all 1,180 Standards sections and all 69 Guidance sections have a
  printed page.

### 1.2 Heading patterns
- **Finding:** the two PDFs use five heading styles: `404.2.3 Clear Width.` (numbered), `403 Walking Surfaces`
  (chapter), `§ 35.151 New construction and alterations.` (regulation), `Section 36.406(f) Assembly Areas`
  (guidance), `213, 603, 604, and 608 Toilet and Bathing` (multi-ID, guidance Appendix B, 30 of them).
- **Decision:** one regex per style, tried in order. Rejected even if they look like headings: lines starting
  with `Advisory` (225 of them; they stay inside the section they explain), running headers containing `--`,
  table-of-contents lines with dot leaders, years (`1991 Standards`), wrapped body text (`402.2 and 403.`).
  Section IDs are stored without `§`, so `35.151` matches the tokenizer output.

### 1.3 Repeated section IDs
- **Finding (pypdf):** 12 IDs appeared 2–3 times (pp. 82 and 85 printed the same block three times).
- **Decision:** a repeat that is the first line of the next page is a continuation and is merged; any other
  repeat becomes a separate section with `parent_id = sha1(source|section_id|occurrence)`, and every repeat
  is logged.
- **Update (2.9):** with PyMuPDF there are **0** repeats: the triple printing was a pypdf extraction artifact.
  The rule stays as a safety net.

### 1.4 Headers and footers
- **Finding:** judging repetition over the whole page would delete real body lines (`EXCEPTION:`).
  Restricted to the first/last 3 lines of each page: at a 10% threshold only page furniture is removed;
  at 5% a guidance bullet list repeated at page edges 12 times was removed too.
- **Decision:** edge lines whose digit-masked form repeats on ≥10% of pages are furniture, never a line that
  matches a heading. Furniture that repeats on fewer pages is handled by an explicit, tested stoplist in
  `config.FURNITURE_STOPLIST` (chapter running heads, the Title III footer, regulation running heads).
  `config.FURNITURE_ANYWHERE` removes known whole-line footers wherever they sit (2.9).

### 1.5 Contents pages
- **Finding:** contents entries look like headings. `§ 36.402 Alterations.` on contents p.21 claimed occurrence 1,
  so `sha1(source|36.402)` pointed at the contents entry and the real text on p.25 became occurrence 2.
  A second contents block (pp.36–40, `221 Assembly Areas 46`, no dot leaders) was absorbed into `36.406(g)`.
- **Decision:** drop contents pages before splitting. A page is a contents page if it has a
  `TABLE OF CONTENTS`/`Contents` title line, ≥3 dot-leader lines, or ≥5 `NNN Title <page>` lines.

### 1.6 Front matter, back matter (the index)
- **Finding:** the Standards index (pp.256–279, ~28k chars) was glued onto the last section, `1010.1`.
- **Decision:** text before the first heading is `level="front-matter"`; from `INDEX TO ...` to the end of the
  file is `level="back-matter"` with no heading detection inside. Both are kept in the docstore only and
  are excluded from all three search indexes (step 2, fix a).

### 1.7 Regulation paragraphs
- **Finding:** `§ 35.151` was 22k chars and `§ 36.406` 20k; people cite `35.151(b)`, not `35.151`.
- **Decision:** inside a `§` section, a line starting `(a)`, `(b)`, … opens section `35.151(b)` with
  `section_type="regulation"`; `(b)(1)` and deeper stay inside. Guards, all needed on this corpus:
  - letters must increase; skipping is allowed (pypdf put `(c)` of 36.403 on its own line);
  - roman-looking letters `i`, `v`, `x` must be exactly the next letter (`(i)` is usually a sub-paragraph);
  - text after the marker must start with a capital (rejects the wrapped line `(b) that were constructed ...`).
  Result: 38 regulation paragraphs; the largest is `35.151(b)` at 6.6k chars.

### 1.8 Heading-only sections and breadcrumbs
- **Finding:** ~127 sections are only a heading (`216 Signs`, 9 chars) followed directly by `216.1`.
- **Decision:** docstore only, not searchable. Every child carries a breadcrumb
  (`ADA 2010 Standards > 216 Signs > 216.2 Designations`) which is prepended to the child text before
  embedding and BM25 tokenizing, and stored as metadata. Chapter headings live on inside their children.

### 1.9 Very large guidance sections
- **Finding:** Appendix B sections are 21–26k chars (`213, 603, 604, 608 Toilet and Bathing`: 25k) because
  their subtopics (`Urinals.`) have no numbers.
- **Decision:** do not split on unnumbered subtopics (not reliable from PDF text). Keep parents whole in the
  docstore; retrieval returns a context window instead (2.8).

### 1.10 Environment
- `HF_HUB_OFFLINE=1`: huggingface_hub reads it at import time, and `langchain_text_splitters` imports it before
  `src.config` ran, so `.env` is loaded in `src/__init__.py`, before anything else. Set it to 0 on a fresh
  machine or in CI so models can be downloaded once.

### 1.11 Known trade-offs
- Hyphen join: a real compound split at a line end (`multi-` / `user`) becomes `multiuser`.
- `Department of Justice` alone on a line is treated as footer furniture wherever it appears.

---

## Step 2 — retrieval

### 2.1 BM25 negative IDF (rank_bm25 `BM25Okapi`)
- **Finding (read from source, pinned in `tests/test_bm25_behaviour.py`):**
  `idf = ln(N − n + 0.5) − ln(n + 0.5)`; every negative idf (term in more than half the documents) is replaced
  by `eps = 0.25 × average_idf`.
  - Not monotonic: a term in 3 of 4 documents (floored to eps) outweighs one in 2 of 4 (idf exactly 0).
  - The floor itself is negative when the average idf is negative (tiny corpora): all matches score < 0.
  - Unknown query terms contribute 0.
- **Decision:** a BM25 score ≤ 0 is not a hit. IDF is computed on the whole corpus, not on the filtered subset.

### 2.2 Filters
- **Decision:** `code_name`, `edition_year`, `section_type` go into Chroma `where=` for both vector searches.
  BM25 scores all children, applies the filter, *then* takes the top 20; a test shows top-20-then-filter can
  return nothing.

### 2.3 Hybrid vs vector-only on "What does 604.5 require?"
- **Finding:** child-vector top 5 were `601.1, 804.4, 302.1, 35.151(j), 210.1`; section-vector top 5 similar.
  Embeddings do not read section numbers. Only BM25 (tokenizer keeps `604.5` whole) ranked `604.5` first.
- **Decision:** exact-reference boost: a section ref in the query (`604.5`, `35.151(b)`, `R302.1`; ≥2 digits
  before the first dot) that exists in the docstore (first occurrence, searchable, passing the filters) is
  pinned at rank 1. `35.151(b)(4)` falls back to `35.151(b)`. Pinned parents also bypass the rerank threshold:
  a bare `35.151(b)` query must not come back empty.
- **Open:** a bare chapter number (`216`) is not detected as a ref, and heading-only parents are never pinned.

### 2.4 Relevance threshold on out-of-corpus questions
- **Finding:** "What is the capital of France?": retrievers always return something (fused list of 13–28
  parents), but cross-encoder scores were −11.17 … −11.23. Relevant passages scored +4.7 … +9.1.
  The ms-marco cross-encoder returns raw logits (activation `Identity`), roughly −11 … +11.
- **Decision:** threshold 0.0 on the raw rerank score, provisional; tune on the golden set in step 4.
  If nothing passes, `retrieve()` returns an empty result and the caller answers "I don't know".

### 2.5 Re-ranker errors
- **Finding:** "grab bar height for toilets": the cross-encoder put `604.9` (children's water closets, 8.29;
  8.85 after 2.9) above `609.4 Position of Grab Bars` (6.57), which holds the answer (33–36 inches).
  RRF alone had `609.4` at rank 1; after rerank it fell to 4th and out of the top 3.
- **Decision:** no hand-tuning on one query. Config options for step 4 to compare:
  `final_k` (3 or 5) and `rerank_mode`:
  - `replace` (default): order by cross-encoder score;
  - `blend`: `w · sigmoid(rerank) + (1 − w) · rrf / best_rrf`, `w = rerank_blend_weight` (0.5).
  Pinned parents stay first in both modes; the threshold always applies to the raw rerank score.
- **Observation for step 4:** with `blend`, `609.4` comes first (0.999) and `604.9` drops out. But sigmoid
  saturates: sigmoid(6.57) = 0.9986 vs sigmoid(8.85) = 0.9999, so among confident candidates `blend` is
  effectively RRF order and the reranker only matters near the threshold. Consider a temperature or rank-based
  normalization if evals favour blending.

### 2.6 BM25 stopwords
- **Decision:** a small English stopword list, applied in `tokenize(..., remove_stopwords=True)` for BM25 only,
  at ingestion and query time. Never removed: numbers, section IDs, single letters (`35.151(a)` tokenizes to
  `35.151`, `a`), normative words (`shall`, `must`, `may`, `not`, `no`).
- **Effect:** "capital of France" BM25 hits 20 → 0 (they had matched only `what/is/the/of`).
  "What does 604.5 require?" now gets `604.5` plus its siblings from BM25; `604.5.2` (2.21) and `604.9.4` (2.16)
  pass the threshold, so 3 contexts instead of 1.

### 2.7 Context: RRF and reranking inputs
- Children are mapped to parents keeping the best child rank; a parent's rank is its position in that
  deduplicated list, comparable with the section-vector list. RRF k = 60, ties broken deterministically.
- The reranker reads (query, breadcrumb + best-matching child). Best child = child-level RRF over the
  vector and BM25 hits.

### 2.8 Context windows
- **Decision:** a parent ≤ 6,000 chars is returned whole; a bigger one as a ~4,000-char window centred on its
  best children (`start_char` recorded at ingestion), citation still pointing at the parent. Both limits in
  `RetrievalConfig`. Window edges snap to line boundaries (never moving more than 200 chars). A section pinned by
  an exact ref starts at the section start.
- **Effect:** pinned `35.151(b)` windows went from 881–4881 / 305–4305 (starting mid-word, `ountains serving`)
  to 0–4029 / 0–4048, starting at the heading.

### 2.9 PDF text extraction: pypdf → PyMuPDF
- **Experiment:** both extractors on the whole corpus and on 6 pages with known breaks.

  | Standards (whole file)             | pypdf | PyMuPDF |
  |------------------------------------|------:|--------:|
  | lines starting with a word fragment |   175 |      55 |
  | single-letter splits (`Th e`)       |   144 |      23 |
  | extraction time                     | 3.9 s |   0.5 s |

  Guidance: roughly equal (26 vs 20 fragment lines, 69 vs 69 splits). Samples: pypdf
  `occupan`/`t`, `EXCEP`/`TION`, `Th e`, `centered on th`/`e`, `ef`/`fort`, `40`/`4.3.2`: all intact in PyMuPDF.
  PyMuPDF also removed the phantom triple printing (1.3).
- **Decision:** switch to PyMuPDF (`pymupdf`, replaces `pypdf`). Note: PyMuPDF is AGPL-3.0 (commercial
  licence available); fine for this project, relevant for a commercial product.
- **Consequences, all fixed and tested:**
  - contents pages now come out as `101 Purpose` / `5` on separate lines → detected by their title line;
  - the DOJ footer is split over two lines and sometimes placed mid-page → printed page read with the footer
    patterns; whole-line footers removed anywhere (0 footer lines left in section text);
  - PyMuPDF splits 3 headings mid-word (`305.1 Gene` / `ral.`, `804.6.5.1 Side-Hinged Doo` / `r Ovens`,
    `804.6.5.3 C` / `ontrols`) → `repair_split_headings` joins two lines only when the result is a heading
    (space if the next line starts with a capital: `... Section 504` / `Regulations.`). It also covers the
    pypdf case `40` / `4.3.2 Maneuvering Clearance` (not triggered under PyMuPDF; kept, tested).
- **Result:** Standards 1,152 → 1,180 sections (e.g. `404.3.2`, `1008.2.2`, `203.12`, `206.2.3.1` recovered;
  `404.3.1` shrank from 636 to 268 chars because it no longer contains `404.3.2`), 127 heading-only.
  Stores: sections 1,119 · children 3,040 · bm25 3,040 · docstore 1,249.

### Open for step 4
- Threshold value (2.4), `final_k` 3 vs 5 and `rerank_mode` replace vs blend (2.5), with the golden set.

---

## Build order change (after step 2)
- **Decision:** step numbers stay as designed, but the order of work changes: 1, 2, **4a**, 3, 4b, 5, ...
  Step 4a is retrieval evals (no LLM); step 3 is the baseline answer chain (Ollama first, via the provider
  switch); step 4b adds faithfulness (LLM judge) once the chain exists.
- **Why:** retrieval evals need no LLM, so they are free and fast locally, and they settle the open questions
  (rerank or not, `rerank_mode`, `final_k`, the threshold) before an answer chain is built on top.

---

## Step 4a — retrieval evals (no LLM)

Full tables: `eval/results.md` (per-question results: `eval/results.json`); harness: `eval/retrieval_eval.py`.

### 4a.1 Golden set
- 34 questions in `eval/golden.json`, drafted from the PDF text (not from retrieval output): 10 paraphrase,
  5 exact ID, 5 numeric/keyword, 4 guidance-only, 2 two-section, 8 out-of-corpus. `expected` is a list of groups:
  every group must be hit, any section inside a group is acceptable (p04 and p08 also accept the Guidance section
  that states the same number: the user is correctly answered either way).
- The 4 extra out-of-domain negatives (sprinkler spacing, live load, U-factor, circuit amperage) were checked by
  grepping both PDFs: `live load`, `U-factor`, `climate zone`, `amp/ampere/amperage` 0 hits; `sprinkler` 2 hits on
  guidance p.90 (sprinklered buildings exempt from areas of rescue assistance, no spacing rule).
- 8 negatives: refusal accuracy moves in 12.5% steps. Still small; every number below is a handful of questions.

### 4a.2 Raw retrievers (exact-ref boost OFF)
| | vector-only | bm25-only | hybrid | +sections |
|---|---:|---:|---:|---:|
| hit@3 (all in-corpus) | 77% | 81% | **88%** | 73% |
| exact-ID hit@3 / MRR | 40% / 0.27 | 80% / 0.70 | **100%** / 0.73 | 40% / 0.40 |
| paraphrase hit@3 / MRR | 70% / 0.50 | 60% / 0.53 | 70% / 0.55 | 70% / **0.70** |
- **Vector search fails on exact IDs; BM25 does not.** The tokenizer keeps `604.5` whole, embeddings do not read it.
  Hybrid gets the best of both (exact-ID 100%, paraphrase no worse than vector).
- **The section-vector path hurts** exact-ID (100% → 40% hit@3) and guidance-only (100% → 75%) while helping
  paraphrase ordering (MRR 0.55 → 0.70): its top-5 sections pull confident-looking neighbours above the right one.
  Open: evaluate rerank *without* the section path.
- Retrievers alone never refuse (0% refusal accuracy; bm25-only 12% only because stopword-only queries have no hits).

### 4a.3 Reranker (boost ON, threshold 0)
| | replace | blend (w=0.5) | rrf |
|---|---:|---:|---:|
| hit@1 / hit@3 / MRR | 65% / 85% / 0.750 | **69%** / 85% / **0.769** | 65% / 85% / 0.750 |
| refusal acc. / false refusals | 88% / 4% | 88% / 4% | 88% / 4% |
| p50 / p95 latency | 57 / 67 ms | 54 / 62 ms | 54 / 63 ms |
- Reranking is what makes refusal possible: 0% → 88% refusal accuracy at 4% false refusals (1 of 26), MRR
  0.699 → 0.75–0.77, for ~35 ms more per query.
- The three modes differ by about one question: blend ranks paraphrases better (MRR 0.65 vs 0.50) but loses g02
  (guidance-only: the right section has the top rerank score 7.93 but is 6th by RRF, and blend's RRF half keeps it
  out of the top 3). This is the sigmoid saturation from 2.5: among confident candidates blend ≈ RRF order.
- `final_k` 3 vs 5: identical on every rerank row. Every miss is a total miss (not in the top 5 either), and few
  candidates pass the threshold anyway.
- Exact-ref boost ablation (blend, k=3): MRR 0.769 → 0.692, hit@3 85% → 77% without it. Keep it on.

### 4a.4 Why the misses miss (blend, k=3)
| q | failure | cause |
|---|---|---|
| p01 doorway width | 404.2.3 at fused rank 22 of 25 | first-stage recall: outside `rerank_top_n` = 10, the reranker never sees it |
| p03 turning space | 304.3.x at fused rank 25 of 25 | same |
| p04 light switch | expected sections not among 34 candidates; all rerank scores < −5 → empty | vocabulary gap: the code says "operable parts", never "light switch" |
| g02 vans | fused rank 6, top rerank score, dropped by blend | rerank mode (replace gets it) |
| o03 GFCI | `205.1` (receptacles at kitchen counters) scores +1.58 > 0 | hard negative close to real content |
- Two of five misses are recall at the first stage, not ranking: raising `rerank_top_n` (10 → 25) is the obvious
  next experiment; a synonym list (light switch → operable parts) would be the query-normalization fix for p04.

### 4a.5 Threshold sweep (blend, k=3)
| threshold | −4 | −3 … +1 | +2 | +3 | +4 |
|---|---:|---:|---:|---:|---:|
| refusal acc. | 75% | 88% | 100% | 100% | 100% |
| false refusals | 4% | 4% | 4% | 8% | 15% |
- A flat region from −3 to +1. +2 reaches 100% refusal at no cost *on this set*, but only because of one
  question (o03 at +1.58), and it is one step away from where false refusals start rising (+3).
- **Overfitting warning:** the threshold is tuned and reported on the same 34 questions. The 4% false refusal
  (p04) is a recall failure no threshold can fix.

### 4a.6 Recommended defaults (not yet applied to `RetrievalConfig`)
- Retrievers: child vector + BM25; exact-ref boost on; section-vector path: keep for now, but test rerank without it.
- Reranker on. `rerank_mode`: **blend** (best MRR and hit@1), with the caveat that its lead over replace/rrf is
  about one question, and it loses the guidance-only case g02.
- `final_k` = **3**: 5 gains nothing here and adds two more parents (~1.7x the context) for the LLM to read.
- Threshold: **keep 0.0**, the middle of the flat region, not the tuned +2. Out-of-corpus questions that slip through
  (like o03) meet a second gate in step 3: the answer chain must refuse when the context does not answer. A false
  refusal, on the other hand, cannot be recovered later. Re-check with a held-out set.
- Next experiments (evidence above): `rerank_top_n` 25; rerank without the section path; a small synonym list.
