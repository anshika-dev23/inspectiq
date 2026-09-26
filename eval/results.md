# Retrieval eval results (step 4a, no LLM)

Golden set: 34 questions (10 paraphrase, 5 exact_id, 5 numeric_keyword, 4 guidance_only, 2 two_sections, 8 out_of_corpus).
Hit@k and MRR are over in-corpus questions; a two-section question counts only when both sections are in the
top k. Refusal accuracy = out-of-corpus questions returned empty; false refusals = in-corpus questions returned
empty. Latency is the whole retrieve() call on this machine, after one warm-up query. With final_k = 3 only 3
contexts are returned, so hit@5 equals hit@3. Configurations without the reranker have no threshold and never
return empty. Exact-ref boost: OFF in the four retriever-comparison rows (raw retrievers), ON in the three
rerank rows (production behaviour); the ablation below turns it off on the best rerank row.

## All configurations

| config | final_k | hit@1 | hit@3 | hit@5 | MRR | refusal acc. | false refusals | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vector-only | 3 | 42% | 77% | 77% | 0.583 | 0% | 0% | 18 | 21 |
| vector-only | 5 | 42% | 77% | 77% | 0.583 | 0% | 0% | 17 | 18 |
| bm25-only | 3 | 58% | 81% | 81% | 0.686 | 12% | 0% | 2 | 3 |
| bm25-only | 5 | 58% | 81% | 85% | 0.694 | 12% | 0% | 2 | 3 |
| hybrid (vector+bm25) | 3 | 54% | 88% | 88% | 0.699 | 0% | 0% | 20 | 22 |
| hybrid (vector+bm25) | 5 | 54% | 88% | 88% | 0.699 | 0% | 0% | 20 | 55 |
| +sections | 3 | 58% | 73% | 73% | 0.654 | 0% | 0% | 20 | 23 |
| +sections | 5 | 58% | 73% | 77% | 0.662 | 0% | 0% | 20 | 22 |
| +rerank replace | 3 | 65% | 85% | 85% | 0.750 | 88% | 4% | 56 | 65 |
| +rerank replace | 5 | 65% | 85% | 85% | 0.750 | 88% | 4% | 53 | 63 |
| +rerank blend | 3 | 69% | 85% | 85% | 0.769 | 88% | 4% | 54 | 62 |
| +rerank blend | 5 | 69% | 85% | 85% | 0.769 | 88% | 4% | 54 | 63 |
| +rerank rrf | 3 | 65% | 85% | 85% | 0.750 | 88% | 4% | 53 | 63 |
| +rerank rrf | 5 | 65% | 85% | 85% | 0.750 | 88% | 4% | 54 | 63 |

## Hit@3 by question type

| config | final_k | paraphrase | exact_id | numeric_keyword | guidance_only | two_sections | out_of_corpus (refusal acc.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector-only | 3 | 70% | 40% | 100% | 100% | 100% | 0% |
| vector-only | 5 | 70% | 40% | 100% | 100% | 100% | 0% |
| bm25-only | 3 | 60% | 80% | 100% | 100% | 100% | 12% |
| bm25-only | 5 | 60% | 80% | 100% | 100% | 100% | 12% |
| hybrid (vector+bm25) | 3 | 70% | 100% | 100% | 100% | 100% | 0% |
| hybrid (vector+bm25) | 5 | 70% | 100% | 100% | 100% | 100% | 0% |
| +sections | 3 | 70% | 40% | 100% | 75% | 100% | 0% |
| +sections | 5 | 70% | 40% | 100% | 75% | 100% | 0% |
| +rerank replace | 3 | 60% | 100% | 100% | 100% | 100% | 88% |
| +rerank replace | 5 | 60% | 100% | 100% | 100% | 100% | 88% |
| +rerank blend | 3 | 70% | 100% | 100% | 75% | 100% | 88% |
| +rerank blend | 5 | 70% | 100% | 100% | 75% | 100% | 88% |
| +rerank rrf | 3 | 70% | 100% | 100% | 75% | 100% | 88% |
| +rerank rrf | 5 | 70% | 100% | 100% | 75% | 100% | 88% |

## MRR by question type

| config | final_k | paraphrase | exact_id | numeric_keyword | guidance_only | two_sections | out_of_corpus (refusal acc.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector-only | 3 | 0.50 | 0.27 | 0.77 | 1.00 | 0.50 | 0% |
| vector-only | 5 | 0.50 | 0.27 | 0.77 | 1.00 | 0.50 | 0% |
| bm25-only | 3 | 0.53 | 0.70 | 0.80 | 1.00 | 0.50 | 12% |
| bm25-only | 5 | 0.53 | 0.74 | 0.80 | 1.00 | 0.50 | 12% |
| hybrid (vector+bm25) | 3 | 0.55 | 0.73 | 0.80 | 1.00 | 0.50 | 0% |
| hybrid (vector+bm25) | 5 | 0.55 | 0.73 | 0.80 | 1.00 | 0.50 | 0% |
| +sections | 3 | 0.70 | 0.40 | 0.80 | 0.75 | 0.50 | 0% |
| +sections | 5 | 0.70 | 0.44 | 0.80 | 0.75 | 0.50 | 0% |
| +rerank replace | 3 | 0.50 | 1.00 | 0.90 | 1.00 | 0.50 | 88% |
| +rerank replace | 5 | 0.50 | 1.00 | 0.90 | 1.00 | 0.50 | 88% |
| +rerank blend | 3 | 0.65 | 1.00 | 0.90 | 0.75 | 0.50 | 88% |
| +rerank blend | 5 | 0.65 | 1.00 | 0.90 | 0.75 | 0.50 | 88% |
| +rerank rrf | 3 | 0.60 | 1.00 | 0.90 | 0.75 | 0.50 | 88% |
| +rerank rrf | 5 | 0.60 | 1.00 | 0.90 | 0.75 | 0.50 | 88% |

## Threshold sweep on the best rerank configuration: +rerank blend, final_k 3

Tuned on the same 34 questions it is reported on: expect it to look better here than on new questions.

| threshold | refusal acc. | false refusals | hit@1 | hit@3 | MRR |
|---:|---:|---:|---:|---:|---:|
| -4 | 75% | 4% | 69% | 85% | 0.769 |
| -3 | 88% | 4% | 69% | 85% | 0.769 |
| -2 | 88% | 4% | 69% | 85% | 0.769 |
| -1 | 88% | 4% | 69% | 85% | 0.769 |
| +0 | 88% | 4% | 69% | 85% | 0.769 |
| +1 | 88% | 4% | 69% | 85% | 0.769 |
| +2 | 100% | 4% | 69% | 85% | 0.769 |
| +3 | 100% | 8% | 65% | 81% | 0.731 |
| +4 | 100% | 15% | 62% | 77% | 0.692 |

## Ablation: +rerank blend, final_k 3, without the exact-ref boost

| config | final_k | hit@1 | hit@3 | hit@5 | MRR | refusal acc. | false refusals | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +rerank blend (no exact-ref boost) | 3 | 62% | 77% | 77% | 0.692 | 88% | 4% | 54 | 62 |

## Every miss of +rerank blend at final_k 3 (threshold +0), with its top 3

In-corpus: the expected section(s) not all in the top 3. Out-of-corpus: anything returned at all.

| id | type | question | expected | returned |
|---|---|---|---|---|
| p01 | paraphrase | How wide does a doorway need to be for a wheelchair user? | 404.2.3/404.3.1 | 802.1.2 [standards], 604.8.1.1 [standards], 221 [guidance] |
| p03 | paraphrase | How much floor space does a wheelchair user need to turn around? | 304.3/304.3.1/304.3.2 | 809.2.2 [standards], 236, 1004 [guidance], 802.1.3 [standards] |
| p04 | paraphrase | How high on the wall can a light switch be mounted? | 308.2.1/308.3.1/309.3/205, 309 | (empty) |
| g02 | guidance_only | Can cars that are not vans park in van accessible parking spaces? | 208, 502 | 502.3.4 [standards], 502.7 [standards], 502.5 [standards] |
| o03 | out_of_corpus | How many GFCI outlets are required on a kitchen countertop? | (empty) | 205.1 [standards] |
