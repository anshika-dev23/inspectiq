# Retrieval experiments (after step 4a)

One change at a time on eval/golden.json (34 questions), final_k = 3, threshold 0.0, exact-ref boost on.
correct = in-corpus answered in the top 3 + out-of-corpus correctly empty.

## (a) Rerank depth 25 instead of 10

| run | configuration | correct | hit@1 | hit@3 | MRR | refusal acc. | false refusals | p50 ms | p95 ms | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| step-4a baseline | depth 10, sections on, replace, glossary off | 29/34 | 65% | 85% | 0.750 | 88% | 4% | 58 | 88 |  |
| (a) depth 25 | depth 25, sections on, replace, glossary off | 29/34 | 65% | 85% | 0.750 | 88% | 4% | 95 | 135 | rejected: no more questions answered |

Latency cost of depth 25: p50 58 → 95 ms (+36 ms: 25 instead of 10 cross-encoder pairs).

## (b) Without the section-vector path, on top of (a)

| run | configuration | correct | hit@1 | hit@3 | MRR | refusal acc. | false refusals | p50 ms | p95 ms | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| winner of (a) | depth 10, sections on, replace, glossary off | 29/34 | 65% | 85% | 0.750 | 88% | 4% | 55 | 64 |  |
| (b) no sections | depth 10, sections off, replace, glossary off | 29/34 | 65% | 85% | 0.750 | 88% | 4% | 54 | 63 | kept (not worse, simpler) |

## (c) Rerank modes on the winner of (a)/(b)

Rule: the simplest mode (order replace, rrf, blend) within one question of the best (best: 30 correct).

| run | configuration | correct | hit@1 | hit@3 | MRR | refusal acc. | false refusals | p50 ms | p95 ms | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| (c) replace | depth 10, sections off, replace, glossary off | 29/34 | 65% | 85% | 0.750 | 88% | 4% | 54 | 63 | chosen |
| (c) rrf | depth 10, sections off, rrf, glossary off | 30/34 | 65% | 88% | 0.769 | 88% | 4% | 54 | 63 |  |
| (c) blend | depth 10, sections off, blend, glossary off | 30/34 | 69% | 88% | 0.788 | 88% | 4% | 53 | 63 |  |

## (d) Query glossary (everyday words → code terms), on the winner of (c)

| run | configuration | correct | hit@1 | hit@3 | MRR | refusal acc. | false refusals | p50 ms | p95 ms | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| winner of (c) | depth 10, sections off, replace, glossary off | 29/34 | 65% | 85% | 0.750 | 88% | 4% | 54 | 63 |  |
| (d) + glossary | depth 10, sections off, replace, glossary on | 30/34 | 73% | 88% | 0.808 | 88% | 4% | 53 | 65 | kept |

## Final configuration

`depth 10, sections off, replace, glossary on`, final_k 3, threshold +0.0.

### Hit@3 by question type (golden)

| run | paraphrase | exact_id | numeric_keyword | guidance_only | two_sections | out_of_corpus (refusal) |
|---|---:|---:|---:|---:|---:|---:|
| step-4a baseline | 60% | 100% | 100% | 100% | 100% | 88% |
| final | 70% | 100% | 100% | 100% | 100% | 88% |

### Every miss of the final configuration on golden, with its top 3

| id | type | question | expected | returned |
|---|---|---|---|---|
| p03 | paraphrase | How much floor space does a wheelchair user need to turn around? | 304.3/304.3.1/304.3.2 | 809.2.2 [standards], 304.2 [standards], 213, 603, 604, 608 [guidance] |
| p04 | paraphrase | How high on the wall can a light switch be mounted? | 308.2.1/308.3.1/309.3/205, 309 | (empty) |
| p07 | paraphrase | How many accessible parking spaces does a parking lot need? | 208.2 | 208, 502 [guidance], 502.3 [standards], 208.3.2 [standards] |
| o03 | out_of_corpus | How many GFCI outlets are required on a kitchen countertop? | (empty) | 205.1 [standards], 205, 309 [guidance] |

## Held-out set (eval/heldout.json: 10 new paraphrase + 2 negatives), report only

Written before the glossary and these experiments; never used for a decision.

| run | configuration | correct | hit@1 | hit@3 | MRR | refusal acc. | false refusals | p50 ms | p95 ms | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| step-4a baseline | depth 10, sections on, replace, glossary off | 8/12 | 60% | 60% | 0.600 | 100% | 10% | 54 | 64 |  |
| final | depth 10, sections off, replace, glossary on | 10/12 | 50% | 80% | 0.633 | 100% | 0% | 55 | 64 |  |

### Every miss of the final configuration on held-out, with its top 3

| id | type | question | expected | returned |
|---|---|---|---|---|
| h03 | paraphrase | How much space does a person in a wheelchair need in front of a fixture or control? | 305.3 | 802.1.3 [standards], 802.1.2 [standards], 1002.4.4.3 [standards] |
| h06 | paraphrase | What is the narrowest a ramp can be? | 405.5 | 405.7.2 [standards] |

### Every miss of the step-4a baseline on held-out, with its top 3

| id | type | question | expected | returned |
|---|---|---|---|---|
| h03 | paraphrase | How much space does a person in a wheelchair need in front of a fixture or control? | 305.3 | 802.1.3 [standards], 802.1.2 [standards], 1002.4.4.3 [standards] |
| h04 | paraphrase | How high can a step up in the floor be before it has to be ramped? | 303.4/303.2/303.3 | (empty) |
| h06 | paraphrase | What is the narrowest a ramp can be? | 405.5 | 405.7.2 [standards] |
| h10 | paraphrase | How much clear floor area is needed around a toilet? | 604.3.1/604.3 | 213, 603, 604, 608 [guidance], 605.3 [standards], 603.2.2 [standards] |
