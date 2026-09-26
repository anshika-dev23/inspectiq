# Answer eval (baseline chain + three-state number grounding; deterministic, no LLM judge)

Model: ollama `llama3.2:3b`, temperature 0.0, num_ctx 8192; context cap 6,000 chars; retrieval: current RetrievalConfig defaults.

Three-state number grounding (src/grounding.py): **verified** = every number is in a source cited in its
sentence; **needs_review** = some number is only in another source of the prompt (answered, flagged);
**refused** = some number is in no source of the prompt (refusal `ungrounded_number`).

- wrong but verified (target 0): verified answers missing an `answer_contains` number;
- correct: every `answer_contains` string appears in a given answer (questions that have one);
- needs_review: share of given answers (verified + needs_review) flagged for review;
- citation valid: at least one cited section is in `expected` (answered in-corpus questions);
- refusal acc.: out-of-corpus questions refused; false refusals: in-corpus questions refused.

| set | wrong but verified | correct among verified | needs_review | correct among needs_review | correct overall | citation valid | refusal acc. | false refusals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| golden | **2** | 80% (8/10) | 13% (3/23) | 1/2 | 69% (9/13) | 91% (21/23) | 100% (8/8) | 12% (3/26) |
| held-out | **0** | 100% (6/6) | 14% (1/7) | 1/1 | 70% (7/10) | 100% (7/7) | 100% (2/2) | 30% (3/10) |

False refusals by reason: golden: {'no_relevant_sources': 1, 'model_not_in_sources': 2}; held-out: {'model_not_in_sources': 3}

Latency (whole answer() call, warm models, this machine) and tokens:

| set | p50 ms | p95 ms | retrieval p50 | LLM p50 | LLM p95 | tokens in/out (mean) |
|---|---:|---:|---:|---:|---:|---:|
| golden | 2815 | 14225 | 174 | 3491 | 13964 | 1090/88 |
| held-out | 1727 | 7421 | 124 | 1817 | 7232 | 799/52 |

## Wrong but verified

Grounding only checks that a number comes from a cited source; these came from the wrong section.

### golden

| id | expected numbers | cited | answer |
|---|---|---|---|
| p03 | ['60'] | 304.2 | According to [S2], a wheelchair turning space shall comply with 304, and [S2] states that floor or ground surfaces of a turning space shall comply with 302, with changes in level not permitted, except for slopes not steeper than 1:48.  [S2] [ada_2010_standards.pdf p.110 §304.2] |
| p10 | ['60'] | 502.2 | According to [S2], the adjacent access aisle must be at least 96 inches (2440 mm) wide minimum, but this exception applies only when the van parking space is 96 inches (2440 mm) wide minimum. |

### held-out

None.

## needs_review and grounding refusals, with each flagged number

`cited → found in`: the sections the sentence cited, and the prompt sections that contain the number.

### golden

| id | status | flagged number: cited → found in | expected numbers | answer correct? | model output |
|---|---|---|---|---|---|
| p09 | needs_review | 36 inches: 505.4 [standards] → 405.8 [standards]; 915 mm: 505.4 [standards] → 405.8 [standards]; 36 inches: 505.4 [standards] → 405.8 [standards]; 915 mm: 505.4 [standards] → 405.8 [standards] | ['34', '38'] | no | According to [S1], the height of handrails on ramps shall be 36 inches (915 mm) minimum and 38 inches (965 mm) maximum above walking surfaces, stair nosings, and ramp surfaces.  However, [S2] states that for ramps with a rise greater than 6 inches (150 mm), handrails shall comply with [S1]'s require […] |
| e03 | needs_review | 26: 35.151(b) [guidance] → 35.151(b) [standards] | – | n/a | According to [S1], Section 35.151(b) Alterations requires that each facility or part of a facility altered by, on behalf of, or for the use of a public entity in a manner that affects or could affect the usability of the facility or part of the facility shall be altered in such manner that the alter […] |
| m02 | needs_review | 36 inches: 604.5.1 [standards] → 604.5.2 [standards]; 915 mm: 604.5.1 [standards] → 604.5.2 [standards] | ['42', '36'] | yes | According to [S1], the side wall grab bar must be at least 42 inches (1065 mm) long, and the rear wall grab bar must be at least 36 inches (915 mm) long. |

### held-out

| id | status | flagged number: cited → found in | expected numbers | answer correct? | model output |
|---|---|---|---|---|---|
| h08 | needs_review | 11 inches: 606.2 [standards] → 306.3.3 [standards]; 280 mm: 606.2 [standards] → 306.3.3 [standards]; 9 inches: 606.2 [standards] → 306.3.3 [standards]; 230 mm: 606.2 [standards] → 306.3.3 [standards]; 8 inches: 606.2 [standards] → 806 [guidance]/306.3.3 [standards]; 205 mm: 606.2 [standards] → 306.3.3 [standards]; 27 inches: 606.2 [standards] → 306.3.3 [standards]; 685 mm: 606.2 [standards] → 306.3.3 [standards] | ['11'] | yes | According to [S3] [ada_2010_standards.pdf p.174-175 §606.2], the knee clearance under a sink is 11 inches (280 mm) deep minimum at 9 inches (230 mm) above the finish floor or ground, and 8 inches (205 mm) deep minimum at 27 inches (685 mm) above the finish floor or ground. |

## Shadow cost per 1,000 questions (estimate: token counts from the llama tokenizer, not Claude's)

What the LLM calls of this run would cost on each model, from Ollama's token counts and the prices in
config.SHADOW_PRICES_USD_PER_MTOK (USD per million tokens, input/output: claude-haiku-4-5 $1/$5, claude-sonnet-5 $2/$10, claude-opus-5 $5/$25). Questions refused before the LLM count as $0.

| set | claude-haiku-4-5 | claude-sonnet-5 | claude-opus-5 |
|---|---:|---:|---:|
| golden | $1.17 | $2.34 | $5.85 |
| held-out | $0.88 | $1.77 | $4.42 |

## Every failure: golden

| id | failure | question | expected | answer / model output | cited |
|---|---|---|---|---|---|
| p03 | wrong or missing number; cited section not in expected | How much floor space does a wheelchair user need to turn around? | 304.3/304.3.1/304.3.2; contains ['60'] | According to [S2], a wheelchair turning space shall comply with 304, and [S2] states that floor or ground surfaces of a turning space shall comply with 302, with changes in level not permitted, except for slopes not steeper than 1:48.  [S2] [ada_2010_standards.pdf p.110 §304.2] | 304.2 |
| p04 | false refusal (no_relevant_sources) | How high on the wall can a light switch be mounted? | 308.2.1/308.3.1/309.3/205, 309; contains ['48'] | I don't know: the indexed code documents do not answer this question. | – |
| p07 | false refusal (model_not_in_sources) | How many accessible parking spaces does a parking lot need? | 208.2 | NOT_IN_SOURCES | – |
| p09 | wrong or missing number | At what height do ramp handrails have to be installed? | 505.4; contains ['34', '38'] | According to [S1], the height of handrails on ramps shall be 36 inches (915 mm) minimum and 38 inches (965 mm) maximum above walking surfaces, stair nosings, and ramp surfaces.  However, [S2] states that for ramps with a rise greater than 6 inches (150 mm), handrails shall comply with [S1]'s require […] | 505.4, 405.8, 1008.2.5.3.2 |
| p10 | wrong or missing number; cited section not in expected | How wide must the aisle next to an accessible parking space be? | 502.3/502.3.1; contains ['60'] | According to [S2], the adjacent access aisle must be at least 96 inches (2440 mm) wide minimum, but this exception applies only when the van parking space is 96 inches (2440 mm) wide minimum. | 502.2 |
| g02 | false refusal (model_not_in_sources) | Can cars that are not vans park in van accessible parking spaces? | 208, 502 | [S2] [ada_2010_standards.pdf p.155 §502.7] states that parking spaces and access aisles shall be designed so that cars and vans, when parked, cannot obstruct the required clear width of adjacent accessible routes. This implies that cars cannot park in van accessible parking spaces, as it would obstr […] | – |

## Every failure: held-out

| id | failure | question | expected | answer / model output | cited |
|---|---|---|---|---|---|
| h03 | false refusal (model_not_in_sources) | How much space does a person in a wheelchair need in front of a fixture or control? | 305.3; contains ['30', '48'] | NOT_IN_SOURCES | – |
| h06 | false refusal (model_not_in_sources) | What is the narrowest a ramp can be? | 405.5; contains ['36'] | NOT_IN_SOURCES | – |
| h10 | false refusal (model_not_in_sources) | How much clear floor area is needed around a toilet? | 604.3.1/604.3; contains ['60', '56'] | According to [S1], the required clear floor space at fixtures and turning space overlap is not specified, but the turning space includes knee and toe clearance at the lavatory, which is not related to the clear floor area around the toilet.  However, [S2] Plan 4B and Plan 5A show a clear floor space […] | – |
