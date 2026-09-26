# Answer eval (step 3 baseline chain, deterministic, no LLM judge)

Model: ollama `llama3.2:3b`, temperature 0.0, num_ctx 8192; context cap 6,000 chars; retrieval: current RetrievalConfig defaults.

- answer correct: all `answer_contains` strings appear in a non-refused answer (only questions that have one);
- citation valid: at least one cited section is in `expected` (answered in-corpus questions);
- refusal acc.: out-of-corpus questions refused; false refusals: in-corpus questions refused;
- latency: the whole answer() call, warm models, on this machine.

| set | answer correct | citation valid | refusal acc. | false refusals | p50 ms | p95 ms | retrieval p50 | LLM p50 | LLM p95 | tokens in/out (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| golden | 62% (8/13) | 90% (18/20) | 100% (8/8) | 23% (6/26) | 2810 | 10824 | 185 | 3585 | 10640 | 1090/88 |
| held-out | 60% (6/10) | 100% (6/6) | 100% (2/2) | 40% (4/10) | 1633 | 8464 | 127 | 1781 | 8269 | 799/52 |

False refusals by reason: golden: {'no_relevant_sources': 1, 'model_not_in_sources': 2, 'ungrounded_number': 3}; held-out: {'model_not_in_sources': 3, 'ungrounded_number': 1}

## Number grounding (strict: True)

Answers blocked because a number is not in a source cited in its sentence (src/grounding.py).

### golden

| id | ungrounded numbers | expected numbers | model's text had them? | model output |
|---|---|---|---|---|
| p09 | ['36 inches', '915 mm', '36 inches', '915 mm'] | ['34', '38'] | no: correct block | According to [S1], the height of handrails on ramps shall be 36 inches (915 mm) minimum and 38 inches (965 mm) maximum above walking surfaces, stair nosings, and ramp surfaces.  However, [S2] states that for ramps with a rise greater than 6 inches (150 mm), handrails shall comply with [S1]'s require […] |
| e03 | ['26'] | – | n/a (no expected number) | According to [S1], Section 35.151(b) Alterations requires that each facility or part of a facility altered by, on behalf of, or for the use of a public entity in a manner that affects or could affect the usability of the facility or part of the facility shall be altered in such manner that the alter […] |
| m02 | ['36 inches', '915 mm'] | ['42', '36'] | yes: right number, not in the cited source (false block for the user) | According to [S1], the side wall grab bar must be at least 42 inches (1065 mm) long, and the rear wall grab bar must be at least 36 inches (915 mm) long. |

### held-out

| id | ungrounded numbers | expected numbers | model's text had them? | model output |
|---|---|---|---|---|
| h08 | ['11 inches', '280 mm', '9 inches', '230 mm', '8 inches', '205 mm', '27 inches', '685 mm'] | ['11'] | yes: right number, not in the cited source (false block for the user) | According to [S3] [ada_2010_standards.pdf p.174-175 §606.2], the knee clearance under a sink is 11 inches (280 mm) deep minimum at 9 inches (230 mm) above the finish floor or ground, and 8 inches (205 mm) deep minimum at 27 inches (685 mm) above the finish floor or ground. |

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
| p09 | false refusal (ungrounded_number) | At what height do ramp handrails have to be installed? | 505.4; contains ['34', '38'] | According to [S1], the height of handrails on ramps shall be 36 inches (915 mm) minimum and 38 inches (965 mm) maximum above walking surfaces, stair nosings, and ramp surfaces.  However, [S2] states that for ramps with a rise greater than 6 inches (150 mm), handrails shall comply with [S1]'s require […] | 505.4, 405.8, 1008.2.5.3.2 |
| p10 | wrong or missing number; cited section not in expected | How wide must the aisle next to an accessible parking space be? | 502.3/502.3.1; contains ['60'] | According to [S2], the adjacent access aisle must be at least 96 inches (2440 mm) wide minimum, but this exception applies only when the van parking space is 96 inches (2440 mm) wide minimum. | 502.2 |
| e03 | false refusal (ungrounded_number) | Explain 35.151(b). | 35.151(b)/35.151(b) | According to [S1], Section 35.151(b) Alterations requires that each facility or part of a facility altered by, on behalf of, or for the use of a public entity in a manner that affects or could affect the usability of the facility or part of the facility shall be altered in such manner that the alter […] | 35.151(b), 35.151(b), 35.151(b)(3) |
| g02 | false refusal (model_not_in_sources) | Can cars that are not vans park in van accessible parking spaces? | 208, 502 | [S2] [ada_2010_standards.pdf p.155 §502.7] states that parking spaces and access aisles shall be designed so that cars and vans, when parked, cannot obstruct the required clear width of adjacent accessible routes. This implies that cars cannot park in van accessible parking spaces, as it would obstr […] | – |
| m02 | false refusal (ungrounded_number) | How long must the side wall and the rear wall grab bars be at a water closet? | 604.5.1 + 604.5.2; contains ['42', '36'] | According to [S1], the side wall grab bar must be at least 42 inches (1065 mm) long, and the rear wall grab bar must be at least 36 inches (915 mm) long. | 604.5.1 |

## Every failure: held-out

| id | failure | question | expected | answer / model output | cited |
|---|---|---|---|---|---|
| h03 | false refusal (model_not_in_sources) | How much space does a person in a wheelchair need in front of a fixture or control? | 305.3; contains ['30', '48'] | NOT_IN_SOURCES | – |
| h06 | false refusal (model_not_in_sources) | What is the narrowest a ramp can be? | 405.5; contains ['36'] | NOT_IN_SOURCES | – |
| h08 | false refusal (ungrounded_number) | How deep must the space for your knees be under a sink? | 306.3.3/606.2; contains ['11'] | According to [S3] [ada_2010_standards.pdf p.174-175 §606.2], the knee clearance under a sink is 11 inches (280 mm) deep minimum at 9 inches (230 mm) above the finish floor or ground, and 8 inches (205 mm) deep minimum at 27 inches (685 mm) above the finish floor or ground. | 606.2 |
| h10 | false refusal (model_not_in_sources) | How much clear floor area is needed around a toilet? | 604.3.1/604.3; contains ['60', '56'] | According to [S1], the required clear floor space at fixtures and turning space overlap is not specified, but the turning space includes knee and toe clearance at the lavatory, which is not related to the clear floor area around the toilet.  However, [S2] Plan 4B and Plan 5A show a clear floor space […] | – |
