# Answer eval: baseline chain vs LangGraph agent (deterministic, no LLM judge)

Model: ollama `llama3.2:3b`, temperature 0.0, num_ctx 8192; context cap 6,000 chars; retrieval: current RetrievalConfig defaults. Setups: baseline chain, graph (grade only), graph (grade + rewrite).

Number grounding (src/grounding.py): **verified** = every number is in a source cited in its sentence;
**needs_review** = some number is only in another source of the prompt (answered, flagged); **refused** =
some number is in no source of the prompt. Graph: one yes/no grading call per source (unparseable → yes);
the rewrite loop runs only when no source is graded relevant (max 2 retries).

- **wrong but verified** (deciding metric, target 0): verified answers missing an `answer_contains` number;
- correct: every `answer_contains` string appears in a given answer (questions that have one);
- needs_review: share of given answers (verified + needs_review) flagged for review;
- citation valid: at least one cited section is in `expected` (answered in-corpus questions);
- refusal acc.: out-of-corpus refused; false refusals: in-corpus refused;
- LLM calls per question and latency (whole call, warm models, this machine).

## golden

| setup | wrong but verified | correct among verified | needs_review | correct overall | citation valid | refusal acc. | false refusals | LLM calls / q (mean, max) | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline chain | **2** | 80% (8/10) | 13% (3/23) | 69% (9/13) | 91% (21/23) | 100% (8/8) | 12% (3/26) | 0.8, 1 | 5727 | 19739 |
| graph (grade only) | **1** | 90% (9/10) | 9% (2/23) | 77% (10/13) | 96% (22/23) | 100% (8/8) | 12% (3/26) | 2.9, 4 | 6306 | 17389 |
| graph (grade + rewrite) | **1** | 90% (9/10) | 9% (2/23) | 77% (10/13) | 96% (22/23) | 100% (8/8) | 12% (3/26) | 4.0, 8 | 4384 | 11776 |

False refusals by reason: baseline chain: {'no_relevant_sources': 1, 'model_not_in_sources': 2}; graph (grade only): {'no_relevant_sources': 1, 'model_not_in_sources': 2}; graph (grade + rewrite): {'model_not_in_sources': 3}

### golden: questions whose outcome differs between setups

| id | question | baseline chain | graph (grade only) | graph (grade + rewrite) |
|---|---|---|---|---|
| p04 | How high on the wall can a light switch be mounted? | refused (no_relevant_sources) | refused (no_relevant_sources) | refused (model_not_in_sources) |
| p09 | At what height do ramp handrails have to be installed? | WRONG [review] | correct | correct |
| p10 | How wide must the aisle next to an accessible parking space be? | WRONG | refused (model_not_in_sources) | refused (model_not_in_sources) |
| g02 | Can cars that are not vans park in van accessible parking spaces? | refused (model_not_in_sources) | answered | answered |
| o03 | How many GFCI outlets are required on a kitchen countertop? | refused (model_not_in_sources) ✓ | refused (graded_not_relevant) ✓ | refused (graded_not_relevant) ✓ |
| o05 | What sprinkler spacing is required in an office? | refused (no_relevant_sources) ✓ | refused (no_relevant_sources) ✓ | refused (graded_not_relevant) ✓ |
| o06 | What live load must an office floor support? | refused (no_relevant_sources) ✓ | refused (no_relevant_sources) ✓ | refused (graded_not_relevant) ✓ |

## held-out

| setup | wrong but verified | correct among verified | needs_review | correct overall | citation valid | refusal acc. | false refusals | LLM calls / q (mean, max) | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline chain | **0** | 100% (6/6) | 14% (1/7) | 70% (7/10) | 100% (7/7) | 100% (2/2) | 30% (3/10) | 0.8, 1 | 2874 | 12553 |
| graph (grade only) | **1** | 83% (5/6) | 0% (0/6) | 50% (5/10) | 100% (6/6) | 100% (2/2) | 40% (4/10) | 2.7, 4 | 3051 | 9170 |
| graph (grade + rewrite) | **1** | 83% (5/6) | 0% (0/6) | 50% (5/10) | 100% (6/6) | 100% (2/2) | 40% (4/10) | 3.9, 8 | 2419 | 9951 |

False refusals by reason: baseline chain: {'model_not_in_sources': 3}; graph (grade only): {'model_not_in_sources': 2, 'graded_not_relevant': 1, 'no_valid_citation': 1}; graph (grade + rewrite): {'model_not_in_sources': 3, 'no_valid_citation': 1}

### held-out: questions whose outcome differs between setups

| id | question | baseline chain | graph (grade only) | graph (grade + rewrite) |
|---|---|---|---|---|
| h06 | What is the narrowest a ramp can be? | refused (model_not_in_sources) | refused (graded_not_relevant) | refused (model_not_in_sources) |
| h07 | How high can the bottom of a bathroom mirror be? | correct | refused (no_valid_citation) | refused (no_valid_citation) |
| h08 | How deep must the space for your knees be under a sink? | correct [review] | WRONG | WRONG |
| hn1 | How deep must footings be below the frost line? | refused (no_relevant_sources) ✓ | refused (no_relevant_sources) ✓ | refused (graded_not_relevant) ✓ |

## Shadow cost per 1,000 questions (estimate: token counts from the llama tokenizer, not Claude's)

All LLM calls of a question (grading, rewrites, answer) at the prices in config.SHADOW_PRICES_USD_PER_MTOK (USD per million tokens, input/output: claude-haiku-4-5 $1/$5, claude-sonnet-5 $2/$10, claude-opus-5 $5/$25). Questions refused before any LLM call count as $0.

| setup | set | claude-haiku-4-5 | claude-sonnet-5 | claude-opus-5 | tokens in/out per question |
|---|---|---:|---:|---:|---:|
| baseline chain | golden | $1.18 | $2.35 | $5.88 | 833/69 |
| baseline chain | held-out | $0.88 | $1.77 | $4.42 | 666/44 |
| graph (grade only) | golden | $1.89 | $3.78 | $9.44 | 1559/66 |
| graph (grade only) | held-out | $1.14 | $2.29 | $5.72 | 989/31 |
| graph (grade + rewrite) | golden | $2.28 | $4.55 | $11.38 | 1917/72 |
| graph (grade + rewrite) | held-out | $1.50 | $3.01 | $7.52 | 1316/38 |

## Details: baseline chain

### golden: wrong but verified

| id | expected numbers | cited | answer |
|---|---|---|---|
| p03 | ['60'] | 304.2 | According to [S2], a wheelchair turning space shall comply with 304, and [S2] states that floor or ground surfaces of a turning space shall comply with 302, with changes in level not permitted, except for slopes not steeper than 1:48.  [S2] [ada_2010_standards.pdf p.110 §304.2] |
| p10 | ['60'] | 502.2 | According to [S2], the adjacent access aisle must be at least 96 inches (2440 mm) wide minimum, but this exception applies only when the van parking space is 96 inches (2440 mm) wide minimum. |

### golden: needs_review and grounding refusals

| id | status | flagged number: cited → found in | expected numbers | answer correct? | model output |
|---|---|---|---|---|---|
| p09 | needs_review | 36 inches: 505.4 [standards] → 405.8 [standards]; 915 mm: 505.4 [standards] → 405.8 [standards]; 36 inches: 505.4 [standards] → 405.8 [standards]; 915 mm: 505.4 [standards] → 405.8 [standards] | ['34', '38'] | no | According to [S1], the height of handrails on ramps shall be 36 inches (915 mm) minimum and 38 inches (965 mm) maximum above walking surfaces, stair nosings, and ramp surfaces.  However, [S2] states that for ramps with a rise greater than 6 inches (150 mm), handrails shall comply with [S1]'s require […] |
| e03 | needs_review | 26: 35.151(b) [guidance] → 35.151(b) [standards] | – | n/a | According to [S1], Section 35.151(b) Alterations requires that each facility or part of a facility altered by, on behalf of, or for the use of a public entity in a manner that affects or could affect the usability of the facility or part of the facility shall be altered in such manner that the alter […] |
| m02 | needs_review | 36 inches: 604.5.1 [standards] → 604.5.2 [standards]; 915 mm: 604.5.1 [standards] → 604.5.2 [standards] | ['42', '36'] | yes | According to [S1], the side wall grab bar must be at least 42 inches (1065 mm) long, and the rear wall grab bar must be at least 36 inches (915 mm) long. |

### golden: every failure

| id | failure | question | expected | answer / model output | cited |
|---|---|---|---|---|---|
| p03 | wrong or missing number; cited section not in expected | How much floor space does a wheelchair user need to turn around? | 304.3/304.3.1/304.3.2; contains ['60'] | According to [S2], a wheelchair turning space shall comply with 304, and [S2] states that floor or ground surfaces of a turning space shall comply with 302, with changes in level not permitted, except for slopes not steeper than 1:48.  [S2] [ada_2010_standards.pdf p.110 §304.2] | 304.2 |
| p04 | false refusal (no_relevant_sources) | How high on the wall can a light switch be mounted? | 308.2.1/308.3.1/309.3/205, 309; contains ['48'] | I don't know: the indexed code documents do not answer this question. | – |
| p07 | false refusal (model_not_in_sources) | How many accessible parking spaces does a parking lot need? | 208.2 | NOT_IN_SOURCES | – |
| p09 | wrong or missing number | At what height do ramp handrails have to be installed? | 505.4; contains ['34', '38'] | According to [S1], the height of handrails on ramps shall be 36 inches (915 mm) minimum and 38 inches (965 mm) maximum above walking surfaces, stair nosings, and ramp surfaces.  However, [S2] states that for ramps with a rise greater than 6 inches (150 mm), handrails shall comply with [S1]'s require […] | 505.4, 405.8, 1008.2.5.3.2 |
| p10 | wrong or missing number; cited section not in expected | How wide must the aisle next to an accessible parking space be? | 502.3/502.3.1; contains ['60'] | According to [S2], the adjacent access aisle must be at least 96 inches (2440 mm) wide minimum, but this exception applies only when the van parking space is 96 inches (2440 mm) wide minimum. | 502.2 |
| g02 | false refusal (model_not_in_sources) | Can cars that are not vans park in van accessible parking spaces? | 208, 502 | [S2] [ada_2010_standards.pdf p.155 §502.7] states that parking spaces and access aisles shall be designed so that cars and vans, when parked, cannot obstruct the required clear width of adjacent accessible routes. This implies that cars cannot park in van accessible parking spaces, as it would obstr […] | – |

### held-out: wrong but verified

None.

### held-out: needs_review and grounding refusals

| id | status | flagged number: cited → found in | expected numbers | answer correct? | model output |
|---|---|---|---|---|---|
| h08 | needs_review | 11 inches: 606.2 [standards] → 306.3.3 [standards]; 280 mm: 606.2 [standards] → 306.3.3 [standards]; 9 inches: 606.2 [standards] → 306.3.3 [standards]; 230 mm: 606.2 [standards] → 306.3.3 [standards]; 8 inches: 606.2 [standards] → 806 [guidance]/306.3.3 [standards]; 205 mm: 606.2 [standards] → 306.3.3 [standards]; 27 inches: 606.2 [standards] → 306.3.3 [standards]; 685 mm: 606.2 [standards] → 306.3.3 [standards] | ['11'] | yes | According to [S3] [ada_2010_standards.pdf p.174-175 §606.2], the knee clearance under a sink is 11 inches (280 mm) deep minimum at 9 inches (230 mm) above the finish floor or ground, and 8 inches (205 mm) deep minimum at 27 inches (685 mm) above the finish floor or ground. |

### held-out: every failure

| id | failure | question | expected | answer / model output | cited |
|---|---|---|---|---|---|
| h03 | false refusal (model_not_in_sources) | How much space does a person in a wheelchair need in front of a fixture or control? | 305.3; contains ['30', '48'] | NOT_IN_SOURCES | – |
| h06 | false refusal (model_not_in_sources) | What is the narrowest a ramp can be? | 405.5; contains ['36'] | NOT_IN_SOURCES | – |
| h10 | false refusal (model_not_in_sources) | How much clear floor area is needed around a toilet? | 604.3.1/604.3; contains ['60', '56'] | According to [S1], the required clear floor space at fixtures and turning space overlap is not specified, but the turning space includes knee and toe clearance at the lavatory, which is not related to the clear floor area around the toilet.  However, [S2] Plan 4B and Plan 5A show a clear floor space […] | – |

## Details: graph (grade only)

### golden: wrong but verified

| id | expected numbers | cited | answer |
|---|---|---|---|
| p03 | ['60'] | 809.2.2 | According to [S1], a wheelchair user needs at least 30 inches (760 mm) of turning space. |

### golden: needs_review and grounding refusals

| id | status | flagged number: cited → found in | expected numbers | answer correct? | model output |
|---|---|---|---|---|---|
| e03 | needs_review | 26: 35.151(b) [guidance] → 35.151(b) [standards] | – | n/a | According to [S1], Section 35.151(b) Alterations requires that each facility or part of a facility altered by, on behalf of, or for the use of a public entity in a manner that affects or could affect the usability of the facility or part of the facility shall be altered in such manner that the alter […] |
| m02 | needs_review | 36 inches: 604.5.1 [standards] → 604.5.2 [standards]; 915 mm: 604.5.1 [standards] → 604.5.2 [standards] | ['42', '36'] | yes | According to [S1], the side wall grab bar must be at least 42 inches (1065 mm) long, and the rear wall grab bar must be at least 36 inches (915 mm) long. |

### golden: grading and rewrite decisions

| id | queries tried | grades (section: verdict, ? = unparsed → yes) | outcome |
|---|---|---|---|
| p01 | `How wide does a doorway need to be for a wheelchair user?` | 404.3.1 [standards]: yes; 1002.4.4.2 [standards]: no; 802.1.2 [standards]: yes | correct |
| p03 | `How much floor space does a wheelchair user need to turn around?` | 809.2.2 [standards]: yes; 304.2 [standards]: no; 213, 603, 604, 608 [guidance]: no | WRONG |
| p05 | `How high should the seat of an accessible toilet be?` | 604.4 [standards]: yes; 604.9.3 [standards]: yes; 604.2 [standards]: no | correct |
| p07 | `How many accessible parking spaces does a parking lot need?` | 208, 502 [guidance]: yes; 502.3 [standards]: yes; 208.3.2 [standards]: no | refused (model_not_in_sources) |
| p09 | `At what height do ramp handrails have to be installed?` | 505.4 [standards]: yes; 405.8 [standards]: no; 1008.2.5.3.2 [standards]: yes | correct |
| p10 | `How wide must the aisle next to an accessible parking space be?` | 502.3.1 [standards]: yes; 502.2 [standards]: no; 503.3.1 [standards]: yes | refused (model_not_in_sources) |
| e02 | `What does section 405.2 say?` | 405.2 [standards]: yes; 235, 1003 [guidance]: yes; 36.405 [guidance]: no | correct |
| n01 | `33 to 36 inches grab bar` | 609.4 [standards]: yes; 104 [guidance]: no; 104.1.1 [standards]: yes | answered |
| g02 | `Can cars that are not vans park in van accessible parking spaces?` | 208, 502 [guidance]: yes; 502.7 [standards]: no; 502.3.4 [standards]: no | answered |
| g03 | `How did the handrail diameter requirement change from the 1991 Standards to the 2010 Standards?` | 505 [guidance]: yes; 505.7.1 [standards]: no; 205, 309 [guidance]: no | answered |
| g04 | `Why did the Department keep the phrase 'other sloped areas' in the curb ramp rule?` | 35.151(i) [guidance]: yes; 35.151(i) [standards]: yes; 205, 309 [guidance]: no | answered |
| o03 | `How many GFCI outlets are required on a kitchen countertop?` | 205.1 [standards]: no; 205, 309 [guidance]: no | refused (graded_not_relevant) ✓ |

### golden: every failure

| id | failure | question | expected | answer / model output | cited |
|---|---|---|---|---|---|
| p03 | wrong or missing number; cited section not in expected | How much floor space does a wheelchair user need to turn around? | 304.3/304.3.1/304.3.2; contains ['60'] | According to [S1], a wheelchair user needs at least 30 inches (760 mm) of turning space. | 809.2.2 |
| p04 | false refusal (no_relevant_sources) | How high on the wall can a light switch be mounted? | 308.2.1/308.3.1/309.3/205, 309; contains ['48'] | I don't know: the indexed code documents do not answer this question. | – |
| p07 | false refusal (model_not_in_sources) | How many accessible parking spaces does a parking lot need? | 208.2 | NOT_IN_SOURCES | – |
| p10 | false refusal (model_not_in_sources) | How wide must the aisle next to an accessible parking space be? | 502.3/502.3.1; contains ['60'] | NOT_IN_SOURCES | – |

### held-out: wrong but verified

| id | expected numbers | cited | answer |
|---|---|---|---|
| h08 | ['11'] | 306.3.3 | According to [S1], the minimum required depth for knee clearance under a sink is 8 inches (205 mm) at 27 inches (685 mm) above the finish floor or ground. |

### held-out: needs_review and grounding refusals

Every answer was verified.

### held-out: grading and rewrite decisions

| id | queries tried | grades (section: verdict, ? = unparsed → yes) | outcome |
|---|---|---|---|
| h02 | `How wide does a van accessible parking space need to be?` | 502.3.1 [standards]: yes; 502.2 [standards]: yes; 502.3.3 [standards]: no | correct |
| h03 | `How much space does a person in a wheelchair need in front of a fixture or control?` | 802.1.3 [standards]: yes; 802.1.2 [standards]: no; 1002.4.4.3 [standards]: no | refused (model_not_in_sources) |
| h06 | `What is the narrowest a ramp can be?` | 405.7.2 [standards]: no | refused (graded_not_relevant) |
| h07 | `How high can the bottom of a bathroom mirror be?` | 603.3 [standards]: yes; 213, 603, 604, 608 [guidance]: no | refused (no_valid_citation) |
| h08 | `How deep must the space for your knees be under a sink?` | 806 [guidance]: no; 306.3.3 [standards]: yes; 606.2 [standards]: no | WRONG |
| h09 | `How high can the rim of a bathroom sink be?` | 606.3 [standards]: yes; 606.2 [standards]: yes; 605.2 [standards]: no | correct |
| h10 | `How much clear floor area is needed around a toilet?` | 604.3.2 [standards]: yes; 806 [guidance]: no; 604.3.1 [standards]: yes | refused (model_not_in_sources) |

### held-out: every failure

| id | failure | question | expected | answer / model output | cited |
|---|---|---|---|---|---|
| h03 | false refusal (model_not_in_sources) | How much space does a person in a wheelchair need in front of a fixture or control? | 305.3; contains ['30', '48'] | NOT_IN_SOURCES | – |
| h06 | false refusal (graded_not_relevant) | What is the narrowest a ramp can be? | 405.5; contains ['36'] | I don't know: the indexed code documents do not answer this question. | – |
| h07 | false refusal (no_valid_citation) | How high can the bottom of a bathroom mirror be? | 603.3; contains ['40'] | The bottom edge of the reflecting surface of a bathroom mirror can be as low as 35 inches (890 mm) above the finish floor or ground, or as low as 40 inches (1015 mm) above the finish floor or ground, depending on its location. | – |
| h08 | wrong or missing number | How deep must the space for your knees be under a sink? | 306.3.3/606.2; contains ['11'] | According to [S1], the minimum required depth for knee clearance under a sink is 8 inches (205 mm) at 27 inches (685 mm) above the finish floor or ground. | 306.3.3 |
| h10 | false refusal (model_not_in_sources) | How much clear floor area is needed around a toilet? | 604.3.1/604.3; contains ['60', '56'] | NOT_IN_SOURCES | – |

## Details: graph (grade + rewrite)

### golden: wrong but verified

| id | expected numbers | cited | answer |
|---|---|---|---|
| p03 | ['60'] | 809.2.2 | According to [S1], a wheelchair user needs at least 30 inches (760 mm) of turning space. |

### golden: needs_review and grounding refusals

| id | status | flagged number: cited → found in | expected numbers | answer correct? | model output |
|---|---|---|---|---|---|
| e03 | needs_review | 26: 35.151(b) [guidance] → 35.151(b) [standards] | – | n/a | According to [S1], Section 35.151(b) Alterations requires that each facility or part of a facility altered by, on behalf of, or for the use of a public entity in a manner that affects or could affect the usability of the facility or part of the facility shall be altered in such manner that the alter […] |
| m02 | needs_review | 36 inches: 604.5.1 [standards] → 604.5.2 [standards]; 915 mm: 604.5.1 [standards] → 604.5.2 [standards] | ['42', '36'] | yes | According to [S1], the side wall grab bar must be at least 42 inches (1065 mm) long, and the rear wall grab bar must be at least 36 inches (915 mm) long. |

### golden: grading and rewrite decisions

| id | queries tried | grades (section: verdict, ? = unparsed → yes) | outcome |
|---|---|---|---|
| p01 | `How wide does a doorway need to be for a wheelchair user?` | 404.3.1 [standards]: yes; 1002.4.4.2 [standards]: no; 802.1.2 [standards]: yes | correct |
| p03 | `How much floor space does a wheelchair user need to turn around?` | 809.2.2 [standards]: yes; 304.2 [standards]: no; 213, 603, 604, 608 [guidance]: no | WRONG |
| p04 | `How high on the wall can a light switch be mounted?` → `light switch mounting height` | 309.3 [standards]: no; 205.1 [standards]: no; 205, 309 [guidance]: yes | refused (model_not_in_sources) |
| p05 | `How high should the seat of an accessible toilet be?` | 604.4 [standards]: yes; 604.9.3 [standards]: yes; 604.2 [standards]: no | correct |
| p07 | `How many accessible parking spaces does a parking lot need?` | 208, 502 [guidance]: yes; 502.3 [standards]: yes; 208.3.2 [standards]: no | refused (model_not_in_sources) |
| p09 | `At what height do ramp handrails have to be installed?` | 505.4 [standards]: yes; 405.8 [standards]: no; 1008.2.5.3.2 [standards]: yes | correct |
| p10 | `How wide must the aisle next to an accessible parking space be?` | 502.3.1 [standards]: yes; 502.2 [standards]: no; 503.3.1 [standards]: yes | refused (model_not_in_sources) |
| e02 | `What does section 405.2 say?` | 405.2 [standards]: yes; 235, 1003 [guidance]: yes; 36.405 [guidance]: no | correct |
| n01 | `33 to 36 inches grab bar` | 609.4 [standards]: yes; 104 [guidance]: no; 104.1.1 [standards]: yes | answered |
| g02 | `Can cars that are not vans park in van accessible parking spaces?` | 208, 502 [guidance]: yes; 502.7 [standards]: no; 502.3.4 [standards]: no | answered |
| g03 | `How did the handrail diameter requirement change from the 1991 Standards to the 2010 Standards?` | 505 [guidance]: yes; 505.7.1 [standards]: no; 205, 309 [guidance]: no | answered |
| g04 | `Why did the Department keep the phrase 'other sloped areas' in the curb ramp rule?` | 35.151(i) [guidance]: yes; 35.151(i) [standards]: yes; 205, 309 [guidance]: no | answered |
| o01 | `What is the capital of France?` → `capital of France` → `france capital` |  | refused (no_relevant_sources) ✓ |
| o02 | `What R-value of insulation is required in an attic?` → `attic insulation R-value requirements` → `attic insulation R-value requirements` |  | refused (no_relevant_sources) ✓ |
| o03 | `How many GFCI outlets are required on a kitchen countertop?` → `2010 ADA kitchen GFCI outlets` → `2010 ADA kitchen GFCI outlet requirements` | 205.1 [standards]: no; 205, 309 [guidance]: no; 205.1 [standards]: no; 205, 309 [guidance]: no; 205.1 [standards]: no; 205, 309 [guidance]: no | refused (graded_not_relevant) ✓ |
| o04 | `What fire-resistance rating do exterior walls need near a property line?` → `fire-resistance rating for exterior walls near property line` → `fire-resistance rating for exterior walls near property line` |  | refused (no_relevant_sources) ✓ |
| o05 | `What sprinkler spacing is required in an office?` → `2010 ADA sprinkler spacing office` → `2010 ADA sprinkler spacing office building` | 705.1.2 [standards]: no; 703.2.8 [standards]: no; 703.5.8 [standards]: no; 207 [guidance]: no; 705.1.2 [standards]: no; 703.2.7 [standards]: no | refused (graded_not_relevant) ✓ |
| o06 | `What live load must an office floor support?` → `2010 ADA Standards live load` → `2010 ADA Standards live load per square foot` | 1002.3 [standards]: no; 234.2 [standards]: no; 234.4.1 [standards]: no; 703.5.5 [standards]: no; 234.2 [standards]: no; 105.2.3 [standards]: no | refused (graded_not_relevant) ✓ |
| o07 | `What U-factor must windows have in climate zone 5?` → `window U-factor climate zone 5` → `window U-factor climate zone 5` |  | refused (no_relevant_sources) ✓ |
| o08 | `What amperage must a kitchen countertop circuit have?` → `Kitchen countertop circuit ampacity` → `Kitchen countertop circuit ampacity` |  | refused (no_relevant_sources) ✓ |

### golden: every failure

| id | failure | question | expected | answer / model output | cited |
|---|---|---|---|---|---|
| p03 | wrong or missing number; cited section not in expected | How much floor space does a wheelchair user need to turn around? | 304.3/304.3.1/304.3.2; contains ['60'] | According to [S1], a wheelchair user needs at least 30 inches (760 mm) of turning space. | 809.2.2 |
| p04 | false refusal (model_not_in_sources) | How high on the wall can a light switch be mounted? | 308.2.1/308.3.1/309.3/205, 309; contains ['48'] | NOT_IN_SOURCES | – |
| p07 | false refusal (model_not_in_sources) | How many accessible parking spaces does a parking lot need? | 208.2 | NOT_IN_SOURCES | – |
| p10 | false refusal (model_not_in_sources) | How wide must the aisle next to an accessible parking space be? | 502.3/502.3.1; contains ['60'] | NOT_IN_SOURCES | – |

### held-out: wrong but verified

| id | expected numbers | cited | answer |
|---|---|---|---|
| h08 | ['11'] | 306.3.3 | According to [S1], the minimum required depth for knee clearance under a sink is 8 inches (205 mm) at 27 inches (685 mm) above the finish floor or ground. |

### held-out: needs_review and grounding refusals

Every answer was verified.

### held-out: grading and rewrite decisions

| id | queries tried | grades (section: verdict, ? = unparsed → yes) | outcome |
|---|---|---|---|
| h02 | `How wide does a van accessible parking space need to be?` | 502.3.1 [standards]: yes; 502.2 [standards]: yes; 502.3.3 [standards]: no | correct |
| h03 | `How much space does a person in a wheelchair need in front of a fixture or control?` | 802.1.3 [standards]: yes; 802.1.2 [standards]: no; 1002.4.4.3 [standards]: no | refused (model_not_in_sources) |
| h06 | `What is the narrowest a ramp can be?` → `minimum clear width of a ramp` | 405.7.2 [standards]: no; 405.5 [standards]: no; 405.8 [standards]: no; 406.4 [standards]: yes | refused (model_not_in_sources) |
| h07 | `How high can the bottom of a bathroom mirror be?` | 603.3 [standards]: yes; 213, 603, 604, 608 [guidance]: no | refused (no_valid_citation) |
| h08 | `How deep must the space for your knees be under a sink?` | 806 [guidance]: no; 306.3.3 [standards]: yes; 606.2 [standards]: no | WRONG |
| h09 | `How high can the rim of a bathroom sink be?` | 606.3 [standards]: yes; 606.2 [standards]: yes; 605.2 [standards]: no | correct |
| h10 | `How much clear floor area is needed around a toilet?` | 604.3.2 [standards]: yes; 806 [guidance]: no; 604.3.1 [standards]: yes | refused (model_not_in_sources) |
| hn1 | `How deep must footings be below the frost line?` → `2010 ADA Standards 404.2.2.1` → `2010 ADA 404.2.2.1 frost line depth` | 404.1 [standards]: no; 404.3 [standards]: no; 36.406(g) [guidance]: no; 703.2.1 [standards]: no; 306.2.2 [standards]: no; 404.2.11 [standards]: no | refused (graded_not_relevant) ✓ |
| hn2 | `What is the maximum spacing of floor joists?` → `floor joist spacing maximum` → `floor joist spacing maximum` |  | refused (no_relevant_sources) ✓ |

### held-out: every failure

| id | failure | question | expected | answer / model output | cited |
|---|---|---|---|---|---|
| h03 | false refusal (model_not_in_sources) | How much space does a person in a wheelchair need in front of a fixture or control? | 305.3; contains ['30', '48'] | NOT_IN_SOURCES | – |
| h06 | false refusal (model_not_in_sources) | What is the narrowest a ramp can be? | 405.5; contains ['36'] | NOT_IN_SOURCES | – |
| h07 | false refusal (no_valid_citation) | How high can the bottom of a bathroom mirror be? | 603.3; contains ['40'] | The bottom edge of the reflecting surface of a bathroom mirror can be as low as 35 inches (890 mm) above the finish floor or ground, or as low as 40 inches (1015 mm) above the finish floor or ground, depending on its location. | – |
| h08 | wrong or missing number | How deep must the space for your knees be under a sink? | 306.3.3/606.2; contains ['11'] | According to [S1], the minimum required depth for knee clearance under a sink is 8 inches (205 mm) at 27 inches (685 mm) above the finish floor or ground. | 306.3.3 |
| h10 | false refusal (model_not_in_sources) | How much clear floor area is needed around a toilet? | 604.3.1/604.3; contains ['60', '56'] | NOT_IN_SOURCES | – |
