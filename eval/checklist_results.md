# Checklist eval: Public restroom, ground floor (sample inspection)

Drafted outcomes (before human review) vs expected. The chain finds and cites the rule; compute_outcome()
compares the measurement. needs_review is never a wrong pass/fail: it goes to the inspector.

- outcome as expected: **3/5**
- wrong pass/fail (the dangerous error): **0**
- needs_review: 2/5
- expected section cited: 4/5

| item | measured | expected | drafted | reason | cited | grounding |
|---|---|---|---|---|---|---|
| Restroom entry door: clear width | 30 in | fail (§404.2.3) | fail ✓ | 30 in is below the minimum 32 inches | 404.2.3 | verified |
| Water closet grab bars: height to top of gripping surface | 35 in | pass (§609.4) | pass ✓ | 35 in is within the limits | 609.4 | verified |
| Water closet: seat height | 20 in | fail (§604.4) | needs_review review | several different limits: minimums ['17 inches'], maximums ['17 inches', '19 inches'] | 604.4, 604.9.3 | verified |
| Lavatory: knee clearance width | 27 in | fail (§306.3.5) | needs_review review | no grounded requirement (refused: model_not_in_sources) | – | – |
| Entrance ramp: running slope | 1:10 | fail (§405.2) | fail ✓ | 1:10 is above the maximum 1:12 (steeper) | 405.2 | verified |

## Requirements as drafted

### Restroom entry door: clear width

> According to [S1], the minimum clear width of a door opening is 32 inches (815 mm).

### Water closet grab bars: height to top of gripping surface

> According to [S3], grab bars must be installed between 33 inches (840 mm) minimum and 36 inches (915 mm) maximum above the finish floor, measured to the top of the gripping surface.

### Water closet: seat height

> According to [S1], the seat height of a water closet above the finish floor shall be 17 inches (430 mm) minimum and 19 inches (485 mm) maximum measured to the top of the seat.
> 
> Note that [S2] does not provide a different minimum height, but rather a maximum height of 17 inches (430 mm) for water closets for children's use.

### Lavatory: knee clearance width

> (refused: model_not_in_sources)

### Entrance ramp: running slope

> According to [S1], the maximum running slope of a ramp is not steeper than 1:12 [S1 405.2].
