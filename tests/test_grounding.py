import pytest

from src.answer import answer_from_contexts
from src.grounding import check_grounding, extract_numbers
from tests.test_answer import FakeLLM, make_context


def keys(text: str) -> list[str]:
    return [n.key for n in extract_numbers(text)]


# --- extraction and normalization -------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("34 inches (865 mm) minimum", ["34 in", "865 mm"]),
    ("34 in", ["34 in"]),
    ("a 60-inch diameter", ["60 in"]),
    ("30 in. by 48 in", ["30 in", "48 in"]),
    ('12" maximum', ["12 in"]),
    ("5 pounds (22.2 N)", ["5 lb", "22.2 N"]),
    ("5 lbf", ["5 lb"]),
    ("a slope of 1:12", ["1:12"]),
    ("1 : 48 cross slope", ["1:48"]),
    ("5% or 5 percent", ["5 %", "5 %"]),
    ("over ½ inch", ["0.5 in"]),
    ("over 1/2 inch", ["0.5 in"]),
    ("1 1/4 inches", ["1.25 in"]),
    ("1 ¼ inches", ["1.25 in"]),
    ("1,000 square feet", ["1000"]),          # area units are not parsed: value-only match
])
def test_extract_numbers_with_units(text, expected):
    assert keys(text) == expected


@pytest.mark.parametrize("text", [
    "According to [S1] and [S2, S3]",
    "[ada_2010_standards.pdf p.155 §502.7] says",
    "section 404.2.3 and sections 208 and 502",
    "under 35.151(b) and § 36.406(f)",
    "see 28 CFR 35.151 and 42 U.S.C. 12183(a)(2)",
    "the 2010 Standards replace the 1991 Standards",
    "Figure 604.5.1 and Table 208.2",
    "1. Interior doors\n2. Fire doors",
    "a turning space shall comply with 304",
    "surfaces complying with 302 and 303",
    "within the reach ranges specified in 308",
])
def test_extract_numbers_skips_references_labels_years_and_list_numbers(text):
    assert keys(text) == []


# --- grounding ----------------------------------------------------------------

SOURCE_505_4 = "505.4 Height. Top of gripping surfaces of handrails shall be 34 inches (865 mm) minimum and 38\ninches (965 mm) maximum vertically above walking surfaces."


def test_p09_misquoted_number_is_ungrounded():
    answer = "The height of handrails shall be 36 inches (915 mm) minimum and 38 inches (965 mm) maximum [S1]."
    result = check_grounding(answer, [SOURCE_505_4])
    assert not result.verified
    assert [n.raw for n in result.ungrounded] == ["36 inches", "915 mm"]


def test_correct_answer_is_grounded_across_a_line_break():
    answer = "Handrails: 34 inches (865 mm) minimum and 38 inches (965 mm) maximum [S1]."
    assert check_grounding(answer, [SOURCE_505_4]).verified


def test_units_are_normalized_between_answer_and_source():
    assert check_grounding("34 in minimum [S1]", [SOURCE_505_4]).verified
    assert check_grounding("a ½ inch lip [S1]", ["Changes in level of 1/2 inch high maximum"]).verified


def test_unit_mismatch_is_ungrounded_but_unitless_matches_any_unit():
    assert not check_grounding("34 mm [S1]", [SOURCE_505_4]).verified
    assert check_grounding("between 34 and 38 [S1]", [SOURCE_505_4]).verified
    assert check_grounding("60 inches [S1]", ["Table 208.2: 60 48 36"]).verified  # table lost its units


def test_only_cited_sources_count():
    # check_grounding receives only the cited sources' texts; an uncited source cannot ground a number.
    assert not check_grounding("42 inches [S1]", [SOURCE_505_4]).verified


def test_answer_without_numbers_is_verified():
    assert check_grounding("Grab bars are required on the side wall and the rear wall [S1].", [SOURCE_505_4]).verified


# --- in the chain -------------------------------------------------------------

HANDRAIL = make_context("505.4", SOURCE_505_4, pages="158")
P09_REPLY = "According to [S1], handrails shall be 36 inches (915 mm) minimum and 38 inches (965 mm) maximum."


def test_strict_mode_refuses_an_ungrounded_number():
    result = answer_from_contexts("How high are ramp handrails?", [HANDRAIL], FakeLLM(P09_REPLY), 6000)
    assert result.refused and result.refusal_reason == "ungrounded_number"
    assert result.verified is False and result.ungrounded_numbers == ["36 inches", "915 mm"]
    assert result.raw_llm_text == P09_REPLY
    assert [c.section_id for c in result.citations] == ["505.4"]  # kept for debugging


def test_non_strict_mode_answers_but_marks_it_unverified():
    result = answer_from_contexts("q", [HANDRAIL], FakeLLM(P09_REPLY), 6000, strict_grounding=False)
    assert not result.refused and result.verified is False
    assert result.ungrounded_numbers == ["36 inches", "915 mm"]


def test_grounded_answer_is_verified():
    reply = "Handrails shall be 34 inches (865 mm) minimum and 38 inches (965 mm) maximum [S1]."
    result = answer_from_contexts("q", [HANDRAIL], FakeLLM(reply), 6000)
    assert not result.refused and result.verified is True and result.ungrounded_numbers == []


def test_number_from_an_uncited_source_is_ungrounded():
    other = make_context("405.8", "405.8 Handrails. Ramp runs with a rise greater than 6 inches shall have handrails.")
    reply = "Ramps with a rise over 6 inches need handrails [S1]."  # the 6 inches is in S2, not the cited S1
    result = answer_from_contexts("q", [HANDRAIL, other], FakeLLM(reply), 6000)
    assert result.refused and result.ungrounded_numbers == ["6 inches"]


# --- per-sentence grounding ---------------------------------------------------

from src.grounding import check_grounding_by_claim, split_claims  # noqa: E402

SOURCE_405_8 = ("405.8 Handrails. Ramp runs with a rise greater than 6 inches (150 mm) shall have handrails. "
                "Ramps shall be designed to maintain a 36 inch (915 mm) minimum clear width when handrails are installed.")
P09_THREE_SOURCES = (
    "According to [S1], the height of handrails on ramps shall be 36 inches (915 mm) minimum and 38 inches "
    "(965 mm) maximum.\n\nHowever, [S2] states that ramps with a rise greater than 6 inches (150 mm) need "
    "handrails complying with [S1]."
)


def test_split_claims_attaches_trailing_citations_and_keeps_decimals():
    assert split_claims("Doors need 32 inches. [S1]\nForce is 22.2 N [S2].") == [
        ("Doors need 32 inches. [S1]", ["S1"]),
        ("Force is 22.2 N [S2].", ["S2"]),
    ]


def test_p09_number_borrowed_from_another_cited_source_is_caught():
    sources = {"S1": SOURCE_505_4, "S2": SOURCE_405_8}
    # Answer-level check: 36 and 915 appear in S2 (clear width), so the misquote passes ...
    assert check_grounding(P09_THREE_SOURCES, list(sources.values())).verified
    # ... per-sentence check: the height sentence cites only S1, which has no 36 or 915.
    result = check_grounding_by_claim(P09_THREE_SOURCES, sources)
    assert [n.raw for n in result.ungrounded] == ["36 inches", "915 mm"]


def test_correct_multi_source_answer_is_verified_per_sentence():
    answer = ("Handrails are 34 inches (865 mm) minimum [S1]. "
              "Ramps with a rise over 6 inches (150 mm) need them [S2].")
    assert check_grounding_by_claim(answer, {"S1": SOURCE_505_4, "S2": SOURCE_405_8}).verified


def test_sentence_without_citation_is_checked_against_all_cited_sources():
    answer = "Handrails are needed on ramps [S2]. The rise limit is 6 inches."
    assert check_grounding_by_claim(answer, {"S1": SOURCE_505_4, "S2": SOURCE_405_8}).verified


def test_invalid_labels_ground_nothing():
    answer = "Handrails are 34 inches minimum [S7]."
    assert not check_grounding_by_claim(answer, {"S1": SOURCE_505_4}).verified


def test_p03_cross_reference_is_not_a_quantity():
    answer = "According to [S2], a turning space shall comply with 304, with slopes not steeper than 1:48 [S2]."
    source = "304.2 Floor or Ground Surfaces. Floor or ground surfaces of a turning space shall comply with 302. Changes in level are not permitted. EXCEPTION: Slopes not steeper than 1:48 shall be permitted."
    assert check_grounding_by_claim(answer, {"S2": source}).verified


def test_e03_date_attributed_to_the_wrong_source_is_caught():
    sources = {"S1": "Section 35.151(b) Alterations. The 1991 title II regulation does not contain ...",
               "S2": "(b) Alterations. ... if the alteration was commenced after January 26, 1992."}
    answer = "Altered portions must be accessible if the alteration was commenced after January 26, 1992 [S1]. See also [S2]."
    result = check_grounding_by_claim(answer, sources)
    assert [n.raw for n in result.ungrounded] == ["26"]   # the year is skipped, the day is not
