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


# --- three-state grounding ---------------------------------------------------

from src.grounding import NEEDS_REVIEW, REFUSED, VERIFIED, assess_grounding, split_claims  # noqa: E402

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


def test_correct_multi_source_answer_is_verified():
    answer = ("Handrails are 34 inches (865 mm) minimum [S1]. "
              "Ramps with a rise over 6 inches (150 mm) need them [S2].")
    result = assess_grounding(answer, {"S1": SOURCE_505_4, "S2": SOURCE_405_8})
    assert result.status == VERIFIED and result.flagged == []


def test_p09_number_from_another_source_needs_review_and_says_where_it_was_found():
    sources = {"S1": SOURCE_505_4, "S2": SOURCE_405_8}
    # An answer-level check passes p09: 36 and 915 are in S2 (clear width) ...
    assert check_grounding(P09_THREE_SOURCES, list(sources.values())).verified
    # ... the sentence stating the height cites only S1, which has neither.
    result = assess_grounding(P09_THREE_SOURCES, sources)
    assert result.status == NEEDS_REVIEW
    assert [(f.raw, f.cited_labels, f.found_in) for f in result.flagged] == [
        ("36 inches", ["S1"], ["S2"]), ("915 mm", ["S1"], ["S2"]),
    ]


def test_number_in_an_uncited_prompt_source_needs_review():
    # m02 / h08: right number, but the source that has it is not cited.
    answer = "The rise limit is 6 inches [S1]."
    result = assess_grounding(answer, {"S1": SOURCE_505_4, "S2": SOURCE_405_8})
    assert result.status == NEEDS_REVIEW and result.flagged[0].found_in == ["S2"]


def test_number_in_no_source_is_refused():
    answer = "Handrails are 42 inches high [S1]."
    result = assess_grounding(answer, {"S1": SOURCE_505_4, "S2": SOURCE_405_8})
    assert result.status == REFUSED and result.flagged[0].found_in == []


def test_one_invented_number_refuses_even_if_others_only_need_review():
    answer = "Handrails are 36 inches [S1] and 42 inches [S1]."
    result = assess_grounding(answer, {"S1": SOURCE_505_4, "S2": SOURCE_405_8})
    assert result.status == REFUSED
    assert [(f.raw, f.found_in) for f in result.flagged] == [("36 inches", ["S2"]), ("42 inches", [])]


def test_sentence_without_citation_is_checked_against_all_cited_sources():
    answer = "Handrails are needed on ramps [S2]. The rise limit is 6 inches."
    assert assess_grounding(answer, {"S1": SOURCE_505_4, "S2": SOURCE_405_8}).status == VERIFIED


def test_invalid_labels_ground_nothing_but_the_number_can_still_be_found():
    result = assess_grounding("Handrails are 34 inches minimum [S7].", {"S1": SOURCE_505_4})
    assert result.status == NEEDS_REVIEW and result.flagged[0].found_in == ["S1"]


def test_answer_without_numbers_is_verified():
    assert assess_grounding("Grab bars go on the side and rear walls [S1].", {"S1": SOURCE_505_4}).status == VERIFIED


def test_p03_cross_reference_is_not_a_quantity():
    answer = "According to [S2], a turning space shall comply with 304, with slopes not steeper than 1:48 [S2]."
    source = "304.2 Floor or Ground Surfaces. Floor or ground surfaces of a turning space shall comply with 302. Changes in level are not permitted. EXCEPTION: Slopes not steeper than 1:48 shall be permitted."
    assert assess_grounding(answer, {"S2": source}).status == VERIFIED


def test_e03_date_attributed_to_the_wrong_source_needs_review():
    sources = {"S1": "Section 35.151(b) Alterations. The 1991 title II regulation does not contain ...",
               "S2": "(b) Alterations. ... if the alteration was commenced after January 26, 1992."}
    answer = "Altered portions must be accessible if the alteration was commenced after January 26, 1992 [S1]. See also [S2]."
    result = assess_grounding(answer, sources)
    assert result.status == NEEDS_REVIEW
    assert [(f.raw, f.found_in) for f in result.flagged] == [("26", ["S2"])]  # the year is skipped, the day is not


# --- in the chain -------------------------------------------------------------

HANDRAIL = make_context("505.4", SOURCE_505_4, pages="158")
RAMP_HANDRAILS = make_context("405.8", SOURCE_405_8, pages="133")


def test_chain_verified_answer():
    reply = "Handrails shall be 34 inches (865 mm) minimum and 38 inches (965 mm) maximum [S1]."
    result = answer_from_contexts("q", [HANDRAIL], FakeLLM(reply), 6000)
    assert not result.refused and result.grounding_status == "verified" and result.flagged_numbers == []


def test_chain_needs_review_answers_with_flags():
    reply = "According to [S1], handrails shall be 36 inches (915 mm) minimum."
    result = answer_from_contexts("How high are ramp handrails?", [HANDRAIL, RAMP_HANDRAILS], FakeLLM(reply), 6000)
    assert not result.refused and result.grounding_status == "needs_review"
    assert [(f.raw, f.found_in) for f in result.flagged_numbers] == [("36 inches", ["S2"]), ("915 mm", ["S2"])]
    assert result.text == reply


def test_chain_refuses_a_number_in_no_source():
    reply = "According to [S1], handrails shall be 42 inches minimum."
    result = answer_from_contexts("q", [HANDRAIL, RAMP_HANDRAILS], FakeLLM(reply), 6000)
    assert result.refused and result.refusal_reason == "ungrounded_number"
    assert result.grounding_status == "refused" and result.raw_llm_text == reply
    assert [c.section_id for c in result.citations] == ["505.4"]  # kept for debugging


def test_chain_earlier_refusals_have_no_grounding_status():
    result = answer_from_contexts("q", [HANDRAIL], FakeLLM("NOT_IN_SOURCES"), 6000)
    assert result.refused and result.grounding_status is None
