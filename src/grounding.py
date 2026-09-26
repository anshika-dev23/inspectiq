"""Number grounding (step 3b): every number in an answer must appear in a source the answer cites.

Catches the most dangerous failure of the answer chain: a confident, cited, WRONG number
(eval p09: "36 inches minimum" citing 505.4, which says 34). Citation checking cannot catch that,
because the cited section is the right one.

Grounding is per sentence: a number must appear in a source cited IN ITS SENTENCE. Checking against all
cited sources is too weak: in p09 the misquoted "36 inch (915 mm)" is the clear width in 405.8 (also cited,
in another sentence), while the sentence stating the handrail height cites only 505.4. A sentence without a
citation of its own is checked against every cited source; a citation standing alone after a sentence
("... minimum. [S1]") belongs to that sentence.

Numbers are compared as (value, unit): "34 inches" = "34 in" = "34-inch"; "½" = "1/2" = 0.5;
"1 1/4" = 1.25; ratios such as "1:12" are one value. An answer number without a unit matches a
source number with any unit; an answer number with a unit matches the same unit, or a unitless
source number (PDF tables often lose their units).
"""
import re
from dataclasses import dataclass
from fractions import Fraction

UNICODE_FRACTIONS = {"¼": Fraction(1, 4), "½": Fraction(1, 2), "¾": Fraction(3, 4), "⅛": Fraction(1, 8),
                     "⅜": Fraction(3, 8), "⅝": Fraction(5, 8), "⅞": Fraction(7, 8), "⅓": Fraction(1, 3)}
UNIT_ALIASES = {
    "in": "in", "in.": "in", "inch": "in", "inches": "in", '"': "in", "″": "in",
    "ft": "ft", "feet": "ft", "foot": "ft", "'": "ft", "′": "ft",
    "mm": "mm", "m": "m", "cm": "cm",
    "lb": "lb", "lbs": "lb", "lbf": "lb", "pound": "lb", "pounds": "lb",
    "n": "N", "%": "%", "percent": "%",
    "degrees": "deg", "degree": "deg", "°": "deg",
}

_FRACTION_CHARS = "".join(UNICODE_FRACTIONS)
# One number: "1:12" | "1 1/4" | "1 ¼" | "3/8" | "½" | "1,000" | "22.2" | "34"
NUMBER = (
    r"(?P<ratio>\d+\s*:\s*\d+)"
    r"|(?P<mixed>\d+\s+\d+/\d+)"
    rf"|(?P<mixed_uni>\d+\s*[{_FRACTION_CHARS}])"
    r"|(?P<fraction>\d+/\d+)"
    rf"|(?P<uni>[{_FRACTION_CHARS}])"
    r"|(?P<plain>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
)
UNIT = r"(?:\s*-?\s*(?P<unit>inches|inch|in\.|in\b|feet|foot|ft\b|mm\b|cm\b|m\b|lbf|lbs|lb\b|pounds|pound|percent|degrees|degree|N\b|%|°|\"|″|'|′))?"
MENTION = re.compile(rf"(?<![\w.:/])(?:{NUMBER}){UNIT}", re.IGNORECASE)

# Not quantities, removed before extraction: citation labels and brackets, section references, CFR / USC refs.
NOT_QUANTITIES = [
    re.compile(r"\[[^\]]*\]"),                                                   # [S1], [file p.12 §404.2.3]
    re.compile(r"\b\d+\s+(?:CFR|U\.S\.C\.)\s+(?:part\s+)?\d+(?:\.\d+)?(?:\([a-z0-9]+\))*", re.IGNORECASE),  # before § refs
    re.compile(r"§+\s*\d+(?:\.\d+)*(?:\([a-z0-9]+\))*", re.IGNORECASE),           # § 35.151(b)
    re.compile(r"\b(?:sections?|sec\.)\s+\d+(?:\.\d+)*(?:\([a-z0-9]+\))*(?:\s*(?:,|and|or)\s*\d+(?:\.\d+)*(?:\([a-z0-9]+\))*)*", re.IGNORECASE),
    re.compile(r"\b\d{3,4}(?:\.\d+)+(?:\([a-z0-9]+\))*"),                       # 404.2.3, 604.5, 35.151 w/ dots
    re.compile(r"\b\d{2}\.\d{3}(?:\([a-z0-9]+\))*"),                            # 35.151, 36.406(f)
    re.compile(r"\b(?:figure|table|chapter|title|plan|exception)\s+[\d.]+[a-z]?", re.IGNORECASE),
    # cross-references: "shall comply with 304", "complying with 302 and 303", "specified in 308"
    re.compile(r"\b(?:compl(?:y|ies|ying)\s+with|in\s+accordance\s+with|specified\s+in|required\s+by|see)\s+"
               r"\d+(?:\.\d+)*(?:\([a-z0-9]+\))*(?:\s*(?:,|and|or)\s*\d+(?:\.\d+)*(?:\([a-z0-9]+\))*)*", re.IGNORECASE),
    re.compile(r"(?m)^\s*\d+[.)]\s"),                                           # list enumerators "1. " at line start
]
YEAR = re.compile(r"^(?:19|20)\d{2}$")

# Citations in an answer: "[S1]", "[S1, S2]", "[S1; S3]". Also used by src/answer.py.
CITATION_GROUP = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
LABEL = re.compile(r"S\d+")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class NumberMention:
    raw: str             # as written: "34 inches"
    value: str           # normalized: "34", "0.5", "1:12"
    unit: str | None     # normalized: "in", "mm", "lb", "%", ...

    @property
    def key(self) -> str:
        return f"{self.value} {self.unit}" if self.unit else self.value


@dataclass(frozen=True)
class GroundingResult:
    numbers: list[NumberMention]       # every number found in the answer
    ungrounded: list[NumberMention]    # those not found in any cited source

    @property
    def verified(self) -> bool:
        return not self.ungrounded


def normalize_value(match: re.Match) -> str:
    """The numeric value as a canonical string: "1 1/4" -> "1.25", "½" -> "0.5", "1,000" -> "1000", "1 : 12" -> "1:12"."""
    if match["ratio"]:
        left, right = re.split(r"\s*:\s*", match["ratio"])
        return f"{int(left)}:{int(right)}"
    if match["mixed"]:
        whole, fraction = match["mixed"].split()
        value = int(whole) + Fraction(fraction)
    elif match["mixed_uni"]:
        whole, char = match["mixed_uni"][:-1].strip(), match["mixed_uni"][-1]
        value = int(whole) + UNICODE_FRACTIONS[char]
    elif match["fraction"]:
        value = Fraction(match["fraction"])
    elif match["uni"]:
        value = UNICODE_FRACTIONS[match["uni"]]
    else:
        value = Fraction(match["plain"].replace(",", ""))
    return str(value.numerator) if value.denominator == 1 else f"{float(value):g}"


def strip_non_quantities(text: str) -> str:
    for pattern in NOT_QUANTITIES:
        text = pattern.sub(" ", text)
    return text


def extract_numbers(text: str) -> list[NumberMention]:
    """Every quantity in the text, in order. Section refs, citation labels, list numbers and bare years are skipped."""
    mentions = []
    for match in MENTION.finditer(strip_non_quantities(text)):
        unit = UNIT_ALIASES.get((match["unit"] or "").lower()) if match["unit"] else None
        value = normalize_value(match)
        if unit is None and YEAR.match(value):
            continue  # "the 2010 Standards", "the 1991 Standards"
        mentions.append(NumberMention(raw=match.group(0).strip(), value=value, unit=unit))
    return mentions


def is_grounded(number: NumberMention, source_numbers: list[NumberMention]) -> bool:
    for candidate in source_numbers:
        if candidate.value != number.value:
            continue
        if number.unit is None or candidate.unit is None or candidate.unit == number.unit:
            return True
    return False


def check_grounding(answer_text: str, cited_source_texts: list[str]) -> GroundingResult:
    """Every number in the answer must appear in at least one of the cited sources' texts."""
    numbers = extract_numbers(answer_text)
    source_numbers = [n for text in cited_source_texts for n in extract_numbers(text)]
    ungrounded = [n for n in numbers if not is_grounded(n, source_numbers)]
    return GroundingResult(numbers=numbers, ungrounded=ungrounded)


def cited_labels(text: str) -> list[str]:
    """Labels cited in the text, in order, without duplicates."""
    labels: list[str] = []
    for group in CITATION_GROUP.finditer(text):
        for label in LABEL.findall(group[1]):
            if label not in labels:
                labels.append(label)
    return labels


def split_claims(text: str) -> list[tuple[str, list[str]]]:
    """(sentence, labels cited in it). A piece that is only citations ("[S1].") joins the sentence before it."""
    claims: list[tuple[str, list[str]]] = []
    for piece in (p.strip() for p in SENTENCE_BREAK.split(text)):
        if not piece:
            continue
        labels = cited_labels(piece)
        only_citations = not CITATION_GROUP.sub("", piece).strip(" .;,")
        if only_citations and claims:
            sentence, previous = claims[-1]
            claims[-1] = (f"{sentence} {piece}", previous + [label for label in labels if label not in previous])
        else:
            claims.append((piece, labels))
    return claims


def check_grounding_by_claim(answer_text: str, source_texts: dict[str, str]) -> GroundingResult:
    """Per sentence: its numbers must appear in the sources it cites (all cited sources if it cites none).

    source_texts maps each valid label ("S1") to the source text the model saw. Labels that are not in it
    (invalid citations) ground nothing.
    """
    all_cited = [label for label in cited_labels(answer_text) if label in source_texts]
    numbers: list[NumberMention] = []
    ungrounded: list[NumberMention] = []
    for sentence, labels in split_claims(answer_text):
        scope = [label for label in labels if label in source_texts] or all_cited
        result = check_grounding(sentence, [source_texts[label] for label in scope])
        numbers += result.numbers
        ungrounded += result.ungrounded
    return GroundingResult(numbers=numbers, ungrounded=ungrounded)
