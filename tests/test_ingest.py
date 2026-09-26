import hashlib
import json
import logging
import pickle
from dataclasses import replace

import chromadb
import pytest

from src.config import CorpusFile, load_settings
from src.ingest import (
    Page,
    Section,
    ancestor_ids,
    build_parents,
    child_index_text,
    clean_pages,
    detect_printed_page,
    edge_line_indexes,
    find_repeated_edge_lines,
    fix_hyphenation,
    is_contents_page,
    is_next_paragraph_letter,
    is_stoplisted,
    make_breadcrumb,
    make_child_id,
    make_child_splitter,
    make_parent_id,
    mask_digits,
    match_heading,
    paragraph_title,
    remove_headers_footers,
    section_embedding_text,
    split_children,
    split_sections,
    write_stores,
)

CORPUS_FILE = CorpusFile(code_name="ADA 2010 Standards", edition_year=2010)


# --- cleaning ---------------------------------------------------------------

def test_mask_digits():
    assert mask_digits("2010 Standards: Titles II and III - 117") == "# Standards: Titles II and III - #"


def test_edge_line_indexes_short_and_long_pages():
    assert edge_line_indexes(2) == {0, 1}
    assert edge_line_indexes(10) == {0, 1, 2, 7, 8, 9}


def make_pages(count: int, body_line: str) -> list[Page]:
    """Pages with a header, a footer carrying the page number, and a body line in the middle.

    Filler lines differ per page ("page one a", "page two a", ...) so they never repeat.
    """
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    return [
        Page(
            number=n,
            lines=["TECHNICAL CHAPTER 4", *[f"page {words[n]} {x}" for x in "abc"], body_line,
                   *[f"page {words[n]} {x}" for x in "def"],
                   f"{n + 100} - 2010 Standards: Titles II and III Department of Justice"],
        )
        for n in range(1, count + 1)
    ]


def test_repeated_edge_lines_found_but_middle_lines_ignored():
    repeated = find_repeated_edge_lines(make_pages(10, "EXCEPTION:"))
    assert "TECHNICAL CHAPTER #" in repeated
    assert "# - # Standards: Titles II and III Department of Justice" in repeated
    assert "EXCEPTION:" not in repeated  # repeated on every page, but never at an edge


def test_remove_headers_footers_keeps_body_and_records_printed_page():
    cleaned = remove_headers_footers(make_pages(10, "EXCEPTION:"))
    assert cleaned[0].lines == ["page one a", "page one b", "page one c", "EXCEPTION:",
                                "page one d", "page one e", "page one f"]
    assert cleaned[0].printed_number == "101"


def test_remove_headers_footers_never_drops_a_heading():
    pages = [Page(n, ["403.1 General.", "body", "more", "text", "end", "x", "y"]) for n in range(1, 11)]
    assert all(page.lines[0] == "403.1 General." for page in remove_headers_footers(pages))


@pytest.mark.parametrize("line", [
    "TECHNICAL CHAPTER 4: ACCESSIBLE ROUTES",
    "CHAPTER 4: ACCESSIBLE ROUTES TECHNICAL",
    "ADA CHAPTER 2: SCOPING REQUIREMENTS AMERICANS WITH DISABILITIES ACT: SCOPING",
    "AMERICANS WITH DISABILITIES ACT: SCOPING ADA CHAPTER 2: SCOPING REQUIREMENTS",
    "TECHNICAL CHAPTER 8: SPECIAL ROOMS, SPACES, AND ELEMENTS",
    "Department of Justice 2010 Standards: Title III - 21",
    "26 - 2010 Standards: Title III Department of Justice",
    "Department of Justice 2010 Standards: Title II  and III- 12",
    "Guidance on 2010 Standards: Title II - 5Department of Justice",
    "Subpart D of 28 CFR Part 36",
    "Section 35.151 of 28 CFR Part 35",
])
def test_stoplist_matches_furniture(line):
    assert is_stoplisted(line)


@pytest.mark.parametrize("line", [
    "The provisions of Chapter 4 shall apply where required by Chapter 2",
    "EXCEPTION:",
    "Section 213.3.1 of the 2010 Standards requires",
    "compliance with the 2010 Standards: Title III entities must",
    "403 Walking Surfaces",
])
def test_stoplist_ignores_body_lines(line):
    assert not is_stoplisted(line)


def test_stoplisted_line_removed_at_edge_but_kept_in_middle():
    page = Page(1, ["TECHNICAL CHAPTER 4: ACCESSIBLE ROUTES", "a", "b", "c",
                    "TECHNICAL CHAPTER 4: ACCESSIBLE ROUTES", "d", "e", "f",
                    "Department of Justice 2010 Standards: Title III - 21"])
    [cleaned] = remove_headers_footers([page])
    assert cleaned.lines == ["a", "b", "c", "TECHNICAL CHAPTER 4: ACCESSIBLE ROUTES", "d", "e", "f"]
    assert cleaned.printed_number == "21"


@pytest.mark.parametrize("lines, expected", [
    (["116 - 2010 Standards: Titles II and III Department of Justice"], "116"),
    (["2010 Standards: Titles II and III - 117Department of Justice"], "117"),
    (["Guidance on the 2010 Standards:  Titles II and III - 93"], "93"),
    (["Titles II and III - 2010 Standards - 87"], None),  # inner numbering, not the DOJ footer
    (["Department of Justice"], None),
    ([], None),
])
def test_detect_printed_page(lines, expected):
    assert detect_printed_page(lines) == expected


def test_fix_hyphenation_joins_words_broken_at_line_end():
    assert fix_hyphenation("resi-\ndential units") == "residential units"


def test_fix_hyphenation_joins_hyphen_on_its_own_line():
    assert fix_hyphenation("tran\n-\nsient lodging") == "transient lodging"


def test_fix_hyphenation_leaves_inline_hyphens_and_dashes():
    text = "multi-user rooms\naccessibility-\n-the two"
    assert fix_hyphenation(text) == text


def test_is_contents_page():
    contents = Page(21, ["§ 36.402 Alterations.", "(a) General..………....21", "(b) Alteration……...21", "(c) Feasible…..22"])
    body = Page(25, ["§ 36.402 Alterations.", "(a) General.", "Any alteration ... shall be made"])
    assert is_contents_page(contents)
    assert not is_contents_page(body)


def test_is_contents_page_without_dot_leaders():
    contents = Page(36, ["101 Purpose 5", "102 Dimensions for Adults and Children 5", "103 Equivalent Facilitation 5",
                         "104 Conventions 5", "105 Referenced Standards 6", "106 Definitions 8"])
    one_entry = Page(90, ["221 Assembly Areas 46", "Assembly areas shall provide wheelchair spaces"])
    assert is_contents_page(contents)
    assert not is_contents_page(one_entry)


def test_clean_pages_drops_contents_pages_so_real_section_gets_occurrence_1():
    pages = [
        Page(21, ["§ 36.402 Alterations.", "(a) General..………....21", "(b) Alteration……...21", "(c) Feasible…..22"]),
        Page(25, ["§ 36.402 Alterations.", "Any alteration shall be made"]),
    ]
    [section] = split_sections(clean_pages(pages), "s.pdf")
    assert (section.section_id, section.occurrence, section.pages) == ("36.402", 1, [25])


def test_clean_pages_removes_furniture_then_fixes_hyphens():
    pages = make_pages(10, "resi-")
    pages[0].lines.insert(5, "dential")
    assert "residential" in clean_pages(pages)[0].lines


# --- headings ---------------------------------------------------------------

@pytest.mark.parametrize("line, expected", [
    ("404.2.3 Clear Width.  Door openings shall provide a clear width", ("404.2.3", "Clear Width", "code")),
    ("604.5 Grab Bars.  Grab bars for water closets shall comply with 609.", ("604.5", "Grab Bars", "code")),
    ("§ 35.151 New construction and alterations.", ("35.151", "New construction and alterations", "regulation")),
    ("Section 36.406(f) Assembly Areas", ("36.406(f)", "Assembly Areas", "code")),
    ("213, 603, 604, and 608 Toilet and Bathing", ("213, 603, 604, 608", "Toilet and Bathing", "code")),
    ("205 and 309 Operable Parts", ("205, 309", "Operable Parts", "code")),
    ("403 Walking Surfaces", ("403", "Walking Surfaces", "code")),
    ("1003.2.1 Boat Slips.", ("1003.2.1", "Boat Slips", "code")),
    ("Section 35.151(k)", ("35.151(k)", "", "code")),
])
def test_match_heading_accepts(line, expected):
    heading = match_heading(line)
    assert (heading.section_id, heading.section_title, heading.section_type) == expected


def test_match_heading_keeps_body_text_of_numbered_heading():
    assert match_heading("309.4 Operation.   Operable parts shall be operable").body == "Operable parts shall be operable"
    assert match_heading("403.1 General.").body == ""


@pytest.mark.parametrize("line", [
    "Advisory 402.2 Components.  Walking surfaces must have running slopes",
    "Advisory 103 Equivalent Facilitation.",
    "Section 802.1.5 of the 2010 Standards",
    "Section 35.151 -- Title II Regulation",
    "§ 36.404 Alterations: Elevator exemption.……………..……….......25",
    "221 Assembly Areas 46",   # table of contents entry with page number
    "402.2 and 403.",          # body text wrapped after "comply with"
    "Figure 308.3.2",
    "1991 Standards require",  # a year, not a section number
    "§ 36.104 are considered sales or rental establishments.",
    "walking surfaces shall comply with 403.",
])
def test_match_heading_rejects(line):
    assert match_heading(line) is None


# --- regulation paragraphs --------------------------------------------------

@pytest.mark.parametrize("letter, last_letter, expected", [
    ("a", None, True),
    ("b", "a", True),
    ("d", "b", True),    # "(c)" was lost by the PDF extractor: skipping is allowed
    ("b", "c", False),   # going backwards: a wrapped line "(b) that were ..."
    ("i", "a", False),   # roman numeral sub-paragraph inside (a)
    ("i", "h", True),    # real paragraph (i) right after (h)
    ("v", "b", False),
])
def test_is_next_paragraph_letter(letter, last_letter, expected):
    assert is_next_paragraph_letter(letter, last_letter) is expected


@pytest.mark.parametrize("rest, expected", [
    ("Alterations.", "Alterations"),
    ("Scope of coverage.  The 1991 Standards and the 2010 Standards apply", "Scope of coverage"),
    ("This section does not require the installation of an elevator in an altered facility that is less", ""),
    (None, ""),
])
def test_paragraph_title(rest, expected):
    assert paragraph_title(rest) == expected


def test_regulation_splits_at_top_level_paragraphs_only():
    pages = [Page(10, [
        "§ 35.151 New construction and alterations.",
        "(a) Design and construction.",
        "(1) Each facility shall be designed",
        "(i) Full compliance with the requirements",    # roman sub-paragraph: stays in (a)
        "(b) Alterations.",
        "(1) Each facility that is altered",            # (b)(1): stays in (b)
        "(b) that were constructed or altered before",  # wrapped body text: stays in (b)
        "(c)",                                          # bare marker, title on next line
        "Accessibility standards and compliance date.",
    ])]
    sections = split_sections(pages, "s.pdf")
    assert [(s.section_id, s.section_title, s.section_type) for s in sections] == [
        ("35.151", "New construction and alterations", "regulation"),
        ("35.151(a)", "Design and construction", "regulation"),
        ("35.151(b)", "Alterations", "regulation"),
        ("35.151(c)", "", "regulation"),
    ]
    assert sections[1].lines[-1] == "(i) Full compliance with the requirements"
    assert sections[2].lines[-1] == "(b) that were constructed or altered before"


def test_paragraph_markers_outside_regulations_are_ordinary_text():
    pages = [Page(1, ["404.2 Doors.", "(a) General text", "(b) More text"])]
    [section] = split_sections(pages, "s.pdf")
    assert section.section_id == "404.2"
    assert len(section.lines) == 3


def test_a_new_heading_closes_the_regulation():
    pages = [Page(1, ["§ 36.402 Alterations.", "(a) General.", "101 Purpose", "(b) Not a paragraph"])]
    sections = split_sections(pages, "s.pdf")
    assert [s.section_id for s in sections] == ["36.402", "36.402(a)", "101"]


# --- section splitting ------------------------------------------------------

def test_split_sections_tags_front_matter():
    pages = [Page(1, ["Table of contents", "Introduction"]), Page(2, ["402.1 General.", "text"])]
    sections = split_sections(pages, "f.pdf")
    assert [(s.section_id, s.level) for s in sections] == [("front-matter", "front-matter"), ("402.1", "section")]


def test_split_sections_records_pages_and_printed_pages():
    pages = [Page(5, ["402.1 General.", "a"], "11"), Page(6, ["b", "402.2 Components.", "c"], "12")]
    first, second = split_sections(pages, "f.pdf")
    assert (first.pages, first.printed_pages, first.text) == ([5, 6], ["11", "12"], "402.1 General.\na\nb")
    assert (second.pages, second.text) == ([6], "402.2 Components.\nc")


def test_split_sections_merges_continuation_on_next_page(caplog):
    pages = [Page(1, ["221 Assembly Areas", "text one"]), Page(2, ["221 Assembly Areas", "text two"])]
    with caplog.at_level(logging.INFO, logger="inspectiq.ingest"):
        sections = split_sections(pages, "f.pdf")
    assert len(sections) == 1
    assert sections[0].text == "221 Assembly Areas\ntext one\ntext two"
    assert sections[0].pages == [1, 2]
    assert sections[0].continuations == 1
    assert "merged as continuation" in caplog.text


def test_split_sections_keeps_non_adjacent_repeat_separate_and_logs_it(caplog):
    pages = [Page(1, ["221 Assembly Areas", "a", "221.1 General.", "b", "221 Assembly Areas", "c"])]
    with caplog.at_level(logging.WARNING, logger="inspectiq.ingest"):
        sections = split_sections(pages, "f.pdf")
    assert [(s.section_id, s.occurrence) for s in sections] == [("221", 1), ("221.1", 1), ("221", 2)]
    assert "repeated section id 221 (occurrence 2)" in caplog.text


def test_split_sections_same_id_mid_page_is_not_a_continuation():
    pages = [Page(1, ["221 Assembly Areas", "a"]), Page(2, ["intro line", "221 Assembly Areas", "b"])]
    assert [s.occurrence for s in split_sections(pages, "f.pdf")] == [1, 2]


def test_split_sections_index_is_back_matter_without_heading_detection():
    pages = [
        Page(255, ["1010.1 Turning Space.", "A circular turning space"]),
        Page(256, ["INDEX TO THE 2010 STANDARDS", "Assembly Areas", "221 Assembly Areas"]),
    ]
    sections = split_sections(pages, "s.pdf")
    assert [(s.section_id, s.level) for s in sections] == [("1010.1", "section"), ("back-matter", "back-matter")]
    assert sections[0].text == "1010.1 Turning Space.\nA circular turning space"
    assert sections[1].pages == [256]


def test_split_sections_falls_back_to_one_section_per_page():
    pages = [Page(1, ["no headings here"]), Page(2, []), Page(3, ["nor here"])]
    sections = split_sections(pages, "f.pdf")
    assert [(s.section_id, s.pages, s.heading_only) for s in sections] == [
        ("page-1", [1], False), ("page-3", [3], False),
    ]


def test_heading_only_sections():
    pages = [Page(1, ["216 Signs", "216.1 General.", "Signs shall comply", "216.2 Designations.  Rooms shall"])]
    sections = split_sections(pages, "s.pdf")
    assert [(s.section_id, s.heading_only) for s in sections] == [
        ("216", True),     # only its heading line
        ("216.1", False),  # body on the following line
        ("216.2", False),  # body on the heading line itself
    ]


# --- IDs, breadcrumbs and parents -------------------------------------------

def test_parent_id_is_sha1_of_source_and_section_id():
    expected = hashlib.sha1(b"ada_2010_standards.pdf|404.2.3").hexdigest()
    assert make_parent_id("ada_2010_standards.pdf", "404.2.3") == expected


def test_repeated_occurrence_gets_its_own_parent_id():
    expected = hashlib.sha1(b"f.pdf|221|2").hexdigest()
    assert make_parent_id("f.pdf", "221", 2) == expected
    assert make_parent_id("f.pdf", "221", 2) != make_parent_id("f.pdf", "221")


def test_child_id_is_deterministic_and_index_dependent():
    assert make_child_id("abc", 0) == hashlib.sha1(b"abc|0").hexdigest()
    assert make_child_id("abc", 0) != make_child_id("abc", 1)


@pytest.mark.parametrize("section_id, expected", [
    ("216.2.1", ["216", "216.2"]),
    ("216", []),
    ("35.151(b)", ["35", "35.151"]),
    ("213, 603, 604, 608", []),
])
def test_ancestor_ids(section_id, expected):
    assert ancestor_ids(section_id) == expected


def test_make_breadcrumb():
    titles = {"216": "Signs", "216.2": "Designations"}
    section = Section("216.2", "Designations")
    assert make_breadcrumb("ADA 2010 Standards", section, titles) == "ADA 2010 Standards > 216 Signs > 216.2 Designations"


def test_make_breadcrumb_skips_missing_ancestors_and_empty_titles():
    titles = {"35.151": "New construction and alterations"}
    assert make_breadcrumb("ADA 2010 Standards", Section("35.151(k)", ""), titles) == (
        "ADA 2010 Standards > 35.151 New construction and alterations > 35.151(k)"
    )


def test_make_breadcrumb_for_front_matter():
    section = Section("front-matter", "Front matter", level="front-matter")
    assert make_breadcrumb("ADA 2010 Guidance", section, {}) == "ADA 2010 Guidance > Front matter"


def test_build_parents_metadata():
    pages = [Page(7, ["404 Doors, Doorways, and Gates", "404.2.3 Clear Width.", "32 inches"], "119")]
    parents = build_parents(split_sections(pages, "s.pdf"), "s.pdf", CORPUS_FILE)
    assert parents[1]["metadata"] == {
        "parent_id": make_parent_id("s.pdf", "404.2.3"),
        "source": "s.pdf",
        "code_name": "ADA 2010 Standards",
        "edition_year": 2010,
        "section_id": "404.2.3",
        "section_title": "Clear Width",
        "section_type": "code",
        "breadcrumb": "ADA 2010 Standards > 404 Doors, Doorways, and Gates > 404.2.3 Clear Width",
        "pages": "7",
        "page_start": 7,
        "page_end": 7,
        "printed_pages": "119",
        "occurrence": 1,
        "level": "section",
        "heading_only": False,
    }
    assert parents[0]["metadata"]["heading_only"] is True


# --- Part 2: children and stores --------------------------------------------

def make_parents(texts: dict[str, str]) -> list[dict]:
    pages = [Page(i + 1, [f"{sid} Title.", *text.split("\n")]) for i, (sid, text) in enumerate(texts.items())]
    return build_parents(split_sections(pages, "s.pdf"), "s.pdf", CORPUS_FILE)


def test_section_embedding_text_is_breadcrumb_plus_prefix():
    [parent] = make_parents({"402.1": "x" * 3000})
    text = section_embedding_text(parent, prefix_chars=1500)
    assert text.startswith("ADA 2010 Standards > 402.1 Title\n402.1 Title.")
    assert len(text) == len("ADA 2010 Standards > 402.1 Title\n") + 1500


def test_children_never_cross_section_boundaries():
    parents = make_parents({"402.1": "alpha " * 200, "402.2": "beta " * 200})
    splitter = make_child_splitter(chunk_size=400, chunk_overlap=60)
    for parent in parents:
        children = split_children(parent, splitter)
        assert len(children) > 1
        for index, child in enumerate(children):
            start = child["metadata"]["start_char"]
            assert parent["text"][start:start + len(child["text"])] == child["text"]
            assert len(child["text"]) <= 400
            assert child["parent_id"] == parent["parent_id"]
            assert child["child_id"] == make_child_id(parent["parent_id"], index)
            assert child["metadata"]["level"] == "child"
            assert child["metadata"]["section_id"] == parent["metadata"]["section_id"]


def test_child_index_text_prepends_breadcrumb():
    [parent] = make_parents({"402.1": "alpha"})
    [child] = split_children(parent, make_child_splitter(400, 60))
    assert child["text"] == "402.1 Title.\nalpha"
    assert child_index_text(child) == "ADA 2010 Standards > 402.1 Title\n402.1 Title.\nalpha"


def test_front_matter_children_keep_section_level():
    pages = [Page(1, ["Preface text"]), Page(2, ["402.1 General.", "body"])]
    parents = build_parents(split_sections(pages, "s.pdf"), "s.pdf", CORPUS_FILE)
    children = split_children(parents[0], make_child_splitter(400, 60))
    assert children[0]["metadata"]["section_level"] == "front-matter"


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic stand-in for the embedding model (no download, no network)."""
    return [[float(len(t) % 7), float(len(t) % 11), 1.0] for t in texts]


@pytest.fixture
def tmp_settings(tmp_path):
    return replace(
        load_settings(),
        store_dir=tmp_path,
        chroma_dir=tmp_path / "chroma",
        bm25_path=tmp_path / "bm25.pkl",
        docstore_path=tmp_path / "docstore.json",
    )


def test_reingestion_does_not_duplicate_and_removes_stale_entries(tmp_settings):
    client = chromadb.PersistentClient(path=str(tmp_settings.chroma_dir))
    parents = make_parents({"402.1": "alpha " * 200, "402.2": "beta " * 200, "403.1": "gamma"})

    first = write_stores(parents, tmp_settings, fake_embed, client)
    second = write_stores(parents, tmp_settings, fake_embed, client)
    assert first == second
    assert first["sections"] == first["docstore"] == 3
    assert first["children"] == first["bm25"] > 3

    # Drop one parent (e.g. a heading rule changed): its section and children must disappear.
    third = write_stores(parents[:2], tmp_settings, fake_embed, client)
    assert third["sections"] == 2
    assert third["children"] == first["children"] - 1

    with tmp_settings.bm25_path.open("rb") as f:
        index = pickle.load(f)
    assert len(index["child_ids"]) == third["bm25"]
    assert index["bm25"].get_scores(["alpha"]).max() > 0


def test_heading_only_parents_go_to_docstore_only(tmp_settings):
    client = chromadb.PersistentClient(path=str(tmp_settings.chroma_dir))
    pages = [Page(1, ["216 Signs", "216.2 Designations.  Interior and exterior signs"])]
    parents = build_parents(split_sections(pages, "s.pdf"), "s.pdf", CORPUS_FILE)

    counts = write_stores(parents, tmp_settings, fake_embed, client)
    assert counts == {"sections": 1, "children": 1, "bm25": 1, "docstore": 2}

    # The chapter heading lives on in the child's breadcrumb, which BM25 indexes.
    with tmp_settings.bm25_path.open("rb") as f:
        index = pickle.load(f)
    assert index["metadatas"][0]["breadcrumb"] == "ADA 2010 Standards > 216 Signs > 216.2 Designations"
    assert "signs" in index["bm25"].doc_freqs[0]  # term counts of the indexed (breadcrumb + chunk) text
    assert len(json.loads(tmp_settings.docstore_path.read_text())) == 2
