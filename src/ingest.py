"""Step 1 — ingestion.

Part 1: files -> parent sections (load page by page, clean, split on code headings).
Part 2: parent sections -> four stores:
    Chroma "sections"  (breadcrumb + opening text, coarse vector search)
    Chroma "children"  (400-char chunks inside each section, fine vector search)
    bm25.pkl           (same children, keyword search)
    docstore.json      (parent_id -> full section text + metadata, lookup only)

Run:  .venv/bin/python -m src.ingest
"""
import hashlib
import json
import logging
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from src.config import CORPUS, FURNITURE_STOPLIST, CorpusFile, Settings, load_settings
from src.text import tokenize

logger = logging.getLogger("inspectiq.ingest")

# pypdf warns about fonts it cannot fully parse; the extracted text is still fine.
logging.getLogger("pypdf").setLevel(logging.ERROR)

EmbedFn = Callable[[list[str]], list[list[float]]]

# Header/footer detection only looks at the first and last EDGE_LINES lines of each page.
EDGE_LINES = 3
# An edge line is page furniture if (digits masked) it sits at the edge of >= 10% of pages.
# Measured on the ADA corpus: 10% catches footers and running headers; 5% also ate body lists.
REPEAT_FRACTION = 0.10
CHROMA_BATCH_SIZE = 500


# =============================================================================
# Part 1a — load files page by page
# =============================================================================

@dataclass
class Page:
    number: int                        # 1-based PDF page index, used for citations "p.X"
    lines: list[str]
    printed_number: str | None = None  # page number printed in the footer, if detected


def split_lines(text: str) -> list[str]:
    """Split text into stripped, non-empty lines."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_pdf_pages(path: Path) -> list[Page]:
    reader = PdfReader(path)
    return [
        Page(number=number, lines=split_lines(pdf_page.extract_text() or ""))
        for number, pdf_page in enumerate(reader.pages, start=1)
    ]


def load_markdown_pages(path: Path) -> list[Page]:
    """Markdown has no pages, so the whole file is page 1."""
    return [Page(number=1, lines=split_lines(path.read_text(encoding="utf-8")))]


def load_pages(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_pages(path)
    if suffix in (".md", ".markdown"):
        return load_markdown_pages(path)
    raise ValueError(f"Unsupported file type: {path}")


# =============================================================================
# Part 1b — clean: drop headers/footers and contents pages, join hyphenated words
# =============================================================================

PRINTED_PAGE_PATTERN = re.compile(
    r"^(\d{1,4}) - "                         # "116 - 2010 Standards: Titles II and III ..."
    r"|- ?(\d{1,4}) ?Department of Justice"  # "... Titles II and III - 117Department of Justice"
    r"|- ?(\d{1,4})$"                        # "Guidance on the 2010 Standards: Titles II and III - 93"
)
STOPLIST_PATTERNS = [re.compile(pattern) for pattern in FURNITURE_STOPLIST]
HYPHEN_AT_LINE_END = re.compile(r"(\w)-\n(\w)")   # "resi-\ndential"  -> "residential"
HYPHEN_ON_OWN_LINE = re.compile(r"(\w)\n-\n(\w)")  # "tran\n-\nsient" -> "transient"
DOT_LEADER = re.compile(r"…|\.{4,}")
MIN_DOT_LEADER_LINES = 3
# Contents entry without dot leaders: "221 Assembly Areas 46"
NUMBERED_CONTENTS_ENTRY = re.compile(r"^(?:10\d{2}|[1-9]\d{2})(?:\.\d+)* [A-Z][^.…]*? \d{1,3}$")
MIN_NUMBERED_CONTENTS_ENTRIES = 5


def mask_digits(line: str) -> str:
    """'2010 Standards - 117' -> '# Standards - #', so footers with changing numbers compare equal."""
    return re.sub(r"\d+", "#", line)


def edge_line_indexes(line_count: int) -> set[int]:
    """Indexes of the first and last EDGE_LINES lines of a page."""
    first = range(min(EDGE_LINES, line_count))
    last = range(max(0, line_count - EDGE_LINES), line_count)
    return set(first) | set(last)


def find_repeated_edge_lines(pages: list[Page]) -> set[str]:
    """Masked edge lines that appear on at least REPEAT_FRACTION of the pages."""
    pages_containing = Counter()
    for page in pages:
        pages_containing.update({mask_digits(page.lines[i]) for i in edge_line_indexes(len(page.lines))})
    min_pages = max(2, REPEAT_FRACTION * len(pages))
    return {line for line, count in pages_containing.items() if count >= min_pages}


def is_stoplisted(line: str) -> bool:
    """Known furniture from config.FURNITURE_STOPLIST."""
    return any(pattern.search(line) for pattern in STOPLIST_PATTERNS)


def detect_printed_page(furniture_lines: list[str]) -> str | None:
    """Best effort: read the printed page number from the DOJ footer lines, else None."""
    for line in furniture_lines:
        if "Department of Justice" not in line and "Guidance" not in line:
            continue
        match = PRINTED_PAGE_PATTERN.search(line)
        if match:
            return next(group for group in match.groups() if group)
    return None


def remove_headers_footers(pages: list[Page]) -> list[Page]:
    """Drop repeated or stoplisted edge lines (never a heading); keep the printed page number they carried."""
    repeated = find_repeated_edge_lines(pages)
    cleaned = []
    for page in pages:
        edges = edge_line_indexes(len(page.lines))
        kept, furniture = [], []
        for i, line in enumerate(page.lines):
            looks_like_furniture = mask_digits(line) in repeated or is_stoplisted(line)
            is_furniture = i in edges and looks_like_furniture and match_heading(line) is None
            (furniture if is_furniture else kept).append(line)
        cleaned.append(Page(page.number, kept, detect_printed_page(furniture)))
    return cleaned


def is_contents_page(page: Page) -> bool:
    """A table-of-contents page has several dot-leader lines ("(a) General..……....21")
    or several numbered entries ending in a page number ("221 Assembly Areas 46")."""
    dot_leader_lines = sum(1 for line in page.lines if DOT_LEADER.search(line))
    numbered_entries = sum(1 for line in page.lines if NUMBERED_CONTENTS_ENTRY.match(line))
    return dot_leader_lines >= MIN_DOT_LEADER_LINES or numbered_entries >= MIN_NUMBERED_CONTENTS_ENTRIES


def fix_hyphenation(text: str) -> str:
    """Join words broken across lines by a hyphen.

    Trade-off: a real compound split at a line end ("multi-\\nuser") becomes "multiuser".
    """
    text = HYPHEN_AT_LINE_END.sub(r"\1\2", text)
    return HYPHEN_ON_OWN_LINE.sub(r"\1\2", text)


def clean_pages(pages: list[Page]) -> list[Page]:
    """Headers/footers first (they are whole lines), then drop contents pages, then fix hyphenation.

    Contents pages must go: their entries ("§ 36.402 Alterations.") look like headings and would
    claim occurrence 1 of the ID ahead of the real section.
    """
    cleaned = []
    for page in remove_headers_footers(pages):
        if is_contents_page(page):
            logger.info("dropping contents page p.%d", page.number)
            continue
        text = fix_hyphenation("\n".join(page.lines))
        cleaned.append(Page(page.number, text.split("\n"), page.printed_number))
    return cleaned


# =============================================================================
# Part 1c — detect code headings and regulation paragraphs
# =============================================================================

# 101-999 and 1001-1099 (ADA chapters 1-10). Excludes years such as "1991 Standards".
SECTION_NUMBER = r"(?:10\d{2}|[1-9]\d{2})"

# "404.2.3 Clear Width.  Door openings shall ..."
NUMBERED_HEADING = re.compile(
    r"^(?P<id>" + SECTION_NUMBER + r"(?:\.\d+)+) (?P<title>[A-Z][^.…]{0,100}?)\.(?:\s|$)"
)
# "213, 603, 604, and 608 Toilet and Bathing"   (guidance Appendix B)
MULTI_ID_HEADING = re.compile(
    r"^(?P<ids>" + SECTION_NUMBER + r"(?:\.\d+)*(?:,? (?:and )?" + SECTION_NUMBER + r"(?:\.\d+)*)+),? "
    r"(?P<title>[A-Z][^.…]{0,100})$"
)
# "403 Walking Surfaces"   (whole line; no digits or period in the title)
CHAPTER_HEADING = re.compile(r"^(?P<id>" + SECTION_NUMBER + r") (?P<title>[A-Z][A-Za-z ,'’/()&-]{2,80})$")
# "§ 35.151 New construction and alterations."   (regulation text)
REGULATION_HEADING = re.compile(r"^§ ?(?P<id>\d+\.\d+(?:\([a-z0-9]+\))*) (?P<title>[A-Z][^…]{0,100}?)\.?$")
# "Section 36.406(f) Assembly Areas"   (guidance; title may wrap onto the next line)
GUIDANCE_SECTION_HEADING = re.compile(
    r"^Section (?P<id>\d+\.\d+(?:\([a-zA-Z0-9]+\))*)(?: (?P<title>[A-Z][^…]{0,100}))?$"
)

# "INDEX TO THE 2010 STANDARDS": everything from here to the end of the file is back matter.
BACK_MATTER_START = re.compile(r"^INDEX TO\b")

# Top-level paragraph of a regulation section: "(b) Alterations." or a bare "(c)" line.
REGULATION_PARAGRAPH = re.compile(r"^\((?P<letter>[a-z])\)(?: +(?P<rest>[A-Z].*))?$")
# Sub-paragraphs use roman numerals "(i)", "(v)", "(x)" that look like letters.
ROMAN_LOOKING_LETTERS = "ivx"
MAX_PARAGRAPH_TITLE_CHARS = 60


@dataclass(frozen=True)
class Heading:
    section_id: str
    section_title: str
    section_type: str  # "regulation" (§ sections) or "code"
    body: str = ""     # text after the heading on the same line ("Door openings shall ...")


def is_rejected_heading_line(line: str) -> bool:
    """Lines that look like headings but must never start a section."""
    return (
        line.startswith("Advisory")     # advisory notes belong to the section they explain
        or "--" in line                 # running header "Section 35.151 -- Title II Regulation"
        or "…" in line or "...." in line  # table-of-contents dot leaders
    )


def match_heading(line: str) -> Heading | None:
    """Return the Heading if the line starts a section, else None."""
    if is_rejected_heading_line(line):
        return None

    match = NUMBERED_HEADING.match(line)
    if match:
        return Heading(match["id"], match["title"].strip(), "code", body=line[match.end():].strip())

    match = MULTI_ID_HEADING.match(line)
    if match:
        ids = re.findall(SECTION_NUMBER + r"(?:\.\d+)*", match["ids"])
        return Heading(", ".join(ids), match["title"].strip(), "code")

    match = CHAPTER_HEADING.match(line)
    if match:
        return Heading(match["id"], match["title"].strip(), "code")

    match = REGULATION_HEADING.match(line)
    if match:
        return Heading(match["id"], match["title"].strip(), "regulation")

    match = GUIDANCE_SECTION_HEADING.match(line)
    if match:
        return Heading(match["id"], (match["title"] or "").strip(), "code")

    return None


def is_next_paragraph_letter(letter: str, last_letter: str | None) -> bool:
    """Accept (a), (b), ... in increasing order; skipping is allowed because pypdf sometimes
    breaks a marker off its line. A roman-looking letter must be exactly the next one."""
    previous = last_letter or chr(ord("a") - 1)
    if letter <= previous:
        return False
    if letter in ROMAN_LOOKING_LETTERS:
        return ord(letter) == ord(previous) + 1
    return True


def paragraph_title(rest: str | None) -> str:
    """'Alterations.  (1) Each facility ...' -> 'Alterations'; long first sentences give ''."""
    if not rest:
        return ""
    first_sentence = rest.split(".")[0].strip()
    return first_sentence if len(first_sentence) <= MAX_PARAGRAPH_TITLE_CHARS else ""


# =============================================================================
# Part 1d — split pages into parent sections
# =============================================================================

@dataclass
class Section:
    section_id: str
    section_title: str
    section_type: str = "code"             # "code", "regulation", "front-matter" or "back-matter"
    level: str = "section"                 # "section", "front-matter" or "back-matter"
    occurrence: int = 1                    # 2, 3, ... when the same ID appears again (not a continuation)
    continuations: int = 0                 # repeats merged because the section resumed on a new page
    heading_body: str = ""                 # text on the heading line after the heading itself
    lines: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    printed_pages: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def heading_only(self) -> bool:
        """Only a heading line, e.g. '216 Signs' immediately followed by '216.1 General.'"""
        return len(self.lines) == 1 and not self.heading_body

    def add_line(self, line: str, page: Page) -> None:
        self.lines.append(line)
        if page.number not in self.pages:
            self.pages.append(page.number)
        if page.printed_number and page.printed_number not in self.printed_pages:
            self.printed_pages.append(page.printed_number)


class SectionSplitter:
    """Walks the lines of one file and opens a new Section at every heading.

    - Text before the first heading is front matter; from "INDEX TO ..." to the end is back matter.
    - Inside a regulation section (§ 35.151), top-level paragraphs "(a)", "(b)" become their own
      sections "35.151(a)", "35.151(b)"; deeper levels such as "(b)(1)" stay inside.
    - The same ID at the top of the next page is a continuation and is merged.
      Any other repeat becomes a separate section with occurrence 2, 3, ... and is logged.
    """

    def __init__(self, source: str):
        self.source = source
        self.sections: list[Section] = []
        self.current: Section | None = None
        self.occurrences = Counter()
        self.first_seen_page: dict[str, int] = {}
        self.regulation_id: str | None = None      # open § section, if any
        self.last_paragraph_letter: str | None = None

    def split(self, pages: list[Page]) -> list[Section]:
        for page in pages:
            for line_index, line in enumerate(page.lines):
                self.add(line, line_index, page)
        if not self.occurrences:  # no headings at all in this file
            return split_one_section_per_page(pages)
        return self.sections

    def add(self, line: str, line_index: int, page: Page) -> None:
        if self.current is not None and self.current.level == "back-matter":
            self.current.add_line(line, page)  # no heading detection inside the index
            return

        if BACK_MATTER_START.match(line):
            self.start(Section("back-matter", "Index", section_type="back-matter", level="back-matter"), line, page)
            return

        heading = match_heading(line)
        if heading is not None:
            self.add_heading(heading, line, line_index, page)
            return

        paragraph = REGULATION_PARAGRAPH.match(line) if self.regulation_id else None
        if paragraph and is_next_paragraph_letter(paragraph["letter"], self.last_paragraph_letter):
            self.last_paragraph_letter = paragraph["letter"]
            section_id = f"{self.regulation_id}({paragraph['letter']})"
            section = Section(section_id, paragraph_title(paragraph["rest"]), section_type="regulation",
                              heading_body=paragraph["rest"] or "")
            self.start_numbered(section, line, page)
            return

        if self.current is None:
            self.start(Section("front-matter", "Front matter", section_type="front-matter", level="front-matter",
                               heading_body=line), line, page)
            return
        self.current.add_line(line, page)

    def add_heading(self, heading: Heading, line: str, line_index: int, page: Page) -> None:
        # Same ID as the open section, at the top of a new page: the section resumes, merge.
        if self.current is not None and self.current.section_id == heading.section_id and line_index == 0:
            self.current.continuations += 1
            logger.info("%s: %s resumes on p.%d, merged as continuation", self.source, heading.section_id, page.number)
            return

        if heading.section_type == "regulation":
            self.regulation_id, self.last_paragraph_letter = heading.section_id, None
        else:
            self.regulation_id = None

        section = Section(heading.section_id, heading.section_title, section_type=heading.section_type,
                          heading_body=heading.body)
        self.start_numbered(section, line, page)

    def start_numbered(self, section: Section, line: str, page: Page) -> None:
        """Count occurrences of the ID, log repeats, then open the section."""
        self.occurrences[section.section_id] += 1
        section.occurrence = self.occurrences[section.section_id]
        if section.occurrence > 1:
            logger.warning(
                "%s: repeated section id %s (occurrence %d) on p.%d, first seen p.%d; kept separate",
                self.source, section.section_id, section.occurrence, page.number,
                self.first_seen_page[section.section_id],
            )
        else:
            self.first_seen_page[section.section_id] = page.number
        self.start(section, line, page)

    def start(self, section: Section, line: str, page: Page) -> None:
        self.sections.append(section)
        self.current = section
        section.add_line(line, page)


def split_sections(pages: list[Page], source: str) -> list[Section]:
    return SectionSplitter(source).split(pages)


def split_one_section_per_page(pages: list[Page]) -> list[Section]:
    """Fallback when a file has no recognisable headings."""
    sections = []
    for page in pages:
        section = Section(f"page-{page.number}", f"Page {page.number}", heading_body="(page)")
        for line in page.lines:
            section.add_line(line, page)
        if section.lines:
            sections.append(section)
    return sections


# =============================================================================
# Part 1e — breadcrumbs and parent records with deterministic IDs
# =============================================================================

def sha1_hex(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def make_parent_id(source: str, section_id: str, occurrence: int = 1) -> str:
    """sha1(source|section_id); a repeated (non-continuation) ID adds |occurrence."""
    if occurrence == 1:
        return sha1_hex(f"{source}|{section_id}")
    return sha1_hex(f"{source}|{section_id}|{occurrence}")


def make_child_id(parent_id: str, index: int) -> str:
    return sha1_hex(f"{parent_id}|{index}")


def ancestor_ids(section_id: str) -> list[str]:
    """Outermost first: '216.2.1' -> ['216', '216.2']; '35.151(b)' -> ['35', '35.151']."""
    ancestors = []
    current = section_id
    while True:
        if current.endswith(")"):
            current = current[:current.rindex("(")]
        elif "." in current:
            current = current.rsplit(".", 1)[0]
        else:
            break
        ancestors.append(current)
    return list(reversed(ancestors))


def section_label(section_id: str, section_title: str) -> str:
    return f"{section_id} {section_title}".strip()


def make_breadcrumb(code_name: str, section: Section, titles: dict[str, str]) -> str:
    """'ADA 2010 Standards > 216 Signs > 216.2 Designations'. Ancestors not in the file are skipped."""
    if section.level != "section":
        return f"{code_name} > {section.section_title}"
    parts = [code_name]
    parts += [section_label(a, titles[a]) for a in ancestor_ids(section.section_id) if a in titles]
    parts.append(section_label(section.section_id, section.section_title))
    return " > ".join(parts)


def first_titles(sections: list[Section]) -> dict[str, str]:
    """section_id -> title of its first occurrence, for breadcrumbs."""
    titles: dict[str, str] = {}
    for section in sections:
        if section.level == "section":
            titles.setdefault(section.section_id, section.section_title)
    return titles


def build_parents(sections: list[Section], source: str, corpus_file: CorpusFile) -> list[dict]:
    """Section -> {"parent_id", "text", "metadata"}. Metadata values are scalars (Chroma requirement)."""
    titles = first_titles(sections)
    parents = []
    for section in sections:
        parent_id = make_parent_id(source, section.section_id, section.occurrence)
        metadata = {
            "parent_id": parent_id,
            "source": source,
            "code_name": corpus_file.code_name,
            "edition_year": corpus_file.edition_year,
            "section_id": section.section_id,
            "section_title": section.section_title,
            "section_type": section.section_type,
            "breadcrumb": make_breadcrumb(corpus_file.code_name, section, titles),
            "pages": ",".join(str(p) for p in section.pages),
            "page_start": section.pages[0],
            "page_end": section.pages[-1],
            "printed_pages": ",".join(section.printed_pages),
            "occurrence": section.occurrence,
            "level": section.level,
            "heading_only": section.heading_only,
        }
        parents.append({"parent_id": parent_id, "text": section.text, "metadata": metadata})
    return parents


def load_file_as_parents(path: Path, corpus_file: CorpusFile) -> tuple[list[Section], list[dict]]:
    """Part 1 for one file: load -> clean -> split -> parent records."""
    pages = clean_pages(load_pages(path))
    sections = split_sections(pages, source=path.name)
    return sections, build_parents(sections, path.name, corpus_file)


# =============================================================================
# Part 2 — parent sections -> four stores
# =============================================================================

def section_embedding_text(parent: dict, prefix_chars: int) -> str:
    """Coarse path: breadcrumb (ends with the section title) + the opening of the section."""
    return f"{parent['metadata']['breadcrumb']}\n{parent['text'][:prefix_chars]}"


def child_index_text(child: dict) -> str:
    """What is embedded and BM25-tokenized: breadcrumb + chunk, so chapter headings live on in children."""
    return f"{child['metadata']['breadcrumb']}\n{child['text']}"


def make_child_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    # add_start_index records where each chunk starts in the parent (for step 2's context windows).
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)


def split_children(parent: dict, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """Fine path: chunks from ONE parent's text, so children never cross section boundaries.

    "text" is the raw chunk (parent_text[start_char:start_char + len(text)]).
    level is "child"; section_level keeps the parent's level so retrieval can skip front/back matter.
    """
    children = []
    for index, document in enumerate(splitter.create_documents([parent["text"]])):
        child_id = make_child_id(parent["parent_id"], index)
        metadata = {
            **parent["metadata"],
            "child_id": child_id,
            "child_index": index,
            "start_char": document.metadata["start_index"],
            "section_level": parent["metadata"]["level"],
            "level": "child",
        }
        children.append({
            "child_id": child_id,
            "parent_id": parent["parent_id"],
            "text": document.page_content,
            "metadata": metadata,
        })
    return children


def make_embedder(model_name: str) -> EmbedFn:
    """Sentence-transformers embedder; normalized vectors so cosine distance is meaningful."""
    from sentence_transformers import SentenceTransformer  # heavy import, only when really embedding

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    return embed


def write_collection(client, name: str, ids: list[str], documents: list[str], embedding_texts: list[str],
                     metadatas: list[dict], embed_fn: EmbedFn) -> int:
    """Upsert by deterministic ID (no duplicates on re-run), then delete IDs from earlier runs that no longer exist.

    documents are stored for display; embedding_texts are what gets embedded (they add the breadcrumb).
    """
    collection = client.get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}})
    for start in range(0, len(ids), CHROMA_BATCH_SIZE):
        end = start + CHROMA_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embed_fn(embedding_texts[start:end]),
        )
    stale_ids = set(collection.get(include=[])["ids"]) - set(ids)
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))
        logger.info("%s: deleted %d stale entries", name, len(stale_ids))
    return collection.count()


def build_bm25_index(children: list[dict]) -> dict:
    """Keyword index over the same children as the vector path; rebuilt in full every run."""
    return {
        "bm25": BM25Okapi([tokenize(child_index_text(child)) for child in children]),
        "child_ids": [child["child_id"] for child in children],
        "parent_ids": [child["parent_id"] for child in children],
        "texts": [child["text"] for child in children],
        "metadatas": [child["metadata"] for child in children],
    }


def write_bm25(children: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(build_bm25_index(children), f)
    return len(children)


def write_docstore(parents: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    docstore = {parent["parent_id"]: {"text": parent["text"], "metadata": parent["metadata"]} for parent in parents}
    path.write_text(json.dumps(docstore, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(docstore)


def write_stores(parents: list[dict], settings: Settings, embed_fn: EmbedFn, chroma_client) -> dict[str, int]:
    """Part 2: write all four stores and return the number of entries in each.

    Heading-only parents ("216 Signs") go to the docstore only; they live on in children's breadcrumbs.
    """
    searchable = [parent for parent in parents if not parent["metadata"]["heading_only"]]
    splitter = make_child_splitter(settings.child_chunk_size, settings.child_chunk_overlap)
    children = [child for parent in searchable for child in split_children(parent, splitter)]

    return {
        "sections": write_collection(
            chroma_client,
            settings.sections_collection,
            ids=[p["parent_id"] for p in searchable],
            documents=[p["text"][:settings.section_prefix_chars] for p in searchable],
            embedding_texts=[section_embedding_text(p, settings.section_prefix_chars) for p in searchable],
            metadatas=[p["metadata"] for p in searchable],
            embed_fn=embed_fn,
        ),
        "children": write_collection(
            chroma_client,
            settings.children_collection,
            ids=[c["child_id"] for c in children],
            documents=[c["text"] for c in children],
            embedding_texts=[child_index_text(c) for c in children],
            metadatas=[c["metadata"] for c in children],
            embed_fn=embed_fn,
        ),
        "bm25": write_bm25(children, settings.bm25_path),
        "docstore": write_docstore(parents, settings.docstore_path),
    }


# =============================================================================
# Report + entry point
# =============================================================================

def section_report(source: str, sections: list[Section]) -> list[str]:
    """Longest sections hint at missed headings, shortest at false headings."""
    searchable = [s for s in sections if not s.heading_only]
    by_length = sorted(searchable, key=lambda s: len(s.text))
    repeated_ids = {s.section_id for s in sections if s.occurrence > 1}
    extra_occurrences = sum(1 for s in sections if s.occurrence > 1)
    merged = sum(s.continuations for s in sections)
    paragraphs = sum(1 for s in sections if s.section_type == "regulation" and s.section_id.endswith(")"))

    def describe(section: Section) -> str:
        return f"    {section.section_id:<22} {section.section_title[:50]:<50} {len(section.text):>7,} chars"

    return [
        f"{source}: {len(sections)} sections "
        f"({len(sections) - len(searchable)} heading-only, docstore only; {paragraphs} regulation paragraphs)",
        "  5 longest:", *[describe(s) for s in reversed(by_length[-5:])],
        "  5 shortest (excluding heading-only):", *[describe(s) for s in by_length[:5]],
        f"  repeated IDs: {len(repeated_ids)} "
        f"({extra_occurrences} extra occurrences kept separate, {merged} continuations merged)",
    ]


def ingest_corpus(settings: Settings, embed_fn: EmbedFn, chroma_client) -> tuple[list[str], dict[str, int]]:
    report: list[str] = []
    all_parents: list[dict] = []

    for filename in sorted(p.name for p in settings.data_dir.iterdir() if p.is_file()):
        if filename not in CORPUS:
            logger.warning("Skipping %s: not listed in config.CORPUS", filename)
            continue
        sections, parents = load_file_as_parents(settings.data_dir / filename, CORPUS[filename])
        report.extend(section_report(filename, sections))
        all_parents.extend(parents)

    return report, write_stores(all_parents, settings, embed_fn, chroma_client)


def main() -> None:
    import chromadb

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("inspectiq").setLevel(logging.INFO)

    settings = load_settings()
    settings.store_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    embed_fn = make_embedder(settings.embedding_model)

    report, counts = ingest_corpus(settings, embed_fn, chroma_client)
    print("\n".join(report))
    print("stores:", ", ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
