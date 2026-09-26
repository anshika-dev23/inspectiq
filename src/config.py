"""Settings for InspectIQ, read from environment variables with sensible defaults.

.env is loaded by src/__init__.py, before any other import, so it is already in os.environ here.
"""
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLM_PROVIDERS = ("ollama", "anthropic")


@dataclass(frozen=True)
class CorpusFile:
    """Metadata attached to every section from one source file."""

    code_name: str
    edition_year: int


# Which files in data/ are ingested, and how they are labelled.
CORPUS = {
    "ada_2010_standards.pdf": CorpusFile(code_name="ADA 2010 Standards", edition_year=2010),
    "ada_2010_guidance.pdf": CorpusFile(code_name="ADA 2010 Guidance", edition_year=2010),
}

# Page furniture that survives the "repeated at the page edge on >= 10% of pages" rule because it
# only appears in one part of a document. Matched against the first/last lines of each page only.
FURNITURE_STOPLIST = (
    # Chapter running heads: "TECHNICAL CHAPTER 4: ACCESSIBLE ROUTES", "ADA CHAPTER 2: SCOPING REQUIREMENTS ..."
    r"^[A-Z ,:]*CHAPTER \d+: [A-Z ,:]+$",
    # DOJ footer, page number first: "26 - 2010 Standards: Title III Department of Justice"
    r"^\d+ - (?:Guidance on )?(?:the )?\d{4} Standards:? +Titles? I",
    # DOJ footer, page number last: "Department of Justice 2010 Standards: Title III - 21"
    r"\d{4} Standards:? +Titles? I[I ]*(?:and III)? ?- ?\d+",
    # Running heads of the regulation text
    r"^Subpart D of 28 CFR Part 36$",
    r"^Section 35\.151 of 28 CFR Part 35$",
)


@dataclass(frozen=True)
class Settings:
    # Paths
    data_dir: Path
    store_dir: Path
    chroma_dir: Path
    bm25_path: Path
    docstore_path: Path

    # Ingestion
    embedding_model: str
    section_prefix_chars: int
    child_chunk_size: int
    child_chunk_overlap: int
    sections_collection: str
    children_collection: str

    # Retrieval context (step 2): a parent up to this size is returned whole ...
    parent_full_text_max_chars: int
    # ... a bigger parent is returned as a window of about this size around its best-matching children.
    context_window_chars: int

    # LLM provider switch (used from step 3; steps 1-2 never call an LLM)
    llm_provider: str
    ollama_model: str
    ollama_base_url: str
    anthropic_model: str


def load_settings() -> Settings:
    store_dir = PROJECT_ROOT / "store"
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if llm_provider not in LLM_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {LLM_PROVIDERS}, got {llm_provider!r}")

    return Settings(
        data_dir=PROJECT_ROOT / "data",
        store_dir=store_dir,
        chroma_dir=store_dir / "chroma",
        bm25_path=store_dir / "bm25.pkl",
        docstore_path=store_dir / "docstore.json",
        embedding_model="BAAI/bge-small-en-v1.5",
        section_prefix_chars=1500,
        child_chunk_size=400,
        child_chunk_overlap=60,
        sections_collection="sections",
        children_collection="children",
        parent_full_text_max_chars=6000,
        context_window_chars=4000,
        llm_provider=llm_provider,
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
    )
