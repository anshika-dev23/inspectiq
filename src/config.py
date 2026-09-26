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

    # Retrieval models (step 2)
    query_instruction: str  # BGE models expect this prefix on queries (not on documents)
    reranker_model: str

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
        query_instruction="Represent this sentence for searching relevant passages: ",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_provider=llm_provider,
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
    )


# Query normalization: abbreviations are expanded in place, "GFCI" -> "GFCI (ground-fault circuit-interrupter)".
ABBREVIATIONS = {
    "gfci": "ground-fault circuit-interrupter",
    "afci": "arc-fault circuit-interrupter",
    "adaag": "ADA Accessibility Guidelines",
    "tty": "text telephone teletypewriter",
    "atm": "automatic teller machine",
    "als": "assistive listening system",
    "ufas": "Uniform Federal Accessibility Standards",
    "cfr": "Code of Federal Regulations",
    "hud": "Department of Housing and Urban Development",
    "wc": "water closet toilet",
}


@dataclass(frozen=True)
class RetrievalConfig:
    """Every retrieval stage is toggleable so evals can compare configurations (step 4)."""

    # Stage toggles
    use_query_normalization: bool = True  # expand ABBREVIATIONS
    use_child_vector: bool = True
    use_bm25: bool = True
    use_section_vector: bool = True
    use_exact_ref_boost: bool = True      # a section ref in the query ("604.5") is pinned at rank 1
    use_rerank: bool = True               # cross-encoder + relevance threshold
    use_context_window: bool = True       # False: always return whole parents

    # Sizes
    child_vector_top_k: int = 20
    bm25_top_k: int = 20
    section_vector_top_k: int = 5
    rrf_k: int = 60
    rerank_top_n: int = 10
    final_top_n: int = 3

    # ms-marco cross-encoder returns raw logits (about -11 .. +11); below this a parent is not relevant.
    # Provisional value, to be tuned with the golden set in step 4.
    rerank_threshold: float = 0.0

    # A parent up to this size is returned whole ...
    parent_full_text_max_chars: int = 6000
    # ... a bigger parent is returned as a window of this size centred on its best-matching children.
    context_window_chars: int = 4000
