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

# Furniture that can appear anywhere on a page (e.g. an inner footer printed above a figure),
# so it is removed wherever it occurs, not only at the page edges.
FURNITURE_ANYWHERE = (
    r"^Titles II and III - 2010 Standards - \d+$",
    # The DOJ footer, which PyMuPDF sometimes places mid-page (above a figure). Whole lines only.
    r"^\d{4} Standards: +Titles? I[I ]*(?:and III)? ?- ?\d+$",
    r"^\d+ - \d{4} Standards: +Titles? I[I ]*(?:and III)?$",
    r"^Guidance on (?:the )?\d{4} Standards: +Titles? I[I ]*(?:and III)? ?- ?\d+$",
    r"^Department of Justice$",
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
    ollama_num_ctx: int          # set explicitly: Ollama's default context is smaller and silently truncates
    anthropic_model: str
    llm_temperature: float

    # Answering (step 3): total characters of source text put in the prompt
    answer_max_context_chars: int
    # Every number in an answer must appear in a cited source; strict: otherwise refuse ("ungrounded_number")
    strict_number_grounding: bool


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
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
        llm_temperature=0.0,
        # ~6,000 chars is ~1,500 tokens: fits llama3.2:3b's 8k context with instructions and answer to spare.
        answer_max_context_chars=int(os.getenv(
            "ANSWER_MAX_CONTEXT_CHARS", "6000" if llm_provider == "ollama" else "20000")),
        strict_number_grounding=os.getenv("STRICT_NUMBER_GROUNDING", "1") != "0",
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


# Query glossary: everyday words -> the terms the Standards use (mostly defined in 106.5 Defined Terms, plus the
# chapter vocabulary: toilet room, water closet, lavatory, turning space). Matched as whole words (plural allowed);
# the code terms are APPENDED to the query, so phrases such as "toilet room" are never broken apart.
GLOSSARY = {
    "light switch": "operable parts, controls",            # 106.5 Operable Part; 205.1 lists light switches
    "outlet": "operable parts, convenience receptacles",
    "thermostat": "operable parts, environmental controls",
    "door handle": "door hardware, operable parts",
    "bathroom": "toilet room",
    "restroom": "toilet room",
    "toilet": "water closet",
    "sink": "lavatory",
    "turn around": "turning space",
    "hallway": "circulation path, walking surface",       # 106.5 Circulation Path
    "sidewalk": "walk, accessible route",                 # 106.5 Walk
    "steep": "running slope",                             # 106.5 Running Slope
    "step": "change in level",
    "stairs": "stairways",
    "curb cut": "curb ramp",                              # 106.5 Curb Ramp
    "how wide": "clear width",
    "hotel room": "transient lodging guest room",         # 106.5 Transient Lodging
    "wheelchair seating": "wheelchair space",             # 106.5 Wheelchair Space
    "raised letters": "tactile characters",               # 106.5 Tactile, Characters
    "water fountain": "drinking fountain",
}


@dataclass(frozen=True)
class RetrievalConfig:
    """Every retrieval stage is toggleable so evals can compare configurations (step 4)."""

    # Stage toggles
    use_query_normalization: bool = True  # expand ABBREVIATIONS
    use_glossary: bool = True             # append GLOSSARY code terms for everyday words
    use_child_vector: bool = True
    use_bm25: bool = True
    use_section_vector: bool = False      # off by default: no gain on the golden set, hurt exact IDs (decisions 4a)
    use_exact_ref_boost: bool = True      # a section ref in the query ("604.5") is pinned at rank 1
    use_rerank: bool = True               # cross-encoder + relevance threshold
    use_context_window: bool = True       # False: always return whole parents

    # Sizes
    child_vector_top_k: int = 20
    bm25_top_k: int = 20
    section_vector_top_k: int = 5
    rrf_k: int = 60
    rerank_top_n: int = 10
    final_k: int = 3                      # parents returned; step 4 compares 3 and 5

    # ms-marco cross-encoder returns raw logits (about -11 .. +11); below this a parent is not relevant.
    # Provisional value, to be tuned with the golden set in step 4. Applies in both rerank modes.
    rerank_threshold: float = 0.0

    # How the reranker orders candidates (pinned exact refs always stay first):
    #   "replace": by cross-encoder score alone
    #   "blend":   rerank_blend_weight * sigmoid(rerank score) + (1 - weight) * (RRF score / best RRF score)
    #   "rrf":     the reranker's ranking is a 4th list in RRF (rank-based, no score normalization)
    rerank_mode: str = "replace"
    rerank_blend_weight: float = 0.5

    # A parent up to this size is returned whole ...
    parent_full_text_max_chars: int = 6000
    # ... a bigger parent is returned as a window of this size centred on its best-matching children.
    context_window_chars: int = 4000


# Shadow cost (step 5): what each LLM call WOULD cost on these Anthropic models, in USD per million tokens
# (input, output). Anthropic first-party prices, from the model table cached 2026-06-24. Applied to the local
# model's token counts, so it is an estimate: the llama tokenizer does not count tokens like Claude's.
SHADOW_PRICES_USD_PER_MTOK = {
    "claude-haiku-4-5": (1.00, 5.00),    # candidate for grading / query rewriting (step 6)
    "claude-sonnet-5": (2.00, 10.00),    # candidate for final answers
    "claude-opus-5": (5.00, 25.00),      # upper bound
}
SHADOW_COST_NOTE = "estimate: token counts from the llama tokenizer, not Claude's"
