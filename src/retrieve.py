"""Step 2 — retrieval as a pure function: retrieve(query, filters, config) -> RetrievalResult.

Pipeline (every stage can be switched off in RetrievalConfig):
  1. normalize the query: expand abbreviations, detect section refs, tokenize for BM25
  2. search three indexes, each with the metadata filters applied INSIDE the search:
       child vector top 20 | BM25 over children top 20 | section vector top 5
  3. map child hits to their parent (dedupe, keep the best rank)
  4. fuse the three parent rankings with Reciprocal Rank Fusion (k=60)
  5. pin parents named by an exact section ref ("604.5") at rank 1
  6. rerank the top 10 with a cross-encoder ("replace" or "blend" with RRF); drop those under the
     threshold (pinned parents stay)
  7. return the top final_k parents as contexts: whole if small, else a line-aligned window around the
     best children (or from the section start, when the section was pinned by an exact ref)
No LLM is called here.
"""
import json
import math
import pickle
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

from src.config import ABBREVIATIONS, RetrievalConfig, Settings, load_settings
from src.text import tokenize

FILTER_KEYS = ("code_name", "edition_year", "section_type")

# Section refs in a query: "404.2.3", "604.5", "35.151(b)", "§ 36.406(f)", "R302.1".
# At least two digits before the first dot, so "1.5 inches" is not a ref.
SECTION_REF = re.compile(r"\b([a-z]?\d{2,4}(?:\.\d+)+(?:\([a-z0-9]+\))*)", re.IGNORECASE)
RERANK_MODES = ("replace", "blend", "rrf")
# Window edges move to the nearest line break, but never further than this.
MAX_SNAP_CHARS = 200


# =============================================================================
# Result types
# =============================================================================

@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    text: str                 # abbreviations expanded; used for embeddings and the reranker
    tokens: list[str]         # used for BM25 (stopwords removed)
    section_refs: list[str]   # lowercased, e.g. ["35.151(b)"]


@dataclass(frozen=True)
class ChildHit:
    child_id: str
    parent_id: str
    rank: int      # 1-based
    score: float   # cosine similarity (vector) or BM25 score


@dataclass(frozen=True)
class ParentHit:
    parent_id: str
    rank: int
    score: float


@dataclass
class Candidate:
    """A parent moving through fusion and reranking."""

    parent_id: str
    rrf_score: float
    ranks: dict[str, int]              # retriever name -> rank of this parent in that retriever
    pinned: bool = False               # named by an exact section ref in the query
    rerank_score: float | None = None
    rerank_rank: int | None = None     # 1-based rank by cross-encoder score among the reranked candidates
    order_score: float | None = None   # the score used for ordering in rerank_mode "blend" or "rrf"
    best_child_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedContext:
    parent_id: str
    citation: str                      # "[ada_2010_standards.pdf p.123 §404.2.3]"
    source: str
    code_name: str
    section_id: str
    section_title: str
    breadcrumb: str
    pages: str
    text: str                          # whole parent, or a window of it
    window: tuple[int, int] | None     # character range in the parent; None = whole parent
    parent_length: int
    rrf_score: float
    rerank_score: float | None
    pinned: bool
    ranks: dict[str, int]


@dataclass
class RetrievalDebug:
    normalized: NormalizedQuery
    child_vector: list[ChildHit]
    bm25: list[ChildHit]
    section_vector: list[ParentHit]
    fused: list[Candidate]
    reranked: list[Candidate]
    pinned_parent_ids: list[str]
    timings_ms: dict[str, float]


@dataclass
class RetrievalResult:
    query: str
    contexts: list[RetrievedContext]
    debug: RetrievalDebug

    @property
    def is_empty(self) -> bool:
        """Empty means the caller must answer "I don't know"."""
        return not self.contexts


# =============================================================================
# Stores and models
# =============================================================================

@dataclass
class RetrievalResources:
    docstore: dict[str, dict]                              # parent_id -> {"text", "metadata"}
    bm25_index: dict                                       # see ingest.build_bm25_index
    children_collection: object                            # Chroma collection "children"
    sections_collection: object                            # Chroma collection "sections"
    embed_query: Callable[[str], list[float]]
    rerank: Callable[[list[tuple[str, str]]], list[float]]

    # Lookups derived from the stores above
    children_by_id: dict[str, dict] = field(init=False)
    children_by_parent: dict[str, list[str]] = field(init=False)
    parents_by_section_id: dict[str, list[str]] = field(init=False)

    def __post_init__(self):
        index = self.bm25_index
        self.children_by_id = {}
        self.children_by_parent = {}
        for child_id, parent_id, text, metadata in zip(
            index["child_ids"], index["parent_ids"], index["texts"], index["metadatas"]
        ):
            self.children_by_id[child_id] = {"parent_id": parent_id, "text": text, "start_char": metadata["start_char"]}
            self.children_by_parent.setdefault(parent_id, []).append(child_id)

        # Exact-ref lookup: only searchable parents, only the first occurrence of an ID.
        self.parents_by_section_id = {}
        for parent_id, parent in self.docstore.items():
            metadata = parent["metadata"]
            if metadata["level"] == "section" and not metadata["heading_only"] and metadata["occurrence"] == 1:
                self.parents_by_section_id.setdefault(metadata["section_id"].lower(), []).append(parent_id)


def load_resources(settings: Settings) -> RetrievalResources:
    """Open the stores written by src/ingest.py and load the embedding and reranking models."""
    import chromadb
    from sentence_transformers import CrossEncoder

    from src.ingest import make_embedder

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    embed = make_embedder(settings.embedding_model)
    cross_encoder = CrossEncoder(settings.reranker_model)
    with settings.bm25_path.open("rb") as f:
        bm25_index = pickle.load(f)

    return RetrievalResources(
        docstore=json.loads(settings.docstore_path.read_text(encoding="utf-8")),
        bm25_index=bm25_index,
        children_collection=client.get_collection(settings.children_collection),
        sections_collection=client.get_collection(settings.sections_collection),
        embed_query=lambda query: embed([settings.query_instruction + query])[0],
        rerank=lambda pairs: [float(score) for score in cross_encoder.predict(pairs)],
    )


@lru_cache(maxsize=1)
def get_default_resources() -> RetrievalResources:
    """Loaded once per process; models take a few seconds to load."""
    return load_resources(load_settings())


# =============================================================================
# 1. Query normalization (no LLM)
# =============================================================================

def expand_abbreviations(text: str, abbreviations: dict[str, str]) -> str:
    """'GFCI outlets' -> 'GFCI (ground-fault circuit-interrupter) outlets'."""
    if not abbreviations:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(a) for a in abbreviations) + r")\b", re.IGNORECASE)
    return pattern.sub(lambda match: f"{match[0]} ({abbreviations[match[0].lower()]})", text)


def detect_section_refs(text: str) -> list[str]:
    """Unique section refs in order of appearance, lowercased: 'See 35.151(B)' -> ['35.151(b)']."""
    refs = []
    for match in SECTION_REF.finditer(text):
        ref = match[1].lower()
        if ref not in refs:
            refs.append(ref)
    return refs


def normalize_query(query: str, config: RetrievalConfig) -> NormalizedQuery:
    text = expand_abbreviations(query, ABBREVIATIONS) if config.use_query_normalization else query
    return NormalizedQuery(
        original=query,
        text=text,
        tokens=tokenize(text, remove_stopwords=True),
        section_refs=detect_section_refs(query),
    )


# =============================================================================
# 2. Metadata filters
# =============================================================================

def validate_filters(filters: dict | None) -> dict:
    filters = {key: value for key, value in (filters or {}).items() if value is not None}
    unknown = set(filters) - set(FILTER_KEYS)
    if unknown:
        raise ValueError(f"Unknown filter keys {sorted(unknown)}; allowed: {FILTER_KEYS}")
    return filters


def chroma_where(filters: dict) -> dict | None:
    """{"code_name": "X", "edition_year": [2010, 2012]} -> Chroma where clause (None = no filter)."""
    clauses = [
        {key: {"$in": list(value)} if isinstance(value, (list, tuple)) else {"$eq": value}}
        for key, value in filters.items()
    ]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def matches_filters(metadata: dict, filters: dict) -> bool:
    """Same semantics as chroma_where, for BM25 and exact refs (which do not go through Chroma)."""
    for key, value in filters.items():
        allowed = value if isinstance(value, (list, tuple)) else [value]
        if metadata.get(key) not in allowed:
            return False
    return True


# =============================================================================
# 3. The three searches
# =============================================================================

def child_vector_search(collection, query_embedding: list[float], where: dict | None, top_k: int) -> list[ChildHit]:
    result = collection.query(
        query_embeddings=[query_embedding], n_results=top_k, where=where, include=["metadatas", "distances"]
    )
    return [
        ChildHit(child_id, metadata["parent_id"], rank, 1.0 - distance)  # cosine distance -> similarity
        for rank, (child_id, metadata, distance) in enumerate(
            zip(result["ids"][0], result["metadatas"][0], result["distances"][0]), start=1
        )
    ]


def section_vector_search(collection, query_embedding: list[float], where: dict | None, top_k: int) -> list[ParentHit]:
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where, include=["distances"])
    return [
        ParentHit(parent_id, rank, 1.0 - distance)  # section ids are parent ids
        for rank, (parent_id, distance) in enumerate(zip(result["ids"][0], result["distances"][0]), start=1)
    ]


def bm25_search(index: dict, tokens: list[str], filters: dict, top_k: int) -> list[ChildHit]:
    """Score ALL children, keep those that match the filters, THEN take the top k.

    (Top k first and filter afterwards would silently return fewer than k, or nothing.)
    A score of 0 means no query term occurs in the child, so it is not a hit.
    """
    scores = index["bm25"].get_scores(tokens)
    candidates = [
        i for i, metadata in enumerate(index["metadatas"])
        if scores[i] > 0 and matches_filters(metadata, filters)
    ]
    candidates.sort(key=lambda i: scores[i], reverse=True)
    return [
        ChildHit(index["child_ids"][i], index["parent_ids"][i], rank, float(scores[i]))
        for rank, i in enumerate(candidates[:top_k], start=1)
    ]


# =============================================================================
# 4. Children -> parents, Reciprocal Rank Fusion, exact refs
# =============================================================================

def parents_from_children(hits: list[ChildHit]) -> list[ParentHit]:
    """Dedupe child hits to parents, ordered by each parent's best child.

    A parent's rank is its position in this deduplicated list (1, 2, 3, ...), so it is
    comparable with the section-vector ranking, which is also a list of parents.
    """
    parents: list[ParentHit] = []
    seen: set[str] = set()
    for hit in sorted(hits, key=lambda h: h.rank):
        if hit.parent_id not in seen:
            seen.add(hit.parent_id)
            parents.append(ParentHit(hit.parent_id, len(parents) + 1, hit.score))
    return parents


def reciprocal_rank_fusion(rankings: dict[str, list[ParentHit]], k: int) -> list[Candidate]:
    """RRF score = sum over retrievers of 1 / (k + rank). Uses ranks only, so scores on
    different scales (cosine, BM25) never need to be compared."""
    fused: dict[str, Candidate] = {}
    for retriever, hits in rankings.items():
        for hit in hits:
            candidate = fused.setdefault(hit.parent_id, Candidate(hit.parent_id, 0.0, {}))
            candidate.rrf_score += 1.0 / (k + hit.rank)
            candidate.ranks[retriever] = hit.rank
    # Ties: better best rank first, then parent_id so the order is deterministic.
    return sorted(fused.values(), key=lambda c: (-c.rrf_score, min(c.ranks.values()), c.parent_id))


def resolve_section_ref(ref: str, parents_by_section_id: dict[str, list[str]]) -> list[str]:
    """Parent ids for a ref; '35.151(b)(4)' falls back to '35.151(b)' when (4) is not its own section."""
    candidate = ref.lower()
    while True:
        if candidate in parents_by_section_id:
            return parents_by_section_id[candidate]
        if not candidate.endswith(")"):
            return []
        candidate = candidate[:candidate.rindex("(")]


def find_pinned_parents(refs: list[str], resources: RetrievalResources, filters: dict) -> list[str]:
    """Parents named by the query's section refs that exist in the docstore and pass the filters."""
    pinned = []
    for ref in refs:
        for parent_id in resolve_section_ref(ref, resources.parents_by_section_id):
            if parent_id not in pinned and matches_filters(resources.docstore[parent_id]["metadata"], filters):
                pinned.append(parent_id)
    return pinned


def pin_to_top(fused: list[Candidate], pinned_ids: list[str]) -> list[Candidate]:
    """Move pinned parents to the front (in query order), adding them if no retriever found them."""
    by_id = {candidate.parent_id: candidate for candidate in fused}
    pinned = []
    for parent_id in pinned_ids:
        candidate = by_id.get(parent_id) or Candidate(parent_id, 0.0, {})
        candidate.pinned = True
        pinned.append(candidate)
    return pinned + [candidate for candidate in fused if candidate.parent_id not in pinned_ids]


# =============================================================================
# 5. Best children, reranking, threshold
# =============================================================================

def child_rrf_scores(child_rankings: list[list[ChildHit]], k: int) -> dict[str, float]:
    """RRF at child level (vector + BM25): which chunks of a parent matched best."""
    scores: dict[str, float] = {}
    for hits in child_rankings:
        for hit in hits:
            scores[hit.child_id] = scores.get(hit.child_id, 0.0) + 1.0 / (k + hit.rank)
    return scores


def best_children(parent_id: str, child_scores: dict[str, float], resources: RetrievalResources) -> list[str]:
    """This parent's matched children, best first. A parent found only by the section vector
    (or pinned) has no matched child, so its first child stands in."""
    children = resources.children_by_parent.get(parent_id, [])
    matched = sorted((c for c in children if c in child_scores), key=lambda c: -child_scores[c])
    return matched or children[:1]


def rerank_passage(candidate: Candidate, resources: RetrievalResources) -> str:
    """What the cross-encoder reads: where the text sits (breadcrumb) + the best-matching chunk."""
    parent = resources.docstore[candidate.parent_id]
    if candidate.best_child_ids:
        text = resources.children_by_id[candidate.best_child_ids[0]]["text"]
    else:
        text = parent["text"][:400]
    return f"{parent['metadata']['breadcrumb']}\n{text}"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def order_candidates(candidates: list[Candidate], mode: str, blend_weight: float, rrf_k: int) -> list[Candidate]:
    """Pinned parents first (query order), the rest by:
      replace: the cross-encoder score;
      blend:   weight * sigmoid(rerank score) + (1 - weight) * rrf_score / best rrf_score.
               Both parts are in 0..1: the ms-marco cross-encoder is trained as a binary classifier,
               so sigmoid(logit) reads as P(relevant); RRF is divided by this query's best RRF.
      rrf:     the reranker's ranking is a 4th list in RRF: rrf_score + 1 / (rrf_k + rerank_rank).
               Rank-based, so no score normalization is needed.
    """
    if mode not in RERANK_MODES:
        raise ValueError(f"rerank_mode must be one of {RERANK_MODES}, got {mode!r}")
    for rank, candidate in enumerate(sorted(candidates, key=lambda c: -c.rerank_score), start=1):
        candidate.rerank_rank = rank
    pinned = [c for c in candidates if c.pinned]
    others = [c for c in candidates if not c.pinned]

    if mode == "replace":
        return pinned + sorted(others, key=lambda c: -c.rerank_score)

    if mode == "blend":
        best_rrf = max((c.rrf_score for c in candidates), default=0.0) or 1.0
        for candidate in candidates:
            candidate.order_score = (
                blend_weight * sigmoid(candidate.rerank_score) + (1 - blend_weight) * candidate.rrf_score / best_rrf
            )
    else:  # "rrf"
        for candidate in candidates:
            candidate.order_score = candidate.rrf_score + 1.0 / (rrf_k + candidate.rerank_rank)
    return pinned + sorted(others, key=lambda c: -c.order_score)


def rerank(query_text: str, candidates: list[Candidate], resources: RetrievalResources,
           config: RetrievalConfig) -> list[Candidate]:
    """Score (query, passage) pairs with the cross-encoder, then order them (see order_candidates)."""
    if not candidates:
        return []
    scores = resources.rerank([(query_text, rerank_passage(c, resources)) for c in candidates])
    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)
    return order_candidates(candidates, config.rerank_mode, config.rerank_blend_weight, config.rrf_k)


def passes_threshold(candidate: Candidate, threshold: float) -> bool:
    """Pinned parents always pass: the user asked for that section by its ID.
    The threshold is on the raw cross-encoder score in both rerank modes: it answers
    "is this relevant at all?", while the mode only decides the order."""
    return candidate.pinned or candidate.rerank_score >= threshold


# =============================================================================
# 6. Contexts: whole parent or a window around the best children
# =============================================================================

def choose_window(parent_length: int, child_spans: list[tuple[int, int]], window_chars: int) -> tuple[int, int]:
    """Character range of about window_chars, centred on the best children.

    child_spans are (start, end), best first. Start with the best child's span and widen it with
    each next child that still fits in the window; then centre the window on that span and shift it
    back inside the parent if it runs off either end.
    """
    low, high = child_spans[0]
    for start, end in child_spans[1:]:
        if max(high, end) - min(low, start) <= window_chars:
            low, high = min(low, start), max(high, end)
    centre = (low + high) // 2
    start = max(0, centre - window_chars // 2)
    end = min(parent_length, start + window_chars)
    start = max(0, end - window_chars)
    return start, end


def snap_to_lines(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a window to whole lines: start back to its line start, end forward to its line end,
    unless the line break is more than MAX_SNAP_CHARS away (then the edge stays where it is)."""
    line_start = text.rfind("\n", 0, start) + 1
    if start - line_start <= MAX_SNAP_CHARS:
        start = line_start
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    if line_end - end <= MAX_SNAP_CHARS:
        end = line_end
    return start, end


def format_citation(metadata: dict) -> str:
    """[file p.X §section], or p.X-Y when the section spans pages."""
    start, end = metadata["page_start"], metadata["page_end"]
    pages = f"{start}" if start == end else f"{start}-{end}"
    return f"[{metadata['source']} p.{pages} §{metadata['section_id']}]"


def build_context(candidate: Candidate, resources: RetrievalResources, config: RetrievalConfig) -> RetrievedContext:
    parent = resources.docstore[candidate.parent_id]
    metadata, text = parent["metadata"], parent["text"]

    window = None
    if config.use_context_window and len(text) > config.parent_full_text_max_chars:
        if candidate.pinned or not candidate.best_child_ids:
            # Asked for by its ID (or nothing matched inside it): show the section from its start.
            window = (0, min(len(text), config.context_window_chars))
        else:
            spans = []
            for child_id in candidate.best_child_ids:
                child = resources.children_by_id[child_id]
                spans.append((child["start_char"], child["start_char"] + len(child["text"])))
            window = choose_window(len(text), spans, config.context_window_chars)
        window = snap_to_lines(text, *window)
        text = text[window[0]:window[1]]

    return RetrievedContext(
        parent_id=candidate.parent_id,
        citation=format_citation(metadata),
        source=metadata["source"],
        code_name=metadata["code_name"],
        section_id=metadata["section_id"],
        section_title=metadata["section_title"],
        breadcrumb=metadata["breadcrumb"],
        pages=metadata["pages"],
        text=text,
        window=window,
        parent_length=len(parent["text"]),
        rrf_score=candidate.rrf_score,
        rerank_score=candidate.rerank_score,
        pinned=candidate.pinned,
        ranks=dict(candidate.ranks),
    )


# =============================================================================
# The pipeline
# =============================================================================

@contextmanager
def timed(timings: dict[str, float], stage: str):
    start = time.perf_counter()
    yield
    timings[stage] = round((time.perf_counter() - start) * 1000, 1)


def retrieve(
    query: str,
    filters: dict | None = None,
    config: RetrievalConfig | None = None,
    resources: RetrievalResources | None = None,
) -> RetrievalResult:
    """Find the parent sections that answer `query`. An empty result means "I don't know".

    filters: any of code_name, edition_year, section_type (a value or a list of values).
    resources: the stores and models; loaded once from store/ when not given (tests pass fakes).
    """
    config = config or RetrievalConfig()
    resources = resources or get_default_resources()
    filters = validate_filters(filters)
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    with timed(timings, "normalize"):
        normalized = normalize_query(query, config)
    where = chroma_where(filters)

    query_embedding = None
    if config.use_child_vector or config.use_section_vector:
        with timed(timings, "embed_query"):
            query_embedding = resources.embed_query(normalized.text)

    child_vector_hits: list[ChildHit] = []
    if config.use_child_vector:
        with timed(timings, "child_vector"):
            child_vector_hits = child_vector_search(
                resources.children_collection, query_embedding, where, config.child_vector_top_k
            )

    bm25_hits: list[ChildHit] = []
    if config.use_bm25:
        with timed(timings, "bm25"):
            bm25_hits = bm25_search(resources.bm25_index, normalized.tokens, filters, config.bm25_top_k)

    section_hits: list[ParentHit] = []
    if config.use_section_vector:
        with timed(timings, "section_vector"):
            section_hits = section_vector_search(
                resources.sections_collection, query_embedding, where, config.section_vector_top_k
            )

    with timed(timings, "fusion"):
        rankings = {}
        if config.use_child_vector:
            rankings["child_vector"] = parents_from_children(child_vector_hits)
        if config.use_bm25:
            rankings["bm25"] = parents_from_children(bm25_hits)
        if config.use_section_vector:
            rankings["section_vector"] = section_hits
        fused = reciprocal_rank_fusion(rankings, config.rrf_k)

        pinned_ids = find_pinned_parents(normalized.section_refs, resources, filters) if config.use_exact_ref_boost else []
        fused = pin_to_top(fused, pinned_ids)

        child_scores = child_rrf_scores([child_vector_hits, bm25_hits], config.rrf_k)
        for candidate in fused:
            candidate.best_child_ids = best_children(candidate.parent_id, child_scores, resources)

    reranked: list[Candidate] = []
    if config.use_rerank:
        with timed(timings, "rerank"):
            reranked = rerank(normalized.text, fused[:config.rerank_top_n], resources, config)
        final = [c for c in reranked if passes_threshold(c, config.rerank_threshold)][:config.final_k]
    else:
        final = fused[:config.final_k]

    with timed(timings, "context"):
        contexts = [build_context(candidate, resources, config) for candidate in final]

    timings["total"] = round((time.perf_counter() - total_start) * 1000, 1)
    debug = RetrievalDebug(
        normalized=normalized,
        child_vector=child_vector_hits,
        bm25=bm25_hits,
        section_vector=section_hits,
        fused=fused,
        reranked=reranked,
        pinned_parent_ids=pinned_ids,
        timings_ms=timings,
    )
    return RetrievalResult(query=query, contexts=contexts, debug=debug)
