"""Versioned candidate artifact contract; retriever APIs remain backward compatible."""
import math
from typing import TypedDict

from src.data.schema import unique_ids

SCHEMA_VERSION = "medical-rag-candidates-v1"
REQUIRED_FIELDS = {"query_id", "chunk_id", "doc_id", "text", "rank", "score", "source",
                   "bm25_rank", "bm25_score", "dense_rank", "dense_score", "fused_score",
                   "source_ranks", "source_scores"}


class HandoffCandidate(TypedDict):
    query_id: str
    chunk_id: str
    doc_id: str
    text: str
    rank: int
    score: float
    source: str
    bm25_rank: int | None
    bm25_score: float | None
    dense_rank: int | None
    dense_score: float | None
    fused_score: float | None
    source_ranks: dict[str, int]
    source_scores: dict[str, float]


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite numeric")
    return float(value)


def _rank(value):
    if type(value) is not int or value < 1:
        raise ValueError("Source rank must be a positive integer")
    return value


def standardize_candidates(query_id, candidates, *, method=None):
    """Add explicit nulls for missing sources, never invent ranks or rewrite IDs/text."""
    rows = []
    for position, candidate in enumerate(candidates, 1):
        if candidate.get("query_id", query_id) != query_id:
            raise ValueError("Candidate query_id mismatch")
        ranks = dict(candidate.get("source_ranks", {}))
        scores = dict(candidate.get("source_scores", {}))
        source = candidate["source"]
        if not ranks and not scores and source not in {"rrf", "cc", "union"}:
            ranks[source] = candidate["rank"]
            scores[source] = candidate["score"]
        for name in ("bm25", "dense"):
            rank, score = candidate.get(f"{name}_rank"), candidate.get(f"{name}_score")
            if rank is not None:
                if name in ranks and ranks[name] != rank:
                    raise ValueError("Conflicting source rank provenance")
                ranks[name] = rank
            if score is not None:
                if name in scores and scores[name] != score:
                    raise ValueError("Conflicting source score provenance")
                scores[name] = score
        if set(ranks) != set(scores):
            raise ValueError("Source rank/score keys disagree")
        ranks = {name: _rank(rank) for name, rank in ranks.items()}
        scores = {name: _number(score, "source score") for name, score in scores.items()}
        fused = candidate.get("fused_score")
        if fused is None and source in {"rrf", "cc"}:
            fused = candidate["score"]
        row = {**candidate, "query_id": query_id, "rank": position,
               "source": "union" if method == "union" else source,
               "score": _number(candidate["score"], "score"),
               "fused_score": None if fused is None else _number(fused, "fused_score"),
               "source_ranks": ranks, "source_scores": scores}
        for name in ("bm25", "dense"):
            row[f"{name}_rank"] = ranks.get(name)
            row[f"{name}_score"] = scores.get(name)
        rows.append(row)
    return rows


def validate_candidate_records(records, queries, registry):
    """Validate coverage and identity without converting internal/prototype IDs."""
    query_ids = unique_ids(queries, "id")
    if unique_ids(records, "id") != query_ids or set(registry["expected_query_ids"]) != query_ids:
        raise ValueError("Candidate/query/registry coverage mismatch")
    if any(not isinstance(q.get("text"), str) or not q["text"].strip() for q in queries):
        raise ValueError("Query text must be nonempty")
    for record in records:
        if not isinstance(record.get("candidates"), list) or any(not isinstance(row, dict) for row in record["candidates"]):
            raise ValueError("Candidates must be a list of objects")
        unique_ids(record["candidates"], "chunk_id")
        for position, row in enumerate(record["candidates"], 1):
            if not REQUIRED_FIELDS <= row.keys():
                raise ValueError("Missing required candidate fields")
            if row["query_id"] != record["id"] or type(row["rank"]) is not int or row["rank"] != position:
                raise ValueError("Candidate query_id/rank mismatch")
            if not isinstance(row["source"], str) or not row["source"]:
                raise ValueError("Candidate source must be nonempty")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("Candidate text must be nonempty")
            if not isinstance(row["doc_id"], str) or not row["doc_id"]:
                raise ValueError("Candidate doc_id must be nonempty")
            official = registry["internal_to_official"].get(row["chunk_id"], row["chunk_id"])
            if official not in registry["chunk_to_doc"] or registry["chunk_to_doc"][official] != row["doc_id"]:
                raise ValueError("Unknown candidate chunk or mismatched parent")
            _number(row["score"], "score")
            if row["fused_score"] is not None:
                _number(row["fused_score"], "fused_score")
            if row["source"] in {"rrf", "cc"} and row["fused_score"] != row["score"]:
                raise ValueError("Fused score must equal the ranking score for RRF/CC")
            ranks, scores = row["source_ranks"], row["source_scores"]
            if not isinstance(ranks, dict) or not isinstance(scores, dict) or set(ranks) != set(scores):
                raise ValueError("Source rank/score keys disagree")
            for name in ranks:
                if not isinstance(name, str) or not name:
                    raise ValueError("Source name must be nonempty")
                _rank(ranks[name])
                _number(scores[name], "source score")
            for name in ("bm25", "dense"):
                if row[f"{name}_rank"] is not None:
                    _rank(row[f"{name}_rank"])
                if row[f"{name}_score"] is not None:
                    _number(row[f"{name}_score"], "source score")
                if row[f"{name}_rank"] != ranks.get(name) or row[f"{name}_score"] != scores.get(name):
                    raise ValueError("Flat and dictionary source provenance disagree")
