# Architecture

Pipeline: **data → retrieval → reranking → scoring → evaluation → submission**.

## Boundaries

| Package | Responsibility | Must not do |
| --- | --- | --- |
| src/data | Canonical loading, source field adaptation, identity checks, mappings, index writing | Modify raw input, invent official IDs |
| src/retrieval | BM25, BGE/Qdrant, candidate union, RRF/CC, optional query variants | Rerank pairs or choose final thresholds |
| src/reranking | Score query/text pairs with BGE cross-encoder | Threshold or fallback |
| src/scoring | Independent chunk/document selection | Load models |
| src/evaluation | Set metrics, ranking metrics, macro per query | Infer competition scoring rules |
| src/submission | Official-ID conversion, corpus validation, single-file ZIP | Accept unverified internal IDs |
| src/pipeline | Connect stages, config/run artifacts, CLI dispatch | Implement retrieval or metric algorithms |
| scripts | Import and invoke CLI | Business logic |

The original implementation consisted of interfaces raising NotImplementedError.
The new runnable baseline adds BM25Okapi (rank-bm25), Qdrant adapters and
cross-encoder adapters using the originally selected BGE models. No working
algorithm or preprocessing implementation was replaced.

## Data contracts

Documents require string doc_id. Chunks require chunk_id, doc_id, text.
Queries require id, text. Additional metadata is retained. When chunk_id is an
internal rechunk ID, official_chunk_id must identify its official parent chunk.
An official chunk may have several internal chunks but only one document.
The authoritative official IDs must come from the organizer's corpus; no regex
can establish their provenance.

Preparation preserves text without normalization, validates unique IDs
and parent documents, then writes canonical JSON and two mandatory mappings.
JSON, JSONL and optionally Parquet inputs are supported. Preparation does not
silently segment, normalize, deduplicate, rechunk or split the dataset.
Optional data.splits.<name>.data_path inputs are checked for document leakage.

A retriever returns dictionaries containing chunk_id, doc_id, score (float),
rank (one-based integer), source; text and metadata are retained. BM25 index
snapshots store corpus plus tokenizer/algorithm parameters in JSON, rebuilding
the in-memory BM25 index on load. No historical pickle implementation existed.

Dense indexes store UUID Qdrant point IDs only as storage keys; official/internal
chunk IDs remain intact in payload. Metadata binds an index to its model,
dimension, text field, maximum sequence length and corpus fingerprint.

## Candidate generation

CandidateGenerator receives named BaseRetriever instances and independent top_k
limits. An optional query_expander callable supplies extra query strings. Query
variants fuse within a source, then sources fuse together. No model-based query
expander is implemented or enabled in the shipped YAML configs.

RRF sums 1 / (rrf_k + rank) for each source; raw BM25 and cosine scores are never
compared. Repeated chunk IDs in one source count once. Conflicting document IDs
for the same chunk are rejected. CC min-max normalizes each source before applying
alpha * dense + (1-alpha) * BM25. Constant-score lists normalize to zero.
Union mode retains first appearance and is order-based, not score-comparable.

## Reranking and selection

The reranker retains retrieval score and adds rerank_score. The pipeline sends
all candidates to reranking and does not truncate there. The legacy top_k argument
remains available for old API callers, while the canonical default returns all.
Scores are the cross-encoder backend's output, not guaranteed calibrated
probabilities. Thresholds are null in baseline configs until calibrated on val.

Before selection, internal rechunks collapse to official chunks using their
highest score. This prevents duplicate mappings from consuming the output limit
or reducing fallback coverage.

Chunk branch: threshold → fill to fallback minimum → cap at chunk_max.
Document branch independently sees all scored official chunks:
group → aggregate → document threshold → fallback → doc_max.

Aggregations:
- max (legacy alias max_p): maximum chunk score.
- mean_top_k (legacy alias top_k_mean): mean of up to K best chunks.
- weighted: weight * max + (1-weight) * mean_top_k.

No requirement is imposed that every selected document has a selected chunk:
the branches have independent thresholds as specified.

## Evaluation

Precision/Recall/F1/F2 use ID sets per query. F2 = 5*TP/(4*|truth|+|prediction|).
Macro is the arithmetic mean of per-query metrics, computed separately for
documents and chunks; it is not F2 of averaged precision and recall.
Empty denominators use configurable zero_division, default 0.
Precision@K divides by K (including short result lists). Recall@K divides by the
number of relevant IDs. Ranking duplicates are removed while preserving order.
Evaluation requires identical query coverage in predictions and ground truth.

These are explicit local conventions. Confirm them against the actual
competition rules before treating the metric as an official leaderboard score.

## State and reproducibility

Paths ending in _path or _dir resolve relative to the YAML file declaring them.
extends supports recursive deep merge and rejects cycles. Run snapshots contain
absolute resolved paths so later stage commands do not depend on working directory.
Each run records config.yaml, registry.json, queries.json, run.log and stage outputs.
The registry snapshot lets packaging validate without re-reading a changed corpus.
Preparation, indexes, runs, result files and ZIP archives use exclusive creation.
A failed stage may leave diagnostic partial artifacts; preserve it and retry with
a new run_name/index path. No automatic cleanup or overwrite is performed.

All model imports are lazy. Pure CPU BM25, metrics and submission do not load
torch, Qdrant or generation dependencies. Install the package in editable mode;
there is no sys.path manipulation in scripts.

## Retained extensions

config/Settings and prompts retain their original values and environment behavior.
Legacy data/bm25_index and data/qdrant_db remain untouched; YAML uses artifacts/.
src/generation, src/multilingual, preprocessing/context-window interfaces and GPU
monitoring remain unfinished. See migration.md and graph_schema.md.
