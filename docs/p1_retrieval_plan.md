# Mai Ngọc Duy — P1-01 through P1-11

Source: [assignment spreadsheet](https://docs.google.com/spreadsheets/d/1TVLcExMp1hzgH6uq4aMBZA1XiqY4hbJ0fmDD0_Rylj0/edit).
The Checklist tab explicitly assigns P1-01…P1-17 to Mai Ngọc Duy. The overview's
“MMD” label beside Person 2 is inconsistent; the Checklist and user clarification
take precedence. No remote sheet is edited.

| Task | Scope | Repository work |
| --- | --- | --- |
| P1-01 | Query/document/chunk schema; triplet adapter; source IDs | src/data/prototype.py + schema definitions |
| P1-02 | Query-positive dedup; merge negatives; statistics | prototype adapter and dedup report |
| P1-03 | Seeded split by document/title | src/data/split.py; shared-positive/query components kept together |
| P1-04 | Chunk corpus with parent/title/text/metadata | prototype outputs and streamed MMedC importer |
| P1-05 | BM25 Top-K and retrieval cache/benchmark | existing BM25 + retrieval cache + benchmark runner |
| P1-06 | BGE-M3 vectors/index, cosine/dot | model-aware prefixes/normalization; configurable Qdrant distance |
| P1-07 | Multilingual E5 comparison | E5 config; same corpus/query fingerprint, recall/latency/memory report |
| P1-08 | Union with source rank/score provenance | fusion union keeps all named source evidence |
| P1-09 | RRF k sweep; stable ranking and union comparison | benchmark run variants and comparable recall reports |
| P1-10 | Chunk/doc Recall@20/50/100/200 + misses | evaluate_retrieval.py; per-query and aggregate reports |
| P1-11 | Fixed candidates schema for Person 2 | src/retrieval/schema.py; portable export + validation in src/pipeline/handoff.py |

P1-11 was requested separately after P1-01…P1-10. Its contract, receiving
instructions and weak-label limitations are in [p1_p2_handoff.md](p1_p2_handoff.md).
Person 2's P2-01 model inference remains separate work.

The requested data source is Henrychur/MMedC, limited to Chinese, English,
Japanese and French. Its dataset card identifies a pretraining TXT corpus, not
query/positive/negative supervision. Do not fabricate official retrieval qrels.
Triplet adaptation is implemented for genuine triplet inputs. Corpus preparation
uses an explicitly user-approved proxy: first 180 characters as query, next
600-character span as weak positive. Query characters are excluded from every
candidate chunk. These labels are not human-verified relevance or competition
scores; adjacent chunks and same-document chunks may be false negatives.
The triplet adapter remains available separately; MMedC does not provide real
anchor/positive/negative rows.

Original raw data and existing Chinese.zip.fdmdownload must be preserved.
New downloads are pinned, resumable, size/SHA256 verified and ignored by Git.
New prototype IDs are explicitly namespaced; they are not organizer official IDs.
Models and full datasets remain outside Git.
