# Working notes outline (P3-17)

**Status:** skeletal structure committed; numbers are filled by
`scripts/export_working_notes.py` from real artifacts. Placeholders must not be
reported as results.

## How to use

```bash
python scripts/export_working_notes.py \
  --p2-run-dir <p2 run with selector_cv_report.json and reranker_benchmark.json> \
  --p3-doc-dir <P3-14 output with doc_pipeline_report.json> \
  --submission-log submissions/submission_log.csv \
  --output-dir outputs/p3_working_notes
```

The command writes `working_notes_outline.md` (filled) and `metrics_summary.json`
into the output directory (exclusive creation). Artifacts that are absent are
marked `n/a (pending official run)`.

## Required sections

1. Problem and constraints
2. System architecture
3. Data preparation and audit (P1)
4. Retrieval and fusion (P1)
5. Reranking and chunk selection (P2)
6. Document aggregation and submission (P3)
7. Metrics and evaluation protocol
8. Error analysis
9. Ablations
10. Reproducibility
11. Member contributions
12. Limitations and pending work

## Required figures

- Figure 1 — End-to-end pipeline (P1 retrieval → P2 reranking/selection → P3 aggregation/submission).
- Figure 2 — Candidate Recall@K before reranking (chunk and document).
- Figure 3 — Macro Chunk F2 vs threshold (plateau / confidence interval).
- Figure 4 — Document aggregation ablation (aggregation × direct-doc).
- Figure 5 — Error composition: P1 retrieval miss vs P2 pruning vs P3 aggregation.

## Required tables

- Table 1 — Dataset audit (queries/documents/chunks, language, duplicates).
- Table 2 — Retriever comparison (BM25 / BGE-M3 / E5 / fusion) with Recall@K and latency.
- Table 3 — Reranker comparison and out-of-fold chunk selector score.
- Table 4 — Document aggregation + direct-doc ablation with holdout F2_doc.
- Table 5 — Graph ablation (graph only vs graph + hybrid).
- Table 6 — Submission log (run IDs, commits, local/public/private scores).

## Member contributions

| Member | Scope | Contribution |
| --- | --- | --- |
| P1 — Mai Ngọc Duy | Data & retrieval | Adapter, dedup, split, corpus/index, BM25, BGE-M3 dense, E5 comparison, union/fusion, candidate recall, P1→P2 handoff |
| P2 — Mạc Duy | Reranking & scoring | Reranker inference, chunk F2 evaluation, threshold/fallback/cap sweeps, training pairs, hard negatives, fine-tuning, CV, FN analysis, P2→P3 handoff |
| P3 — Quế | Graph, aggregation & submission | Hierarchy, doc aggregation/ablation, doc selector, consistency, submission generator/validator/ZIP, graph schema/NER/store/GraphRetriever, label audit, official aggregation/submission, submission log |

## Rules

- Every number must trace to a listed source artifact; unverified results stay `n/a`.
- Report out-of-fold/holdout numbers for generalisation, not in-sample tuning scores.
- Confirm the official metric against the competition rules before calling any score final.
