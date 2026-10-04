# P3-13 — Label hierarchy report

**Status:** methodology implemented; the decision is **pending official labels (P1-12)**.
No hierarchy rule is hard-coded into the pipeline.

## Question

The official labels contain both `relevant_docs` and `relevant_chunks`. Before any
hierarchical constraint can be applied (P3-05 policy, P3-14 aggregation) we must know:

1. Does every `relevant_chunk` imply its parent document is in `relevant_docs`? (chunk ⇒ doc)
2. Does every `relevant_doc` have at least one labeled `relevant_chunk`? (doc ⇒ chunk)

## How to run

```bash
python scripts/audit_label_hierarchy.py \
  --labels <labels.json> \
  --handoff-dir <p2_to_p3_handoff> \
  --output-dir outputs/p3_label_audit
```

The command writes `label_hierarchy_report.md`, `label_hierarchy_report.csv` and
`label_hierarchy_report.json` to the output directory (exclusive creation). A
`chunk_to_doc` mapping is required: pass `--chunk-to-doc`, `--registry`, or a
`--handoff-dir` containing `registry.json`.

## Decision rule (implemented)

| chunk ⇒ doc | doc ⇒ chunk | Recommended policy |
| --- | --- | --- |
| no | any | `add_parent_docs` — never prune chunks by selected docs |
| yes | no | `prune_chunks` — safe to drop chunks without a selected parent doc |
| yes | yes | `report_only` — either constraint is safe; wait for the competition rule |
| unknown (no labeled chunks) | unknown | `pending_official_labels` |

These are local recommendations; confirm against the competition rules before
turning any constraint on. Until then `check_doc_chunk_consistency.py` stays
report-only by default.
