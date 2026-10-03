# Medical graph schema (P3-08)

**Status:** schema defined and validated in code (`src/graph/schema.py`). Entity
extraction (P3-09) and normalization (P3-10) are implemented; the graph store and
`GraphRetriever` (P3-11/P3-12) target **Neo4j** (see `notes/22_09_26.txt`).

The graph is derived from the P1 corpus and the P3 hierarchy mapping. It never
invents official IDs and never rewrites P1/P2 artifacts.

## Nodes

| Node | Key (`node_id`) | Main properties |
| --- | --- | --- |
| `Document` | official `doc_id` | `title`, `metadata`, `language` |
| `Chunk` | official `chunk_id` | `doc_id`, `text`, `rank`/`metadata` |
| `Disease` | canonical entity ID | `canonical_name`, `aliases`, `source` |
| `Drug` | canonical entity ID | `canonical_name`, `aliases`, `source` |
| `Symptom` | canonical entity ID | `canonical_name`, `aliases`, `source` |
| `Treatment` | canonical entity ID | `canonical_name`, `aliases`, `source` |

`Document`/`Chunk` IDs come from the official corpus. `Disease`/`Drug`/`Symptom`/
`Treatment` IDs are canonical entity IDs from the lexicon; the original surface
form is kept in `aliases`.

## Edges

| Edge | Source → Target | Required evidence |
| --- | --- | --- |
| `HAS_CHUNK` | Document → Chunk | none (parent mapping from P3-01) |
| `MENTIONS` | Chunk → Disease/Drug/Symptom/Treatment | `provenance.chunk_id` |
| `TREATS` | Drug/Treatment → Disease | explicit cue text (`evidence`) + `provenance.chunk_id` |
| `HAS_SYMPTOM` | Disease → Symptom | explicit cue text (`evidence`) + `provenance.chunk_id` |

Rules enforced by `validate_edge`:

1. `source`, `target`, `edge_type` are required and `source != target`.
2. `MENTIONS`, `TREATS`, `HAS_SYMPTOM` require `provenance.chunk_id`.
3. **`TREATS` and `HAS_SYMPTOM` require a non-empty `evidence` string.** A clinical
   relation is created **only** when an explicit cue (e.g. "điều trị", "gây")
   links two mentions within a bounded window — never from co-occurrence alone.
4. `HAS_CHUNK` connects an existing document to one of its chunks.

## Provenance

Every edge stores the `chunk_id` (and `doc_id` when known) that generated it.
Entity mentions store `chunk_id`, `doc_id`, `start`, `end`, `surface`, `type`,
`canonical_candidate`, `confidence` (`canonical`=1.0, `alias`=0.9,
`abbreviation`=0.7) and the match kind, so any graph fact traces back to a
corpus span.

## Extraction and normalization (P3-09 / P3-10)

```bash
python scripts/extract_entities.py --chunks <chunks.json> --output-dir outputs/p3_entities
python scripts/normalize_entities.py --entities outputs/p3_entities/entities.jsonl \
  --output-dir outputs/p3_normalized_entities
```

`configs/graph/lexicon_seed.yaml` is a **minimal seed lexicon**, not a medical
ontology. Replace or extend it with an official dictionary before drawing medical
conclusions. Normalization keeps the original surface form and adds
`canonical_id`/`canonical_name`/`canonical_type`, reducing duplicate nodes.

## Limitations

- Extraction quality is bounded by the lexicon; the seed lexicon is illustrative.
- Abbreviations can be ambiguous ("MI"); the lexicon is the only disambiguation.
- The Neo4j store, constraints and `GraphRetriever` are P3-11/P3-12; nothing here
  requires a running database.

