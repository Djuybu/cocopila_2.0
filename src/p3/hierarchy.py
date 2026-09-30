"""P3-01: stable chunk -> parent-document hierarchy.

The mapping is built from the P1 corpus (authoritative) and must reconcile with
the P2 handoff registry; any contradiction fails. P1/P2 modules are imported
read-only and their artifacts are never modified. Outputs use exclusive writes
so a previous hierarchy is never overwritten.
"""
from pathlib import Path

from src.data.loader import load_records
from src.utils.io import read_json, write_json

CHUNK_TO_DOC_FILE = "chunk_to_doc.json"
INTERNAL_TO_OFFICIAL_FILE = "internal_to_official.json"
HIERARCHY_REPORT_FILE = "hierarchy_report.json"


def _document_ids(documents):
    if not isinstance(documents, list):
        raise ValueError("documents must be a list")
    doc_ids = set()
    for row in documents:
        if not isinstance(row, dict):
            raise ValueError(f"Document must be an object: {row!r}")
        doc_id = row.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"Missing/non-string doc_id: {row!r}")
        if doc_id in doc_ids:
            raise ValueError(f"Duplicate doc_id: {doc_id}")
        doc_ids.add(doc_id)
    if not doc_ids:
        raise ValueError("The corpus must contain at least one document")
    return doc_ids


def build_hierarchy(documents, chunks):
    """Build the official ``chunk_id -> doc_id`` mapping and its provenance.

    Enforced P3-01 invariants:
    - every chunk has exactly one existing parent document (no orphan chunk);
    - one chunk ID never belongs to more than one document;
    - internal rechunk IDs are unique and do not collide with an official ID.

    Returns a dict with ``chunk_to_doc`` (official namespace),
    ``internal_to_official`` and a ``report`` of counts.
    """
    if not isinstance(chunks, list):
        raise ValueError("chunks must be a list")
    doc_ids = _document_ids(documents)
    chunk_to_doc, internal_to_official = {}, {}
    seen_internal = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk must be an object: {chunk!r}")
        chunk_id, doc_id = chunk.get("chunk_id"), chunk.get("doc_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"Missing/non-string chunk_id: {chunk!r}")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"Missing/non-string doc_id for chunk: {chunk!r}")
        if doc_id not in doc_ids:
            raise ValueError(f"Orphan chunk {chunk_id}: unknown parent document {doc_id}")
        if chunk_id in seen_internal:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")
        seen_internal.add(chunk_id)
        official = chunk.get("official_chunk_id", chunk_id)
        if not isinstance(official, str) or not official:
            raise ValueError(f"Official chunk ID must be a nonempty string: {chunk!r}")
        known = chunk_to_doc.setdefault(official, doc_id)
        if known != doc_id:
            raise ValueError(f"Chunk {official} belongs to multiple documents: {known}, {doc_id}")
        internal_to_official[chunk_id] = official
    for internal, official in internal_to_official.items():
        if internal in chunk_to_doc and internal != official:
            raise ValueError(f"Internal ID collides with an official chunk ID: {internal}")
    report = {
        "document_count": len(doc_ids),
        "chunk_row_count": len(chunks),
        "official_chunk_count": len(chunk_to_doc),
        "internal_rechunk_count": sum(1 for i, o in internal_to_official.items() if i != o),
        # Always zero when build succeeds; explicit for the completion criteria.
        "orphan_chunk_count": 0,
        "multi_parent_chunk_count": 0,
    }
    return {"chunk_to_doc": chunk_to_doc, "internal_to_official": internal_to_official, "report": report}


def load_registry(handoff_dir):
    """Load the P2 handoff ``registry.json`` (chunk_to_doc + internal mapping)."""
    registry_path = Path(handoff_dir) / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing P2 handoff registry: {registry_path}")
    registry = read_json(registry_path)
    if not isinstance(registry, dict) or not isinstance(registry.get("chunk_to_doc"), dict) or not registry["chunk_to_doc"]:
        raise ValueError(f"registry.json has no usable chunk_to_doc mapping: {registry_path}")
    return registry


def reconcile_with_registry(hierarchy, registry, *, allow_registry_superset=False, strict=True):
    """Compare the corpus hierarchy with the registry; contradiction => failure.

    ``allow_registry_superset`` permits official chunks that exist in the registry
    but were not part of the corpus passed in (subset runs); parent mismatches,
    corpus chunks missing from the registry and internal-ID conflicts always fail
    when ``strict`` is true.
    """
    registry_map = registry.get("chunk_to_doc", registry)
    if not isinstance(registry_map, dict) or not registry_map:
        raise ValueError("registry chunk_to_doc mapping is empty")
    corpus_map, internal = hierarchy["chunk_to_doc"], hierarchy["internal_to_official"]
    missing, mismatched = [], []
    for official, doc_id in corpus_map.items():
        if official not in registry_map:
            missing.append(official)
        elif registry_map[official] != doc_id:
            mismatched.append({"chunk_id": official, "corpus_doc_id": doc_id,
                               "registry_doc_id": registry_map[official]})
    registry_internal = registry.get("internal_to_official", {}) or {}
    internal_conflicts = [
        {"internal_chunk_id": internal_id, "corpus_official_id": official,
         "registry_official_id": registry_internal[internal_id]}
        for internal_id, official in internal.items()
        if internal_id in registry_internal and registry_internal[internal_id] != official
    ]
    registry_only = sorted(set(registry_map) - set(corpus_map))
    report = {
        "corpus_official_chunk_count": len(corpus_map),
        "registry_official_chunk_count": len(registry_map),
        "missing_from_registry": sorted(missing),
        "parent_mismatches": mismatched,
        "internal_id_conflicts": internal_conflicts,
        "registry_only_chunk_ids": registry_only,
        "allow_registry_superset": bool(allow_registry_superset),
        "ok": not missing and not mismatched and not internal_conflicts
              and (allow_registry_superset or not registry_only),
    }
    if strict and not report["ok"]:
        problems = []
        if missing:
            problems.append(f"{len(missing)} corpus chunks missing from registry")
        if mismatched:
            problems.append(f"{len(mismatched)} parent mismatches")
        if internal_conflicts:
            problems.append(f"{len(internal_conflicts)} internal-ID conflicts")
        if registry_only and not allow_registry_superset:
            problems.append(f"{len(registry_only)} registry-only chunks not in the corpus")
        raise ValueError("Corpus hierarchy does not match P2 registry: " + "; ".join(problems))
    return report


def write_hierarchy(output_dir, hierarchy, reconciliation=None):
    """Write the mapping, provenance and report with exclusive creation."""
    output = Path(output_dir)
    targets = [output / CHUNK_TO_DOC_FILE, output / INTERNAL_TO_OFFICIAL_FILE, output / HIERARCHY_REPORT_FILE]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Hierarchy output already exists; choose a new directory: " + ", ".join(existing))
    write_json(output / CHUNK_TO_DOC_FILE, hierarchy["chunk_to_doc"])
    write_json(output / INTERNAL_TO_OFFICIAL_FILE, hierarchy["internal_to_official"])
    report = dict(hierarchy["report"])
    if reconciliation is not None:
        report["registry_reconciliation"] = reconciliation
    write_json(output / HIERARCHY_REPORT_FILE, report)
    return {"chunk_to_doc": str(output / CHUNK_TO_DOC_FILE),
            "internal_to_official": str(output / INTERNAL_TO_OFFICIAL_FILE),
            "report": str(output / HIERARCHY_REPORT_FILE)}


def run_hierarchy(documents_path, chunks_path, output_dir, handoff_dir=None, *,
                  allow_registry_superset=False, strict=True):
    """Build from the corpus, reconcile with the handoff registry, then write."""
    hierarchy = build_hierarchy(load_records(documents_path), load_records(chunks_path))
    reconciliation = None
    if handoff_dir is not None:
        reconciliation = reconcile_with_registry(
            hierarchy, load_registry(handoff_dir),
            allow_registry_superset=allow_registry_superset, strict=strict)
    return write_hierarchy(output_dir, hierarchy, reconciliation)
