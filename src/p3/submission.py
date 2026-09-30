"""P3-06: build the final submission JSON from independent chunk/document branches.

IDs are sorted ascending and deduplicated so the artifact is deterministic. The
read-only P1/P2 ``SubmissionValidator`` is reused whenever the official registry
is known. Output uses exclusive creation.
"""
from src.submission.validator import SubmissionValidator
from src.utils.io import write_json

VALIDATOR_KEYS = ("expected_query_ids", "doc_ids", "chunk_to_doc", "internal_to_official")


def validator_from_registry(registry):
    """Build the corpus-aware validator from a P2 handoff ``registry.json`` payload."""
    if not isinstance(registry, dict):
        raise ValueError("Registry must be an object")
    return SubmissionValidator(**{key: registry.get(key) for key in VALIDATOR_KEYS})


def _normalize(values):
    if not isinstance(values, list):
        raise ValueError("Prediction lists must be arrays")
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Prediction IDs must be nonempty strings: {values!r}")
    return sorted(set(values))


def normalize_predictions(predictions):
    """Sort + dedup IDs and keep only the three submission fields, per query."""
    output, seen = [], set()
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError(f"Prediction must be an object: {row!r}")
        query_id = row.get("id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"Prediction needs a nonempty id: {row!r}")
        if query_id in seen:
            raise ValueError(f"Duplicate query id: {query_id}")
        seen.add(query_id)
        output.append({"id": query_id,
                       "relevant_docs": _normalize(row.get("relevant_docs", [])),
                       "relevant_chunks": _normalize(row.get("relevant_chunks", []))})
    if not output:
        raise ValueError("Submission must contain at least one query")
    return output


def build_submission(predictions, validator=None):
    """Normalize then (optionally) validate; never returns an invalid submission."""
    normalized = normalize_predictions(predictions)
    if validator is not None:
        valid, errors = validator.validate(normalized)
        if not valid:
            raise ValueError("Invalid submission: " + "; ".join(errors))
    return normalized


def write_submission(path, predictions, validator=None):
    """Write ``submission.json`` with IDs sorted/dedup (exclusive creation)."""
    normalized = build_submission(predictions, validator)
    write_json(path, normalized)
    return normalized
