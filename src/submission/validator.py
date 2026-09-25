"""Submission validation against the complete official query/corpus registry."""
import logging
from src.utils.io import read_json


class SubmissionValidator:
    SUBMISSION_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "MedicalRAGSubmission", "type": "array",
        "items": {"type": "object", "required": ["id", "relevant_docs", "relevant_chunks"],
                  "properties": {"id": {"type": "string"},
                                 "relevant_docs": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                                 "relevant_chunks": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}},
                  "additionalProperties": False},
    }

    def __init__(self, expected_query_ids=None, doc_ids=None, chunk_to_doc=None, internal_to_official=None):
        self.expected_query_ids = None if expected_query_ids is None else set(expected_query_ids)
        self.doc_ids = None if doc_ids is None else set(doc_ids)
        self.chunk_to_doc = chunk_to_doc
        self.internal_to_official = internal_to_official or {}
        if chunk_to_doc is not None and self.doc_ids is not None:
            if not set(chunk_to_doc.values()) <= self.doc_ids:
                raise ValueError("Official chunk mapping contains unknown parent documents")
            for internal, official in self.internal_to_official.items():
                if official not in chunk_to_doc:
                    raise ValueError(f"Mapping points to unknown official chunk: {official}")
                if internal != official and internal in chunk_to_doc:
                    raise ValueError("Internal and official ID namespaces collide")

    def validate_schema(self, data):
        errors = []
        if not isinstance(data, list):
            return False, ["Submission must be an array"]
        for index, row in enumerate(data):
            if not isinstance(row, dict) or set(row) != {"id", "relevant_docs", "relevant_chunks"}:
                errors.append(f"Record {index}: require exactly id, relevant_docs, relevant_chunks")
                continue
            if not isinstance(row["id"], str) or not row["id"]:
                errors.append(f"Record {index}: invalid query ID")
            for field in ("relevant_docs", "relevant_chunks"):
                values = row[field]
                if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values):
                    errors.append(f"Record {index}: {field} must be a list of nonempty string IDs")
                elif len(values) != len(set(values)):
                    errors.append(f"Record {index}: duplicate IDs in {field}")
        return not errors, errors

    def check_duplicates(self, data):
        errors, seen = [], set()
        for row in data:
            if row["id"] in seen:
                errors.append(f"Duplicate query ID: {row['id']}")
            seen.add(row["id"])
            for field in ("relevant_docs", "relevant_chunks"):
                if len(row[field]) != len(set(row[field])):
                    errors.append(f"Duplicate IDs in {row['id']}.{field}")
        return bool(errors), errors

    def validate(self, data):
        valid, errors = self.validate_schema(data)
        if not valid:
            return False, errors
        if self.expected_query_ids is None or self.doc_ids is None or self.chunk_to_doc is None:
            return False, ["Full validation requires expected query IDs, document IDs and official chunk_to_doc mapping"]
        _, errors = self.check_duplicates(data)
        present = {row["id"] for row in data}
        if present != self.expected_query_ids:
            errors.append(f"Query coverage mismatch: missing={sorted(self.expected_query_ids - present)}, unexpected={sorted(present - self.expected_query_ids)}")
        for row in data:
            unknown_docs = set(row["relevant_docs"]) - self.doc_ids
            unknown_chunks = set(row["relevant_chunks"]) - set(self.chunk_to_doc)
            internal = {item for item in row["relevant_chunks"]
                        if item in self.internal_to_official and self.internal_to_official[item] != item}
            if unknown_docs:
                errors.append(f"{row['id']}: unknown document IDs {sorted(unknown_docs)}")
            if unknown_chunks or internal:
                errors.append(f"{row['id']}: unknown/internal chunk IDs {sorted(unknown_chunks | internal)}")
        return not errors, errors

    def validate_file(self, file_path):
        try:
            return self.validate(read_json(file_path))
        except (OSError, ValueError) as error:
            return False, [str(error)]

    def validate_and_report(self, file_path):
        valid, errors = self.validate_file(file_path)
        if not valid:
            raise ValueError("; ".join(errors))
        logging.getLogger(__name__).info("Submission validated: %s", file_path)
