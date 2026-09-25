"""Translate internal retrieval IDs to official IDs before submission."""
from src.utils.io import write_json


def generate_submission(predictions, validator, internal_to_official=None):
    mapping = internal_to_official or {}
    output = []
    for row in predictions:
        official = []
        for chunk_id in row["relevant_chunks"]:
            if chunk_id in mapping:
                chunk_id = mapping[chunk_id]
            elif chunk_id not in (validator.chunk_to_doc or {}):
                raise ValueError(f"No official mapping for chunk: {chunk_id}")
            official.append(chunk_id)
        output.append({"id": row["id"], "relevant_docs": list(dict.fromkeys(row["relevant_docs"])),
                       "relevant_chunks": list(dict.fromkeys(official))})
    valid, errors = validator.validate(output)
    if not valid:
        raise ValueError("; ".join(errors))
    return output


def save_submission(path, predictions, validator, internal_to_official=None):
    output = generate_submission(predictions, validator, internal_to_official)
    write_json(path, output)
    return output
