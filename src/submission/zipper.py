"""A submission ZIP contains exactly one JSON file at archive root."""
import json
from pathlib import Path
import zipfile


def validate_zip(path, validator=None):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != 1 or "/" in names[0] or "\\" in names[0]
                or not names[0].endswith(".json") or archive.testzip() is not None):
            raise ValueError("ZIP must contain exactly one valid JSON at root")
        data = json.loads(archive.read(names[0]))
    if validator is not None:
        valid, errors = validator.validate(data)
        if not valid:
            raise ValueError("; ".join(errors))
    return data


def validate_json(file_path, validator=None):
    if validator is None:
        raise ValueError("Corpus-aware validator is required")
    return validator.validate_file(file_path)[0]


def pack_submission(json_path, output_zip_path, validator=None):
    if validator is None:
        raise ValueError("Corpus-aware validator is required")
    valid, errors = validator.validate_file(json_path)
    if not valid:
        raise ValueError("; ".join(errors))
    source, output = Path(json_path), Path(output_zip_path)
    if source.suffix != ".json":
        raise ValueError("Submission source must be a JSON file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(source, arcname=source.name)
    validate_zip(output, validator)
    return str(output.resolve())
