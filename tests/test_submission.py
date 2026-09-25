import copy
import json
import zipfile
import pytest
from src.submission.validator import SubmissionValidator
from src.submission.generator import generate_submission
from src.submission.zipper import pack_submission, validate_zip
from src.utils.io import write_json


@pytest.fixture
def validator():
    return SubmissionValidator(["q1", "q2"], ["d1"], {"c1": "d1"}, {"internal1": "c1", "c1": "c1"})


@pytest.fixture
def valid():
    return [{"id": "q1", "relevant_docs": ["d1"], "relevant_chunks": ["c1"]},
            {"id": "q2", "relevant_docs": [], "relevant_chunks": []}]


def test_valid_empty_lists(valid, validator):
    assert validator.validate(valid) == (True, [])


@pytest.mark.parametrize("mutation", ["missing_field", "duplicate_query", "duplicate_doc", "duplicate_chunk",
                                     "unknown_doc", "unknown_chunk", "internal", "missing_query", "extra_query"])
def test_reject_invalid(valid, validator, mutation):
    data = copy.deepcopy(valid)
    if mutation == "missing_field":
        del data[0]["relevant_docs"]
    elif mutation == "duplicate_query":
        data.append(data[0])
    elif mutation == "duplicate_doc":
        data[0]["relevant_docs"] *= 2
    elif mutation == "duplicate_chunk":
        data[0]["relevant_chunks"] *= 2
    elif mutation == "unknown_doc":
        data[0]["relevant_docs"] = ["missing"]
    elif mutation == "unknown_chunk":
        data[0]["relevant_chunks"] = ["missing"]
    elif mutation == "internal":
        data[0]["relevant_chunks"] = ["internal1"]
    elif mutation == "missing_query":
        data.pop()
    else:
        data.append({"id": "q3", "relevant_docs": [], "relevant_chunks": []})
    assert not validator.validate(data)[0]


def test_requires_official_registry(valid):
    validator = SubmissionValidator()
    assert validator.validate_schema(valid)[0]
    assert not validator.validate(valid)[0]


def test_convert_internal_ids(valid, validator):
    valid[0]["relevant_chunks"] = ["internal1", "c1"]
    converted = generate_submission(valid, validator, validator.internal_to_official)
    assert converted[0]["relevant_chunks"] == ["c1"]


def test_zip_root_and_exclusive_write(tmp_path, valid, validator):
    source, target = tmp_path / "result.json", tmp_path / "submission.zip"
    write_json(source, valid)
    pack_submission(source, target, validator)
    assert validate_zip(target, validator) == valid
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == ["result.json"]
    with pytest.raises(FileExistsError):
        pack_submission(source, target, validator)


@pytest.mark.parametrize("names", [["folder/result.json"], ["a.json", "b.json"], ["a.txt"], ["folder\\result.json"]])
def test_zip_bad_structure(tmp_path, names):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "x") as archive:
        for name in names:
            archive.writestr(name, "[]")
    with pytest.raises(ValueError):
        validate_zip(path)


def test_invalid_submission_never_packed(tmp_path, valid, validator):
    valid.pop()
    source, target = tmp_path / "bad.json", tmp_path / "bad.zip"
    write_json(source, valid)
    with pytest.raises(ValueError):
        pack_submission(source, target, validator)
    assert not target.exists()
