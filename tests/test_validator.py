"""Compatibility and malformed-schema tests for the historical validator path."""
import pytest
from src.utils.validator import SubmissionValidator


class TestSubmissionValidator:
    def test_valid_submission(self):
        validator = SubmissionValidator(["q"], ["d"], {"c": "d"})
        assert validator.validate([{"id": "q", "relevant_docs": ["d"], "relevant_chunks": ["c"]}])[0]

    def test_invalid_missing_field(self):
        assert not SubmissionValidator().validate_schema([{"id": "q"}])[0]

    def test_duplicate_ids(self):
        validator = SubmissionValidator()
        has_duplicates, errors = validator.check_duplicates([
            {"id": "q", "relevant_docs": ["d", "d"], "relevant_chunks": []}])
        assert has_duplicates and errors

    def test_empty_arrays_valid(self):
        assert SubmissionValidator().validate_schema([
            {"id": "q", "relevant_docs": [], "relevant_chunks": []}])[0]

    @pytest.mark.parametrize("data", [None, {}, [None], [{"id": "q", "relevant_docs": None, "relevant_chunks": []}]])
    def test_malformed_schema(self, data):
        assert not SubmissionValidator().validate_schema(data)[0]

    def test_large_file_validation(self, tmp_path):
        from src.utils.io import write_json
        queries = [f"q{i}" for i in range(2000)]
        data = [{"id": q, "relevant_docs": [], "relevant_chunks": []} for q in queries]
        path = tmp_path / "large.json"
        write_json(path, data)
        assert SubmissionValidator(queries, [], {}).validate_file(path)[0]
