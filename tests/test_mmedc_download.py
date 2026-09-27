"""Downloads keep raw files intact, resume exact offsets and verify content."""
import hashlib
import io
import pytest
from src.data import download


def test_parallel_download_resumes_prefix_and_verifies_sha(tmp_path, monkeypatch):
    content = bytes(range(100)) * 8
    requests = []
    class Response(io.BytesIO):
        status = 206
        def __init__(self, start, end):
            super().__init__(content[start:end + 1])
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}
    def respond(request, **kwargs):
        value = request.get_header("Range").removeprefix("bytes=")
        start, end = map(int, value.split("-"))
        requests.append((start, end))
        return Response(start, end)
    monkeypatch.setattr(download, "urlopen", respond)
    partial = tmp_path / "English.zip.part"
    partial.write_bytes(content[:71])
    entry = {"path": "English.zip", "size": len(content), "lfs": {"oid": hashlib.sha256(content).hexdigest()}}
    download._download_file("https://example.test", tmp_path, entry, 2, range_workers=3, range_bytes=101)
    assert (tmp_path / "English.zip").read_bytes() == content
    assert not partial.exists()
    assert min(start for start, _ in requests) == 71
    count = len(requests)
    download._download_file("https://example.test", tmp_path, entry, 2)
    assert len(requests) == count


def test_existing_corrupt_archive_is_never_replaced(tmp_path):
    existing = tmp_path / "Chinese.zip"
    existing.write_bytes(b"keep this")
    entry = {"path": "Chinese.zip", "size": 9, "lfs": {"oid": "wrong"}}
    with pytest.raises(ValueError, match="Existing archive fails"):
        download._download_file("https://example.test", tmp_path, entry, 2)
    assert existing.read_bytes() == b"keep this"


@pytest.mark.parametrize("languages", [["spanish"], ["english", "English"]])
def test_downloader_rejects_unrequested_or_duplicate_languages(languages):
    with pytest.raises(ValueError, match="Only distinct"):
        download.download_mmedc({"download": {"languages": languages}})
