import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from src.data.kaggle_crawl import SOURCE_FILES, PACKAGE_FILES, build_notebook, prepare_bundle

PROJECT = Path(__file__).resolve().parents[1]


def test_notebook_is_private_cpu_and_bundles_current_crawler_without_secrets():
    nb, metadata = build_notebook(PROJECT, owner="testowner", sample_per_host=10)
    assert metadata["is_private"] is True and metadata["enable_gpu"] is False and metadata["enable_tpu"] is False
    assert metadata["enable_internet"] is True
    config_tree = ast.parse(nb["cells"][3]["source"])
    dataset_metadata_call = next(n.value for n in config_tree.body if isinstance(n, ast.Assign)
                                 and n.targets[0].id == "DATASET_METADATA")
    dataset_metadata = json.loads(ast.literal_eval(dataset_metadata_call.args[0]))
    dataset_title = dataset_metadata["title"]
    # Kaggle rejects dataset titles already used by the crawler notebook.
    assert dataset_title != metadata["title"] and len(dataset_title) <= 50
    assert dataset_metadata["id"] != metadata["id"]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    tree = ast.parse(nb["cells"][2]["source"])
    payload_call = next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "payload")
    payload = base64.b64decode(ast.literal_eval(payload_call.args[0]))
    assert hashlib.sha256(payload).hexdigest() == nb["metadata"]["vibiomir_tool_sha256"]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == set(SOURCE_FILES + PACKAGE_FILES)
        for name in SOURCE_FILES:
            assert archive.read(name) == (PROJECT / name).read_bytes()
        for name in PACKAGE_FILES:
            assert archive.read(name) == b""


def test_embedded_parquet_export_streams_multiple_batches_with_nullable_metadata(tmp_path):
    nb, _ = build_notebook(PROJECT, owner="testowner")
    tree = ast.parse(nb["cells"][7]["source"])
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    namespace = {"pa": pa, "pq": pq, "json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "parquet_cell", "exec"), namespace)
    source, target = tmp_path / "documents.jsonl", tmp_path / "documents.parquet"
    rows = [{"doc_id": str(i), "official_id": i, "url": "https://example.org/a", "title": "", "text": "中文 y khoa",
             "metadata": {"language": None if i < 500 else "zh"}} for i in range(600)]
    source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    schema = pa.schema([(n, pa.int64() if n == "official_id" else pa.string()) for n in
                        ("doc_id", "official_id", "url", "title", "text", "metadata_json")])
    namespace["jsonl_to_parquet"](source, target, schema)
    table = pq.read_table(target)
    assert len(table) == 600
    assert json.loads(table["metadata_json"][550].as_py())["language"] == "zh"
    assert table["text"][0].as_py() == "中文 y khoa"


@pytest.mark.parametrize("owner,slug", [("../escape", "valid-name"), ("owner", "../escape")])
def test_invalid_account_path_rejected(owner, slug):
    with pytest.raises(ValueError, match="owner or slug"):
        build_notebook(PROJECT, owner=owner, slug=slug)


def test_bundle_never_overwrites_existing_notebook(tmp_path):
    prepare_bundle(PROJECT, tmp_path, owner="testowner")
    with pytest.raises(FileExistsError):
        prepare_bundle(PROJECT, tmp_path, owner="testowner")
