"""Build a self-contained Kaggle crawler notebook from the current workspace."""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import textwrap
import zipfile

from src.utils.config import load_config
from src.utils.io import write_json

SOURCE_FILES = (
    "src/data/download.py", "src/data/vibiomir.py", "src/data/web_corpus.py",
    "src/data/rag_dataset.py", "src/data/browser_crawl.py", "src/data/full_crawl.py", "src/utils/io.py", "src/utils/config.py",
)
PACKAGE_FILES = ("src/__init__.py", "src/data/__init__.py", "src/utils/__init__.py")


def _cell(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build_notebook(project, *, owner, slug="vibiomir-rag-crawl-pilot-20261003", sample_per_host=10, source_files=SOURCE_FILES):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", owner) or not re.fullmatch(r"[a-z0-9-]{3,50}", slug):
        raise ValueError("Invalid Kaggle owner or slug")
    if not 1 <= sample_per_host <= 100:
        raise ValueError("Pilot sample_per_host must be between 1 and 100")
    project = Path(project)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in source_files:
            bundle.writestr(name, (project / name).read_bytes())
        for name in PACKAGE_FILES:
            bundle.writestr(name, b"")
    payload = base64.b64encode(archive.getvalue()).decode()
    digest = hashlib.sha256(archive.getvalue()).hexdigest()
    config = load_config(project / "configs/data/vibiomir_rag.yaml")
    config["data"].update(raw_dir="/kaggle/working/vibiomir_raw", output_dir="/kaggle/working/vibiomir_prepared",
                           report_dir="/kaggle/working/vibiomir_estimate")
    config["rag"]["sample_per_host"] = sample_per_host
    kernel_ref = f"{owner}/{slug}"
    dataset_ref = f"{owner}/{slug[:42]}-dataset"
    notebook_title = slug.replace("-", " ").title()
    dataset_meta = {"id": dataset_ref, "title": notebook_title[:42] + " Dataset",
                    "licenses": [{"name": "other"}],
                    "subtitle": "Crawled multilingual biomedical passages for RAG research",
                    "description": "Private RAG pilot crawled on Kaggle from https://huggingface.co/datasets/AIGuruTinix/ViBioMIR. "
                    "Official document/query IDs are retained; rag:* chunks are local IDs. No relevance labels. "
                    "The source query/URL catalogue states CC BY-NC 4.0; crawled page content remains attributable to its original authors and source terms. "
                    "Files include documents/chunks/queries in JSONL or JSON and Parquet, local mappings, fetch log and size estimate."}
    cells = [
        _cell("markdown", f"""
        # ViBioMIR RAG Crawl Pilot

        Chạy trên Kaggle CPU với **Internet bật**. Notebook nhúng code crawler hiện tại,
        tải hai Parquet đã pin SHA256, lấy tối đa **{sample_per_host} URL/hostname**,
        rồi xuất dataset RAG ở `/kaggle/working/vibiomir_rag_dataset`.

        Đây là pilot phủ domain; không phải crawl toàn bộ 4,4 triệu URL.
        Official document IDs được giữ; chunk IDs `rag:*` dành cho RAG nội bộ.
        Khi Save & Run All hoàn tất, dùng **Output → Create Dataset**, hoặc upload
        thư mục output bằng Kaggle CLI. Dataset tạo từ CLI mặc định private.
        """),
        _cell("code", """
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "numpy>=1.26", "PyYAML>=6", "pyarrow>=16", "requests>=2.32",
                        "trafilatura>=2.1,<3", "pypdf>=5,<7", "protego>=0.6,<1"], check=True)
        """),
        _cell("code", f"""
        import base64
        import hashlib
        import io
        import json
        import logging
        from pathlib import Path
        import shutil
        import zipfile

        TOOL_ROOT = Path("/tmp/vibiomir_kaggle_tool")
        payload = base64.b64decode({payload!r})
        assert hashlib.sha256(payload).hexdigest() == {digest!r}, "Tool payload checksum mismatch"
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name in archive.namelist():
                target = (TOOL_ROOT / name).resolve()
                if not target.is_relative_to(TOOL_ROOT.resolve()):
                    raise ValueError("Unexpected embedded path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        sys.path.insert(0, str(TOOL_ROOT))
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
        TOOL_SHA256 = {digest!r}
        print("Embedded crawler verified:", TOOL_SHA256)
        """),
        _cell("code", f"""
        CONFIG = json.loads({json.dumps(config, ensure_ascii=False)!r})
        STATE_DIR = Path("/kaggle/working/vibiomir_estimate")
        OUTPUT_DIR = Path("/kaggle/working/vibiomir_rag_dataset")
        DATASET_METADATA = json.loads({json.dumps(dataset_meta)!r})
        print("Pilot URLs per hostname:", CONFIG["rag"]["sample_per_host"])
        print("Source revision:", CONFIG["data"]["revision"])
        """),
        _cell("code", """
        from src.data.vibiomir import download_catalogue
        from src.data.rag_dataset import estimate, export
        download_catalogue(CONFIG["data"])
        """),
        _cell("code", """
        from IPython.display import Markdown, display
        report = estimate(CONFIG, STATE_DIR)
        display(Markdown((STATE_DIR / "estimate.md").read_text()))
        print("Actual sample status counts:", report["sample_status_counts"])
        """),
        _cell("code", """
        manifest = export(CONFIG, STATE_DIR, OUTPUT_DIR)
        # Flatten mappings so Kaggle's default directory-skip upload retains them.
        for name in ("chunk_to_doc.json", "internal_to_official_id.json"):
            shutil.copy2(OUTPUT_DIR / "mappings" / name, OUTPUT_DIR / name)
        import yaml
        config_path = OUTPUT_DIR / "rag_config.yaml"
        local_config = yaml.safe_load(config_path.read_text())
        local_config["data"]["mappings_dir"] = "."
        config_path.write_text(yaml.safe_dump(local_config, allow_unicode=True, sort_keys=False))
        for name in ("estimate.json", "estimate.md", "sample_plan.json"):
            shutil.copy2(STATE_DIR / name, OUTPUT_DIR / name)
        print("Exported RAG dataset:", manifest["counts"])
        """),
        _cell("code", """
        import pyarrow as pa
        import pyarrow.parquet as pq

        def jsonl_to_parquet(source, target, schema):
            with pq.ParquetWriter(target, schema, compression="zstd") as writer:
                batch = []
                with source.open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        row["metadata_json"] = json.dumps(row.pop("metadata", {}), ensure_ascii=False)
                        batch.append(row)
                        if len(batch) >= 500:
                            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                            batch = []
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))

        document_schema = pa.schema([(name, pa.int64() if name == "official_id" else pa.string())
                                    for name in ("doc_id", "official_id", "url", "title", "text", "metadata_json")])
        chunk_schema = pa.schema([(name, pa.string()) for name in
                                 ("chunk_id", "doc_id", "text", "title", "url", "metadata_json")])
        jsonl_to_parquet(OUTPUT_DIR / "documents.jsonl", OUTPUT_DIR / "documents.parquet", document_schema)
        jsonl_to_parquet(OUTPUT_DIR / "chunks.jsonl", OUTPUT_DIR / "chunks.parquet", chunk_schema)
        queries = json.loads((OUTPUT_DIR / "queries.json").read_text())
        pq.write_table(pa.Table.from_pylist(queries), OUTPUT_DIR / "queries.parquet", compression="zstd")
        assert pq.ParquetFile(OUTPUT_DIR / "documents.parquet").metadata.num_rows == manifest["counts"]["documents"]
        assert pq.ParquetFile(OUTPUT_DIR / "chunks.parquet").metadata.num_rows == manifest["counts"]["chunks"]
        """),
        _cell("code", """
        from src.utils.io import write_json
        write_json(OUTPUT_DIR / "dataset-metadata.json", DATASET_METADATA)
        write_json(OUTPUT_DIR / "kaggle_crawl_complete.json",
                   {"counts": manifest["counts"], "tool_sha256": TOOL_SHA256,
                    "source_revision": CONFIG["data"]["revision"],
                    "population_urls": report["population_urls"], "sample_urls": report["sample_urls"],
                    "sample_status_counts": report["sample_status_counts"], "execution_platform": "Kaggle"})
        total_bytes = sum(p.stat().st_size for p in OUTPUT_DIR.rglob("*") if p.is_file())
        print("Dataset output:", OUTPUT_DIR)
        print("Output MiB:", round(total_bytes / 2**20, 2))
        print("Save & Run All, then Output -> Create Dataset; upload the files in this folder.")
        display(pa.Table.from_pylist(queries[:3]).to_pandas())
        """),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "vibiomir_tool_sha256": digest}, "nbformat": 4, "nbformat_minor": 5}
    for index, cell in enumerate(cells):
        cell["id"] = f"vibiomir-{index:02d}"
    metadata = {"id": kernel_ref, "title": notebook_title, "code_file": "ViBioMIR_RAG_Kaggle.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    return notebook, metadata


def prepare_bundle(project, output_dir, **kwargs):
    notebook, metadata = build_notebook(project, **kwargs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / metadata["code_file"], notebook)
    write_json(output / "kernel-metadata.json", metadata)
    return output


def build_full_notebook(project, *, owner, slug, shard, config, resume_kernel=None, retry_transient_only=False, catalogue_dataset=None):
    notebook, metadata = build_notebook(project, owner=owner, slug=slug)
    config = json.loads(json.dumps(config))
    config["data"]["raw_dir"] = "/kaggle/working/vibiomir_raw"
    config["execution_platform"] = "Kaggle"
    config["full_crawl"]["retry_transient_only"] = retry_transient_only
    cells = notebook["cells"][:4]
    cells[0] = _cell("markdown", "# ViBioMIR full-catalogue crawler\n\nThis worker processes one exact row range from the complete catalogue. It saves a resumable SQLite checkpoint and Parquet RAG shards. Missing content remains explicit; document labels are official integer IDs.")
    cells[1] = _cell("code", """
        import os, subprocess, sys
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/tmp/vibiomir_chromium"
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "numpy>=1.26", "PyYAML>=6",
                        "pyarrow>=16", "requests>=2.32", "trafilatura>=2.1,<3", "pypdf>=5,<7",
                        "protego>=0.6,<1", "playwright>=1.50,<2"], check=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        """)
    cells[3] = _cell("code", f"""
        CONFIG = json.loads({json.dumps(config, ensure_ascii=False)!r})
        CONFIG["tool_sha256"] = TOOL_SHA256
        CATALOGUE_DATASET = {catalogue_dataset!r}
        SHARD = json.loads({json.dumps(shard)!r})
        STATE_DIR = Path("/kaggle/working/full_state")
        OUTPUT_DIR = Path("/kaggle/working/full_shard")
        RESUME = {bool(resume_kernel)!r}
        if RESUME:
            previous = list(Path("/kaggle/input").rglob("full_state/state.sqlite"))
            if len(previous) != 1:
                raise ValueError(f"Expected one checkpoint input, found {{len(previous)}}")
            shutil.copytree(previous[0].parent, STATE_DIR)
        print("Full-catalogue shard:", SHARD, "resume:", RESUME)
        """)
    cells += [_cell("code", """
        from src.data.vibiomir import download_catalogue
        from src.data.full_crawl import run_shard
        from src.data.download import sha256_file
        if CATALOGUE_DATASET:
            raw = Path(CONFIG["data"]["raw_dir"])
            raw.mkdir(parents=True, exist_ok=True)
            for candidate in Path("/kaggle/input").rglob("links_corpus.parquet"):
                parent = candidate.parent
                if all((parent/name).is_file() and sha256_file(parent/name) == spec["sha256"]
                       for name, spec in CONFIG["data"]["files"].items()):
                    for name in CONFIG["data"]["files"]:
                        shutil.copy2(parent/name, raw/name)
                    break
            else:
                raise ValueError("Pinned official catalogue was not found in Kaggle inputs")
        download_catalogue(CONFIG["data"])
        manifest = run_shard(CONFIG, SHARD, STATE_DIR, OUTPUT_DIR,
                             budget_seconds=CONFIG["full_crawl"]["budget_seconds"],
                             retry_failed=CONFIG["full_crawl"].get("retry_transient_only", False),
                             retry_transient_only=CONFIG["full_crawl"].get("retry_transient_only", False))
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        print("Tool SHA256:", TOOL_SHA256)
        """)]
    for index, cell in enumerate(cells):
        cell["id"] = f"full-vibiomir-{index:02d}"
    notebook["cells"] = cells
    if resume_kernel:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-z0-9-]+/[0-9]+", resume_kernel):
            raise ValueError("Resume input must pin owner/kernel/version")
        metadata["kernel_sources"] = [resume_kernel]
    if catalogue_dataset:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-z0-9-]+/[0-9]+", catalogue_dataset):
            raise ValueError("Catalogue input must pin owner/dataset/version")
        metadata["dataset_sources"] = [catalogue_dataset]
    return notebook, metadata
