"""Load canonical records and prepare validated, unchanged-text data."""
import json
from pathlib import Path

from src.data.adapter import adapt_rows, build_mappings
from src.data.schema import validate_corpus
from src.data.split import validate_split_leakage
from src.utils.io import read_json, write_json


def load_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix == ".parquet":
        import pandas as pd
        rows = pd.read_parquet(path).to_dict("records")
    else:
        rows = read_json(path)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a list of records: {path}")
    return rows


def prepare_data(config):
    cfg = config.get("data", config)
    fields = cfg.get("fields", {})
    records = {
        kind: adapt_rows(load_records(cfg[f"{kind}_path"]), fields.get(kind, {}))
        for kind in ("documents", "chunks", "queries")
    }
    validate_corpus(**records)
    if cfg.get("splits"):
        validate_split_leakage({
            split: load_records(info["data_path"])
            for split, info in cfg["splits"].items()
        })
    chunk_to_doc, internal = build_mappings(records["chunks"])
    output, mappings = Path(cfg["output_dir"]), Path(cfg["mappings_dir"])
    if cfg.get("raw_dir"):
        raw_dir = Path(cfg["raw_dir"]).resolve()
        for directory in (output, mappings):
            if directory.resolve().is_relative_to(raw_dir):
                raise ValueError("Prepared outputs must be outside data/raw")
    targets = [output / f"{kind}.json" for kind in records]
    targets += [mappings / "chunk_to_doc.json", mappings / "internal_to_official_id.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Prepared dataset/mappings already exist; use a new output directory")
    raw_paths = [Path(cfg[f"{kind}_path"]).resolve() for kind in records]
    if any(target.resolve() in raw_paths for target in targets):
        raise ValueError("Prepared outputs must not overwrite raw inputs")
    for kind, rows in records.items():
        write_json(output / f"{kind}.json", rows)
    write_json(mappings / "chunk_to_doc.json", chunk_to_doc)
    write_json(mappings / "internal_to_official_id.json", internal)
    return records
