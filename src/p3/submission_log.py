"""P3-16: submission log — traceable record of every leaderboard submission.

Extends ``submissions/submission_log.csv`` to the P3 schema, migrates the legacy
``score`` column, optionally exports ``.xlsx``, and enforces the maximum of five
private submissions. Standard library + optional openpyxl.
"""
from pathlib import Path
import csv

LOG_COLUMNS = ("run_id", "split", "archive", "git_commit", "config", "model",
               "local_score", "public_score", "private_score", "notes")
LEGACY_COLUMNS = ("run_id", "split", "archive", "score", "notes")
PRIVATE_LIMIT = 5
SPLITS = ("public", "private")


def read_log(path):
    """Read the log, migrating the legacy ``score`` column when present."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        rows = []
        for raw in reader:
            row = {column: (raw.get(column) or "") for column in LOG_COLUMNS}
            if not row["local_score"] and raw.get("score"):
                row["local_score"] = raw["score"]
            rows.append(row)
    return rows


def write_log(path, rows):
    """Rewrite the CSV with the P3 header; the log is append-only by semantics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOG_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in LOG_COLUMNS})
    return rows


def append_submission(path, record, *, enforce_private_limit=True):
    """Validate, then append one submission row; refuses to exceed 5 private runs."""
    if not isinstance(record, dict):
        raise ValueError("Submission log record must be a mapping")
    run_id, split = record.get("run_id"), record.get("split")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id is required")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    rows = read_log(path)
    if any(row["run_id"] == run_id and row["split"] == split for row in rows):
        raise ValueError(f"Submission already logged: run_id={run_id}, split={split}")
    if split == "private":
        used = sum(1 for row in rows if row["split"] == "private")
        if used >= PRIVATE_LIMIT and enforce_private_limit:
            raise ValueError(f"Private submission limit reached ({PRIVATE_LIMIT}); "
                             "record why an extra run is justified before overriding")
    row = {column: record.get(column, "") for column in LOG_COLUMNS}
    rows.append(row)
    write_log(path, rows)
    return row


def summarize_log(rows):
    """Counts per split, best local score and remaining private submissions."""
    private = [row for row in rows if row["split"] == "private"]
    public = [row for row in rows if row["split"] == "public"]
    scores = [float(row["local_score"]) for row in rows if str(row["local_score"]).strip()]
    return {
        "total": len(rows),
        "public_count": len(public),
        "private_count": len(private),
        "private_remaining": max(0, PRIVATE_LIMIT - len(private)),
        "best_local_score": max(scores) if scores else None,
    }


def export_xlsx(path, rows, sheet_name="submission_log"):
    """Write an ``.xlsx`` mirror of the log (exclusive creation; needs openpyxl)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"XLSX log already exists: {path}")
    try:
        from openpyxl import Workbook
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("openpyxl is required to export the XLSX log") from error
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    sheet.append(list(LOG_COLUMNS))
    for row in rows:
        sheet.append([row.get(column, "") for column in LOG_COLUMNS])
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return str(path)
