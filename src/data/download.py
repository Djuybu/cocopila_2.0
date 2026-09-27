"""Pinned, resumable downloads of explicitly selected MMedC language archives."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

from src.utils.io import read_json, write_json

LOGGER = logging.getLogger(__name__)
LANGUAGES = {"chinese": "Chinese.zip", "english": "English.zip",
             "japanese": "Japanese.zip", "french": "French.zip"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_url(url):
    with urlopen(Request(url, headers={"User-Agent": "medical-rag-competition/0.1"}), timeout=60) as response:
        return json.load(response)


def _download_segment(url, path, start, end, attempts):
    """Resume one bounded range; never share this file between workers."""
    size = end - start + 1
    for attempt in range(attempts):
        present = path.stat().st_size if path.exists() else 0
        if present > size:
            raise ValueError(f"Oversized range fragment: {path}")
        if present == size:
            return path
        offset = start + present
        request = Request(url, headers={"Range": f"bytes={offset}-{end}"})
        try:
            with urlopen(request, timeout=120) as response:
                expected = f"bytes {offset}-{end}/"
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                    raise ValueError("Server did not honor bounded range")
                with path.open("ab" if present else "wb") as handle:
                    remaining = size - present
                    while remaining:
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            raise OSError("Incomplete bounded response")
                        handle.write(block)
                        remaining -= len(block)
                    if response.read(1):
                        raise ValueError("Oversized bounded response")
            return path
        except (OSError, TimeoutError) as error:
            LOGGER.warning("range %s attempt %d: %s", path.name, attempt + 1, error)
            if attempt + 1 == attempts:
                raise
            time.sleep(min(30, 2**attempt))


def _parallel_resume(url, partial, size, attempts, workers, range_bytes):
    """Assemble complete ranges in order, keeping the contiguous resume prefix."""
    fragments = partial.parent / f"{partial.name}.ranges"
    fragments.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while (partial.stat().st_size if partial.exists() else 0) < size:
            start = partial.stat().st_size if partial.exists() else 0
            ranges = [(offset, min(offset + range_bytes, size) - 1)
                      for offset in range(start, min(size, start + workers * range_bytes), range_bytes)]
            tasks = [(begin, end, fragments / f"{begin}-{end}.part") for begin, end in ranges]
            futures = [pool.submit(_download_segment, url, path, begin, end, attempts)
                       for begin, end, path in tasks]
            for future in futures:
                future.result()
            with partial.open("ab") as target:
                for begin, end, fragment in tasks:
                    if target.tell() != begin or fragment.stat().st_size != end - begin + 1:
                        raise ValueError("Range assembly offset mismatch")
                    with fragment.open("rb") as source:
                        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                            target.write(block)
                    target.flush()
                    # Only disposable verified range fragments are removed.
                    fragment.unlink()
            LOGGER.info("%s %.2f/%.2f GiB", partial.name, partial.stat().st_size / 2**30, size / 2**30)


def _download_file(base_url, destination, entry, attempts, range_workers=1, range_bytes=64 * 1024 * 1024):
    target = destination / entry["path"]
    expected_size, expected_hash = entry["size"], entry["lfs"]["oid"]
    if target.exists():
        if target.stat().st_size != expected_size or sha256_file(target) != expected_hash:
            raise ValueError(f"Existing archive fails integrity check: {target}")
        LOGGER.info("verified existing %s", target.name)
        return entry
    partial = target.with_suffix(target.suffix + ".part")
    if range_workers > 1:
        if range_bytes < 1:
            raise ValueError("range_bytes must be positive")
        if partial.exists() and partial.stat().st_size > expected_size:
            raise ValueError(f"Partial archive is larger than source: {partial}")
        url = base_url + "/" + quote(entry["path"]) + "?download=true"
        _parallel_resume(url, partial, expected_size, attempts, range_workers, range_bytes)
    for attempt in range(attempts):
        start = partial.stat().st_size if partial.exists() else 0
        if start > expected_size:
            raise ValueError(f"Partial archive is larger than source: {partial}")
        if start == expected_size:
            break
        url = base_url + "/" + quote(entry["path"]) + "?download=true"
        request = Request(url, headers={"Range": f"bytes={start}-", "User-Agent": "medical-rag-competition/0.1"})
        try:
            with urlopen(request, timeout=120) as response:
                if start and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-")):
                    raise ValueError("Server did not honor resume offset; partial file kept")
                last_log = time.monotonic()
                with partial.open("ab" if start else "wb") as handle:
                    while block := response.read(4 * 1024 * 1024):
                        handle.write(block)
                        if time.monotonic() - last_log >= 30:
                            LOGGER.info("%s %.2f/%.2f GiB", target.name, handle.tell() / 2**30, expected_size / 2**30)
                            last_log = time.monotonic()
                if partial.stat().st_size != expected_size:
                    raise OSError("Incomplete response")
            break
        except (OSError, TimeoutError) as error:
            LOGGER.warning("%s attempt %d/%d: %s", target.name, attempt + 1, attempts, error)
            if attempt + 1 == attempts:
                raise
            time.sleep(min(30, 2**attempt))
    if not partial.exists() or partial.stat().st_size != expected_size or sha256_file(partial) != expected_hash:
        raise ValueError(f"Downloaded archive fails SHA256/size check: {partial}")
    # Exclusive target; preserve the partial file if somebody created target meanwhile.
    os.link(partial, target)
    partial.unlink()
    LOGGER.info("download complete and verified: %s", target.name)
    return entry


def download_mmedc(config):
    cfg = config["download"]
    names = [name.lower() for name in cfg["languages"]]
    if len(names) != len(set(names)) or any(name not in LANGUAGES for name in names):
        raise ValueError("Only distinct Chinese/English/Japanese/French languages are supported")
    repo = cfg.get("repo", "Henrychur/MMedC")
    revision = cfg.get("revision", "main")
    info = _json_url(f"https://huggingface.co/api/datasets/{repo}/revision/{revision}")
    commit = info["sha"]
    entries = _json_url(f"https://huggingface.co/api/datasets/{repo}/tree/{commit}?recursive=true")
    wanted = {LANGUAGES[name] for name in names}
    selected = [entry for entry in entries if entry["path"] in wanted]
    if {entry["path"] for entry in selected} != wanted:
        raise ValueError("Requested language archive missing from dataset repository")
    destination = Path(cfg["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Pinned %s@%s; downloading %.2f GiB", repo, commit, sum(e["size"] for e in selected) / 2**30)
    base_url = f"https://huggingface.co/datasets/{repo}/resolve/{commit}"
    with ThreadPoolExecutor(max_workers=cfg.get("workers", 4)) as pool:
        futures = [pool.submit(_download_file, base_url, destination, entry, cfg.get("attempts", 10),
                               cfg.get("range_workers", 1), cfg.get("range_bytes", 64 * 1024 * 1024))
                   for entry in selected]
        verified = [future.result() for future in futures]
    manifest = {"repo": repo, "revision": commit, "languages": names,
                "files": [{"path": entry["path"], "size": entry["size"], "sha256": entry["lfs"]["oid"]}
                          for entry in verified]}
    path = destination / "download_manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("Existing download manifest differs")
    else:
        write_json(path, manifest)
    return manifest
