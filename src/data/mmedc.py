"""Bounded MMedC TXT prototype. Adjacent-span qrels are explicitly weak labels."""
from collections import OrderedDict, defaultdict
import hashlib
import io
from pathlib import Path
import random
from urllib.request import Request, urlopen
import zipfile

from src.data.download import LANGUAGES, _json_url, sha256_file
from src.data.prototype import prototype_id, text_key, write_prototype
from src.data.schema import validate_corpus


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only HTTP source for ZIP central directory/member sampling."""
    def __init__(self, url, size, page_size=256 * 1024):
        self.url, self.size, self.page_size = url, size, page_size
        self.position, self.pages = 0, OrderedDict()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if value < 0:
            raise ValueError("Negative seek")
        self.position = value
        return value

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Read on closed source")
        remaining = max(0, self.size - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if size > 512 * 1024 * 1024:
            raise ValueError("Remote ZIP reads must be bounded")
        if size > self.page_size:
            start, end = self.position, self.position + size - 1
            request = Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(request, timeout=120) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("Remote source does not support safe ranged reads")
                data = response.read(size + 1)
            if len(data) != size:
                raise OSError("Incomplete/oversized range response")
            self.position += size
            return data
        output = bytearray()
        while size:
            page = self.position // self.page_size
            start = page * self.page_size
            if page not in self.pages:
                end = min(self.size - 1, start + self.page_size - 1)
                request = Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urlopen(request, timeout=120) as response:
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                        raise ValueError("Remote source does not support safe ranged reads")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise OSError("Incomplete/oversized range response")
                self.pages[page] = data
                if len(self.pages) > 8:
                    self.pages.popitem(last=False)
            self.pages.move_to_end(page)
            data = self.pages[page]
            offset = self.position - start
            take = min(size, len(data) - offset)
            output.extend(data[offset:offset + take])
            self.position += take
            size -= take
        return bytes(output)


def _sample_archive(archive, language, cfg, source):
    members = sorted(info.filename for info in archive.infolist()
                     if not info.is_dir() and info.filename.lower().endswith(".txt")
                     and "cultural_filtered_data_used" not in info.filename
                     and "__MACOSX" not in info.filename)
    random.Random(f"{cfg['seed']}:{language}").shuffle(members)
    samples = []
    for member in members:
        if len(samples) >= cfg["max_documents_per_language"]:
            break
        with archive.open(member) as handle:
            # Decode only a bounded prefix; no archive extraction or whole-file loading.
            raw = handle.read(cfg["max_chars_per_document"] * 4)
        content = raw.decode("utf-8-sig", errors="replace")[:cfg["max_chars_per_document"]]
        if len(content.strip()) < cfg["query_chars"] + cfg["chunk_chars"]:
            continue
        info = archive.getinfo(member)
        samples.append({"text": content, "language": language, "title": Path(member).stem,
                        "source": {**source, "member": member, "member_crc32": info.CRC,
                                   "member_uncompressed_bytes": info.file_size,
                                   "sample_char_count": len(content), "sample_sha256": hashlib.sha256(raw).hexdigest(),
                                   "truncated": info.file_size > len(raw)}})
    return samples, len(members)


def prepare_mmedc(config):
    cfg = config["data"]
    if not cfg.get("weak_labels_approved", False):
        raise ValueError("MMedC has no retrieval qrels; explicitly approve weak_labels_approved before deriving a proxy")
    languages = [name.lower() for name in cfg["languages"]]
    if any(name not in LANGUAGES for name in languages) or len(languages) != len(set(languages)):
        raise ValueError("Unsupported or duplicate language")
    if cfg["chunk_chars"] <= cfg["overlap_chars"] or cfg["overlap_chars"] < 0 or cfg["query_chars"] < 1:
        raise ValueError("Invalid chunk/query sizes")
    repo, revision = cfg["repo"], cfg["revision"]
    info = _json_url(f"https://huggingface.co/api/datasets/{repo}/revision/{revision}")
    revision = info["sha"]
    entries = {entry["path"]: entry for entry in _json_url(
        f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true")}
    archives_dir = Path(cfg["archives_dir"])
    samples, language_stats = [], {}
    for language in languages:
        filename = LANGUAGES[language]
        entry = entries[filename]
        source = {"repo": repo, "revision": revision, "archive": filename,
                  "archive_sha256": entry["lfs"]["oid"]}
        local = archives_dir / filename
        if local.exists():
            if local.stat().st_size != entry["size"] or sha256_file(local) != entry["lfs"]["oid"]:
                raise ValueError(f"Local archive fails integrity check: {local}")
            with zipfile.ZipFile(local) as archive:
                rows, available = _sample_archive(archive, language, cfg, source)
        elif cfg.get("allow_remote_sampling", False):
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}?download=true"
            with HTTPRangeFile(url, entry["size"]) as ranged, zipfile.ZipFile(ranged) as archive:
                rows, available = _sample_archive(archive, language, cfg, source)
        else:
            raise FileNotFoundError(f"Archive not downloaded: {local}")
        if not rows:
            raise ValueError(f"No usable TXT documents in {filename}")
        samples.extend(rows)
        language_stats[language] = {"eligible_txt_members": available, "sampled_documents": len(rows)}

    documents, chunks, queries, labels = [], [], [], []
    source_mappings = {"documents": {}, "chunks": {}, "queries": {}}
    seen, duplicates = {}, 0
    for sample in samples:
        content = sample["text"]
        normalized = text_key(content)
        if normalized in seen:
            duplicates += 1
            source_mappings["documents"][seen[normalized]].append(sample["source"])
            continue
        source = sample["source"]
        identity = [repo, revision, source["archive"], source["member"]]
        doc_id = prototype_id("doc", "mmedc", identity)
        seen[normalized] = doc_id
        metadata = {"language": sample["language"], "source": source, "title_is_source_filename": True}
        documents.append({"doc_id": doc_id, "title": sample["title"], "metadata": metadata,
                          "id_namespace": "prototype"})
        source_mappings["documents"][doc_id] = [source]
        # The query span is excluded from every candidate chunk, including overlap.
        anchor = content[:cfg["query_chars"]].strip()
        query_id = prototype_id("query", "mmedc", [doc_id, anchor])
        queries.append({"id": query_id, "query_id": query_id, "text": anchor,
                        "language": sample["language"], "label_quality": "weak_adjacent_span",
                        "id_namespace": "prototype"})
        source_mappings["queries"][query_id] = {**source, "span_start": 0, "span_end": cfg["query_chars"]}
        doc_chunks = []
        stride = cfg["chunk_chars"] - cfg["overlap_chars"]
        for start in range(cfg["query_chars"], len(content), stride):
            text = content[start:start + cfg["chunk_chars"]]
            if not text.strip():
                continue
            end = start + len(text)
            chunk_id = prototype_id("chunk", "mmedc", [doc_id, start, end])
            chunks.append({"chunk_id": chunk_id, "doc_id": doc_id, "text": text, "title": sample["title"],
                           "language": sample["language"], "metadata": {**metadata, "span_start": start, "span_end": end},
                           "id_namespace": "prototype"})
            source_mappings["chunks"][chunk_id] = {**source, "span_start": start, "span_end": end}
            doc_chunks.append(chunk_id)
            if end == len(content):
                break
        if not doc_chunks:
            raise ValueError("Document has no candidate chunks")
        labels.append({"id": query_id, "relevant_docs": [doc_id], "relevant_chunks": [doc_chunks[0]]})
    validate_corpus(documents, chunks, queries)
    bundle = {"documents": documents, "chunks": chunks, "queries": queries, "labels": labels,
              "source_mappings": source_mappings, "label_quality": "weak_adjacent_span",
              "source": {"repo": repo, "revision": revision, "languages": languages},
              "dedup_report": {"documents_before": len(samples), "duplicate_documents_removed": duplicates,
                               "documents_after": len(documents), "chunks_after": len(chunks), "queries_after": len(queries),
                               "languages": language_stats,
                               "limitations": "Prefix-as-query, next span as positive; not human relevance labels or competition scores."}}
    return write_prototype(bundle, cfg)
