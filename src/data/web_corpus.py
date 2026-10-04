"""Bounded, resumable URL acquisition and text extraction for a RAG corpus."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import urljoin, urlsplit, urlunsplit
import re

LOGGER = logging.getLogger(__name__)


def encoded(row):
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def chunk_document(document, size=1200, overlap=150):
    """Character windows also cover CJK text; IDs explicitly use a local namespace."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require chunk_chars > chunk_overlap >= 0")
    text, start, index = document["text"], 0, 0
    version = hashlib.sha256(f"{size}:{overlap}".encode()).hexdigest()[:8]
    content_hash = document["metadata"]["text_sha256"][:16]
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            floor = start + max(overlap + 1, int(size * 0.65))
            boundaries = [text.rfind(mark, floor, end) for mark in ("\n", ". ", "。", "! ", "? ")]
            boundary = max(boundaries)
            if boundary >= floor:
                end = boundary + 1
        yield {"chunk_id": f"rag:{document['doc_id']}:{content_hash}:{version}:{index:06d}",
               "doc_id": document["doc_id"], "text": text[start:end],
               "title": document["title"], "url": document["url"],
               "metadata": {"start_char": start, "end_char": end,
                            "chunk_id_namespace": "local_rag", "language": document["metadata"].get("language")}}
        if end == len(text):
            break
        start, index = end - overlap, index + 1


def extract_document(body, content_type, url, official_id, min_chars=200):
    if "application/pdf" in content_type or body.startswith(b"%PDF"):
        from pypdf import PdfReader
        pdf = PdfReader(io.BytesIO(body))
        if len(pdf.pages) > 200:
            raise ValueError("PDF exceeds 200-page extraction limit")
        text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        title, language = str((pdf.metadata or {}).get("/Title") or ""), None
    elif "text/plain" in content_type:
        from trafilatura.utils import decode_file
        text, title, language = decode_file(body), "", None
    elif "html" in content_type or (not content_type and b"<html" in body[:2000].lower()):
        import trafilatura
        from lxml import html
        from trafilatura.utils import decode_file
        markup = decode_file(body)
        # Some publishers close <html> before <head>; lxml otherwise drops the article.
        markup = re.sub(r"</html>(\s*)(?=<head(?:\s|>))", r"\1", markup, count=1, flags=re.I)
        tree = html.fromstring(markup)
        page_title = " ".join(tree.xpath("//title/text()")).strip()
        if challenge_page(body, page_title):
            raise ValueError("Browser verification/challenge page detected")
        result = trafilatura.bare_extraction(markup, url=url, include_comments=False,
                                           include_tables=True, with_metadata=True)
        data = result if isinstance(result, dict) else result.as_dict() if result else {}
        text, title, language = data.get("text") or "", data.get("title") or "", data.get("language")
        # Prefer an explicit article body over indiscriminate page text fallback.
        if len(text.strip()) < min_chars:
            bodies = []
            def article_bodies(value):
                if isinstance(value, dict):
                    if isinstance(value.get("articleBody"), str):
                        bodies.append(value["articleBody"])
                    for child in value.values():
                        article_bodies(child)
                elif isinstance(value, list):
                    for child in value:
                        article_bodies(child)
            for script in tree.xpath('//script[@type="application/ld+json"]/text()'):
                try:
                    article_bodies(json.loads(script))
                except (ValueError, RecursionError):
                    pass
            for node in tree.xpath('//*[@itemprop="articleBody"]'):
                for unwanted in node.xpath('.//script|.//style|.//nav|.//aside'):
                    unwanted.drop_tree()
                bodies.append(node.text_content())
            selectors = {"baotayninh.vn": "content-detail", "food.39.net": "art_content",
                         "gan.39.net": "art_content"}
            article_class = selectors.get(urlsplit(url).hostname)
            if article_class:
                for node in tree.xpath('//*[contains(concat(" ", normalize-space(@class), " "), $token)]', token=" " + article_class + " "):
                    for unwanted in node.xpath('.//script|.//style|.//nav|.//aside'):
                        unwanted.drop_tree()
                    bodies.append("\n".join(x.strip() for x in node.itertext() if x.strip()))
            if bodies:
                text = max(bodies, key=len).strip()
                title = title or page_title
        if not language:
            language = tree.get("lang") or None
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    text = text.strip()
    if len(text) < min_chars:
        raise ValueError("Extracted text is shorter than min_text_chars")
    if any(term in title.lower() for term in ("just a moment", "access denied", "verify you are human", "captcha")):
        raise ValueError("Challenge/error page detected")
    return {"doc_id": str(official_id), "official_id": official_id, "url": url, "title": title,
            "text": text, "metadata": {"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                         "language": language, "extraction": "main_text"}}


def challenge_page(body, title=""):
    sample = body[:30000].lower()
    return (any(x in title.lower() for x in ("just a moment", "access denied", "verify you are human", "captcha"))
            or b'cf-chl-' in sample or b'challenge-platform' in sample
            or (b'document.cookie=' in sample and b'location.reload' in sample and len(body) < 3000))


class WebFetcher:
    """One request per hostname at a time; robots checks also apply to redirects."""
    def __init__(self, settings, renderer=None):
        self.settings = settings
        self.guard, self.hosts = threading.Lock(), {}
        self.local = threading.local()
        self.renderer = renderer
        if renderer is None and settings.get("browser_fallback", False):
            from src.data.browser_crawl import BrowserRenderer
            self.renderer = BrowserRenderer(settings)

    def _state(self, host):
        with self.guard:
            return self.hosts.setdefault(host, {"lock": threading.Lock(), "last": 0, "robots": {}, "browser_failed_until": 0})

    def _session(self):
        import requests
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            self.local.session.headers.update({"User-Agent": self.settings["user_agent"], "Accept-Encoding": "identity"})
        return self.local.session

    def _get(self, url, cap):
        """Return a bounded decoded body and the measured encoded response bytes."""
        timeout = self.settings["timeout_seconds"]
        with self._session().get(url, timeout=(min(5, timeout), timeout), stream=True,
                                 allow_redirects=False) as response:
            chunks, length, started, too_large = [], 0, time.monotonic(), False
            # raw.read tracks encoded bytes for chunked responses; iter_content's
            # read_chunked path can leave raw.tell() at zero after a successful fetch.
            while True:
                block = response.raw.read(min(cap + 1, 65536), decode_content=True)
                if not block:
                    break
                length += len(block)
                if length > cap or time.monotonic() - started > timeout:
                    too_large = True
                    break
                chunks.append(block)
            return {"code": response.status_code, "headers": {k.lower(): v for k, v in response.headers.items()},
                    "body": b"".join(chunks), "wire_bytes": response.raw.tell(),
                    "too_large": too_large}

    def _request(self, url, cap, result):
        """Retry transport errors, 429 and 5xx; never retry a permanent 404/403."""
        attempts = self.settings.get("retry_attempts", 3)
        for attempt in range(attempts):
            result["request_attempts"] = result.get("request_attempts", 0) + 1
            try:
                response = self._get(url, cap)
                if response["code"] != 429 and not 500 <= response["code"] < 600:
                    return response
                if attempt + 1 == attempts:
                    return response
                retry_after = response["headers"].get("retry-after", "")
                wait_seconds = self.settings.get("retry_backoff_seconds", 1) * 2**attempt
                if retry_after:
                    try:
                        wait_seconds = max(wait_seconds, float(retry_after))
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        try:
                            wait_seconds = max(wait_seconds, parsedate_to_datetime(retry_after).timestamp() - time.time())
                        except (ValueError, TypeError):
                            pass
                if wait_seconds > self.settings.get("max_retry_wait_seconds", 30):
                    return response
            except Exception:
                if attempt + 1 == attempts:
                    raise
                wait_seconds = self.settings.get("retry_backoff_seconds", 1) * 2**attempt
            time.sleep(max(0, wait_seconds))

    def _render(self, url, state, result):
        if self.renderer is None or time.monotonic() < state["browser_failed_until"]:
            return None
        try:
            response = self.renderer.render(url)
            if response["too_large"] or response["code"] != 200 or challenge_page(response["body"], response.get("title", "")):
                state["browser_failed_until"] = time.monotonic() + self.settings.get("browser_failure_ttl_seconds", 3600)
                return None
            result["browser_rendered"] = True
            return response
        except Exception as error:
            result["browser_error"] = f"{type(error).__name__}: {error}"
            state["browser_failed_until"] = time.monotonic() + self.settings.get("browser_failure_ttl_seconds", 3600)
            return None

    def _robots(self, origin, state, result):
        from protego import Protego
        cached = state["robots"].get(origin)
        if cached and cached["expires"] > time.monotonic():
            result["robots"] = cached["evidence"]
            return cached["parser"]
        evidence = {"url": origin + "/robots.txt"}
        parser = None
        try:
            robots_url = evidence["url"]
            for redirect in range(6):
                rr = self._request(robots_url, 512 * 1024, result)
                result["robots_wire_bytes"] += rr["wire_bytes"]
                if rr["code"] not in {301, 302, 303, 307, 308}:
                    break
                if redirect == 5 or not rr["headers"].get("location"):
                    raise ValueError("Too many robots redirects or missing Location")
                robots_url = urljoin(robots_url, rr["headers"]["location"])
                if urlsplit(robots_url).scheme not in {"http", "https"}:
                    raise ValueError("Non-HTTP(S) robots redirect")
            if rr["code"] == 403 or (rr["code"] == 200 and challenge_page(rr["body"])):
                rendered = self._render(robots_url, state, result)
                if rendered:
                    rr = rendered
                    rr["body"] = rendered.get("body_text", "").encode()
                elif rr["code"] == 200:
                    raise ValueError("Robots endpoint requires browser verification")
            evidence.update(final_url=robots_url, http_status=rr["code"], body_sha256=hashlib.sha256(rr["body"]).hexdigest())
            if rr["code"] in {401, 403}:
                evidence["error"] = "Robots endpoint refused access; no robots rule was read"
            elif 400 <= rr["code"] < 500 and rr["code"] != 429:
                parser = Protego.parse("")
            elif rr["code"] == 200 and not rr["too_large"]:
                parser = Protego.parse(rr["body"].decode("utf-8-sig", errors="replace"))
            else:
                evidence["error"] = "Robots endpoint unreachable, rate limited or too large"
        except Exception as error:
            evidence["error"] = f"{type(error).__name__}: {error}"
        state["last"] = time.monotonic()
        ttl = self.settings.get("robots_cache_seconds", 3600) if parser is not None else self.settings.get("robots_failure_ttl_seconds", 60)
        state["robots"][origin] = {"parser": parser, "evidence": evidence, "expires": time.monotonic() + min(ttl, 86400)}
        result["robots"] = evidence
        return parser

    def fetch(self, row):
        original = row["url"]
        result = {"id": row["id"], "url": original, "host": row["host"],
                  "status": "error", "wire_bytes": 0, "robots_wire_bytes": 0, "response_body_bytes": 0,
                  "raw_gzip_bytes": 0, "text_bytes": 0, "document_jsonl_bytes": 0,
                  "chunks_jsonl_bytes": 0, "chunk_count": 0,
                  "fetched_at": datetime.now(timezone.utc).isoformat()}
        current = original
        if self.settings.get("prefer_https", False) and current.startswith("http://"):
            current = "https://" + current[7:]
        try:
            for _ in range(6):
                parsed = urlsplit(current)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise ValueError("Non-HTTP(S) URL/redirect")
                state = self._state(parsed.hostname)
                origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
                with state["lock"]:
                    robots = self._robots(origin, state, result)
                    if robots is None:
                        refused = result["robots"].get("http_status") in {401, 403}
                        result.update(status="robots_http_denied" if refused else "robots_unavailable",
                                      retryable=not refused, failure_stage="robots", error=result["robots"]["error"])
                        return result
                    if not robots.can_fetch(current, self.settings["user_agent"]):
                        result.update(status="robots_denied", retryable=False, failure_stage="robots", error="Disallowed by robots.txt")
                        return result
                    rate = robots.request_rate(self.settings["user_agent"])
                    delay = max(self.settings["delay_per_host_seconds"], robots.crawl_delay(self.settings["user_agent"]) or 0,
                                rate.seconds / rate.requests if rate else 0)
                    time.sleep(max(0, delay - (time.monotonic() - state["last"])))
                    result["failure_stage"] = "page"
                    response = self._request(current, self.settings["max_response_bytes"], result)
                    if response["code"] == 403 or (response["code"] == 200 and challenge_page(response["body"])):
                        response = self._render(current, state, result) or response
                    state["last"] = time.monotonic()
                result["wire_bytes"] += response["wire_bytes"]
                result["response_body_bytes"] += len(response["body"])
                result.update(http_status=response["code"], final_url=current)
                if response["too_large"]:
                    result.update(status="response_limit", error="Response exceeded byte/time limit; not indexed")
                    return result
                if response["code"] in {301, 302, 303, 307, 308}:
                    location = response["headers"].get("location")
                    if not location:
                        raise ValueError("Redirect without Location")
                    current = urljoin(current, location)
                    continue
                if response["code"] != 200:
                    result.update(status="access_blocked" if response["code"] in {401, 403} else "http_error",
                                  retryable=response["code"] == 429 or response["code"] >= 500, error=f"HTTP {response['code']}")
                    return result
                body = response["body"]
                result["failure_stage"] = "extraction"
                try:
                    document = extract_document(body, response["headers"].get("content-type", ""), current,
                                                row["id"], self.settings["min_text_chars"])
                except ValueError:
                    rendered = self._render(current, state, result)
                    if not rendered:
                        raise
                    body = rendered["body"]
                    document = extract_document(body, "text/html", current, row["id"], self.settings["min_text_chars"])
                document["url"] = original
                document["metadata"].update(final_url=current, fetched_at=result["fetched_at"])
                if len(document["text"].encode()) > self.settings.get("max_extracted_text_bytes", 2097152):
                    result.update(status="response_limit", retryable=False, error="Extracted document exceeds text byte limit; not truncated")
                    return result
                if result.get("browser_rendered"):
                    document["metadata"]["extraction"] = "browser_main_text"
                chunks = list(chunk_document(document, self.settings["chunk_chars"], self.settings["chunk_overlap"]))
                result.pop("failure_stage", None)
                result.update(status="ok", retryable=False, document=document, raw_gzip_bytes=len(gzip.compress(body, mtime=0)),
                              text_bytes=len(document["text"].encode()), document_jsonl_bytes=len(encoded(document)),
                              chunks_jsonl_bytes=sum(len(encoded(c)) for c in chunks), chunk_count=len(chunks))
                if self.settings.get("keep_raw"):
                    result["raw"] = gzip.compress(body, mtime=0)
                return result
            raise ValueError("Too many redirects")
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            result["retryable"] = not isinstance(error, ValueError)
            return result


@contextmanager
def open_state(directory, identity):
    """Exclusive process lock; SQLite commits make interrupted crawls resumable."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = sqlite3.connect(directory / "state.sqlite")
        try:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS pages (id INTEGER PRIMARY KEY, status TEXT, result TEXT)")
            old = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            value = json.dumps(identity, sort_keys=True)
            if old and old[0] != value:
                raise ValueError("State belongs to different source/chunk/fetch settings; use a new state directory")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (value,))
            db.commit()
            yield db
        finally:
            db.close()


def acquire(rows, db, settings, state_dir, *, retry_failed=False, fetcher=None, stop_after_seconds=None, retry_transient_only=False):
    """Bound pending work and memory independently of the 4M+ source URLs."""
    fetcher = fetcher or WebFetcher(settings)
    def pending_rows():
        for row in rows:
            old = db.execute("SELECT status, result FROM pages WHERE id=?", (row["id"],)).fetchone()
            if old is None or (retry_failed and old[0] != "ok" and
                               (not retry_transient_only or json.loads(old[1]).get("retryable", False))):
                yield row
    iterator, completed = iter(pending_rows()), 0
    deadline = time.monotonic() + stop_after_seconds if stop_after_seconds is not None else float("inf")
    with ThreadPoolExecutor(max_workers=settings["workers"]) as pool:
        futures = set()
        exhausted = False
        while futures or not exhausted:
            if time.monotonic() >= deadline:
                exhausted = True
            while not exhausted and len(futures) < settings["workers"] * 2:
                row = next(iterator, None)
                if row is None:
                    exhausted = True
                else:
                    futures.add(pool.submit(fetcher.fetch, row))
            if not futures:
                break
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                raw = result.pop("raw", None)
                if raw is not None:
                    target = Path(state_dir) / "raw" / str(result["id"] // 10000)
                    target.mkdir(parents=True, exist_ok=True)
                    (target / f"{result['id']}.gz").write_bytes(raw)
                db.execute("INSERT OR REPLACE INTO pages VALUES (?, ?, ?)",
                           (result["id"], result["status"], encoded(result).decode()))
                db.commit()
                completed += 1
                LOGGER.info("URL %s: %s (%s completed this invocation)", result["id"], result["status"], completed)
    return completed
