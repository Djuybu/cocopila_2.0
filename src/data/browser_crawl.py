"""Render ordinary JavaScript pages; challenge pages remain acquisition failures."""
import threading
from urllib.parse import urlsplit


class BrowserRenderer:
    def __init__(self, settings):
        self.settings = settings
        self.local = threading.local()

    def render(self, url):
        from playwright.sync_api import sync_playwright
        if not hasattr(self.local, "browser"):
            self.local.playwright = sync_playwright().start()
            self.local.browser = self.local.playwright.chromium.launch(headless=True)
            self.local.contexts = {}
        origin = urlsplit(url)
        def route_request(route):
            request = route.request
            destination = urlsplit(request.url)
            # Cross-origin top-level redirects need a fresh robots decision.
            if request.is_navigation_request() and request.frame.parent_frame is None and (destination.scheme, destination.netloc) != (origin.scheme, origin.netloc):
                route.abort()
            elif request.resource_type in {"image", "media", "font"}:
                route.abort()
            else:
                route.continue_()
        key = (origin.scheme, origin.netloc)
        context = self.local.contexts.get(key)
        if context is None:
            context = self.local.browser.new_context(user_agent=self.settings["user_agent"])
            context.route("**/*", route_request)
            self.local.contexts[key] = context
        page = context.new_page()
        timeout = self.settings.get("browser_timeout_seconds", 25) * 1000
        try:
            response = page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            try:
                page.wait_for_function("document.body && document.body.innerText.trim().length > 200", timeout=min(timeout, 5000))
            except Exception:
                pass
            body = page.content().encode("utf-8")
            if len(body) > self.settings["max_response_bytes"]:
                body, oversized = b"", True
            else:
                oversized = False
            return {"code": response.status if response else 0, "headers": {"content-type": "text/html; charset=utf-8"},
                    "body": body, "body_text": page.locator("body").inner_text(timeout=1000), "title": page.title(),
                    "wire_bytes": 0, "too_large": oversized, "final_url": page.url}
        finally:
            page.close()
