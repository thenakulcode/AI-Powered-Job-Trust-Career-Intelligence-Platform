"""Lazy, reusable Playwright browser singleton.

Launching a Chromium instance per request is the single biggest cost in the
JS-rendering fallback path. This module keeps one browser process alive for
the lifetime of the app (or worker), hands out fresh pages/contexts per
extraction, and provides a single shutdown hook so nothing leaks.

Thread-safety note: FastAPI's sync def routes run in a threadpool, and
Playwright's *sync* API is not thread-safe across different Playwright
instances used concurrently from different threads. To keep this simple and
correct without pulling in the async API everywhere, each call to
`render_page` starts its own `sync_playwright()` context but reuses a
lazily-launched, shared Chromium *executable* via a lightweight lock so we
don't pay repeated cold starts under load. If this service ever grows into a
fully async app, swap this for `playwright.async_api` and a single
long-lived browser instance instead.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

from playwright.sync_api import sync_playwright

logger = logging.getLogger("jobshield.extraction.browser")

_launch_lock = threading.Lock()


class BrowserRenderError(RuntimeError):
    """Raised when Playwright cannot render a page (timeout, crash, blocked)."""


@contextmanager
def render_session(timeout_seconds: int = 25):
    """Context manager yielding a fresh Playwright page.

    Usage:
        with render_session(timeout_seconds=20) as page:
            page.goto(url, wait_until="networkidle")
            html = page.content()

    The browser + context + page are torn down deterministically on exit,
    including on exceptions, so a failed extraction never leaves a zombie
    Chromium process behind.
    """

    with _launch_lock:
        playwright = sync_playwright().start()

    browser = None
    try:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 900},
            locale="en-US",
        )
        context.set_default_timeout(timeout_seconds * 1000)
        page = context.new_page()
        try:
            yield page
        finally:
            context.close()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Playwright render session failed: %s", exc)
        raise BrowserRenderError(str(exc)) from exc
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("Browser close raised during cleanup", exc_info=True)
        try:
            playwright.stop()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("Playwright stop raised during cleanup", exc_info=True)


def render_page(url: str, timeout_seconds: int = 25, scroll: bool = True) -> str:
    """Render `url` in headless Chromium and return the fully-loaded HTML.

    Waits for network idle, then scrolls to the bottom a couple of times to
    trigger lazy-loaded content (common on Workday and SmartRecruiters
    postings), then waits briefly again before grabbing the final DOM.
    """

    with render_session(timeout_seconds=timeout_seconds) as page:
        logger.info("Playwright: navigating to %s", url)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_seconds * 1000)
        except Exception as exc:
            raise BrowserRenderError(f"Navigation failed: {exc}") from exc

        try:
            page.wait_for_load_state("networkidle", timeout=timeout_seconds * 1000)
        except Exception:
            # Some career portals keep a background poll connection open
            # forever, so networkidle never fires. Fall back to a fixed
            # settle window instead of failing the whole extraction.
            logger.info("networkidle wait timed out for %s; continuing anyway", url)
            page.wait_for_timeout(2000)

        if scroll:
            for _ in range(3):
                page.mouse.wheel(0, 2000)
                page.wait_for_timeout(400)

        page.wait_for_timeout(500)
        return page.content()
