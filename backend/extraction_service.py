"""URL-based job posting extraction service.

Extraction strategy, in order (cheapest/most-reliable first, each one only
attempted if the previous one didn't produce a usable result):

  1. Site-specific API handlers (Microsoft Careers, Greenhouse, Lever,
     Ashby, ...) — see `extraction.site_handlers`. These hit a stable JSON
     API instead of scraping HTML, so they're both faster and more accurate
     than anything DOM-based.
  2. schema.org JobPosting JSON-LD in the static HTML (present on most
     LinkedIn, Indeed, Naukri, Glassdoor, and many company career pages).
  3. trafilatura's readability-style main-content extraction on the static
     HTML.
  4. A generic BeautifulSoup pass over the static HTML.
  5. Playwright (headless Chromium): re-fetch with full JS rendering and
     retry steps 2-4 against the rendered DOM. This is the expensive path
     and is what unlocks Microsoft Careers' HTML (as a fallback if the API
     handler above ever changes), Workday, SmartRecruiters, SAP
     SuccessFactors, Oracle Careers, and other client-side-rendered portals.

Public interface (`JobExtractionService.extract(url) -> ExtractionResult`)
is unchanged from the previous version, so `main.py` did not need to change
its call site. `ExtractionResult` gained new *optional* fields (skills,
requirements, benefits, department, industry, posting_date,
application_deadline, source, confidence) — all default to None/empty so
any code that only reads the original fields keeps working untouched.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

from backend.extraction import cache
from backend.extraction.browser import BrowserRenderError, render_page
from backend.extraction.site_handlers import get_handler_for, needs_js_rendering

try:
    import trafilatura
except ImportError:  # pragma: no cover - optional dependency guard
    trafilatura = None

logger = logging.getLogger("jobshield.extraction")

REQUEST_TIMEOUT = 15  # seconds, static HTTP fetch
PLAYWRIGHT_TIMEOUT = 25  # seconds, JS render fallback
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "iframe", "noscript"]
_NOISE_KEYWORDS = re.compile(
    r"(related job|recommended job|similar job|you may also like|sign in|log in|"
    r"cookie|privacy policy|terms of service|advertisement|sponsored|subscribe to|"
    r"share this job|apply now|save job|report this job)",
    re.IGNORECASE,
)
_CHALLENGE_PAGE = re.compile(
    r"(captcha|are you a human|access denied|checking your browser|cloudflare)", re.IGNORECASE
)

_SECTION_LABELS: dict[str, list[str]] = {
    "responsibilities": ["responsibilities", "what you'll do", "what you will do", "your role", "job responsibilities"],
    "requirements": ["requirements", "qualifications", "what we're looking for", "what we are looking for", "minimum qualifications"],
    "skills": ["skills", "preferred skills", "core skills", "competencies"],
    "benefits": ["benefits", "why join us", "perks", "company benefits"],
}


@dataclass(slots=True)
class ExtractionResult:
    """Container for URL-based job extraction output.

    Original fields are unchanged (same names/types) so the existing API
    response and frontend keep working without modification. New fields are
    additive and optional.
    """

    success: bool
    description: str | None = None
    title: str | None = None
    company: str | None = None
    location: str | None = None
    employment_type: str | None = None
    experience: str | None = None
    salary: str | None = None
    error: str | None = None

    # --- new, optional fields (additive; safe defaults) ---
    skills: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    benefits: list[str] = field(default_factory=list)
    department: str | None = None
    industry: str | None = None
    education: str | None = None
    posting_date: str | None = None
    application_deadline: str | None = None
    source: str | None = None  # which strategy produced this result
    confidence: float | None = None


class JobExtractionError(RuntimeError):
    """Raised for extraction failures that should surface as a clean message
    rather than a 500, mirroring how ModelLoadError is used in service.py."""


class JobExtractionService:
    """Fetches a job-posting URL and extracts structured job data."""

    def extract(self, url: str) -> ExtractionResult:
        cached = cache.get(url)
        if cached is not None:
            logger.info("Extraction cache hit for %s", url)
            return cached

        result = self._extract_uncached(url)
        if result.success:
            cache.set(url, result)
        return result

    # -- orchestration -------------------------------------------------

    def _extract_uncached(self, url: str) -> ExtractionResult:
        logger.info("Starting extraction for %s", url)

        # Strategy 1: site-specific API handler, if one exists for this domain.
        handler = get_handler_for(url)
        if handler is not None:
            logger.info("Using site handler '%s' for %s", handler.name, url)
            try:
                handler_data = handler.extract(url)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Site handler '%s' raised: %s", handler.name, exc)
                handler_data = None
            if handler_data and handler_data.get("success") and handler_data.get("description"):
                return self._to_result(handler_data)
            logger.info("Site handler '%s' did not produce a usable result; falling back", handler.name)

        # Strategy 2-4: static HTML fetch, then JSON-LD / trafilatura / soup.
        static_result = self._try_static_html(url)
        if static_result is not None:
            return static_result

        # Strategy 5: Playwright JS render, then re-run 2-4 against the
        # rendered DOM. Also the first strategy attempted for domains we
        # already know are SPA-rendered, to avoid wasting the static
        # request first (see needs_js_rendering()).
        rendered_result = self._try_playwright(url)
        if rendered_result is not None:
            return rendered_result

        logger.warning("All extraction strategies exhausted for %s", url)
        return ExtractionResult(
            success=False,
            error=(
                "Couldn't extract this job posting after trying direct fetch, "
                "structured-data parsing, and JavaScript rendering. "
                "Please paste the description manually."
            ),
        )

    def _try_static_html(self, url: str) -> ExtractionResult | None:
        if needs_js_rendering(url):
            logger.info("Skipping static fetch for known JS-rendered domain: %s", url)
            return None

        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
        except requests.exceptions.Timeout:
            logger.info("Static fetch timed out for %s", url)
            return None
        except requests.exceptions.ConnectionError:
            logger.info("Static fetch connection error for %s", url)
            return None
        except requests.exceptions.RequestException as exc:
            logger.info("Static fetch failed for %s: %s", url, exc)
            return None

        if response.status_code == 403:
            # Could be a real block, or could just mean "needs JS" — let the
            # Playwright fallback have a shot before giving up.
            logger.info("Static fetch got 403 for %s; will try JS rendering", url)
            return None
        if response.status_code == 404:
            return ExtractionResult(success=False, error="That job posting couldn't be found (it may have been removed).")
        if response.status_code == 429:
            return ExtractionResult(success=False, error="Too many extraction requests right now. Please wait a moment and try again.")
        if not response.ok:
            logger.info("Static fetch got status %s for %s; will try JS rendering", response.status_code, url)
            return None

        html = response.text
        if len(html) < 5000 and _CHALLENGE_PAGE.search(html):
            logger.info("Challenge/CAPTCHA page detected for %s; will try JS rendering", url)
            return None

        parsed = self._parse_html(html, url, source="static_html")
        if parsed is not None and parsed.success:
            return parsed
        return None

    def _try_playwright(self, url: str) -> ExtractionResult | None:
        logger.info("Attempting Playwright render for %s", url)
        try:
            html = render_page(url, timeout_seconds=PLAYWRIGHT_TIMEOUT)
        except BrowserRenderError as exc:
            logger.warning("Playwright render failed for %s: %s", url, exc)
            return ExtractionResult(
                success=False,
                error="This page requires a browser to render and it timed out or failed to load. Please paste the description manually.",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unexpected Playwright failure for %s: %s", url, exc)
            return ExtractionResult(
                success=False,
                error="Couldn't render this page. Please paste the description manually.",
            )

        if not html or len(html) < 100:
            return ExtractionResult(success=False, error="The rendered page came back empty. Please paste the description manually.")

        if _CHALLENGE_PAGE.search(html[:8000]):
            return ExtractionResult(success=False, error="This site requires human verification (CAPTCHA). Please paste the description manually.")

        parsed = self._parse_html(html, url, source="playwright")
        if parsed is not None and parsed.success:
            return parsed

        return ExtractionResult(
            success=False,
            error="Couldn't identify a job description on that page, even after rendering it. Please paste it manually.",
        )

    # -- HTML parsing (shared by static + rendered paths) ---------------

    def _parse_html(self, html: str, url: str, source: str) -> ExtractionResult | None:
        soup = BeautifulSoup(html, "lxml")

        job_ld = self._extract_json_ld_jobposting(soup)
        if job_ld:
            result = self._parse_json_ld(job_ld, source=f"{source}+json_ld")
            if result.success:
                return result

        text = self._fallback_trafilatura(html, url) or self._fallback_soup(soup)
        if not text:
            return None

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        sections = self._extract_labeled_sections(soup)

        return ExtractionResult(
            success=True,
            description=text,
            title=title,
            skills=sections.get("skills", []),
            requirements=sections.get("requirements", []),
            benefits=sections.get("benefits", []),
            source=f"{source}+content_extraction",
            confidence=0.65,
        )

    # -- internals --------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines()]
        seen: set[str] = set()
        cleaned: list[str] = []
        for ln in lines:
            if not ln or _NOISE_KEYWORDS.search(ln):
                continue
            key = ln.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(ln)
        result = "\n".join(cleaned)
        result = re.sub(r"[ \t]+", " ", result)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    @staticmethod
    def _extract_json_ld_jobposting(soup: BeautifulSoup) -> dict | None:
        for tag in soup.find_all("script", type="application/ld+json"):
            raw = tag.string or tag.text
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            candidates = data if isinstance(data, list) else [data]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                graph = candidate.get("@graph")
                if isinstance(graph, list):
                    candidates.extend(g for g in graph if isinstance(g, dict))
                type_field = candidate.get("@type")
                types = type_field if isinstance(type_field, list) else [type_field]
                if types and "JobPosting" in types:
                    return candidate
        return None

    def _parse_json_ld(self, job: dict, source: str = "json_ld") -> ExtractionResult:
        title = job.get("title")

        company = None
        org = job.get("hiringOrganization")
        if isinstance(org, dict):
            company = org.get("name")
        elif isinstance(org, str):
            company = org

        location = None
        loc = job.get("jobLocation")
        if isinstance(loc, list) and loc:
            loc = loc[0]
        if isinstance(loc, dict):
            address = loc.get("address")
            if isinstance(address, dict):
                parts = [address.get("addressLocality"), address.get("addressRegion"), address.get("addressCountry")]
                location = ", ".join(p for p in parts if p)
        elif isinstance(loc, str):
            location = loc

        employment_type = job.get("employmentType")
        if isinstance(employment_type, list):
            employment_type = ", ".join(employment_type)

        salary = None
        base_salary = job.get("baseSalary")
        if isinstance(base_salary, dict):
            value = base_salary.get("value")
            currency = base_salary.get("currency", "")
            if isinstance(value, dict):
                min_v, max_v = value.get("minValue"), value.get("maxValue")
                unit = value.get("unitText", "")
                if min_v and max_v:
                    salary = f"{currency} {min_v}-{max_v} {unit}".strip()
                elif value.get("value"):
                    salary = f"{currency} {value.get('value')} {unit}".strip()

        experience = job.get("experienceRequirements")
        if isinstance(experience, dict):
            experience = experience.get("description") or experience.get("monthsOfExperience")

        education = job.get("educationRequirements")
        if isinstance(education, dict):
            education = education.get("description")

        raw_description = job.get("description") or ""
        description_text = self._clean_text(BeautifulSoup(raw_description, "lxml").get_text("\n"))

        return ExtractionResult(
            success=bool(description_text),
            title=title,
            company=company,
            location=location,
            employment_type=str(employment_type) if employment_type else None,
            experience=str(experience) if experience else None,
            education=str(education) if education else None,
            salary=salary,
            industry=job.get("industry") if isinstance(job.get("industry"), str) else None,
            department=job.get("department") if isinstance(job.get("department"), str) else None,
            posting_date=job.get("datePosted"),
            application_deadline=job.get("validThrough"),
            description=description_text or None,
            source=source,
            confidence=0.9,
        )

    def _fallback_trafilatura(self, html: str, url: str) -> str | None:
        if trafilatura is None:
            return None
        extracted = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=False, favor_precision=True
        )
        return self._clean_text(extracted) if extracted else None

    def _fallback_soup(self, soup: BeautifulSoup) -> str | None:
        # Work on a copy's tags in place is fine here since `soup` isn't
        # reused after this call in the current call sites.
        for tag_name in _NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        candidates = soup.find_all(["main", "article", "section", "div"])
        best_text = ""
        for candidate in candidates:
            text = candidate.get_text("\n", strip=True)
            if len(text) > len(best_text):
                best_text = text
        if not best_text:
            best_text = soup.get_text("\n", strip=True)

        return self._clean_text(best_text) or None

    def _extract_labeled_sections(self, soup: BeautifulSoup) -> dict[str, list[str]]:
        """Best-effort pull of skills/requirements/benefits list items near
        a recognizable heading, for portals without JSON-LD."""

        found: dict[str, list[str]] = {}
        for section_name, labels in _SECTION_LABELS.items():
            for label in labels:
                heading = soup.find(string=re.compile(re.escape(label), re.IGNORECASE))
                if heading is None or not hasattr(heading, "parent"):
                    continue
                parent = heading.parent
                container = parent.parent if parent.name in {"h1", "h2", "h3", "h4", "h5", "h6"} else parent
                if container is None:
                    continue
                items = [
                    self._clean_text(li.get_text(" ", strip=True))
                    for li in container.find_all("li")
                ]
                items = [i for i in items if i and i.lower() != label.lower()]
                if items:
                    found[section_name] = items[:20]
                    break
        return found

    def _to_result(self, data: dict) -> ExtractionResult:
        """Normalize a site-handler dict into an ExtractionResult."""

        skills = data.get("skills") or []
        if isinstance(skills, str):
            skills = [s.strip() for s in re.split(r"[,;\n]", skills) if s.strip()]

        return ExtractionResult(
            success=bool(data.get("success")),
            description=data.get("description"),
            title=data.get("title"),
            company=data.get("company"),
            location=data.get("location"),
            employment_type=data.get("employment_type"),
            experience=data.get("experience"),
            salary=data.get("salary"),
            error=data.get("error"),
            skills=skills,
            requirements=data.get("requirements") or [],
            benefits=data.get("benefits") or [],
            department=data.get("department"),
            industry=data.get("industry"),
            education=data.get("education"),
            posting_date=data.get("posting_date"),
            application_deadline=data.get("application_deadline"),
            source=data.get("source"),
            confidence=data.get("confidence"),
        )