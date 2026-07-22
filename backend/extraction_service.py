"""URL-based job posting extraction service."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:  # pragma: no cover - optional dependency guard
    trafilatura = None

# These could be promoted into backend/config.py alongside FRAUD_THRESHOLD
# if you want them environment-configurable later; kept local for now since
# they're only used here.
REQUEST_TIMEOUT = 15  # seconds
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
    r"(captcha|are you a human|access denied|checking your browser)", re.IGNORECASE
)


@dataclass(slots=True)
class ExtractionResult:
    """Container for URL-based job extraction output."""

    success: bool
    description: str | None = None
    title: str | None = None
    company: str | None = None
    location: str | None = None
    employment_type: str | None = None
    experience: str | None = None
    salary: str | None = None
    error: str | None = None


class JobExtractionError(RuntimeError):
    """Raised for extraction failures that should surface as a clean message
    rather than a 500, mirroring how ModelLoadError is used in service.py."""


class JobExtractionService:
    """Fetches a job-posting URL and extracts structured job data.

    Extraction strategy (cheapest/most-reliable first):
      1. schema.org JobPosting JSON-LD (present on most LinkedIn, Indeed,
         Naukri, Glassdoor, Greenhouse and Lever pages) — structured, so no
         guessing about layout.
      2. trafilatura's readability-style main-content extraction.
      3. A generic BeautifulSoup pass that strips boilerplate tags and
         returns the largest remaining text block.

    Playwright/Selenium are deliberately not used here — per spec, those are
    only worth the overhead for sites that render the job description via
    client-side JS. If you find a specific domain always returns empty
    content through this pipeline, that's the place to add a targeted
    Playwright fallback rather than making every request pay that cost.
    """

    def extract(self, url: str) -> ExtractionResult:
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
        except requests.exceptions.Timeout:
            return ExtractionResult(success=False, error="The site took too long to respond. Please try again or paste manually.")
        except requests.exceptions.ConnectionError:
            return ExtractionResult(success=False, error="Couldn't connect to that site. Check the URL and try again.")
        except requests.exceptions.RequestException:
            return ExtractionResult(success=False, error="Couldn't fetch that URL. Please paste the description manually.")

        if response.status_code == 403:
            return ExtractionResult(success=False, error="This site is blocking automated access. Please paste the description manually.")
        if response.status_code == 404:
            return ExtractionResult(success=False, error="That job posting couldn't be found (it may have been removed).")
        if response.status_code == 429:
            return ExtractionResult(success=False, error="Too many extraction requests right now. Please wait a moment and try again.")
        if not response.ok:
            return ExtractionResult(success=False, error=f"The site returned an error (status {response.status_code}).")

        html = response.text
        if len(html) < 5000 and _CHALLENGE_PAGE.search(html):
            return ExtractionResult(success=False, error="This site requires human verification (CAPTCHA). Please paste the description manually.")

        soup = BeautifulSoup(html, "lxml")

        job_ld = self._extract_json_ld_jobposting(soup)
        if job_ld:
            result = self._parse_json_ld(job_ld)
            if result.success:
                return result

        text = self._fallback_trafilatura(html, url) or self._fallback_soup(soup)
        if not text:
            return ExtractionResult(success=False, error="Couldn't identify a job description on that page. Please paste it manually.")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        return ExtractionResult(success=True, description=text, title=title)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Collapse whitespace, drop duplicate/noise lines from raw HTML text."""

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
        """Find a schema.org JobPosting block in <script type="application/ld+json">."""

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

    def _parse_json_ld(self, job: dict) -> ExtractionResult:
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

        raw_description = job.get("description") or ""
        description_text = self._clean_text(BeautifulSoup(raw_description, "lxml").get_text("\n"))

        return ExtractionResult(
            success=bool(description_text),
            title=title,
            company=company,
            location=location,
            employment_type=str(employment_type) if employment_type else None,
            experience=str(experience) if experience else None,
            salary=salary,
            description=description_text or None,
        )

    def _fallback_trafilatura(self, html: str, url: str) -> str | None:
        if trafilatura is None:
            return None
        extracted = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=False, favor_precision=True
        )
        return self._clean_text(extracted) if extracted else None

    def _fallback_soup(self, soup: BeautifulSoup) -> str | None:
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