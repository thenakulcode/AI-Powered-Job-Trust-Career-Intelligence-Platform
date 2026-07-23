"""Site-specific extraction strategies.

Several major ATS platforms expose a stable JSON API that's far more
reliable than scraping their (often client-side-rendered) HTML. Each
handler below:

  - detects whether a URL belongs to it (`matches`)
  - fetches structured data directly from the platform's API where one
    exists, or provides a tuned Playwright wait-strategy where it doesn't
  - returns a partially-filled `ExtractionResult`-shaped dict, using the
    same keys as `extraction_service.ExtractionResult` so the orchestrator
    can merge/normalize results identically regardless of source

Adding a new site: subclass `SiteHandler`, implement `matches` + `extract`,
and add it to `HANDLERS` at the bottom. Nothing else needs to change.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("jobshield.extraction.sites")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15


def _clean_html_to_text(html: str) -> str:
    if not html:
        return ""
    text = BeautifulSoup(html, "lxml").get_text("\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class SiteHandler:
    name = "generic"

    def matches(self, url: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def extract(self, url: str) -> dict[str, Any] | None:
        """Return a partial result dict on success, or None to let the
        orchestrator fall through to the next strategy."""
        raise NotImplementedError  # pragma: no cover - interface


class MicrosoftCareersHandler(SiteHandler):
    """Microsoft Careers (apply.careers.microsoft.com) renders entirely via
    client-side JS but is backed by a public REST API keyed by job id (the
    `pid` query param on job pages)."""

    name = "microsoft_careers"
    API_URL = "https://gcsservices.careers.microsoft.com/search/api/v1/job/{job_id}"

    def matches(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return "careers.microsoft.com" in host

    def _job_id_from_url(self, url: str) -> str | None:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        for key in ("pid", "jobId", "job_id"):
            if key in query and query[key]:
                return query[key][0]
        match = re.search(r"/job/(\d+)", parsed.path)
        if match:
            return match.group(1)
        return None

    def extract(self, url: str) -> dict[str, Any] | None:
        job_id = self._job_id_from_url(url)
        if not job_id:
            logger.info("Microsoft Careers handler: no job id found in %s", url)
            return None

        try:
            response = requests.get(
                self.API_URL.format(job_id=job_id),
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.info("Microsoft Careers API failed for job %s: %s", job_id, exc)
            return None

        job = payload.get("result") or payload.get("job") or payload
        if not isinstance(job, dict):
            return None

        title = job.get("title")
        description_html = job.get("description") or job.get("descriptionExternal") or ""
        qualifications = job.get("qualifications") or ""
        responsibilities = job.get("responsibilities") or ""

        description_parts = [
            _clean_html_to_text(description_html),
            _clean_html_to_text(responsibilities),
            _clean_html_to_text(qualifications),
        ]
        description = "\n\n".join(p for p in description_parts if p)

        locations = job.get("primaryWorkLocation") or job.get("workLocations") or {}
        if isinstance(locations, list) and locations:
            locations = locations[0]
        location = None
        if isinstance(locations, dict):
            city = locations.get("city") or locations.get("cityName")
            state = locations.get("state") or locations.get("region")
            country = locations.get("country") or locations.get("countryName")
            location = ", ".join(p for p in (city, state, country) if p) or None

        employment_type = job.get("employmentType")

        return {
            "success": bool(description or title),
            "title": title,
            "company": "Microsoft",
            "location": location,
            "employment_type": employment_type,
            "description": description or None,
            "source": self.name,
            "confidence": 0.95 if description else 0.6,
        }


class GreenhouseHandler(SiteHandler):
    """boards.greenhouse.io and job-boards.greenhouse.io both expose a
    stable JSON endpoint per job id."""

    name = "greenhouse"

    def matches(self, url: str) -> bool:
        return "greenhouse.io" in urlparse(url).netloc.lower()

    def extract(self, url: str) -> dict[str, Any] | None:
        parsed = urlparse(url)
        match = re.search(r"/jobs/(\d+)", parsed.path)
        board_match = re.match(r"^/([^/]+)/", parsed.path) or re.match(r"^([^.]+)\.", parsed.netloc)
        if not match:
            return None
        job_id = match.group(1)

        board_token = None
        segments = [s for s in parsed.path.split("/") if s]
        if segments:
            board_token = segments[0]
        if not board_token and "." in parsed.netloc:
            board_token = parsed.netloc.split(".")[0]

        if not board_token:
            return None

        api_url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs/{job_id}"
        try:
            response = requests.get(api_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            job = response.json()
        except Exception as exc:
            logger.info("Greenhouse API failed for %s: %s", url, exc)
            return None

        location = None
        if isinstance(job.get("location"), dict):
            location = job["location"].get("name")

        company = None
        if isinstance(job.get("departments"), list) and job["departments"]:
            company = job.get("company_name")

        return {
            "success": True,
            "title": job.get("title"),
            "company": job.get("company_name") or company,
            "location": location,
            "description": _clean_html_to_text(job.get("content", "")),
            "source": self.name,
            "confidence": 0.95,
        }


class LeverHandler(SiteHandler):
    """jobs.lever.co exposes `/{company}/{posting_id}.json`."""

    name = "lever"

    def matches(self, url: str) -> bool:
        return "lever.co" in urlparse(url).netloc.lower()

    def extract(self, url: str) -> dict[str, Any] | None:
        parsed = urlparse(url)
        segments = [s for s in parsed.path.split("/") if s]
        if len(segments) < 2:
            return None
        company, posting_id = segments[0], segments[1]
        api_url = f"https://api.lever.co/v0/postings/{company}/{posting_id}?mode=json"

        try:
            response = requests.get(api_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            job = response.json()
        except Exception as exc:
            logger.info("Lever API failed for %s: %s", url, exc)
            return None

        if isinstance(job, list):
            job = job[0] if job else {}
        if not job:
            return None

        categories = job.get("categories", {}) or {}
        description_parts = [
            _clean_html_to_text(job.get("descriptionPlain") or job.get("description", "")),
        ]
        for list_item in job.get("lists", []) or []:
            heading = list_item.get("text", "")
            content = _clean_html_to_text(list_item.get("content", ""))
            if content:
                description_parts.append(f"{heading}\n{content}" if heading else content)

        return {
            "success": True,
            "title": job.get("text"),
            "company": company.replace("-", " ").title(),
            "location": categories.get("location"),
            "employment_type": categories.get("commitment"),
            "description": "\n\n".join(p for p in description_parts if p),
            "source": self.name,
            "confidence": 0.95,
        }


class AshbyHandler(SiteHandler):
    """jobs.ashbyhq.com postings are served by a public GraphQL-ish JSON API
    keyed by org + job slug."""

    name = "ashby"

    def matches(self, url: str) -> bool:
        return "ashbyhq.com" in urlparse(url).netloc.lower()

    def extract(self, url: str) -> dict[str, Any] | None:
        parsed = urlparse(url)
        segments = [s for s in parsed.path.split("/") if s]
        if len(segments) < 2:
            return None
        org, job_slug = segments[0], segments[-1]

        api_url = "https://api.ashbyhq.com/posting-api/job-board/" + org
        try:
            response = requests.get(api_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            board = response.json()
        except Exception as exc:
            logger.info("Ashby board API failed for %s: %s", url, exc)
            return None

        jobs = board.get("jobs", []) if isinstance(board, dict) else []
        job = next((j for j in jobs if j.get("id") == job_slug or job_slug in (j.get("jobUrl") or "")), None)
        if not job:
            return None

        return {
            "success": True,
            "title": job.get("title"),
            "company": org.replace("-", " ").title(),
            "location": job.get("location"),
            "employment_type": job.get("employmentType"),
            "description": _clean_html_to_text(job.get("descriptionHtml", "")),
            "source": self.name,
            "confidence": 0.9,
        }


# Ordered: more specific / higher-confidence handlers first.
HANDLERS: list[SiteHandler] = [
    MicrosoftCareersHandler(),
    GreenhouseHandler(),
    LeverHandler(),
    AshbyHandler(),
]


def get_handler_for(url: str) -> SiteHandler | None:
    host = urlparse(url).netloc.lower()
    if not host:
        return None
    for handler in HANDLERS:
        try:
            if handler.matches(url):
                return handler
        except Exception:  # pragma: no cover - defensive
            continue
    return None


# Domains known to require JS rendering (no public API, but SPA-rendered).
# Used by the orchestrator to skip straight to Playwright instead of wasting
# a request on a static-HTML attempt that will always come back empty.
JS_RENDERED_DOMAINS = (
    "careers.microsoft.com",  # handled by API above, but kept as a safety net
    "myworkdayjobs.com",
    "wd1.myworkdayjobs.com",
    "smartrecruiters.com",
    "successfactors.com",
    "oraclecloud.com",
    "linkedin.com",
)


def needs_js_rendering(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(domain in host for domain in JS_RENDERED_DOMAINS)
