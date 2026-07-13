from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


class ScrapingError(RuntimeError):
    """Raised when a job posting cannot be extracted from a public page."""


def is_supported_page(url: str) -> bool:
    """Quick heuristic to block obvious login/auth pages and support common job boards."""

    lowered = url.lower()
    if any(token in lowered for token in ["/login", "/signin", "/auth", "/account"]):
        return False
    if any(domain in lowered for domain in ["linkedin.com", "indeed.com", "naukri.com", "glassdoor.com", "lever.co", "greenhouse.io", "jobs", "career"]):
        return True
    return True


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(value: str | None) -> str:
    return normalize_whitespace(value or "")


def extract_location_from_text(text: str | None) -> str:
    value = clean_text(text)
    if not value:
        return ""

    for delimiter in ["·", "|", "–", "-", "→"]:
        if delimiter in value:
            parts = [part.strip() for part in value.split(delimiter)]
            for part in reversed(parts):
                if part and not part.lower().startswith(("remote", "hybrid", "onsite", "on-site")):
                    return part
    if re.search(r"\b(?:Remote|Hybrid|On-site|Onsite)\b", value, re.I):
        for part in re.split(r"\b(?:Remote|Hybrid|On-site|Onsite)\b", value, flags=re.I):
            if part.strip():
                return clean_text(part)
    return value


def _extract_by_labels(soup: BeautifulSoup, labels: list[str]) -> str:
    for label in labels:
        for element in soup.find_all(string=re.compile(label, re.IGNORECASE)):
            parent = element.parent if hasattr(element, "parent") else None
            if parent is None:
                continue
            text = normalize_whitespace(parent.get_text(" ", strip=True))
            if text:
                return text
    return ""


def extract_job_data_from_html(html: str, url: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    title_candidates = [soup.title.string if soup.title and soup.title.string else ""]
    title_candidates.extend(
        [
            clean_text(soup.find("h1").get_text(" ", strip=True)) if soup.find("h1") else "",
            clean_text(soup.find(["h2", "h3"], class_=re.compile(r"title|job|heading", re.I)).get_text(" ", strip=True))
            if soup.find(["h2", "h3"], class_=re.compile(r"title|job|heading", re.I))
            else "",
        ]
    )
    for candidate in title_candidates:
        if candidate:
            title = candidate
            break

    company = ""
    company_candidates = [
        clean_text(soup.find(attrs={"data-company-name": True}).get("data-company-name")) if soup.find(attrs={"data-company-name": True}) else "",
        clean_text(soup.find(class_=re.compile(r"company|employer|org", re.I)).get_text(" ", strip=True)) if soup.find(class_=re.compile(r"company|employer|org", re.I)) else "",
    ]
    for heading in soup.find_all(["h2", "h3", "h4"]):
        heading_text = clean_text(heading.get_text(" ", strip=True))
        if heading_text and heading_text.lower() not in {title.lower(), "responsibilities", "requirements", "benefits", "skills", "salary"}:
            company_candidates.append(heading_text)
    for candidate in company_candidates:
        if candidate:
            company = candidate
            break

    location = ""
    location_candidates = [
        extract_location_from_text(clean_text(soup.find(string=re.compile(r"(Remote|Hybrid|On-site|New York|Los Angeles|London|Berlin|Delhi|Mumbai)", re.I)))),
        extract_location_from_text(clean_text(soup.find(class_=re.compile(r"location|geo", re.I)).get_text(" ", strip=True))) if soup.find(class_=re.compile(r"location|geo", re.I)) else "",
    ]
    for candidate in company_candidates:
        if candidate:
            company = candidate
            break

    for candidate in location_candidates:
        if candidate:
            location = candidate
            break

    description = ""
    meta_desc = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
    if meta_desc and meta_desc.get("content"):
        description = clean_text(meta_desc.get("content"))

    responsibilities = ""
    requirements = ""
    skills = ""
    benefits = ""
    salary = ""

    section_map = {
        "responsibilities": ["responsibilities", "what you'll do", "what you will do", "your role", "job responsibilities"],
        "requirements": ["requirements", "qualifications", "what we're looking for", "what we are looking for", "minimum qualifications"],
        "skills": ["skills", "preferred skills", "core skills", "competencies"],
        "benefits": ["benefits", "why join us", "perks", "company benefits"],
        "salary": ["salary", "compensation", "pay range", "base pay"],
    }

    def collect_section_text(section_name: str, labels: list[str]) -> str:
        for label in labels:
            heading = soup.find(string=re.compile(label, re.I))
            if heading is None:
                continue
            parent = heading.parent if hasattr(heading, "parent") else None
            if parent is None:
                continue
            section = parent.parent if parent.name in {"h1", "h2", "h3", "h4", "h5", "h6"} else parent
            if section is None:
                continue
            text_parts = []
            for child in section.find_all(["li", "p", "span"]):
                chunk = clean_text(child.get_text(" ", strip=True))
                if chunk and chunk.lower() not in {label.lower(), "responsibilities", "requirements", "benefits", "skills", "salary"}:
                    text_parts.append(chunk)
            if text_parts:
                return "; ".join(text_parts)
            fallback_text = normalize_whitespace(section.get_text(" ", strip=True))
            if fallback_text and fallback_text.lower() != label.lower():
                return fallback_text
        return ""

    responsibilities = collect_section_text("responsibilities", section_map["responsibilities"])
    requirements = collect_section_text("requirements", section_map["requirements"])
    skills = collect_section_text("skills", section_map["skills"])
    benefits = collect_section_text("benefits", section_map["benefits"])
    salary = collect_section_text("salary", section_map["salary"])

    description_parts = []
    if description:
        description_parts.append(description)
    if responsibilities:
        description_parts.append(f"Responsibilities: {responsibilities}")
    if requirements:
        description_parts.append(f"Requirements: {requirements}")
    if skills:
        description_parts.append(f"Skills: {skills}")
    if benefits:
        description_parts.append(f"Benefits: {benefits}")
    if salary:
        description_parts.append(f"Salary: {salary}")

    combined_text = " ".join(part for part in description_parts if part).strip()

    return {
        "title": title,
        "company": company,
        "location": location,
        "description": combined_text or f"Job posting from {url}",
        "responsibilities": responsibilities,
        "requirements": requirements,
        "skills": skills,
        "benefits": benefits,
        "salary": salary,
    }


def fetch_page_with_requests(url: str, timeout: int = 10) -> str:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_page_with_playwright(url: str, timeout: int = 20) -> str:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0")
        page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
        html = page.content()
        browser.close()
        return html


def extract_job_post(url: str, timeout: int = 15) -> dict[str, str]:
    if not is_supported_page(url):
        raise ScrapingError("Unable to extract job description from this website.")

    try:
        html = fetch_page_with_requests(url, timeout=timeout)
    except Exception:
        try:
            html = fetch_page_with_playwright(url, timeout=timeout)
        except Exception as exc:
            raise ScrapingError("Unable to extract job description from this website.") from exc

    if not html or "<html" not in html.lower():
        raise ScrapingError("Unable to extract job description from this website.")

    data = extract_job_data_from_html(html, url)
    if all(not data.get(field, "") for field in ["title", "company", "location", "description"]):
        raise ScrapingError("Unable to extract job description from this website.")
    return data
