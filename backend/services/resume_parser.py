"""Resume parsing utilities for PDF and DOCX files."""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any

import fitz
from docx import Document

SECTION_LABELS: dict[str, tuple[str, ...]] = {
    "skills": ("skills", "technical skills", "core skills", "tools", "technologies"),
    "education": ("education", "academic background", "academics"),
    "projects": ("projects", "selected projects", "personal projects", "portfolio"),
    "experience": ("experience", "work experience", "professional experience", "employment"),
    "certifications": ("certifications", "licenses", "certificates"),
    "achievements": ("achievements", "awards", "accomplishments", "highlights"),
}

SKILL_TERMS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "node",
    "fastapi",
    "django",
    "flask",
    "sql",
    "postgres",
    "mysql",
    "mongodb",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "linux",
    "git",
    "rest",
    "api",
    "microservices",
    "machine learning",
    "ai",
    "data science",
    "nlp",
    "spark",
    "hadoop",
    "tableau",
    "power bi",
    "excel",
    "pytest",
    "jenkins",
    "ci/cd",
    "html",
    "css",
]


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    document = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = [page.get_text("text") for page in document]
    document.close()
    return "\n".join(chunks)


def _extract_text_from_docx(file_bytes: bytes) -> str:
    document = Document(BytesIO(file_bytes))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n".join(paragraphs)


def extract_resume_text(file_name: str, file_bytes: bytes) -> str:
    name = (file_name or "").lower()
    if name.endswith(".pdf"):
        return _extract_text_from_pdf(file_bytes)
    if name.endswith(".docx"):
        return _extract_text_from_docx(file_bytes)
    raise ValueError("Unsupported file type. Please upload a PDF or DOCX resume.")


def parse_resume_sections(resume_text: str) -> dict[str, Any]:
    text = normalize_text(resume_text or "")
    if not text:
        return {
            "skills": [],
            "education": [],
            "projects": [],
            "experience": [],
            "certifications": [],
            "achievements": [],
            "summary": "",
        }

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections: dict[str, list[str]] = {key: [] for key in SECTION_LABELS}
    current_section: str | None = None

    for line in lines:
        lowered = line.lower()
        matched_section = None
        for key, labels in SECTION_LABELS.items():
            if any(lowered.startswith(label + ":") or lowered == label or lowered.startswith(label + " ") for label in labels):
                matched_section = key
                break
        if matched_section:
            current_section = matched_section
            if line.lower() not in {label for labels in SECTION_LABELS.values() for label in labels}:
                sections[matched_section].append(line)
            continue

        if current_section:
            sections[current_section].append(line)

    def clean_items(items: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in items:
            candidate = re.sub(r"^[-•*]\s*", "", item).strip()
            if candidate:
                cleaned.append(candidate)
        return cleaned

    skills = _extract_skills(text, clean_items(sections["skills"]))
    education = clean_items(sections["education"])
    projects = clean_items(sections["projects"])
    experience = clean_items(sections["experience"])
    certifications = clean_items(sections["certifications"])
    achievements = clean_items(sections["achievements"])

    if not skills and not education and not projects and not experience and not certifications and not achievements:
        # Fallback to a light heuristic from the whole text when no explicit sections exist.
        skills = _extract_skills(text, [])
        if not experience:
            experience = [sent for sent in re.split(r"(?<=[.!?])\s+", text) if len(sent.split()) >= 4][:3]

    return {
        "skills": skills,
        "education": education,
        "projects": projects,
        "experience": experience,
        "certifications": certifications,
        "achievements": achievements,
        "summary": text[:500],
    }


def _extract_skills(text: str, section_items: list[str]) -> list[str]:
    candidates: list[str] = []
    for item in section_items:
        candidates.extend([part.strip() for part in re.split(r"[,;|/]+", item) if part.strip()])

    for term in SKILL_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
            candidates.append(term)

    unique_items: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        value = normalize_text(item).lower()
        if not value or value in seen:
            continue
        seen.add(value)
        unique_items.append(value.title() if len(value.split()) == 1 else value)

    return unique_items
