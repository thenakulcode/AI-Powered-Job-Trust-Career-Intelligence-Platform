"""Dynamic education matching, independent of any specific field of study.

Instead of checking for "computer science / software / engineering", this
extracts (degree_level, field_of_study) pairs from both the job description
and the resume's education section, then scores overlap by degree-level rank
and field-of-study phrase similarity (lexical/semantic via skill_extraction).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.services.skill_extraction import lexical_similarity, normalize_phrase

_DEGREE_LEVELS: dict[str, int] = {
    "diploma": 1,
    "associate": 2,
    "bachelor": 3,
    "b.tech": 3,
    "btech": 3,
    "b.sc": 3,
    "bsc": 3,
    "b.a": 3,
    "b.e": 3,
    "master": 4,
    "m.tech": 4,
    "mtech": 4,
    "m.sc": 4,
    "msc": 4,
    "m.a": 4,
    "mba": 4,
    "phd": 5,
    "ph.d": 5,
    "doctorate": 5,
    "md": 5,
    "m.d": 5,
    "jd": 5,
    "j.d": 5,
}

_DEGREE_PATTERN = re.compile(
    r"\b(diploma|associate(?:'s)?|bachelor(?:'s)?|b\.?\s?tech|b\.?\s?sc|b\.?\s?a|b\.?\s?e|"
    r"master(?:'s)?|m\.?\s?tech|m\.?\s?sc|m\.?\s?a|mba|ph\.?\s?d|doctorate|m\.?\s?d|j\.?\s?d)\b"
    r"(?:\s+(?:of|in|degree in))?\s*([a-z][a-z\s&/-]{2,60})?",
    re.I,
)


@dataclass
class EducationRequirement:
    degree_level: str | None
    degree_rank: int
    field: str | None


def _degree_rank(token: str) -> int:
    key = re.sub(r"[.\s]", "", token.lower())
    return _DEGREE_LEVELS.get(key, _DEGREE_LEVELS.get(token.lower().strip(), 0))


def extract_education_requirements(text_lines: list[str] | str) -> list[EducationRequirement]:
    if isinstance(text_lines, list):
        text = " . ".join(text_lines)
    else:
        text = text_lines or ""
    requirements: list[EducationRequirement] = []
    for match in _DEGREE_PATTERN.finditer(text):
        degree_token = match.group(1)
        field = (match.group(2) or "").strip(" .,-")
        rank = _degree_rank(degree_token)
        requirements.append(
            EducationRequirement(
                degree_level=degree_token.lower(),
                degree_rank=rank,
                field=normalize_phrase(field) if field else None,
            )
        )
    return requirements


class EducationMatcher:
    """Scores resume education against job education requirements without
    assuming any particular field of study is the "correct" one."""

    def score(self, job_description: str, resume_education_lines: list[str]) -> tuple[float, str]:
        job_requirements = extract_education_requirements(job_description)
        resume_requirements = extract_education_requirements(resume_education_lines)

        if not job_requirements:
            # Job doesn't specify a degree requirement; presence of any
            # education is a mild positive signal, absence is neutral.
            return (70.0 if resume_requirements else 55.0), "No explicit degree requirement detected in the job description."

        if not resume_requirements:
            return 20.0, "The job specifies an education requirement that the resume does not show."

        best_score = 0.0
        best_reason = "Education does not align with the job's requirements."
        for job_req in job_requirements:
            for resume_req in resume_requirements:
                level_score = self._level_score(job_req.degree_rank, resume_req.degree_rank)
                field_score = self._field_score(job_req.field, resume_req.field)
                combined = level_score * 0.5 + field_score * 0.5
                if combined > best_score:
                    best_score = combined
                    best_reason = self._describe(job_req, resume_req, level_score, field_score)

        return best_score, best_reason

    def _level_score(self, job_rank: int, resume_rank: int) -> float:
        if job_rank == 0:
            return 70.0
        if resume_rank >= job_rank:
            return 100.0
        if resume_rank == job_rank - 1:
            return 55.0
        return 25.0

    def _field_score(self, job_field: str | None, resume_field: str | None) -> float:
        if not job_field:
            return 70.0
        if not resume_field:
            return 30.0
        similarity = lexical_similarity(job_field, resume_field)
        return 30.0 + similarity * 70.0

    def _describe(self, job_req: EducationRequirement, resume_req: EducationRequirement, level_score: float, field_score: float) -> str:
        field_part = f" in {resume_req.field}" if resume_req.field else ""
        job_field_part = f" in {job_req.field}" if job_req.field else ""
        if field_score >= 80 and level_score >= 80:
            return f"Education closely matches the job's requirement ({job_req.degree_level}{job_field_part})."
        if field_score < 50:
            return f"The resume shows a {resume_req.degree_level}{field_part}, which differs from the job's requirement ({job_req.degree_level}{job_field_part})."
        return f"Education partially aligns with the job's requirement ({job_req.degree_level}{job_field_part})."
