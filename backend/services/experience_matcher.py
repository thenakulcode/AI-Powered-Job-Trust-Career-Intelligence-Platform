"""Domain-independent experience matching.

Scores resume experience against a job description using three signals:
  - years of experience vs. the years required (if stated)
  - role-title relevance (lexical/semantic similarity of role phrases)
  - industry/domain relevance (shared key phrases between experience text
    and the job description, using the dynamic extractor rather than a
    fixed "developer/engineer/backend" keyword list)
"""

from __future__ import annotations

import re

from backend.services.skill_extraction import (
    extract_key_phrases,
    lexical_similarity,
    match_phrases,
    normalize_text,
)

_YEARS_RE = re.compile(r"(\d+)\s*\+?\s*(?:years?|yrs?|yr)", re.I)
_ROLE_TITLE_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}(?:\s*[-–]\s*[A-Z][a-zA-Z]+)?)\b"
)


class ExperienceMatcher:
    def score(self, job_description: str, resume_experience_lines: list[str], resume_total_years: int) -> tuple[float, str]:
        job_text = normalize_text(job_description)
        experience_text = normalize_text(" ".join(resume_experience_lines or []))

        years_score, years_reason = self._years_score(job_text, resume_total_years)
        relevance_score, relevance_reason = self._relevance_score(job_text, experience_text)

        combined = years_score * 0.4 + relevance_score * 0.6
        reason = f"{years_reason} {relevance_reason}".strip()
        return combined, reason

    def _years_score(self, job_text: str, resume_years: int) -> tuple[float, str]:
        matches = _YEARS_RE.findall(job_text)
        target_years = int(matches[0]) if matches else None
        if target_years is None:
            if resume_years >= 3:
                return 80.0, "The resume shows solid tenure even though the job does not specify a required duration."
            if resume_years >= 1:
                return 65.0, "The resume shows some experience; the job does not specify a required duration."
            return 45.0, "The job does not specify a required duration; limited experience is shown."
        if resume_years >= target_years:
            return 100.0, f"Meets or exceeds the {target_years}+ year requirement."
        if resume_years >= max(target_years - 1, 0):
            return 65.0, f"Close to the {target_years}+ year requirement."
        return max(20.0, 50.0 * (resume_years / max(target_years, 1))), f"Below the {target_years}+ year requirement."

    def _relevance_score(self, job_text: str, experience_text: str) -> tuple[float, str]:
        if not experience_text:
            return 20.0, "No experience section detected."
        job_phrases = extract_key_phrases(job_text, top_n=20)
        experience_phrases = extract_key_phrases(experience_text, top_n=25)
        if not job_phrases:
            return 60.0, "No specific role or industry signals found in the job description."
        matched, missing, _ = match_phrases(experience_phrases, job_phrases, threshold=0.55)
        ratio = len(matched) / max(len(job_phrases), 1)
        score = min(100.0, 30.0 + ratio * 70.0)
        if ratio >= 0.6:
            return score, "Experience closely relates to the responsibilities in the job description."
        if ratio >= 0.3:
            return score, "Experience partially overlaps with the role's responsibilities."
        return score, "Experience shows limited overlap with the role's responsibilities."
