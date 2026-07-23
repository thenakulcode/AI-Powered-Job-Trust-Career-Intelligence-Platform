"""Domain-aware project/portfolio-equivalent matching.

Different professions have different "portfolio evidence": software has
projects, research has publications, teaching has workshops/curricula,
marketing has campaigns, healthcare has clinical work, law has cases. This
matcher looks at the detected domain to decide which resume sections count
as portfolio evidence, then scores overlap with the job's key phrases using
the dynamic extractor — never assuming "project" means a coding project.
"""

from __future__ import annotations

from backend.services.domain_detector import DomainMatch
from backend.services.skill_extraction import extract_key_phrases, match_phrases, normalize_text

# Which resume sections count as "portfolio evidence" per domain, and how to
# describe that evidence in generated text. Falls back to a generic set for
# domains not explicitly listed.
_DOMAIN_EVIDENCE_SECTIONS: dict[str, tuple[str, ...]] = {
    "software_engineering": ("projects", "experience", "achievements"),
    "ai_ml_data_science": ("projects", "achievements", "experience"),
    "research_phd": ("achievements", "certifications", "experience"),
    "teaching_academia": ("achievements", "experience", "certifications"),
    "marketing": ("achievements", "projects", "experience"),
    "sales": ("achievements", "experience"),
    "healthcare_medicine": ("experience", "certifications"),
    "nursing": ("experience", "certifications"),
    "law_legal": ("experience", "achievements"),
    "forensic_science": ("experience", "achievements", "certifications"),
}

_EVIDENCE_LABEL: dict[str, str] = {
    "software_engineering": "projects",
    "ai_ml_data_science": "projects",
    "research_phd": "publications and research work",
    "teaching_academia": "teaching materials and workshops",
    "marketing": "campaigns",
    "sales": "sales achievements",
    "healthcare_medicine": "clinical experience",
    "nursing": "clinical experience",
    "law_legal": "case work",
    "forensic_science": "casework and lab experience",
}


class ProjectMatcher:
    def score(
        self,
        job_description: str,
        parsed_resume: dict,
        domain_match: DomainMatch,
    ) -> tuple[float, str]:
        sections = _DOMAIN_EVIDENCE_SECTIONS.get(domain_match.domain, ("projects", "achievements", "experience"))
        evidence_lines: list[str] = []
        for section in sections:
            evidence_lines.extend(parsed_resume.get(section, []) or [])

        label = _EVIDENCE_LABEL.get(domain_match.domain, "relevant portfolio evidence")

        if not evidence_lines:
            return 20.0, f"No {label} were found in the resume."

        job_text = normalize_text(job_description)
        evidence_text = normalize_text(" ".join(evidence_lines))
        job_phrases = extract_key_phrases(job_text, top_n=20)
        evidence_phrases = extract_key_phrases(evidence_text, top_n=25)

        if not job_phrases:
            return 65.0, f"The resume includes {label}, but the job description does not specify concrete evidence requirements."

        matched, missing, _ = match_phrases(evidence_phrases, job_phrases, threshold=0.55)
        ratio = len(matched) / max(len(job_phrases), 1)
        score = min(100.0, 35.0 + ratio * 65.0)
        if ratio >= 0.5:
            return score, f"The {label} shown align well with the job's requirements."
        if ratio >= 0.25:
            return score, f"The {label} shown partially align with the job's requirements."
        return score, f"The {label} shown have limited overlap with the job's requirements."
