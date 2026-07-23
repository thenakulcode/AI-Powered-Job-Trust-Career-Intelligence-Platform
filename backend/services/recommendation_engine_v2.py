"""Domain-independent recommendation generation.

Replaces the old RecommendationEngine, which only had templates for a fixed
list of software skills. This version generates suggestions, courses,
projects, and roadmap steps for *any* missing phrase, using the detected
domain to phrase things naturally (e.g. "workshop" for teaching, "campaign"
for marketing, "case study" for law) without hardcoding a skill catalog.
"""

from __future__ import annotations

from backend.services.domain_detector import DomainMatch
from backend.services.skill_extraction import display_phrase, normalize_phrase

# Per-domain vocabulary used only to phrase generic recommendations more
# naturally — this is NOT a skill catalog, it's just wording. Any domain not
# listed here still gets fully functional, if slightly more generic, output.
_DOMAIN_VOCAB: dict[str, dict[str, str]] = {
    "software_engineering": {"practice_unit": "project", "course_style": "hands-on course", "artifact": "one small working project"},
    "ai_ml_data_science": {"practice_unit": "model/notebook", "course_style": "applied course", "artifact": "one trained model with a short writeup"},
    "cybersecurity": {"practice_unit": "lab exercise", "course_style": "hands-on lab", "artifact": "one documented security assessment"},
    "ui_ux_design": {"practice_unit": "design mockup", "course_style": "design course", "artifact": "one polished case study"},
    "hr": {"practice_unit": "process document", "course_style": "certification course", "artifact": "one process improvement writeup"},
    "marketing": {"practice_unit": "campaign", "course_style": "marketing course", "artifact": "one small campaign with measurable results"},
    "sales": {"practice_unit": "pitch/playbook", "course_style": "sales training", "artifact": "one sales playbook or case study"},
    "finance_accounting": {"practice_unit": "financial model", "course_style": "professional course", "artifact": "one financial model or audit sample"},
    "healthcare_medicine": {"practice_unit": "clinical case review", "course_style": "continuing education course", "artifact": "one documented clinical case study"},
    "nursing": {"practice_unit": "clinical skill", "course_style": "continuing education course", "artifact": "one documented clinical rotation summary"},
    "pharmacy": {"practice_unit": "case review", "course_style": "continuing education course", "artifact": "one medication management case study"},
    "mechanical_engineering": {"practice_unit": "CAD design", "course_style": "technical course", "artifact": "one design project with drawings"},
    "civil_engineering": {"practice_unit": "design drawing", "course_style": "technical course", "artifact": "one structural or site design project"},
    "electrical_engineering": {"practice_unit": "circuit design", "course_style": "technical course", "artifact": "one working prototype or design writeup"},
    "law_legal": {"practice_unit": "case brief", "course_style": "CLE course", "artifact": "one written case brief or memo"},
    "teaching_academia": {"practice_unit": "lesson plan", "course_style": "pedagogy course", "artifact": "one full course syllabus or workshop"},
    "research_phd": {"practice_unit": "literature review", "course_style": "research methodology course", "artifact": "one short paper or preprint"},
    "biotechnology": {"practice_unit": "lab protocol", "course_style": "lab-focused course", "artifact": "one documented lab protocol or assay result"},
    "forensic_science": {"practice_unit": "case analysis", "course_style": "forensic science course", "artifact": "one mock case analysis with chain-of-custody notes"},
    "government_public_sector": {"practice_unit": "policy brief", "course_style": "public policy course", "artifact": "one policy brief"},
    "mba_business": {"practice_unit": "case study", "course_style": "business course", "artifact": "one strategy case study"},
    "product_management": {"practice_unit": "product spec", "course_style": "product course", "artifact": "one product requirements document"},
    "business_analysis": {"practice_unit": "requirements doc", "course_style": "business analysis course", "artifact": "one requirements/gap-analysis document"},
}

_DEFAULT_VOCAB = {"practice_unit": "practical exercise", "course_style": "course", "artifact": "one concrete work sample"}


class DomainRecommendationEngine:
    """Generates suggestions/courses/projects/roadmap for any missing phrase
    in any domain, without a hardcoded per-skill template table."""

    def generate(
        self,
        missing_phrases: list[str],
        domain_match: DomainMatch,
        resume_context: dict | None = None,
    ) -> dict[str, list[str]]:
        vocab = _DOMAIN_VOCAB.get(domain_match.domain, _DEFAULT_VOCAB)
        ranked = self._rank(missing_phrases)

        suggestions: list[str] = []
        courses: list[str] = []
        projects: list[str] = []
        interview: list[str] = []

        for phrase in ranked:
            label = display_phrase(phrase)
            if not label:
                continue
            suggestions.append(f"Strengthen {label} through a {vocab['practice_unit']} that demonstrates practical understanding.")
            courses.append(f"{label} — {vocab['course_style']}")
            projects.append(f"{vocab['artifact'].capitalize()} focused on {label}")
            interview.append(f"Explain your experience with {label} and how you've applied it.")

        return {
            "improvementSuggestions": self._dedupe(suggestions)[:8],
            "recommendedCourses": self._dedupe(courses)[:8],
            "recommendedProjects": self._dedupe(projects)[:8],
            "interviewPreparation": self._dedupe(interview)[:10],
            "learningRoadmap": self._build_roadmap(ranked, vocab),
        }

    def _rank(self, missing_phrases: list[str]) -> list[str]:
        seen: set[str] = set()
        ranked: list[str] = []
        for phrase in missing_phrases:
            normalized = normalize_phrase(phrase)
            if normalized and normalized not in seen:
                seen.add(normalized)
                ranked.append(normalized)
        return ranked

    def _build_roadmap(self, missing_phrases: list[str], vocab: dict[str, str]) -> list[str]:
        if not missing_phrases:
            return ["Week 1: Review core fundamentals and produce one small work sample."]
        roadmap: list[str] = []
        for index, phrase in enumerate(missing_phrases[:6], start=1):
            label = display_phrase(phrase)
            roadmap.append(f"Week {index}: Build depth in {label} through a {vocab['practice_unit']}.")
        roadmap.append(f"Final Week: Assemble {vocab['artifact']} for your portfolio.")
        return roadmap

    def _dedupe(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(value.strip())
        return unique
