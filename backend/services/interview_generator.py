"""Generates interview preparation questions based on the detected domain
and the specific skills/phrases involved, rather than a fixed software-only
question bank."""

from __future__ import annotations

from backend.services.domain_detector import DomainMatch
from backend.services.skill_extraction import display_phrase

_DOMAIN_QUESTION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "software_engineering": (
        "Walk me through the architecture of your most complex project.",
        "How do you approach debugging a production issue under time pressure?",
        "Describe a time you improved system performance or reliability.",
    ),
    "ai_ml_data_science": (
        "How do you decide which model architecture fits a given problem?",
        "Describe how you evaluate a model beyond accuracy.",
        "Walk me through your feature engineering process on a recent project.",
    ),
    "teaching_academia": (
        "What is your teaching philosophy?",
        "How do you adapt your teaching style for different learners?",
        "Describe your approach to mentoring students or supervising research.",
    ),
    "research_phd": (
        "Walk me through your most significant publication and its impact.",
        "How do you approach peer review and incorporating feedback?",
        "Describe how you secured funding or resources for a research project.",
    ),
    "forensic_science": (
        "Walk me through proper chain-of-custody procedures for evidence.",
        "How do you approach DNA analysis in a contaminated sample scenario?",
        "Describe a time your forensic findings were challenged in court.",
    ),
    "law_legal": (
        "Walk me through how you build a case strategy.",
        "How do you handle conflicting client and ethical obligations?",
        "Describe your approach to legal research on an unfamiliar topic.",
    ),
    "healthcare_medicine": (
        "Walk me through your approach to a complex differential diagnosis.",
        "How do you communicate difficult news to patients or families?",
        "Describe a time you had to make a rapid clinical decision.",
    ),
    "marketing": (
        "Walk me through a campaign you ran end-to-end and its results.",
        "How do you measure ROI on a marketing initiative?",
        "Describe how you adjust strategy based on analytics.",
    ),
    "finance_accounting": (
        "Walk me through how you'd close a month-end reporting cycle.",
        "How do you approach identifying discrepancies in financial statements?",
        "Describe a time you improved a financial process.",
    ),
}

_GENERIC_QUESTIONS = (
    "Walk me through your most significant piece of relevant work.",
    "How do you stay current in your field?",
    "Describe a challenge you faced in this domain and how you resolved it.",
)


class InterviewGenerator:
    def generate(self, domain_match: DomainMatch, matched_phrases: list[str], missing_phrases: list[str]) -> list[str]:
        questions = list(_DOMAIN_QUESTION_TEMPLATES.get(domain_match.domain, _GENERIC_QUESTIONS))
        for phrase in matched_phrases[:4]:
            label = display_phrase(phrase)
            if label:
                questions.append(f"Tell me about your experience with {label}.")
        for phrase in missing_phrases[:3]:
            label = display_phrase(phrase)
            if label:
                questions.append(f"How would you get up to speed on {label} if hired?")
        return self._dedupe(questions)[:10]

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
