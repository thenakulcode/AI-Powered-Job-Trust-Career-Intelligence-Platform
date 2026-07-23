"""Detects the professional domain of a job description or resume.

Domains are described declaratively as (name -> anchor phrases) so adding a
new profession is a data change, not a code change. Detection scores each
domain by anchor-phrase presence in the text (word-boundary matches) and
falls back to "general" when nothing scores meaningfully — the system never
assumes "software" as a default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.services.skill_extraction import normalize_text

# Anchor phrases are intentionally broad and not exhaustive; they exist to
# *classify* the domain, not to *score* skills. Skill scoring still relies on
# the dynamic extractor, so an unlisted domain still works reasonably (it
# just falls back to "general" instead of a wrong specific label).
DOMAIN_ANCHORS: dict[str, tuple[str, ...]] = {
    "software_engineering": (
        "software", "developer", "engineer", "backend", "frontend", "full stack",
        "programming", "code", "api", "microservices", "devops", "cloud",
    ),
    "ai_ml_data_science": (
        "machine learning", "data science", "artificial intelligence", "deep learning",
        "data scientist", "neural network", "nlp", "computer vision", "model training",
    ),
    "cybersecurity": (
        "cybersecurity", "penetration testing", "information security", "vulnerability",
        "threat intelligence", "soc analyst", "incident response", "siem",
    ),
    "ui_ux_design": (
        "ui/ux", "user experience", "user interface", "wireframe", "figma", "usability",
        "interaction design", "product design",
    ),
    "hr": (
        "human resources", "recruitment", "talent acquisition", "onboarding", "payroll",
        "employee relations", "hr generalist", "hr business partner",
    ),
    "marketing": (
        "marketing", "seo", "campaign", "brand management", "social media", "content strategy",
        "google analytics", "digital marketing",
    ),
    "sales": (
        "sales", "quota", "pipeline", "account executive", "business development",
        "lead generation", "crm", "closing deals",
    ),
    "finance_accounting": (
        "finance", "accounting", "financial analyst", "gaap", "audit", "bookkeeping",
        "accounts payable", "budgeting", "forecasting", "cpa",
    ),
    "healthcare_medicine": (
        "clinical", "patient care", "diagnosis", "physician", "healthcare", "hospital",
        "medical", "treatment plan", "doctor",
    ),
    "nursing": (
        "nursing", "registered nurse", "patient monitoring", "clinical rotations", "vitals",
        "nurse practitioner",
    ),
    "pharmacy": (
        "pharmacy", "pharmacist", "dispensing", "pharmaceutical", "medication management",
        "drug interactions",
    ),
    "mechanical_engineering": (
        "mechanical engineering", "cad", "thermodynamics", "manufacturing", "solidworks",
        "hvac", "mechanical design",
    ),
    "civil_engineering": (
        "civil engineering", "structural design", "autocad civil", "construction management",
        "site supervision", "surveying",
    ),
    "electrical_engineering": (
        "electrical engineering", "circuit design", "power systems", "pcb design", "embedded systems",
        "control systems",
    ),
    "law_legal": (
        "law", "legal", "litigation", "contract review", "attorney", "paralegal", "compliance",
        "legal research", "bar exam",
    ),
    "teaching_academia": (
        "professor", "lecturer", "teaching", "curriculum development", "academic", "pedagogy",
        "classroom management", "syllabus",
    ),
    "research_phd": (
        "research", "phd supervision", "publications", "peer review", "grant writing",
        "principal investigator", "research methodology", "thesis",
    ),
    "biotechnology": (
        "biotechnology", "molecular biology", "genomics", "bioinformatics", "cell culture",
        "laboratory", "assay",
    ),
    "forensic_science": (
        "forensic", "crime scene", "dna analysis", "toxicology", "evidence collection",
        "chain of custody", "csi", "fingerprint analysis",
    ),
    "government_public_sector": (
        "government", "public sector", "civil service", "policy", "public administration",
        "regulatory affairs",
    ),
    "mba_business": (
        "mba", "business strategy", "operations management", "consulting", "stakeholder management",
    ),
    "product_management": (
        "product management", "product roadmap", "product owner", "user stories", "product strategy",
        "go-to-market",
    ),
    "business_analysis": (
        "business analyst", "requirements gathering", "stakeholder analysis", "process improvement",
        "gap analysis",
    ),
}


@dataclass
class DomainMatch:
    domain: str
    confidence: float
    matched_anchors: list[str] = field(default_factory=list)


def _contains(text: str, phrase: str) -> bool:
    pattern = re.escape(phrase).replace(r"\ ", r"(?:\s|-|/)+")
    return bool(re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text))


def detect_domain(text: str) -> DomainMatch:
    """Score every declared domain against the text and return the best
    match, or 'general' with zero confidence if nothing meaningfully
    matches. This function performs no software-specific defaulting."""
    lowered = normalize_text(text).lower()
    if not lowered:
        return DomainMatch(domain="general", confidence=0.0)

    scores: dict[str, list[str]] = {}
    for domain, anchors in DOMAIN_ANCHORS.items():
        hits = [a for a in anchors if _contains(lowered, a)]
        if hits:
            scores[domain] = hits

    if not scores:
        return DomainMatch(domain="general", confidence=0.0)

    best_domain = max(scores, key=lambda d: len(scores[d]))
    hits = scores[best_domain]
    confidence = min(1.0, len(hits) / 4.0)
    return DomainMatch(domain=best_domain, confidence=confidence, matched_anchors=hits)


def humanize_domain(domain: str) -> str:
    return domain.replace("_", " ").title() if domain != "general" else "General Professional"
