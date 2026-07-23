"""Infers recommended job titles from the detected domain, rather than
returning a fixed "Software Engineer / Backend Developer" fallback."""

from __future__ import annotations

from backend.services.domain_detector import DomainMatch, humanize_domain
from backend.services.skill_extraction import display_phrase

_DOMAIN_JOB_TITLES: dict[str, tuple[str, ...]] = {
    "software_engineering": ("Software Engineer", "Backend Developer", "Full Stack Developer"),
    "ai_ml_data_science": ("Machine Learning Engineer", "Data Scientist", "AI Engineer"),
    "cybersecurity": ("Security Analyst", "Penetration Tester", "SOC Analyst"),
    "ui_ux_design": ("UX Designer", "UI Designer", "Product Designer"),
    "hr": ("HR Generalist", "Talent Acquisition Specialist", "HR Business Partner"),
    "marketing": ("Marketing Specialist", "Digital Marketing Manager", "Brand Manager"),
    "sales": ("Account Executive", "Sales Representative", "Business Development Manager"),
    "finance_accounting": ("Financial Analyst", "Accountant", "Finance Manager"),
    "healthcare_medicine": ("Physician", "Clinical Associate", "Medical Officer"),
    "nursing": ("Registered Nurse", "Nurse Practitioner", "Clinical Nurse"),
    "pharmacy": ("Pharmacist", "Clinical Pharmacist", "Pharmacy Manager"),
    "mechanical_engineering": ("Mechanical Engineer", "Design Engineer", "Manufacturing Engineer"),
    "civil_engineering": ("Civil Engineer", "Structural Engineer", "Site Engineer"),
    "electrical_engineering": ("Electrical Engineer", "Embedded Systems Engineer", "Power Systems Engineer"),
    "law_legal": ("Associate Attorney", "Paralegal", "Legal Counsel"),
    "teaching_academia": ("Assistant Professor", "Associate Professor", "Lecturer"),
    "research_phd": ("Research Associate", "Postdoctoral Researcher", "Research Scientist"),
    "biotechnology": ("Research Scientist", "Lab Technician", "Bioinformatics Analyst"),
    "forensic_science": ("Forensic Analyst", "Crime Scene Investigator", "Forensic Research Associate"),
    "government_public_sector": ("Policy Analyst", "Government Program Officer", "Public Administrator"),
    "mba_business": ("Business Strategy Analyst", "Management Consultant", "Operations Manager"),
    "product_management": ("Product Manager", "Associate Product Manager", "Product Owner"),
    "business_analysis": ("Business Analyst", "Systems Analyst", "Process Analyst"),
}


class JobRecommender:
    def recommend(self, domain_match: DomainMatch, matched_phrases: list[str], missing_phrases: list[str]) -> list[str]:
        titles = list(_DOMAIN_JOB_TITLES.get(domain_match.domain, ()))
        if titles:
            return titles[:4]

        # No declared title list for this domain: build a generic fallback
        # from the domain name itself plus the strongest matched phrase,
        # rather than defaulting to a software title.
        base = humanize_domain(domain_match.domain)
        generic = [f"{base} Specialist", f"{base} Associate"]
        if matched_phrases:
            top_phrase = display_phrase(matched_phrases[0])
            generic.insert(0, f"{top_phrase} Specialist")
        return generic[:4]
